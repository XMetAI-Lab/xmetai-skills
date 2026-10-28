"""
核心数据契约

定义 DataRequest、DataIndex、DataBundle、EvaluationBatch、MetricResult
等核心对象，按照 README 第 13.1 节的最小字段要求实现。
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from enum import Enum

import xarray as xr

from reporting.core.semantics import (
    TemporalKind,
    DataKind,
)


@dataclass
class DataRequest:
    """
    表示"希望读取什么"，不表示"已经读到了什么"

    按照 README 第 13.1 节：DataRequest 不允许放 mean()、插值算法或指标名称。
    指标需求由 Planner 转换为数据请求。
    """

    source_id: str  # 数据源配置ID
    variables: List[str]  # 标准变量名列表
    init_times: Optional[List[datetime]] = None  # 起报时间选择
    lead_times: Optional[List[int]] = None  # 时效选择（小时），使用时长而非文件序号
    members: Optional[List[Any]] = None  # 成员选择；None表示按产品策略选择
    levels: Optional[List[Any]] = None  # 层次选择，可为空
    region: Optional[Dict[str, Any]] = None  # 目标区域或裁剪规则

    def __post_init__(self):
        """基础校验"""
        if not self.source_id:
            raise ValueError("source_id is required")
        if not self.variables:
            raise ValueError("variables is required")


@dataclass
class DataIndex:
    """
    表示 Catalog 的发现结果

    按照 README 第 13.1 节：Catalog 发现到多个可能文件时，状态必须是 ambiguous
    并阻断执行，不能按文件名排序后静默选一个。

    Change 5 简化版本：直接存储文件路径列表。
    """

    source_id: str  # 数据源ID
    available: List[Any]  # 可用文件路径列表
    ambiguous: List[Any] = field(default_factory=list)  # 有歧义的文件


@dataclass
class SemanticMetadata:
    """数据语义元数据"""

    units: Dict[str, str]  # 变量 → 单位
    temporal_kind: Union[TemporalKind, Dict[str, TemporalKind]]  # 所有变量的时间语义或每个变量独立
    grid_type: Optional[str] = None  # regular_ll / rotated / stations / unstructured
    member_count: Optional[int] = None  # 成员数
    coordinate_names: Optional[Dict[str, str]] = None  # 标准名 → 源坐标名
    timezone: str = "UTC"  # 时区


@dataclass
class Provenance:
    """数据来源记录"""

    input_files: List[str]  # 实际读取的文件列表
    reader_id: str  # Reader ID
    reader_version: str  # Reader版本
    transform_chain: List[str] = field(default_factory=list)  # 转换步骤记录
    source_variables: Optional[Dict[str, str]] = None  # 标准名 → 源变量名


@dataclass
class DataBundle:
    """
    表示已读取并经过最低限度结构标准化的数据

    按照 README 第 13.1 节：Reader 不得返回只在某个调用者内部有效的裸数组。
    payload 必须能通过数据契约校验。
    """

    payload: Union[xr.Dataset, xr.DataArray]  # 标准化数据
    kind: DataKind  # forecast / observation / reference / derived
    source_id: str  # 数据源ID
    standard_vars: Dict[str, str]  # 源变量 → 标准变量映射
    semantic: SemanticMetadata  # 单位、时间语义、网格和成员语义
    provenance: Provenance  # 输入文件、读取器版本、转换记录
    quality: Optional[Dict[str, Any]] = None  # 缺测、异常值、质量标记

    def __post_init__(self):
        """基础校验"""
        if not isinstance(self.payload, (xr.Dataset, xr.DataArray)):
            raise TypeError("payload must be xarray.Dataset or DataArray")


@dataclass
class EvaluationBatch:
    """
    表示已经可以交给指标计算的一批配对数据

    按照 README 第 13.1 节：Metric 不负责构造 EvaluationBatch。
    如果传入的数据没有 alignment 或 valid_mask，应在边界校验阶段失败。

    sample_dim 是"配对样本单位"维度：格点评测用 lat/lon（或展平后的 sample），
    站点评测用 station，台风路径评测用 storm。指标据此判断可以沿哪些维度聚合。
    """

    forecast: Union[DataBundle, xr.DataArray]  # 预报 DataBundle 或标准 DataArray
    observation: Union[DataBundle, xr.DataArray]  # 观测 DataBundle 或标准 DataArray
    sample_keys: List[Dict[str, Any]]  # 每个样本的稳定标识
    valid_mask: xr.DataArray  # 预报、观测和参考共同有效掩码
    members: Optional[xr.DataArray] = None  # 集合成员（概率指标用）；确定性场景为 None
    reference: Optional[Union[DataBundle, xr.DataArray]] = None  # 气候态/概率/投影基底等
    weights: Optional[xr.DataArray] = None  # 空间/站点/样本权重
    sample_dim: str = "sample"  # 样本单位维度：sample / station / storm / time
    alignment: Optional[Dict[str, Any]] = None  # 时间、空间、变量和单位对齐记录
    protocol_id: Optional[str] = None  # 验证协议标识

    def __post_init__(self):
        """基础校验"""
        if self.alignment is None:
            raise ValueError("alignment record is required")
        if not isinstance(self.valid_mask, xr.DataArray):
            raise TypeError("valid_mask must be xarray.DataArray")


class ResultStatus(Enum):
    """结果状态"""

    SUCCESS = "success"  # 有效结果
    PARTIAL = "partial"  # 有效但部分样本/成员缺失，符合协议
    NOT_APPLICABLE = "not_applicable"  # 产品类型不支持该指标
    NO_VALID_DATA = "no_valid_data"  # 请求存在但没有有效配对
    UNDEFINED = "undefined"  # 数学定义遇到零分母/零方差
    FAILED = "failed"  # 实现或输入错误


@dataclass
class MetricResult:
    """
    表示一项指标的最终结果

    按照 README 第 13.1 节：value 不建议把大型空间场直接塞入 JSON；
    ReportWriter 应将其写入 NetCDF/Zarr 等诊断产物，并在 MetricResult 中保存引用和摘要。

    按照 README 第 16.2 节：value = NaN 不是状态协议。结果必须同时有 status。
    """

    metric_name: str  # 稳定指标名
    metric_version: str  # 公式/实现版本
    value: Union[float, str, Dict[str, Any]]  # 标量、诊断数据引用或摘要
    status: ResultStatus  # 结果状态
    n_requested: int  # 请求样本或评分点数量
    n_valid: int  # 实际有效数量
    coordinates: Optional[Dict[str, Any]] = None  # threshold、lead、region等结果坐标
    weights_sum: Optional[float] = None  # 有效权重和（适用时）
    aggregation: Optional[str] = None  # 统计口径与聚合方式
    protocol_id: Optional[str] = None  # 验证协议ID
    provenance: Optional[Dict[str, Any]] = None  # 数据和参考来源
    warnings: List[str] = field(default_factory=list)  # 非致命问题列表
    unit: Optional[str] = None  # 结果单位（无量纲指标写 "1"）
    product_kind: Optional[str] = None  # 产品类型：deterministic/categorical/probabilistic/ensemble/index
    # 逐点曲线类诊断量（目前只有功率谱）。**不能塞进 value**：value 里每个分组键
    # 都会在 build_tables 里展开成明细行，720 个波数 × 3 个字段 = 2165 行/样本，
    # 全年段 20880 个样本就是四千多万行。曲线在这里原样带着（numpy 数组），
    # 由 build_tables 收进 ResultTables.curves，writer 直接消费。
    # 形状约定：{"kind": ..., 其余键由指标自定}。
    curve: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """基础校验"""
        if not self.metric_name:
            raise ValueError("metric_name is required")
        if not self.metric_version:
            raise ValueError("metric_version is required")
        if self.n_valid < 0 or self.n_requested < 0:
            raise ValueError("n_valid and n_requested must be non-negative")
        if self.n_valid > self.n_requested:
            raise ValueError("n_valid cannot exceed n_requested")


@dataclass
class ResultBundle:
    """多个指标结果的集合"""

    run_id: str
    results: List[MetricResult]
    manifest: Dict[str, Any]  # 输入、协议、版本、状态、产物索引
    resolved_config: Dict[str, Any]  # 完整解析后配置
