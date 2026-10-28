"""
错误分类体系

按照 README 第 14.2 节的分类，所有错误都应该携带足够的上下文信息，
不能用裸 Exception 作为用户最终看到的科学错误。
"""

from typing import Optional, Any


class EvaluationError(Exception):
    """评测框架基础错误，所有自定义错误都应继承此类"""

    def __init__(
        self,
        message: str,
        *,
        source_id: Optional[str] = None,
        sample_key: Optional[Any] = None,
        variable: Optional[str] = None,
        path: Optional[str] = None,
        step_id: Optional[str] = None,
        cause: Optional[Exception] = None,
    ):
        """
        Args:
            message: 用户可读的错误描述
            source_id: 数据源ID
            sample_key: 样本标识（起报、时效等）
            variable: 变量名
            path: 文件路径
            step_id: 执行步骤ID
            cause: 底层异常，保留为 __cause__
        """
        super().__init__(message)
        self.source_id = source_id
        self.sample_key = sample_key
        self.variable = variable
        self.path = path
        self.step_id = step_id
        if cause is not None:
            self.__cause__ = cause

    def __str__(self) -> str:
        parts = [super().__str__()]
        if self.source_id:
            parts.append(f"source={self.source_id}")
        if self.variable:
            parts.append(f"variable={self.variable}")
        if self.sample_key:
            parts.append(f"sample={self.sample_key}")
        if self.path:
            parts.append(f"path={self.path}")
        if self.step_id:
            parts.append(f"step={self.step_id}")
        return " | ".join(parts)


class ConfigError(EvaluationError):
    """配置错误：字段缺失、类型错误、未知组件

    默认行为：计划阶段失败，不重试
    """

    pass


class ContractError(EvaluationError):
    """数据契约错误：单位/维度/时间语义不符

    默认行为：当前步骤失败，不重试
    """

    pass


class DiscoveryError(EvaluationError):
    """发现错误：文件缺失、重复匹配、路径歧义

    默认行为：按协议决定失败或记录缺失
    """

    pass


class DecodeError(EvaluationError):
    """解码错误：GRIB/NetCDF/站点文件损坏

    默认行为：可重试一次后失败
    """

    pass


class AlignmentError(EvaluationError):
    """对齐错误：起报、有效时段、网格无法匹配

    默认行为：步骤失败，保留样本诊断
    """

    pass


class MetricError(EvaluationError):
    """指标错误：零分母、参数非法、公式无法计算

    默认行为：生成明确结果状态或失败
    """

    pass


class ResourceError(EvaluationError):
    """资源错误：内存、磁盘、并发资源不足

    默认行为：可调整分块后重试
    """

    pass


class OutputError(EvaluationError):
    """输出错误：写入失败、校验失败、发布失败

    默认行为：运行失败，不发布半成品
    """

    pass
