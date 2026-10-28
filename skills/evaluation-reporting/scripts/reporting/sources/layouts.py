# -*- coding: utf-8 -*-
"""格点数据源的布局声明。

一个数据源与另一个数据源的差别只有三件事：

1. 文件长什么样（目录模板、文件名模式、时效从哪来）；
2. 变量怎么映射（源变量名 -> 标准名）；
3. 单位怎么换算（scale / offset）。

把这三件事写成 ``GriddedLayout``，读取逻辑就只需要 ``io/gridded.py`` 里的一份实现。
新增模型（NJU-Earth / PuYun / W2S / AIFS 等）只需在这里加一条声明。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Optional, Sequence, Tuple

from reporting.core.semantics import DataKind, TemporalKind

GRAVITY = 9.80665


@dataclass(frozen=True)
class VarSpec:
    """标准变量的解码声明。

    Attributes:
        source: 文件里的变量名；``None`` 表示"文件里唯一的变量"（如 cfgrib 按过滤器取出的单变量）。
        unit: 换算后的标准单位。
        scale: 数值换算系数，标准值 = 源值 * scale + offset。
        offset: 数值换算偏移。
        source_unit: 换算前的单位，写入 ``attrs["original_units"]``。
        temporal_kind: 该变量的时间语义。
    """

    source: Optional[str] = None
    unit: str = "unknown"
    scale: float = 1.0
    offset: float = 0.0
    source_unit: str = ""
    temporal_kind: TemporalKind = TemporalKind.INSTANTANEOUS


@dataclass(frozen=True)
class PatternSpec:
    """文件名模式。

    ``init`` 命名组缺省时，时刻取所在目录代表的时刻（如 ``root/YYYYMMDD/001.nc``）；
    ``lead`` 命名组在 ``lead_from="filename"`` 的预报布局里必需。
    分组名 ``group`` 可以来自正则，也可以由 ``group`` 字段直接声明。
    """

    regex: str
    group: Optional[str] = None


@dataclass(frozen=True)
class GriddedLayout:
    """一个格点数据源的完整布局声明。"""

    name: str
    patterns: Tuple[PatternSpec, ...]
    variables: Dict[str, VarSpec] = field(default_factory=dict)
    kind: DataKind = DataKind.GRIDDED_FORECAST
    engine: str = "netcdf"
    #: 默认后端打开失败时的回退后端（Windows 非 ASCII 路径下 netCDF4 会报 ENOENT）
    engine_fallback: Optional[str] = "h5netcdf"
    root_template: str = "{root}/{init:%Y%m%d}"
    init_format: str = "%Y%m%d%H"
    time_kind: str = "forecast"
    lead_from: str = "filename"
    step_hours: float = 6.0
    member_dim: Optional[str] = None
    ensure_member_dim: bool = False
    #: 成员子目录通配（如 "member_*"）；设置后成员从子目录名解析
    member_glob: Optional[str] = None
    #: 成员维度名（成员来自目录结构时使用）
    member_coord: str = "member"
    groups: Dict[str, str] = field(default_factory=dict)
    grib_filters: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    squeeze_dims: Tuple[str, ...] = ()
    transpose_dims: Tuple[str, ...] = ()
    case_insensitive: bool = False

    @property
    def per_variable_open(self) -> bool:
        """是否需要按变量分别打开文件（cfgrib 需要按 filter_by_keys 过滤）。"""
        return bool(self.grib_filters)

    def groups_for(self, variables: Sequence[str]) -> set:
        """请求变量涉及的文件分组。"""
        return {
            self.groups[name]
            for name in (str(variable).lower() for variable in variables)
            if name in self.groups
        }

    def spec_for(self, standard_name: str) -> Optional[VarSpec]:
        return self.variables.get(str(standard_name))

    def standard_name_for(self, file_variable: str) -> Optional[str]:
        """源变量名 -> 标准名；未声明时返回 None（由调用方按小写兜底）。"""
        for standard_name, spec in self.variables.items():
            if spec.source is None:
                continue
            if spec.source == file_variable:
                return standard_name
            if self.case_insensitive and spec.source.lower() == str(file_variable).lower():
                return standard_name
        return None

    def with_step_hours(self, step_hours: float) -> "GriddedLayout":
        return replace(self, step_hours=float(step_hours))


FUXI_LAYOUT = GriddedLayout(
    name="fuxi",
    patterns=(PatternSpec(r"(?P<lead_index>\d{3})\.nc"),),
    variables={
        "tp": VarSpec(source="tp", unit="mm", temporal_kind=TemporalKind.INTERVAL_ACCUMULATION),
        "t2m": VarSpec(source="t2m", unit="K"),
        "z500": VarSpec(source="z500", unit="m", scale=1.0 / GRAVITY, source_unit="m^2/s^2"),
        "q700": VarSpec(source="q700", unit="g/kg", source_unit="g/kg"),
        "q2m": VarSpec(source="q2m", unit="g/kg", source_unit="g/kg"),
    },
    kind=DataKind.GRIDDED_FORECAST,
    lead_from="index",
    step_hours=6.0,
    case_insensitive=True,
)

#: FuXi 集合预报：root/YYYYMMDD/member_*/001.nc…，时效按成员目录内文件序号 * step
FUXI_ENS_LAYOUT = GriddedLayout(
    name="fuxi_ens",
    patterns=(PatternSpec(r"(?P<lead_index>\d{3})\.nc"),),
    variables={
        "tp": VarSpec(source="tp", unit="mm", temporal_kind=TemporalKind.INTERVAL_ACCUMULATION),
        "t2m": VarSpec(source="t2m", unit="K"),
        "z500": VarSpec(source="z500", unit="m", scale=1.0 / GRAVITY, source_unit="m^2/s^2"),
        "q700": VarSpec(source="q700", unit="g/kg", source_unit="g/kg"),
        "q2m": VarSpec(source="q2m", unit="g/kg", source_unit="g/kg"),
    },
    kind=DataKind.GRIDDED_FORECAST,
    lead_from="index",
    step_hours=6.0,
    case_insensitive=True,
    member_glob="member_*",
    member_dim="member",
)

#: xu 复刻口径的要素表：标准名与 xu 的 ``--vars`` 逐字一致，值是**报告单位**。
#:
#: 与 ``FUXI_LAYOUT`` 只有一处实质差别——z500 **不除 g**，直接报 m²/s²
#: （xu 的 z500 单位就是 ``m2 s-2``）。q 源文件已是 g/kg，两边都不换算：
#: xu 内部转 kg/kg、报告层再 ×1000，两步相消，落点仍是 g/kg。
FUXI_PHYS_VARIABLES: Dict[str, VarSpec] = {
    "z500": VarSpec(source="z500", unit="m^2/s^2", source_unit="m^2/s^2"),
    "t700": VarSpec(source="t700", unit="K"),
    "t850": VarSpec(source="t850", unit="K"),
    "q700": VarSpec(source="q700", unit="g/kg", source_unit="g/kg"),
    "q2m": VarSpec(source="q2m", unit="g/kg", source_unit="g/kg"),
    "u200": VarSpec(source="u200", unit="m/s"),
    "v200": VarSpec(source="v200", unit="m/s"),
    "u850": VarSpec(source="u850", unit="m/s"),
    "v850": VarSpec(source="v850", unit="m/s"),
    "u10m": VarSpec(source="u10m", unit="m/s"),
    "v10m": VarSpec(source="v10m", unit="m/s"),
    "t2m": VarSpec(source="t2m", unit="K"),
    "msl": VarSpec(source="msl", unit="Pa"),
    "tp": VarSpec(source="tp", unit="mm", temporal_kind=TemporalKind.INTERVAL_ACCUMULATION),
}

#: FuXi 单卡（xu ``single_fuxi.sh`` 口径）
FUXI_PHYS_LAYOUT = GriddedLayout(
    name="fuxi_phys",
    patterns=FUXI_LAYOUT.patterns,
    variables=dict(FUXI_PHYS_VARIABLES),
    kind=DataKind.GRIDDED_FORECAST,
    lead_from="index",
    step_hours=6.0,
    case_insensitive=True,
)

#: FuXi 集合（xu ``ensemble_fuxi.sh`` 口径）
FUXI_ENS_PHYS_LAYOUT = GriddedLayout(
    name="fuxi_ens_phys",
    patterns=FUXI_ENS_LAYOUT.patterns,
    variables=dict(FUXI_PHYS_VARIABLES),
    kind=DataKind.GRIDDED_FORECAST,
    lead_from="index",
    step_hours=6.0,
    case_insensitive=True,
    member_glob="member_*",
    member_dim="member",
)

FENGQING_LAYOUT = GriddedLayout(
    name="fengqing",
    patterns=(
        PatternSpec(
            r"Fengqing_1\.0_GLB_(?P<group>PLEVELS|SURFACE)_OP25_6HOR_ENS_FCST_"
            r"(?P<init>\d{10})_(?P<lead>\d{3})\.nc"
        ),
    ),
    variables={
        "z500": VarSpec(source="Z500", unit="m", scale=1.0 / GRAVITY, source_unit="m^2/s^2"),
        "tp": VarSpec(source="TP", unit="mm", temporal_kind=TemporalKind.INTERVAL_ACCUMULATION),
        "t2m": VarSpec(source="T2M", unit="K"),
        "msl": VarSpec(source="MSL", unit="Pa"),
        "u10": VarSpec(source="U10", unit="m/s"),
        "v10": VarSpec(source="V10", unit="m/s"),
    },
    kind=DataKind.GRIDDED_FORECAST,
    member_dim="member",
    ensure_member_dim=True,
    groups={
        "z500": "PLEVELS",
        "z": "PLEVELS",
        "gh": "PLEVELS",
        "t": "PLEVELS",
        "u": "PLEVELS",
        "v": "PLEVELS",
        "q": "PLEVELS",
        "tp": "SURFACE",
        "t2m": "SURFACE",
        "msl": "SURFACE",
        "u10": "SURFACE",
        "v10": "SURFACE",
    },
    squeeze_dims=("level", "time", "dtime"),
    transpose_dims=("member", "lat", "lon"),
)

#: 风清单卡（xu ``single_fengqing.sh`` 口径）。
#:
#: 与 ``FENGQING_LAYOUT`` 的差别有两点：z500 不除 g（m²/s²）；地面风的标准名改成
#: ``u10m``/``v10m``（xu 与 ERA5 zarr 都用这个名，两边才配得上对）。
#: 文件里的拼写是 ``U10``/``V10``，由 ``source`` 负责翻译。
#: 注意不能同时留 ``u10`` 与 ``u10m``——同一个 ``source`` 只认 ``variables`` 里
#: 字典序靠前的那一个，两个并存会让 ``u10m`` 永远解析不到。
FENGQING_PHYS_LAYOUT = GriddedLayout(
    name="fengqing_phys",
    patterns=FENGQING_LAYOUT.patterns,
    variables={
        "z500": VarSpec(source="Z500", unit="m^2/s^2", source_unit="m^2/s^2"),
        "t700": VarSpec(source="T700", unit="K"),
        "t850": VarSpec(source="T850", unit="K"),
        "q700": VarSpec(source="Q700", unit="g/kg", source_unit="g/kg"),
        "q2m": VarSpec(source="Q2M", unit="g/kg", source_unit="g/kg"),
        "u200": VarSpec(source="U200", unit="m/s"),
        "v200": VarSpec(source="V200", unit="m/s"),
        "u850": VarSpec(source="U850", unit="m/s"),
        "v850": VarSpec(source="V850", unit="m/s"),
        "u10m": VarSpec(source="U10", unit="m/s"),
        "v10m": VarSpec(source="V10", unit="m/s"),
        "t2m": VarSpec(source="T2M", unit="K"),
        "msl": VarSpec(source="MSL", unit="Pa"),
        "tp": VarSpec(source="TP", unit="mm", temporal_kind=TemporalKind.INTERVAL_ACCUMULATION),
    },
    kind=DataKind.GRIDDED_FORECAST,
    member_dim="member",
    ensure_member_dim=True,
    groups={
        "z500": "PLEVELS",
        "z": "PLEVELS",
        "gh": "PLEVELS",
        "t": "PLEVELS",
        "u": "PLEVELS",
        "v": "PLEVELS",
        "q": "PLEVELS",
        "t700": "PLEVELS",
        "t850": "PLEVELS",
        "q700": "PLEVELS",
        "u200": "PLEVELS",
        "v200": "PLEVELS",
        "u850": "PLEVELS",
        "v850": "PLEVELS",
        "tp": "SURFACE",
        "t2m": "SURFACE",
        "msl": "SURFACE",
        "q2m": "SURFACE",
        "u10m": "SURFACE",
        "v10m": "SURFACE",
    },
    squeeze_dims=("level", "time", "dtime"),
    transpose_dims=("member", "lat", "lon"),
)

CRA_LAYOUT = GriddedLayout(
    name="cra",
    patterns=(
        PatternSpec(r"ART_ATM_GLB_0P25_6HOR_ANAL_(?P<init>\d{10})\.grib2", group="ATM"),
        PatternSpec(
            r"CRA40LAND_SURFACE_(?P<init>\d{10})_GLB_0P25_HOUR_V1_0_0\.grib",
            group="SURFACE",
        ),
    ),
    variables={
        "z500": VarSpec(unit="m"),
        "gh": VarSpec(unit="m"),
        "t2m": VarSpec(unit="K"),
        "msl": VarSpec(unit="Pa"),
        "u10": VarSpec(unit="m/s"),
        "v10": VarSpec(unit="m/s"),
        "tp": VarSpec(unit="mm", temporal_kind=TemporalKind.INTERVAL_ACCUMULATION),
    },
    kind=DataKind.GRIDDED_OBSERVATION,
    engine="cfgrib",
    time_kind="observation",
    groups={
        "z500": "ATM",
        "z": "ATM",
        "gh": "ATM",
        "t": "ATM",
        "u": "ATM",
        "v": "ATM",
        "q": "ATM",
        "t2m": "SURFACE",
        "msl": "SURFACE",
        "u10": "SURFACE",
        "v10": "SURFACE",
        "tp": "SURFACE",
    },
    grib_filters={
        "z500": {"shortName": "gh", "typeOfLevel": "isobaricInhPa", "level": 500},
        "gh": {"shortName": "gh", "typeOfLevel": "isobaricInhPa", "level": 500},
        "t2m": {"shortName": "2t", "typeOfLevel": "heightAboveGround", "level": 2},
        "msl": {"shortName": "msl", "typeOfLevel": "meanSea"},
        "u10": {"shortName": "10u", "typeOfLevel": "heightAboveGround", "level": 10},
        "v10": {"shortName": "10v", "typeOfLevel": "heightAboveGround", "level": 10},
        "tp": {"shortName": "tp", "stepType": "accum"},
    },
)

#: ERA5 再分析实况（zarr store，见 ``io/era5_zarr_reader.py``）。
#:
#: 两个 store 分工由 ``groups`` 声明：气压层用 ``pl``、地面用 ``sfc``。
#: store 里**没有层次轴**，层次编码在通道名里（``z_500`` / ``u_200``），
#: 所以 ``source`` 写通道名，配置侧仍用标准名 ``z500``。
#:
#: 单位口径对齐参考实现：q 报 g/kg（源是 kg/kg，×1000）、z500 报 m²/s²（不除 g）、
#: tp 源是 m，×1000 换成 mm。换算烘焙在这里，下游全程是报告单位。
ERA5_ZARR_LAYOUT = GriddedLayout(
    name="era5_zarr",
    patterns=(),
    variables={
        "z500": VarSpec(source="z_500", unit="m^2/s^2"),
        "t700": VarSpec(source="t_700", unit="K"),
        "t850": VarSpec(source="t_850", unit="K"),
        "q700": VarSpec(source="q_700", unit="g/kg", scale=1000.0, source_unit="kg/kg"),
        "u200": VarSpec(source="u_200", unit="m/s"),
        "v200": VarSpec(source="v_200", unit="m/s"),
        "u850": VarSpec(source="u_850", unit="m/s"),
        "v850": VarSpec(source="v_850", unit="m/s"),
        "msl": VarSpec(source="msl", unit="Pa"),
        "t2m": VarSpec(source="t2m", unit="K"),
        "d2m": VarSpec(source="d2m", unit="K"),
        "q2m": VarSpec(source="q2m", unit="g/kg", scale=1000.0, source_unit="kg/kg"),
        "u10m": VarSpec(source="u10m", unit="m/s"),
        "v10m": VarSpec(source="v10m", unit="m/s"),
        # store 自带 10m 风速通道。不声明也能读到（``era5_zarr_reader`` 在
        # ``spec is None`` 时把单位兜底成 "unknown"），但那样长表的 unit 列就空着；
        # 声明只为拿单位，数值取的是 store 里那一份，与现状一致。
        # ws850 / ws200 store 里没有，仍由 ``matcher`` 从分量现合成。
        "ws10m": VarSpec(source="ws10m", unit="m/s"),
        "tp": VarSpec(
            source="tp",
            unit="mm",
            scale=1000.0,
            source_unit="m",
            temporal_kind=TemporalKind.INTERVAL_ACCUMULATION,
        ),
    },
    kind=DataKind.GRIDDED_OBSERVATION,
    engine="zarr",
    time_kind="observation",
    groups={
        "z500": "pl",
        "t700": "pl",
        "t850": "pl",
        "q700": "pl",
        "u200": "pl",
        "v200": "pl",
        "u850": "pl",
        "v850": "pl",
        "msl": "sfc",
        "t2m": "sfc",
        "d2m": "sfc",
        "q2m": "sfc",
        "u10m": "sfc",
        "v10m": "sfc",
        "tp": "sfc",
    },
)

LAYOUTS: Dict[str, GriddedLayout] = {
    FUXI_LAYOUT.name: FUXI_LAYOUT,
    FUXI_PHYS_LAYOUT.name: FUXI_PHYS_LAYOUT,
    FUXI_ENS_LAYOUT.name: FUXI_ENS_LAYOUT,
    FUXI_ENS_PHYS_LAYOUT.name: FUXI_ENS_PHYS_LAYOUT,
    FENGQING_LAYOUT.name: FENGQING_LAYOUT,
    FENGQING_PHYS_LAYOUT.name: FENGQING_PHYS_LAYOUT,
    CRA_LAYOUT.name: CRA_LAYOUT,
    ERA5_ZARR_LAYOUT.name: ERA5_ZARR_LAYOUT,
}
