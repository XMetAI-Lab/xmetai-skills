#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""产物目录 → 报告：认能力 + 派发。

命令行入口 ``skills/evaluation-reporting/scripts/render_report.py``；本模块只提供
实现，不自带 CLI。

**能力身份从产物自己认，不靠目录名**：``manifest.json`` 里的
``resolved_config.pipeline`` 是权威判据（``runner`` 无条件写它）。目录名与
``run_id`` 都不可靠——配置名（``weather_rmse_single_fuxi``）与流程名
（``weather_field_scores``）本来就不绑定。

没有 ``resolved_config`` 的老产物走退路判据：``protocol_id`` + ``manifest["metrics"]``
的组合。注意 ``manifest["metrics"]`` 是**原始**指标名（``activity``），
不是长表里的 ``activity_ratio``。

认得出的能力如果还没有报告模板，**明确报错**，不静默回落到别的模板上——
拿错模板比报错危险得多。

能力先归到**报告家族**（``IMPLEMENTED`` 的值），再按家族派发：同一家族的流程
写出的产物表结构相同，共用一套模板与图表。目前三个家族——``field``
（连续场，吃 ``scores.csv``）、``ts``（降水分类检验，吃
``diagnostics/categorical_wide.csv``）与 ``typhoon``（场次曲线与元数据）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Sequence

from reporting.reports.field_report import manifest_info_from

#: 已经有报告模板的能力：流程名 -> 所属**报告家族**。
#: 同一家族的流程共用一套模板与图表，因为它们写出的产物表结构相同
#: （三条 TS 流程都写 ``diagnostics/categorical_wide.csv``）。
IMPLEMENTED: Dict[str, str] = {
    "weather_field_scores": "field",
    "weather_ts_det": "ts",
    "weather_ts_ens": "ts",
    "fdp_precip_ts": "ts",
    "weather_typhoon_det": "typhoon",
    "weather_typhoon_ens": "typhoon",
}

#: 家族 -> 一句话说明，只用于日志与报错信息。
FAMILY_LABELS: Dict[str, str] = {
    "field": "确定性连续场检验（RMSE / ACC / 活跃度 / 纬向谱）",
    "ts": "站点降水分类检验（TS / POD / FAR / 频率偏差）",
    "typhoon": "台风路径、分量与强度误差（逐指标逐时效严格配对）",
}

#: 退路判据 ``(流程名, protocol_id, 指标名集合)``，只在 manifest 里没有
#: ``resolved_config.pipeline`` 时启用；指标集合要求**完全相等**，
#: 多一个少一个都不算（``weather_ens_field_scores`` 就是多一个 ``crps``）。
_FALLBACK_RULES = (
    (
        "weather_field_scores",
        "grid_valid_time",
        frozenset({"rmse", "acc", "activity", "zonal_spectrum"}),
    ),
    # det / ens / fdp 三类 TS 产物在这组判据下区分不开，但三者同属 "ts" 家族、
    # 共用一套模板，所以落到哪个名字上都不影响出什么报告。
    ("weather_ts_det", "station_valid_time", frozenset({"ts"})),
)

CATEGORICAL_WIDE_RELATIVE_PATH = Path("diagnostics") / "categorical_wide.csv"

#: 集合产物的概率评分宽表（``weather_ts_ens_prob`` 那一段写出）。确定性产物没有
#: 这个文件，对比模型旁边也未必有——两处都是**有就用、没有就跳过**，不是错误。
PROBABILITY_WIDE_RELATIVE_PATH = Path("diagnostics") / "probability_wide.csv"


def parse_compare_specs(specs: Optional[Sequence[str]]) -> Dict[str, Path]:
    """把 ``["FuXi=a.csv", "AIFS=b.csv"]`` 解析成 ``{名字: 路径}``。

    对比模型**只能是裸结果表**，不能是产物目录——它们没有 manifest，
    认不出能力，名字也没法从 ``run_id`` 推断（``ts_multi_*`` 那几个目录就是这种）。
    名字由调用方显式给出，不从目录名猜：这个仓库的规矩是身份不靠目录名。

    Raises:
        ValueError: 条目没有 ``=``，或文件不存在。
    """
    parsed: Dict[str, Path] = {}
    for spec in specs or []:
        name, separator, raw_path = str(spec).partition("=")
        name, raw_path = name.strip(), raw_path.strip()
        if not separator or not name or not raw_path:
            raise ValueError(
                f"--compare 的写法是「名字=CSV路径」，收到的是 {spec!r}。"
                f"例：--compare \"FuXi=evaluation_results/ts_multi_fuxi/ts_fuxi_2025.csv\""
            )
        path = Path(raw_path)
        if not path.is_file():
            raise ValueError(f"--compare 里的 {name} 指向的 {path} 不存在")
        parsed[name] = path
    return parsed


def load_manifest(output_dir: Path) -> Dict[str, object]:
    """读产物目录里的 ``manifest.json``。

    Raises:
        ValueError: 文件不存在或不是合法 JSON。
    """
    path = Path(output_dir) / "manifest.json"
    if not path.is_file():
        raise ValueError(
            f"{path} 不存在：这里看着不像评测产物目录。"
            f"该给的是 ``--config`` 的 output_dir（里面有 scores.csv 和 manifest.json）。"
        )
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path} 不是合法 JSON：{error}") from error


def detect_capability(output_dir: Path) -> str:
    """认出产物目录属于哪条流程。

    Raises:
        ValueError: 目录里没有 manifest，或认不出 / 认得但没有报告模板。
    """
    manifest = load_manifest(Path(output_dir))
    resolved = manifest.get("resolved_config")
    pipeline = resolved.get("pipeline") if isinstance(resolved, dict) else None
    if pipeline:
        if str(pipeline) in IMPLEMENTED:
            return str(pipeline)
        raise ValueError(
            f"流程 '{pipeline}' 的报告模板还没做"
            f"（已有模板的流程：{'、'.join(sorted(IMPLEMENTED))}）。"
            f"产物本身是完整的，只是没有对应的出图与报告代码。"
        )

    protocol = str(manifest.get("protocol_id") or "")
    metrics = frozenset(str(name) for name in (manifest.get("metrics") or []))
    for name, want_protocol, want_metrics in _FALLBACK_RULES:
        if protocol == want_protocol and metrics == want_metrics:
            return name
    raise ValueError(
        f"认不出这个产物属于哪条流程：manifest 里既没有 `resolved_config.pipeline`，"
        f"`protocol_id={protocol!r}` + `metrics={sorted(metrics)}` 也不匹配任何已知判据。"
        f"已有模板的流程：{'、'.join(sorted(IMPLEMENTED))}。"
    )


def _default_output_dir(output_dir: Path, info: Dict[str, object]) -> Path:
    run_id = str(info.get("run_id") or Path(output_dir).name)
    return Path("reports") / run_id


def build_report(
    output_dir: Path,
    *,
    out_dir: Optional[Path] = None,
    model_name: Optional[str] = None,
    spectrum_variable: Optional[str] = None,
    spectrum_init: Optional[str] = None,
    change_description: Optional[str] = None,
    compare: Optional[Sequence[str]] = None,
    members: Optional[int] = None,
    primary_model: Optional[str] = None,
    previous_model: Optional[str] = None,
    primary_leads: Optional[Sequence[float]] = None,
    exploratory_leads: Optional[Sequence[float]] = None,
    leads: Optional[Sequence[float]] = None,
    fixed_leads: Optional[Sequence[float]] = None,
    n_bootstrap: Optional[int] = None,
    seed: Optional[int] = None,
    small_cases: Optional[int] = None,
    small_storms: Optional[int] = None,
    forecast_type: Optional[str] = None,
    verify_legacy_members: bool = False,
    command: Optional[str] = None,
) -> Dict[str, Path]:
    """按产物目录里的能力出图并写报告，返回 ``{名称: 路径}``。

    Args:
        output_dir: ``--config`` 里的 output_dir，里面应有 ``manifest.json``
            以及该能力对应的结果表——连续场是 ``scores.csv``，
            TS 系列是 ``diagnostics/categorical_wide.csv``，台风是场次曲线与元数据。
        out_dir: 报告输出目录；默认 ``reports/<run_id>``（相对当前目录）。
        model_name: 报告标题里的模型名；默认取 ``run_id``。**不要用 manifest 的
            ``forecast_source``**——那是 reader 类型（如 ``"fuxi"``），不是模型名，
            拿它当标题会张冠李戴。
        compare: TS 对比模型为 ``["名字=CSV路径", …]``；
            台风对比模型为 ``["名字=产物目录", …]``，读取目录中的场次曲线与元数据。
            给出来的模型会和主模型一起进箱线图、差值图和对比表。每个对比路径还会
            去**它的同目录**找概率评分表——``probability_wide.csv`` 与老命名的
            ``aroc_bss*.csv`` 都认（见 :func:`_probability_sibling`）；找得到就
            一起进概率评分一节，找不到就只进 TS 那几节（老归档给的是裸 CSV，
            旁边没有这张表）。
        members: 集合成员数（``--members``）。集合产物的 ``manifest.json`` **不写**
            这个字段，所以只能从命令行给；不给则报告里凡是要写成员数的句子里
            省略数字。只在集合产物（``weather_ts_ens``）上有意义。
        spectrum_variable / spectrum_init: 只在连续场链路、且要点名看**某一个起报**
            的谱曲线时才给（读 ``diagnostics/spectrum_by_init.csv``）。
            ``spectrum_init`` 省略时取最后一个起报。
        leads / fixed_leads / n_bootstrap / seed / small_cases / small_storms /
            forecast_type / verify_legacy_members: 只对台风报告生效，原样转给
            :func:`build_typhoon_report`。``None`` 表示使用台风默认值。
        command: 报告「复现与产物」一节记录的命令行；统一入口应传入真实 CLI 命令。

    Raises:
        ValueError: 结果表不存在、认不出能力，或 ``compare`` 不合法。
        ImportError: 没装 matplotlib（``pip install -e .[viz]``）。
    """
    output_dir = Path(output_dir)
    capability = detect_capability(output_dir)
    family = IMPLEMENTED[capability]
    info = manifest_info_from(load_manifest(output_dir))
    target = Path(out_dir) if out_dir is not None else _default_output_dir(output_dir, info)
    full_model_name = model_name or str(info.get("run_id") or output_dir.name)

    if family == "typhoon":
        from reporting.reports.typhoon_report import (
            build_typhoon_report,
        )

        if any(value is not None for value in (
            spectrum_variable, spectrum_init, change_description, members
        )):
            raise ValueError("台风入口不接受谱、改动说明或成员数覆盖；成员数读取实际产物")
        models = [(full_model_name, output_dir)]
        for value in compare or []:
            name, separator, path = value.partition("=")
            if not separator or not name.strip() or not path.strip():
                raise ValueError("台风 --compare 必须为 NAME=产物目录")
            models.append((name.strip(), Path(path.strip())))
        expected = "ens" if capability == "weather_typhoon_ens" else "det"
        if forecast_type is not None and forecast_type != expected:
            raise ValueError(
                f"--forecast-type={forecast_type!r} 与产物能力 {capability!r} 冲突；"
                f"该产物应为 {expected!r}"
            )
        return build_typhoon_report(
            models, target,
            leads=leads, fixed_leads=fixed_leads,
            n_bootstrap=2000 if n_bootstrap is None else n_bootstrap,
            seed=20250910 if seed is None else seed,
            small_cases=10 if small_cases is None else small_cases,
            small_storms=3 if small_storms is None else small_storms,
            command=command,
            forecast_type=expected,
            verify_legacy_members=verify_legacy_members,
            primary_model=primary_model,
            previous_model=previous_model,
            primary_leads=primary_leads,
            exploratory_leads=exploratory_leads,
        )

    if any(value is not None for value in (
        primary_model, previous_model, primary_leads, exploratory_leads,
        leads, fixed_leads, n_bootstrap, seed, small_cases, small_storms,
        forecast_type,
    )) or verify_legacy_members:
        raise ValueError("主模型/旧版本/讨论时效/台风专用参数只适用于台风报告")

    # 先把 compare 校验干净再往下走：任何错误都必须在 create_report 建目录之前抛出来，
    # 否则会留下一个只有半张图的空报告目录。
    comparisons = parse_compare_specs(compare)
    if comparisons and family != "ts":
        raise ValueError(
            f"只有 TS 系列支持 --compare（对比对象要的是同量级、同阈值的宽表）；"
            f"当前产物是 {capability}（{FAMILY_LABELS[family]}）。"
        )
    if full_model_name in comparisons:
        raise ValueError(
            f"--compare 里的名字 {full_model_name!r} 和主模型重名了，"
            f"会把主模型的来源路径覆盖掉。换个名字，或用 --model 改主模型名。"
        )

    print(f"能力：{capability} —— {FAMILY_LABELS[family]}")
    if family == "ts":
        return _render_ts(
            output_dir, target, full_model_name, change_description, comparisons,
            ensemble=capability == "weather_ts_ens",
            members=members,
        )
    return _render_field(
        output_dir,
        target,
        full_model_name,
        info,
        spectrum_variable,
        spectrum_init,
        change_description,
    )


def _probability_sibling(csv_path: Path) -> Optional[Path]:
    """在结果表所在目录里找概率评分表。

    两种命名都收：新的 ``diagnostics/probability_wide.csv``（和结果表同目录），
    以及老归档把两张裸 CSV 并排放在一起时的 ``aroc_bss*.csv``。**认的是列不是
    名字**（``load_probability`` 会校验列），所以老命名直接读得进来。
    找不到返回 ``None``——那几份只进 TS 那几节，不是错误。
    """
    parent = Path(csv_path).parent
    named = parent / PROBABILITY_WIDE_RELATIVE_PATH.name
    if named.is_file():
        return named
    legacy = sorted(parent.glob("aroc_bss*.csv"))
    return legacy[0] if legacy else None


def _coverage_of(wide) -> int:
    """最小阈值档的**全时效配对样本合计**——与报告第二节「样本合计」列同一口径。

    报告要拿它做「各份来源的样本量差多少」那句话。取值失败（缺列、全是空）返回 0，
    调用方会把它当作「这份没报样本量」跳过，而不是在报告里写个 0。
    """
    if "n_pairs" not in wide.columns or "threshold_mm" not in wide.columns:
        return 0
    import pandas as pd

    thresholds = pd.to_numeric(wide["threshold_mm"], errors="coerce")
    if not thresholds.notna().any():
        return 0
    pairs = pd.to_numeric(wide.loc[thresholds == thresholds.min(), "n_pairs"], errors="coerce")
    return int(pairs.dropna().sum())


def _render_ts(
    output_dir: Path,
    target: Path,
    model_name: str,
    change_description: Optional[str],
    comparisons: Optional[Dict[str, Path]] = None,
    *,
    ensemble: bool = False,
    members: Optional[int] = None,
) -> Dict[str, Path]:
    """TS 系列：宽表 → 曲线图 + 报告。``comparisons`` 是 ``{模型名: 对比结果表路径}``。

    ``ensemble`` 由 :func:`build_report` 按能力名给定：``weather_ts_ens`` 才打开
    集合专属措辞。**不靠目录名或 manifest 里的 reader 名去猜**——那些在这个仓库里
    已经被证明不可靠（本批的 ``weather_ts_ens_s2b`` 里 ``run_id`` 就写着另一个名字）。
    """
    wide_path = output_dir / CATEGORICAL_WIDE_RELATIVE_PATH
    if not wide_path.is_file():
        raise ValueError(
            f"{wide_path} 不存在，没有可出报告的结果表。TS 系列的图表吃的是宽表，"
            f"这里**不做长表→宽表转换**（转出来的口径和评测流程自己写的那份未必一致）。"
            f'在配置里加上 writers=["csv_long", "categorical_wide"] 重跑评测即可产出。'
        )
    try:
        import pandas as pd

        from reporting.reports.precipitation_plots import PrecipitationPlotter
    except ImportError as error:  # pragma: no cover - 取决于环境
        raise ImportError(
            "出图需要 matplotlib（可选依赖），先装：pip install -e .[viz]。"
            f"原始错误：{error}"
        ) from error

    comparisons = comparisons or {}
    # 来源要连对比模型一起写进报告头，否则每个对比模型都标成"未提供"
    sources = {model_name: str(wide_path)}
    sources.update({name: str(path) for name, path in comparisons.items()})
    # 纬度带的边界只在 manifest 里（宽表只有带名），报告要写成「带名 [南界, 北界]」
    # 就得从这里读。读不到（老产物没写这个键）时返回空，报告只显示带名。
    from reporting.reports.ts_report import region_labels_from

    # 每份来源的表只读一次：宽表要连用三次（主数据、概率同目录、样本量）。
    frames = {model_name: pd.read_csv(wide_path)}
    frames.update({name: pd.read_csv(path) for name, path in comparisons.items()})

    # 概率评分：主模型读自己的 diagnostics/，对比模型读**它自己那一行的同目录**
    # （老归档给的是裸 CSV，旁边没有这张表，自然缺席）。缺表不是错误——那几份只出 TS。
    probability: Dict[str, object] = {}
    for name, csv_path in [(model_name, wide_path), *comparisons.items()]:
        sibling = _probability_sibling(csv_path)
        if sibling is not None:
            probability[name] = pd.read_csv(sibling)
    # 样本量披露：最小阈值档的全时效 n_pairs 合计。取不到的不进覆盖表，
    # 免得报告里出现「某某 0」这种会被读成"零样本"的数字。
    coverage = {name: count for name, count in ((n, _coverage_of(f)) for n, f in frames.items()) if count}

    return PrecipitationPlotter().create_report(
        frames[model_name],
        output_dir=target,
        model_name=model_name,
        baselines={name: frames[name] for name in comparisons} or None,
        sources=sources,
        change_description=change_description,
        region_labels=region_labels_from(output_dir),
        probability=probability or None,
        members=members,
        coverage=coverage or None,
        ensemble=ensemble,
    )


def _render_field(
    output_dir: Path,
    target: Path,
    model_name: str,
    info: Dict[str, object],
    spectrum_variable: Optional[str],
    spectrum_init: Optional[str],
    change_description: Optional[str],
) -> Dict[str, Path]:
    """连续场：长表 → 曲线图 + 报告。"""
    scores_path = output_dir / "scores.csv"
    if not scores_path.is_file():
        raise ValueError(f"{scores_path} 不存在，没有可出报告的结果表")
    try:
        import pandas as pd

        from reporting.reports.field_plots import FieldScorePlotter
    except ImportError as error:  # pragma: no cover - 取决于环境
        raise ImportError(
            "出图需要 matplotlib（可选依赖），先装：pip install -e .[viz]。"
            f"原始错误：{error}"
        ) from error

    return FieldScorePlotter().create_report(
        # low_memory=False：长表的 ``unit`` 列是混合类型（``m^2/s^2`` / ``1`` / 空），
        # 分块推断会一边读一边告警"列有混合类型"，也让 dtype 随块变化。
        pd.read_csv(scores_path, low_memory=False),
        output_dir=target,
        spectrum_dir=output_dir,
        spectrum_variable=spectrum_variable,
        spectrum_init=spectrum_init,
        model_name=model_name,
        sources={"scores": str(scores_path)},
        manifest_info=info,
        change_description=change_description,
    )
