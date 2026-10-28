# -*- coding: utf-8 -*-
"""报告模块共用的小工具：中文数字、来源标注、数值格式化。

与报告口径无关，只是 ens / wave / field / ts / rainstorm 几份报告里逐字相同的
小函数；集中在这里，改口径时不会出现「五份里改了三份」。
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

CN_NUM = {
    1: "一", 2: "二", 3: "三", 4: "四", 5: "五",
    6: "六", 7: "七", 8: "八", 9: "九", 10: "十",
}


def cn_number(number: int) -> str:
    """中文数字（写「二模型」而不是「2模型」）；超过十退回阿拉伯数字。"""
    return CN_NUM.get(int(number), str(int(number)))


def cn_section(number: int) -> str:
    """章节号的中文数字。超范围直接 KeyError——编号错要立刻暴露，不静默降级。"""
    return CN_NUM[int(number)]


def source_of(sources: Optional[Dict[str, str]], name: str) -> str:
    """来源标注：给了路径就写路径，否则如实写"未提供"。"""
    if sources and sources.get(name):
        return f"`{sources[name]}`"
    return "未提供（调用方应在 sources 里给出文件路径）"


def finite_or_none(value: object) -> Optional[float]:
    """转成 float；转不动或非有限值时返回 None。"""
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def is_finite_number(value: object) -> bool:
    """能转成有限 float 就是 True（表格筛选用的布尔版）。"""
    return finite_or_none(value) is not None


def fmt_number(value: object, digits: int = 3) -> str:
    """定点格式；None / 非有限值写成 "—"（有产物但无有效值）。"""
    number = finite_or_none(value)
    return "—" if number is None else f"{number:.{digits}f}"


def table_lines(header: Sequence[str], rows: Sequence[Sequence[str]]) -> List[str]:
    """Markdown 表格：表头 + 分隔行 + 数据行。"""
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return lines


def bullet_lines(items: Sequence[str], empty: str = "• 无。") -> List[str]:
    """Markdown 无序列表；空列表给一句占位，不留空白。"""
    return [f"• {item}" for item in items] if items else [empty]


def date_span(dates: Sequence[str]) -> str:
    """日期跨度写成集合形式 ``20250102，20251215``（全角逗号，不带括号）。"""
    if not dates:
        return "—"
    return f"{dates[0]}，{dates[-1]}" if len(dates) > 1 else dates[0]


def lead_span(leads: Sequence[float]) -> Tuple[str, str, int]:
    """时效范围 ``("6", "360", 60)``；全非有限时返回 ``("—", "—", 0)``。"""
    finite = [float(x) for x in leads if math.isfinite(float(x))]
    if not finite:
        return "—", "—", 0
    return f"{min(finite):g}", f"{max(finite):g}", len(finite)


def own_dates_mode(bundle: Any) -> bool:
    """本次统计是不是「各模型用自己的全部起报日」（数据驱动，不看开关）。"""
    common = set(bundle.dates)
    return any(set(own) != common for own in bundle.dates_by_model.values())


def date_scope_word(bundle: Any) -> str:
    """句子里的短口径词：``26 个共同日期`` / ``各模型各自的全部起报日``。"""
    if not own_dates_mode(bundle):
        return f"{len(bundle.dates)} 个共同日期"
    return "各模型各自的全部起报日"


def date_scope_counts(bundle: Any) -> str:
    """各模型的天数清单（``AIFS 27 天、FGVP 228 天``），按模型名升序。"""
    return "、".join(
        f"{name} {len(bundle.dates_by_model[name])} 天" for name in sorted(bundle.names)
    )


def date_scope_detail(bundle: Any) -> str:
    """正文交代日期口径的短语（自带括号）；各用各的时括号里是天数清单。"""
    if not own_dates_mode(bundle):
        return f"{len(bundle.dates)} 个共同日期（{date_span(bundle.dates)}）"
    return f"各模型各自的全部起报日（{date_scope_counts(bundle)}）"


def sph_coverage_notes(bundle: Any) -> List[str]:
    """球谐带缺样本的披露句，逐模型逐变量写明；一格不缺时返回空列表。"""
    drops = []
    for name in bundle.names:
        short = [
            (variable, used, requested)
            for variable, (used, requested) in sorted(
                bundle.sph_coverage.get(name, {}).items()
            )
            if used < requested
        ]
        if short:
            drops.append(
                f"{name} 的 "
                + "、".join(
                    f"{variable} 用 {used}/{requested} 天"
                    for variable, used, requested in short
                )
            )
    if not drops:
        return []
    return [
        "**球谐带的样本不是每一格都满**：" + "；".join(drops)
        + "。缺的那些起报日在产物里本来就没有该变量的球谐带结果，"
        "本节按**实际有数的日子**统计，不插补、不折算；"
        "这些格子和满覆盖的模型/变量**不同源**，横向比高低之前先看这一条。"
    ]
