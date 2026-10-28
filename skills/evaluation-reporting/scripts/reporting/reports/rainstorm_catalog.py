# -*- coding: utf-8 -*-
"""2025 年暴雨过程编目：编号 → 起止 / 等级，**这份表是唯一真值来源**。

过程来自《2025年暴雨过程纪要表（修订）-1.docx》（2025-10-08 版），编号
202501–202543，共 43 个过程。等级是纪要表自己给的（``特强`` / ``强`` /
``较强`` / ``中等``），**不是从评分推出来的**——它只用来做过程分档与排序，
不参与任何技巧判定。

三种消费方，口径必须一致，所以都从这里取：

- 评测配置（``configs/rainstorm_ts_single_fuxi.py`` 等）：只要
  ``(编号, 开始日, 结束日)``，用 :func:`process_tuples`；
- 汇总脚本（``skills/.../scripts/render_rainstorm_summary.py``）：要起止标签与等级，
  用 :data:`CATALOG`；
- 报告渲染器（``visualization/rainstorm_report.py``）：要全套，还要按
  :func:`expected_days` 推口径 A 应有的起报日。

**改纪要表就改这里一处**，别在别处再抄一份——历史上这份表在评测配置和汇总
脚本里各有一份，等级只有后一份有，改一头另一头不会跟着动。

两点纪要表本身的坑，照抄时保留了原样：

1. **202519 与 202520 起止完全相同**（7月18-22日）——纪要表把同一时段的两个
   侧面分列两条，**各评各的、不合并**，所以编目里两条都在。
2. 开始日/结束日是纪要表的 **0–0 日历口径**。评测实际用的是 08–08 北京时日界，
   口径 B 因此头尾各多 8 小时——那是评测口径的事，编目只记纪要表原值。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

#: 等级的强弱次序（强 → 弱），只用于报告里的分档展示与排序。
GRADE_ORDER: Tuple[str, ...] = ("特强", "强", "较强", "中等")


@dataclass(frozen=True)
class Process:
    """一个暴雨过程。

    Attributes:
        code: 纪要表编号，``202501``–``202543``。
        span: 起止的中文写法，如 ``"3月12-15日"``，报告里直接引用。
        start: 开始日 ``YYYYMMDD``（纪要表 0–0 口径）。
        end: 结束日 ``YYYYMMDD``（纪要表 0–0 口径）。
        grade: 纪要表给的等级，取值见 :data:`GRADE_ORDER`。
    """

    code: str
    span: str
    start: str
    end: str
    grade: str


#: 43 个过程，按编号升序。
CATALOG: Tuple[Process, ...] = (
    Process("202501", "3月12-15日", "20250312", "20250315", "较强"),
    Process("202502", "4月11-13日", "20250411", "20250413", "较强"),
    Process("202503", "4月18-19日", "20250418", "20250419", "较强"),
    Process("202504", "4月21-25日", "20250421", "20250425", "较强"),
    Process("202505", "5月3-5日", "20250503", "20250505", "较强"),
    Process("202506", "5月7-9日", "20250507", "20250509", "较强"),
    Process("202507", "5月13-16日", "20250513", "20250516", "较强"),
    Process("202508", "5月17-20日", "20250517", "20250520", "较强"),
    Process("202509", "5月21-24日", "20250521", "20250524", "较强"),
    Process("202510", "5月27-29日", "20250527", "20250529", "较强"),
    Process("202511", "5月31日-6月2日", "20250531", "20250602", "较强"),
    Process("202512", "6月6-10日", "20250606", "20250610", "较强"),
    Process("202513", "6月11-15日", "20250611", "20250615", "强"),
    Process("202514", "6月17日夜-23日", "20250617", "20250623", "特强"),
    Process("202515", "6月24-27日", "20250624", "20250627", "较强"),
    Process("202516", "6月29-30日", "20250629", "20250630", "强"),
    Process("202517", "7月2-6日", "20250702", "20250706", "强"),
    Process("202518", "7月5-12日", "20250705", "20250712", "强"),
    Process("202519", "7月18-22日", "20250718", "20250722", "强"),
    Process("202520", "7月18-22日", "20250718", "20250722", "强"),
    Process("202521", "7月23-29日", "20250723", "20250729", "特强"),
    Process("202522", "7月29日-8月2日", "20250729", "20250802", "强"),
    Process("202523", "8月3-7日", "20250803", "20250807", "强"),
    Process("202524", "8月9-12日", "20250809", "20250812", "强"),
    Process("202525", "8月13-15日", "20250813", "20250815", "较强"),
    Process("202526", "8月17-19日", "20250817", "20250819", "强"),
    Process("202527", "8月18-19日", "20250818", "20250819", "较强"),
    Process("202528", "8月20-24日", "20250820", "20250824", "较强"),
    Process("202529", "8月23-25日", "20250823", "20250825", "较强"),
    Process("202530", "8月25日夜-28日", "20250825", "20250828", "强"),
    Process("202531", "9月2-5日", "20250902", "20250905", "强"),
    Process("202532", "9月10-12日", "20250910", "20250912", "较强"),
    Process("202533", "9月15-17日", "20250915", "20250917", "强"),
    Process("202534", "9月18-21日", "20250918", "20250921", "较强"),
    Process("202535", "9月18-22日", "20250918", "20250922", "较强"),
    Process("202536", "9月23-25日", "20250923", "20250925", "强"),
    Process("202537", "9月27-29日", "20250927", "20250929", "较强"),
    Process("202538", "9月30日-10月2日", "20250930", "20251002", "中等"),
    Process("202539", "10月3-6日", "20251003", "20251006", "中等"),
    Process("202540", "10月4-7日", "20251004", "20251007", "中等"),
    Process("202541", "10月7-9日", "20251007", "20251009", "较强"),
    Process("202542", "10月10-12日", "20251010", "20251012", "较强"),
    Process("202543", "10月16-17日", "20251016", "20251017", "较强"),
)

#: 编号 → 过程，供按编号点名取值。
BY_CODE: Dict[str, Process] = {process.code: process for process in CATALOG}


def process_tuples() -> List[Tuple[str, str, str]]:
    """``[(编号, 开始日, 结束日), …]``，给评测配置用。

    评测配置只关心评哪几段，不需要起止标签与等级——那两个是报告层的事。
    """
    return [(process.code, process.start, process.end) for process in CATALOG]


def codes() -> List[str]:
    """全部编号（升序），即 :data:`CATALOG` 的顺序。"""
    return [process.code for process in CATALOG]


def get(code: str) -> Process:
    """按编号取一个过程。

    Raises:
        KeyError: 编号不在编目里（附上全部可用编号的提示）。
    """
    try:
        return BY_CODE[str(code)]
    except KeyError:
        raise KeyError(
            f"编目里没有过程 {code!r}；可用编号 {codes()[0]}–{codes()[-1]}，"
            f"共 {len(CATALOG)} 个"
        ) from None


def window_days(code: str) -> int:
    """口径 B 的累积窗长（天），``(结束日 − 开始日) + 2``。

    ``+2`` 是 08–08 日界的代价：窗为 ``[S-1 日 08:00, E+1 日 08:00]``，
    比纪要表的 0–0 日历口径头尾各多 8 小时。
    """
    process = get(code)
    start = datetime.strptime(process.start, "%Y%m%d")
    end = datetime.strptime(process.end, "%Y%m%d")
    return (end - start).days + 2


def expected_days(code: str) -> List[str]:
    """口径 A 应有的起报日：``[S-1, E]`` 逐日 ``YYYYMMDD``。

    口径 A 是"每个 08–08 窗由当日起报的 day-1 预报评一次"，所以起报日从过程
    开始日**前一天**起、到结束日为止，一天一个结果、互不合并。
    报告拿它当"这段产物该有哪些"的基准，缺的起报日在表里记「缺」而不是静默跳过。
    """
    process = get(code)
    first = datetime.strptime(process.start, "%Y%m%d") - timedelta(days=1)
    last = datetime.strptime(process.end, "%Y%m%d")
    return [
        (first + timedelta(days=offset)).strftime("%Y%m%d")
        for offset in range((last - first).days + 1)
    ]


def sort_key(code: str):
    """报告里按等级强弱再按编号排；等级不在 :data:`GRADE_ORDER` 里时排最后。"""
    grade = get(code).grade
    rank = GRADE_ORDER.index(grade) if grade in GRADE_ORDER else len(GRADE_ORDER)
    return (rank, str(code))
