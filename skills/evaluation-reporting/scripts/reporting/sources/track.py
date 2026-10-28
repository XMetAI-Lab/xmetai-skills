# -*- coding: utf-8 -*-
"""台风检验层：预报场台风中心诊断 + 对 babj 实况的路径/强度误差。

算法对齐参考实现（track_diag_* / track_errs_* 两脚本），差异有三处
（均经确认）：
  1. 强度取最大风的 ±5° 框以**当前诊断中心**为中心（参考脚本用前一时次
     位置）；
  2. 首时次也输出真实诊断量（中心/强度/气压，参考脚本存 0 占位）；
  3. 方框搜索用**修正后的坐标数组**定位（处理 lon 文件头错误与 0/360
     回绕；参考脚本按标准 0.25° 索引算术，行为一致但更脆）。

``match_errors`` 刻意保持**逐时效标量循环**、不向量化：大圆距离与 at/ct 的
浮点结果要对得上旧链路的归档，向量化的写法末位会有出入。

babj diamond7 格式（GBK，样例 /mnt/d/babj/babjYYNN.dat）：
  第 1 行 diamond 7 头；第 2 行 台风名 编号 …；第 3 行空；
  数据行  yy mm dd hh 时效 lon lat 中心气压 最大风速(m/s) …
  **时效列 = 000 的行是分析实况**，其余为 BABJ 主观预报（不用）。
  时间为**北京时**（UTC pred 需 +8h 对时）。
"""
from __future__ import annotations

import datetime as _dt
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from reporting.sources.errors import DecodeError, EvaluationError

#: 大圆距离用的地球半径（km），与旧链路一致。
R_EARTH_KM = 6371.0


class TyphoonDataError(EvaluationError):
    """台风数据/匹配问题（搜索框越出网格、报文缺实况等）。"""


# ---------------------------------------------------------------- babj 实况

def read_babj_analyses(path) -> Tuple[str, str, Dict[_dt.datetime, Dict[str, float]]]:
    """读 babj diamond7，只取时效 000 的实况记录。

    返回 ``(tcname, tcid, {北京时datetime: dict(lon, lat, pmin_hpa, vmax_ms)})``。

    列序是 **lon 在前、lat 在后**（第 6、7 列），别按经纬顺序读反。
    """
    try:
        with open(path, "rb") as f:
            f.readline()
            head = f.readline().decode("gbk", "ignore").split()
        tcname, tcid = head[0], head[1]
        rows = np.loadtxt(path, skiprows=3, encoding="gbk", ndmin=2)
    except (OSError, ValueError, IndexError) as e:
        raise DecodeError("解析 babj 报文 %s 失败: %s" % (path, e)) from e
    analyses: Dict[_dt.datetime, Dict[str, float]] = {}
    for r in rows:
        if int(round(r[4])) != 0:  # 只要实况行
            continue
        t = _dt.datetime(int(r[0]), int(r[1]), int(r[2]), int(r[3]))
        analyses[t] = {
            "lon": float(r[5]),
            "lat": float(r[6]),
            "pmin_hpa": float(r[7]),
            "vmax_ms": float(r[8]),
        }
    if not analyses:
        raise TyphoonDataError("%s 中没有时效 000 的实况记录" % path)
    return tcname, tcid, analyses


# ---------------------------------------------------------------- 几何

def great_circle_km(lat1, lon1, lat2, lon2):
    """球面大圆距离（km，r=6371）。向量化；近距用 haversine 保精度。"""
    lat1, lon1, lat2, lon2 = (
        np.deg2rad(np.asarray(x, dtype="f8")) for x in (lat1, lon1, lat2, lon2)
    )
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R_EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _box_mask(glat, glon, clat, clon, half_lat, half_lon):
    """以 (clat, clon) 为中心的经纬方框掩膜（经度回绕）。"""
    glat = np.asarray(glat, dtype="f8")
    glon = np.mod(np.asarray(glon, dtype="f8"), 360.0)
    m_lat = np.abs(glat - clat) <= half_lat
    dlon = np.mod(glon - (clon % 360.0) + 180.0, 360.0) - 180.0
    m_lon = np.abs(dlon) <= half_lon
    return m_lat[:, None] & m_lon[None, :]


# ---------------------------------------------------------------- 诊断

def find_center_by_slp(msl, glat, glon, clat, clon, half_lat, half_lon):
    """方框内 msl 最小值点为台风中心。返回 (lat, lon, pmin)。"""
    m = _box_mask(glat, glon, clat, clon, half_lat, half_lon)
    if not m.any():
        raise TyphoonDataError(
            "搜索框 (%g,%g)±(%g,%g) 越出网格" % (clat, clon, half_lat, half_lon)
        )
    idx = np.nanargmin(np.where(m, msl, np.nan))
    i, j = np.unravel_index(idx, m.shape)
    return float(glat[i]), float(glon[j]), float(msl[i, j])


def max_wind_in_box(wind, glat, glon, clat, clon, half_lat, half_lon):
    """方框内风速最大值（强度，参考口径：±5° 方框最大 10m 风）。"""
    m = _box_mask(glat, glon, clat, clon, half_lat, half_lon)
    return float(np.nanmax(wind[m]))


def diagnose_track(
    msl,
    glat,
    glon,
    init_lat,
    init_lon,
    leads_hours,
    wind=None,
    center_half=3.0,
    early_half=4.0,
    early_hours=12.0,
    intensity_half=5.0,
):
    """从实况初始位置起链式诊断台风中心与强度。

    msl: (T, lat, lon) Pa；wind: 同形风速(m/s) 或 None；
    leads_hours: (T,) 各时次的真实时效（小时）；
    前 early_hours 小时用 early_half 搜索框（防初期漂移），其后 center_half；
    强度框以**当前诊断中心**为中心（±intensity_half）。

    返回 dict of list: lat, lon, pmin_hpa, vmax(无风场为 nan)。

    **链式**：每一时次的搜索框中心是上一时次诊断出的中心，不是实况位置——
    所以同一段场按不同起点切进来会得到不同结果，时效序列必须完整。
    """
    T = msl.shape[0]
    out = {"lat": [], "lon": [], "pmin_hpa": [], "vmax": []}
    plat, plon = float(init_lat), float(init_lon)
    for k in range(T):
        half = early_half if leads_hours[k] < early_hours else center_half
        lat_c, lon_c, pmin = find_center_by_slp(msl[k], glat, glon, plat, plon, half, half)
        if wind is not None:
            vmax = max_wind_in_box(wind[k], glat, glon, lat_c, lon_c, intensity_half, intensity_half)
        else:
            vmax = float("nan")
        out["lat"].append(lat_c)
        out["lon"].append(lon_c)
        out["pmin_hpa"].append(pmin / 100.0)
        out["vmax"].append(vmax)
        plat, plon = lat_c, lon_c
    return out


# ---------------------------------------------------------------- 误差

def along_cross_track(lat0, lon0, lat_b, lon_b, lat_p, lon_p):
    """沿路径(at)/横路径(ct)误差分解（参考脚本公式，km）。

    ``(lat0,lon0)`` 是**前一个配对实况**位置，``(lat_b,lon_b)`` 是当前实况、
    ``(lat_p,lon_p)`` 是当前预报——方向由实况路径段定，不是预报-实况连线。
    """
    def _alpha(la0, lo0, la, lo):
        dlat = (la - la0) / 180.0 * math.pi
        dlon = (lo - lo0) / 180.0 * math.pi
        return math.atan2(dlon * math.cos((la + la0) / 2 / 180 * math.pi), dlat)

    dalpha = _alpha(lat0, lon0, lat_p, lon_p) - _alpha(lat0, lon0, lat_b, lon_b)
    len_b = float(great_circle_km(lat0, lon0, lat_b, lon_b))
    len_p = float(great_circle_km(lat0, lon0, lat_p, lon_p))
    ct = len_p * math.sin(dalpha)
    at = len_p * math.cos(dalpha) - len_b
    return at, ct


def match_errors(
    track: Dict[str, List[float]],
    leads_hours: Sequence[float],
    init_utc: _dt.datetime,
    babj: Dict[_dt.datetime, Dict[str, float]],
    tz_shift: float = 8.0,
) -> List[Dict[str, float]]:
    """诊断路径对 babj 实况逐时效配对求误差。

    track: diagnose_track 的输出；babj: read_babj_analyses 的字典
    （北京时）。验证时刻(北京时) = init_utc + 时效 + tz_shift。
    无实况记录的时效各项为 NaN（路径/强度误差保持 NaN，不外推）。

    首个**配对成功**的时效 ``at_km``/``ct_km`` 为空——没有前一个实况位置就
    无从定向；实况有缺口时，缺的是那个时效，不是固定 ``lead_h=6``。
    """
    prev_babj: Optional[Tuple[float, float]] = None
    rows: List[Dict[str, float]] = []
    for k, tau in enumerate(leads_hours):
        v_bjt = init_utc + _dt.timedelta(hours=float(tau) + tz_shift)
        ob = babj.get(v_bjt)
        f_lat, f_lon = track["lat"][k], track["lon"][k]
        f_p, f_v = track["pmin_hpa"][k], track["vmax"][k]
        if ob is None:
            rows.append(
                {
                    "lead_h": float(tau),
                    "valid_bjt": str(v_bjt),
                    "fcst_lat": f_lat,
                    "fcst_lon": f_lon,
                    "fcst_pmin_hpa": f_p,
                    "fcst_vmax_ms": f_v,
                    "obs_lat": np.nan,
                    "obs_lon": np.nan,
                    "obs_pmin_hpa": np.nan,
                    "obs_vmax_ms": np.nan,
                    "track_err_km": np.nan,
                    "at_km": np.nan,
                    "ct_km": np.nan,
                    "wind_err_ms": np.nan,
                    "pmin_err_hpa": np.nan,
                }
            )
            continue
        o_lat, o_lon = ob["lat"], ob["lon"]
        err = float(great_circle_km(f_lat, f_lon, o_lat, o_lon))
        if prev_babj is None:  # 首个配对时次
            at = ct = float("nan")
        else:
            at, ct = along_cross_track(prev_babj[0], prev_babj[1], o_lat, o_lon, f_lat, f_lon)
        rows.append(
            {
                "lead_h": float(tau),
                "valid_bjt": str(v_bjt),
                "fcst_lat": f_lat,
                "fcst_lon": f_lon,
                "fcst_pmin_hpa": f_p,
                "fcst_vmax_ms": f_v,
                "obs_lat": o_lat,
                "obs_lon": o_lon,
                "obs_pmin_hpa": ob["pmin_hpa"],
                "obs_vmax_ms": ob["vmax_ms"],
                "track_err_km": err,
                "at_km": at,
                "ct_km": ct,
                "wind_err_ms": f_v - ob["vmax_ms"] if np.isfinite(f_v) else float("nan"),
                "pmin_err_hpa": f_p - ob["pmin_hpa"],
            }
        )
        prev_babj = (o_lat, o_lon)
    return rows
