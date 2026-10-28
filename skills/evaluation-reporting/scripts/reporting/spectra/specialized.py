# -*- coding: utf-8 -*-
"""专项指标：Z500 活跃度比、纬向功率谱与球谐带功率。

口径与参考实现一致（逐项对齐，便于对拍）：

* **活跃度比**：口径同 ``ref/fdp/verify/verify/activity_spectrum_verifier.py``
  ——距平（场 − 气候态）的纬度加权标准差之比 ``AR = std_w(F−C) / std_w(O−C)``，
  权重 ``w = cos(lat)``，标准差为"减加权均值"的加权标准差；
* **功率谱**：对**原始场**做二维 FFT 得到纬向波数谱
  ``P(k) = Σ_l [A_t(l,k)² + B_t(l,k)²]``（按参考实现的等效 FFT 推导），
  单边谱取 k=0..N/2，缺测用全场均值填充；
* **球谐带功率**：口径同老仓 ``vfc/metrics/spectrum.py::spherical_band_power``
  （等价参考 xmetai ``SphericalBandPowerLoss.band_power``）——associated
  Legendre 递推 + Clenshaw-Curtis 求积的球谐展开，按**总波数**区间分带，
  只对全球含极网格有定义（区域网格在这里被拦下，不是静默算错）。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import xarray as xr

from reporting.sources.contracts import (
    EvaluationBatch,
    MetricResult,
    ResultStatus,
)
from reporting.sources.errors import MetricError
from reporting.spectra.base import (
    Metric,
    MetricRequirements,
    MetricState,
    ProductType,
    single_variable,
)

DEFAULT_MAX_WAVENUMBER = 30


def _lat_from_batch(batch: EvaluationBatch, metric_name: str) -> np.ndarray:
    """从 forecast 的坐标里取纬度（度）。谱类指标的加权/校验都靠它。"""
    src = batch.forecast
    if hasattr(src, "payload"):
        src = src.payload
    if isinstance(src, (xr.DataArray, xr.Dataset)):
        lat = src.coords.get("lat")
        if lat is not None:
            return np.asarray(
                lat.values if hasattr(lat, "values") else lat, dtype="f8"
            )
    raise MetricError(
        f"{metric_name} 需要 forecast 携带 lat 坐标（用于纬度加权/网格校验）",
        variable=metric_name,
    )


def _values(data: Any) -> np.ndarray:
    if isinstance(data, xr.DataArray):
        return np.asarray(data.values, dtype="f8")
    if isinstance(data, xr.Dataset):
        return np.asarray(single_variable(data, "输入").values, dtype="f8")
    if hasattr(data, "payload"):
        payload = data.payload
        if isinstance(payload, xr.Dataset):
            payload = single_variable(payload, "输入")
        return np.asarray(payload.values, dtype="f8")
    return np.asarray(data, dtype="f8")


def _weights(batch: EvaluationBatch, shape) -> np.ndarray:
    if batch.weights is not None:
        weights = _values(batch.weights)
        if weights.shape == shape:
            return weights
    return np.ones(shape, dtype="f8")


def power_spectrum(field: np.ndarray, max_wavenumber: Optional[int] = None):
    """纬向波数功率谱 P(k)。

    Args:
        field: (M, N) 二维场（M=纬度方向格点数，N=经度方向格点数）。
        max_wavenumber: 返回的最大波数；None 表示 N//2。

    Returns:
        (wavenumbers, power)
    """
    values = np.asarray(field, dtype="f8")
    if values.ndim != 2:
        raise MetricError(f"功率谱需要二维场，实际 ndim={values.ndim}")
    if np.any(np.isnan(values)):
        values = np.where(np.isnan(values), np.nanmean(values), values)

    nlat, nlon = values.shape
    n_freq = nlon // 2 + 1
    max_k = n_freq - 1 if max_wavenumber is None else min(int(max_wavenumber), n_freq - 1)

    vertical = np.fft.ifft(values, axis=0)
    lat_index = np.arange(nlat)[:, None]
    lon_index = np.arange(nlon)[None, :]
    modulated = vertical * np.exp(-2j * np.pi * lat_index * lon_index / nlat)
    transformed = np.fft.ifft(modulated, axis=1)
    power_per_wavenumber = (nlat * nlon) ** 2 * np.abs(transformed) ** 2
    full = power_per_wavenumber.sum(axis=0)

    power = np.zeros(max_k + 1)
    for k in range(max_k + 1):
        if k == 0 or (nlon % 2 == 0 and k == nlon // 2):
            power[k] = full[k]
        else:
            power[k] = full[k] + full[nlon - k]
    return np.arange(max_k + 1), power


class ActivityRatio(Metric):
    """活跃度比（距平纬度加权标准差之比），需要气候态参考。"""

    PRODUCT_KIND = "specialized"

    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(name="activity", version="1.0.0", params=params or {})

    def requirements(self) -> MetricRequirements:
        return MetricRequirements(
            product_type=ProductType.DETERMINISTIC_FIELD,
            variables=["*"],
        )

    def needs_reference(self) -> bool:
        """距平统计，没有气候态直接算不了（_anomalies 会报错）。"""
        return True

    def _anomalies(self, batch: EvaluationBatch):
        if batch.reference is None:
            raise MetricError(
                "活跃度比是距平统计，需要气候态参考（请为流程配置 reference 数据源）",
                variable=self.name,
            )
        forecast = _values(batch.forecast)
        observation = _values(batch.observation)
        climatology = _values(batch.reference)
        if not (forecast.shape == observation.shape == climatology.shape):
            raise MetricError(
                f"形状不一致：fc={forecast.shape}, obs={observation.shape}, "
                f"clim={climatology.shape}",
                variable=self.name,
            )
        keep = (
            np.isfinite(forecast) & np.isfinite(observation) & np.isfinite(climatology)
        )
        weights = _weights(batch, forecast.shape)
        if batch.valid_mask is not None:
            mask = _values(batch.valid_mask)
            if mask.shape == keep.shape:
                keep = keep & mask.astype(bool)
        return forecast - climatology, observation - climatology, weights, keep

    @staticmethod
    def _moments(values, weights, keep):
        w = np.where(keep, weights, 0.0)
        v = np.where(keep, values, 0.0)
        return {
            "w": float(w.sum()),
            "w1": float((w * v).sum()),
            "w2": float((w * v * v).sum()),
        }

    def accumulate(self, batch: EvaluationBatch) -> MetricState:
        forecast_anomaly, observation_anomaly, weights, keep = self._anomalies(batch)
        return MetricState(
            metric_name=self.name,
            metric_version=self.version,
            data={
                "forecast": self._moments(forecast_anomaly, weights, keep),
                "observation": self._moments(observation_anomaly, weights, keep),
                "n_valid": int(keep.sum()),
            },
            n_accumulated=1,
        )

    def merge(self, states: List[MetricState]) -> MetricState:
        if not states:
            raise MetricError("Cannot merge empty states list")
        first = states[0]
        merged = {
            "forecast": {"w": 0.0, "w1": 0.0, "w2": 0.0},
            "observation": {"w": 0.0, "w1": 0.0, "w2": 0.0},
            "n_valid": 0,
        }
        for state in states:
            if state.metric_name != first.metric_name:
                raise MetricError(
                    f"Cannot merge states from different metrics: "
                    f"{first.metric_name} vs {state.metric_name}",
                    variable=self.name,
                )
            for side in ("forecast", "observation"):
                for key in ("w", "w1", "w2"):
                    merged[side][key] += state.data[side][key]
            merged["n_valid"] += state.data["n_valid"]
        return MetricState(
            metric_name=self.name,
            metric_version=self.version,
            data=merged,
            n_accumulated=sum(state.n_accumulated for state in states),
        )

    @staticmethod
    def _std(moments):
        total_weight = moments["w"]
        if total_weight <= 0:
            return float("nan")
        mean = moments["w1"] / total_weight
        variance = moments["w2"] / total_weight - mean**2
        return float(np.sqrt(max(variance, 0.0)))

    def finalize(self, state: MetricState) -> MetricResult:
        forecast_std = self._std(state.data["forecast"])
        observation_std = self._std(state.data["observation"])
        if not np.isfinite(forecast_std) or not np.isfinite(observation_std):
            status = ResultStatus.NO_VALID_DATA
            ratio = float("nan")
        elif observation_std == 0:
            status = ResultStatus.UNDEFINED
            ratio = float("nan")
        else:
            status = ResultStatus.SUCCESS
            ratio = float(forecast_std / observation_std)

        return MetricResult(
            metric_name=self.name,
            metric_version=self.version,
            value={
                "ACTIVITY_RATIO": ratio,
                "FC_ACTIVITY": forecast_std,
                "OBS_ACTIVITY": observation_std,
                "FC_OBS_BIAS": float(forecast_std - observation_std),
            },
            status=status,
            n_requested=int(state.data["n_valid"]),
            n_valid=int(state.data["n_valid"]),
            aggregation="area_weighted",
            unit="1",
            product_kind=self.PRODUCT_KIND,
        )


class PowerSpectrum(Metric):
    """纬向功率谱（对原始场）：曲线进明细表，总功率比进长表。

    ⚠ 这个谱函数**对纬度朝向敏感**，不是换行序那么无害：相位项
    ``exp(-2πi·l·i/M)`` 里的 ``l`` 是数组**行号**（0,1,2,…）而不是纬度值，
    同一纬度翻转后拿到不同的行号，逐波数的 ``P(k)`` 就变了（不是差一个常数
    倍数，是逐波数各差各的，奇数波数能差到 2 倍）。

    默认 ``lat_order=None`` = **按传进来的形态原样算**，也就是既有行为 ——
    weather 全系列的谱逐位不变。要跟标准脚本
    ``activity_spectrum_verifier.py`` 对表，就在 metric_options 里写
    ``"spectrum": {"lat_order": "descending"}``：它把模式场和实况**都归到
    纬度降序**（90 → −90）再算，而框架侧通常是升序（−90 → 90）。
    """

    PRODUCT_KIND = "specialized"

    def __init__(
        self,
        max_wavenumber: int = DEFAULT_MAX_WAVENUMBER,
        lat_order: Optional[str] = None,
        params=None,
    ):
        super().__init__(name="spectrum", version="1.0.0", params=params or {})
        self.max_wavenumber = int(max_wavenumber)
        if lat_order not in (None, "ascending", "descending"):
            raise MetricError(
                f"spectrum 的 lat_order 只能取 None（按原样）/ 'ascending' / "
                f"'descending'，收到 {lat_order!r}",
                variable="spectrum",
            )
        self.lat_order = lat_order

    def requirements(self) -> MetricRequirements:
        return MetricRequirements(
            product_type=ProductType.DETERMINISTIC_FIELD,
            variables=["*"],
        )

    @staticmethod
    def _lat_values(source):
        """从 batch 那一侧取纬度坐标；取不到返回 None（那就无从判断朝向）。"""
        payload = getattr(source, "payload", source)
        coords = getattr(payload, "coords", None)
        if coords is not None and "lat" in coords:
            return np.asarray(coords["lat"].values, dtype="f8")
        return None

    def _oriented(self, values, source):
        """按 ``self.lat_order`` 把场归到指定纬度朝向。

        只有配置显式写了 ``lat_order`` 才可能动，且**只在朝向确实相反时翻一次**
        —— 已经是目标朝向就是恒等，所以这个动作是幂等的。默认 None 直接原样返回。
        """
        if self.lat_order is None or values.ndim < 2:
            return values
        lat = self._lat_values(source)
        if lat is None or lat.size < 2:
            return values
        ascending = bool(lat[0] < lat[-1])
        if ascending == (self.lat_order == "ascending"):
            return values
        return values[..., ::-1, :]

    def _spectra(self, batch: EvaluationBatch):
        forecast = self._oriented(_values(batch.forecast), batch.forecast)
        observation = self._oriented(_values(batch.observation), batch.observation)
        forecast_grid = (
            forecast.reshape(-1, forecast.shape[-1]) if forecast.ndim > 2 else forecast
        )
        observation_grid = (
            observation.reshape(-1, observation.shape[-1])
            if observation.ndim > 2
            else observation
        )
        _, forecast_power = power_spectrum(forecast_grid, self.max_wavenumber)
        _, observation_power = power_spectrum(observation_grid, self.max_wavenumber)
        return forecast_power, observation_power

    def accumulate(self, batch: EvaluationBatch) -> MetricState:
        forecast_power, observation_power = self._spectra(batch)
        return MetricState(
            metric_name=self.name,
            metric_version=self.version,
            data={
                "forecast_power": forecast_power,
                "observation_power": observation_power,
                "n_samples": 1,
            },
            n_accumulated=1,
        )

    def merge(self, states: List[MetricState]) -> MetricState:
        if not states:
            raise MetricError("Cannot merge empty states list")
        forecast_total = np.zeros_like(states[0].data["forecast_power"])
        observation_total = np.zeros_like(states[0].data["observation_power"])
        for state in states:
            forecast_total = forecast_total + state.data["forecast_power"]
            observation_total = observation_total + state.data["observation_power"]
        return MetricState(
            metric_name=self.name,
            metric_version=self.version,
            data={
                "forecast_power": forecast_total,
                "observation_power": observation_total,
                "n_samples": sum(state.data["n_samples"] for state in states),
            },
            n_accumulated=sum(state.n_accumulated for state in states),
        )

    def finalize(self, state: MetricState) -> MetricResult:
        count = max(int(state.data["n_samples"]), 1)
        forecast_power = state.data["forecast_power"] / count
        observation_power = state.data["observation_power"] / count

        # 总功率比作为标量进长表；逐波数曲线走 curve（见 MetricResult.curve）
        value, curve = _spectrum_summary_and_curve(forecast_power, observation_power)

        return MetricResult(
            metric_name=self.name,
            metric_version=self.version,
            value=value,
            status=ResultStatus.SUCCESS,
            n_requested=count,
            n_valid=count,
            aggregation="wavenumber_spectrum",
            unit="1",  # 进长表的是无量纲的 power_ratio
            product_kind=self.PRODUCT_KIND,
            curve=curve,
        )


def zonal_spectrum(
    field: np.ndarray,
    lat: np.ndarray,
    weighted: bool = True,
    remove_zonal_mean: bool = True,
) -> np.ndarray:
    """纬向功率谱（去纬向均值 → rfft → cos 纬度加权平均）。

    口径与参考实现 ``vfc/metrics/spectrum.py::zonal_spectrum`` 一致。

    Args:
        field: (..., lat, lon) 场，最后两维为 (lat, lon)。
        lat: 纬度数组（度），长度 = field 的 lat 维。
        weighted: True 时按 cos(lat) 加权平均。
        remove_zonal_mean: True 时先去除纬向均值、k=0 置零。

    Returns:
        (..., nlon // 2 + 1) 纬向功率谱。
    """
    x = np.asarray(field, dtype="f8")
    nlon = x.shape[-1]
    if remove_zonal_mean:
        x = x - x.mean(axis=-1, keepdims=True)
    F = np.fft.rfft(x, axis=-1)
    psd = (F.real**2 + F.imag**2) / (nlon**2)
    if remove_zonal_mean:
        psd[..., 0] = 0.0
    w = (
        np.cos(np.deg2rad(np.asarray(lat, dtype="f8")))
        if weighted
        else np.ones(lat.size, dtype="f8")
    )
    wshape = (1,) * (psd.ndim - 2) + (lat.size, 1)
    return (psd * w.reshape(wshape)).sum(axis=-2) / w.sum()


def _spectrum_summary_and_curve(
    forecast_power: np.ndarray, observation_power: np.ndarray
):
    """把逐波数功率曲线拆成「标量摘要 + 曲线」两半。

    摘要进 ``MetricResult.value``（``power_ratio`` 落进长表）；曲线走
    ``MetricResult.curve``，**不进 value**——value 里每个分组键都会在
    ``build_tables`` 里展开成明细行，720 个波数 × 3 个字段就是 2165 行/样本，
    全年段能撑到四千多万行。
    """
    forecast_total = float(forecast_power.sum())
    observation_total = float(observation_power.sum())
    summary = {
        "total_power_forecast": forecast_total,
        "total_power_observation": observation_total,
        "power_ratio": (
            float(forecast_total / observation_total)
            if observation_total > 0
            else float("nan")
        ),
    }
    curve = {
        "kind": "wavenumber_spectrum",
        "wavenumber": np.arange(forecast_power.size, dtype="i8"),
        "power_forecast": np.asarray(forecast_power, dtype="f8"),
        "power_observation": np.asarray(observation_power, dtype="f8"),
    }
    return {"summary": summary}, curve


class ZonalSpectrum(Metric):
    """纬向功率谱（去纬向均值后 rfft，cos 纬度加权），逐样本平均曲线。"""

    PRODUCT_KIND = "specialized"

    def __init__(self, max_wavenumber: int = DEFAULT_MAX_WAVENUMBER, params=None):
        super().__init__(name="zonal_spectrum", version="1.0.0", params=params or {})
        self.max_wavenumber = int(max_wavenumber)

    def requirements(self) -> MetricRequirements:
        return MetricRequirements(
            product_type=ProductType.DETERMINISTIC_FIELD,
            variables=["*"],
        )

    @staticmethod
    def _lat(batch: EvaluationBatch) -> np.ndarray:
        return _lat_from_batch(batch, "zonal_spectrum")

    def _spectra(self, batch: EvaluationBatch):
        forecast = _values(batch.forecast)
        observation = _values(batch.observation)
        if forecast.ndim < 2 or observation.ndim < 2:
            raise MetricError(
                f"纬向谱需要 (..., lat, lon) 场，实际 ndim={forecast.ndim}",
                variable=self.name,
            )
        lat = _lat_from_batch(batch, self.name)
        f = forecast.reshape(-1, forecast.shape[-2], forecast.shape[-1])
        o = observation.reshape(-1, observation.shape[-2], observation.shape[-1])
        forecast_psd = zonal_spectrum(f, lat).sum(axis=0)
        observation_psd = zonal_spectrum(o, lat).sum(axis=0)
        max_k = min(self.max_wavenumber, forecast_psd.size - 1)
        return (
            forecast_psd[: max_k + 1],
            observation_psd[: max_k + 1],
            int(f.shape[0]),
        )

    def accumulate(self, batch: EvaluationBatch) -> MetricState:
        forecast_power, observation_power, n_samples = self._spectra(batch)
        return MetricState(
            metric_name=self.name,
            metric_version=self.version,
            data={
                "forecast_power": forecast_power,
                "observation_power": observation_power,
                "n_samples": n_samples,
            },
            n_accumulated=1,
        )

    def merge(self, states: List[MetricState]) -> MetricState:
        if not states:
            raise MetricError("Cannot merge empty states list")
        forecast_total = np.zeros_like(states[0].data["forecast_power"])
        observation_total = np.zeros_like(states[0].data["observation_power"])
        for state in states:
            forecast_total = forecast_total + state.data["forecast_power"]
            observation_total = observation_total + state.data["observation_power"]
        return MetricState(
            metric_name=self.name,
            metric_version=self.version,
            data={
                "forecast_power": forecast_total,
                "observation_power": observation_total,
                "n_samples": sum(state.data["n_samples"] for state in states),
            },
            n_accumulated=sum(state.n_accumulated for state in states),
        )

    def finalize(self, state: MetricState) -> MetricResult:
        count = max(int(state.data["n_samples"]), 1)
        forecast_power = state.data["forecast_power"] / count
        observation_power = state.data["observation_power"] / count

        # 总功率比作为标量进长表；逐波数曲线走 curve（见 MetricResult.curve）
        value, curve = _spectrum_summary_and_curve(forecast_power, observation_power)

        return MetricResult(
            metric_name=self.name,
            metric_version=self.version,
            value=value,
            status=ResultStatus.SUCCESS,
            n_requested=count,
            n_valid=count,
            aggregation="wavenumber_spectrum",
            unit="1",  # 进长表的是无量纲的 power_ratio
            product_kind=self.PRODUCT_KIND,
            curve=curve,
        )


# --------------------------------------------------------------------------
# 球谐带功率（自老仓 vfc/metrics/spectrum.py 移植，口径逐位对齐）
# --------------------------------------------------------------------------

#: 与参考 xmetai ForecastFidelity 保持一致的球谐总阶数分段。
DEFAULT_SPHERICAL_BANDS: Tuple[Tuple[int, int], ...] = (
    (1, 4), (5, 20), (21, 40), (41, 64), (65, 128),
)


class _SphericalBandPower:
    """球谐带功率计算器（等价于参考 SphericalBandPowerLoss.band_power）。

    数值代码自老仓 ``vfc/metrics/spectrum.py::_SphericalBandPower`` 逐行移植：
    Clenshaw-Curtis 求积权重、稳定的 associated Legendre 三阶递推、
    float32 基函数缓存。**不要"顺手优化"这里的精度/类型**——任何一位的
    差别都会破坏与老档案的逐位对拍。
    """

    def __init__(self, nlat: int, nlon: int, bands: Sequence[Tuple[int, int]]):
        self.nlat = int(nlat)
        self.nlon = int(nlon)
        self.bands = tuple((int(lo), int(hi)) for lo, hi in bands)
        if not self.bands:
            raise ValueError("bands must not be empty")
        if any(lo < 0 or hi < lo for lo, hi in self.bands):
            raise ValueError("invalid spectral band: %r" % (self.bands,))
        self.lmax = self.bands[-1][1]
        n = self.nlat - 1
        if n < 2 or 2 * self.lmax >= n or self.lmax >= self.nlon // 2:
            raise ValueError(
                "grid too small for lmax=%d: nlat=%d nlon=%d"
                % (self.lmax, self.nlat, self.nlon)
            )

        # Clenshaw-Curtis 求积权重，与参考实现一致。
        theta = np.arange(self.nlat, dtype=np.float64) * (np.pi / n)
        weights = np.zeros(self.nlat, dtype=np.float64)
        if n % 2 == 0:
            weights[[0, -1]] = 1.0 / (n * n - 1)
        else:
            weights[[0, -1]] = 1.0 / (n * n)
        interior = np.ones(n - 1, dtype=np.float64)
        for k in range(1, (n + 1) // 2):
            interior -= 2.0 * np.cos(2.0 * k * theta[1:-1]) / (4.0 * k * k - 1.0)
        if n % 2 == 0:
            interior -= np.cos(n * theta[1:-1]) / (n * n - 1.0)
        weights[1:-1] = 2.0 * interior / n
        self.latitude_weights = (weights / 2.0).astype(np.float32)

        # 稳定的 associated Legendre 递推。
        basis = np.zeros((self.lmax + 1, self.lmax + 1, self.nlat),
                         dtype=np.float64)
        basis[0, 0] = 1.0 / np.sqrt(4.0 * np.pi)
        x = np.cos(theta)
        sint = np.sin(theta)
        for m in range(self.lmax + 1):
            if m:
                basis[m, m] = (
                    -np.sqrt((2.0 * m + 1.0) / (2.0 * m))
                    * sint * basis[m - 1, m - 1]
                )
            if m < self.lmax:
                basis[m + 1, m] = np.sqrt(2.0 * m + 3.0) * x * basis[m, m]
            for degree in range(m + 2, self.lmax + 1):
                a = np.sqrt(
                    (4.0 * degree * degree - 1.0)
                    / (degree * degree - m * m)
                )
                b = np.sqrt(
                    ((degree - 1) ** 2 - m * m)
                    / (4.0 * (degree - 1) ** 2 - 1.0)
                )
                basis[degree, m] = a * (
                    x * basis[degree - 1, m] - b * basis[degree - 2, m]
                )
        self.weighted_basis = (basis * weights).astype(np.float32)
        multiplicity = np.full(self.lmax + 1, 2.0, dtype=np.float32)
        multiplicity[0] = 1.0
        self.multiplicity = multiplicity

    def power(self, field, block: int = 16):
        """(..., nlat, nlon) -> (..., nband) 球谐带能量。"""
        arr = np.asarray(field, dtype=np.float32)
        if arr.ndim < 2:
            raise ValueError("field must have at least 2 dimensions")
        if tuple(arr.shape[-2:]) != (self.nlat, self.nlon):
            raise ValueError(
                "expected field grid (%d, %d), got %r"
                % (self.nlat, self.nlon, tuple(arr.shape[-2:]))
            )
        out_shape = tuple(arr.shape[:-2])
        x = arr.reshape(-1, self.nlat, self.nlon)
        nfield = x.shape[0]
        chunks = []
        blk = max(1, int(block))
        for i0 in range(0, nfield, blk):
            xb = x[i0:i0 + blk]
            mean = (xb.mean(axis=-1) * self.latitude_weights).sum(axis=-1)
            xb = xb - mean[:, None, None]
            # np.fft 无 forward norm，除以 nlon 等价于 norm="forward"。
            fft = np.fft.rfft(xb, axis=-1) / float(self.nlon)
            fft = fft[..., :self.lmax + 1]
            real = np.einsum(
                "lmh,nhm->nlm", self.weighted_basis, fft.real,
                optimize=True,
            )
            imag = np.einsum(
                "lmh,nhm->nlm", self.weighted_basis, fft.imag,
                optimize=True,
            )
            power = (
                (real * real + imag * imag) * self.multiplicity[None, None, :]
            ).sum(axis=-1) * (2.0 * np.pi) ** 2
            bands = np.stack(
                [power[:, lo:hi + 1].sum(axis=-1)
                 for lo, hi in self.bands],
                axis=-1,
            )
            chunks.append(bands)
        result = np.concatenate(chunks, axis=0).reshape(
            out_shape + (len(self.bands),)
        )
        return result


#: (nlat, nlon, bands) -> 计算器缓存：基函数是网格形状的纯函数，整 run 复用。
_SPH_CACHE: Dict[Tuple[int, int, Tuple[Tuple[int, int], ...]], _SphericalBandPower] = {}


def validate_global_lat(lat, atol: float = 1e-6) -> None:
    """校验「全球、等距、含南北极」的纬度网格，不满足则抛 ValueError。

    球谐展开只对全球球面有定义：连带勒让德函数只在整个纬圈上正交，把
    纬度截断成区域后 _SphericalBandPower 仍会把它当成"从极到极"的球面
    来算，结果是错的且不会报错。因此这里显式拒绝区域/非等距/不含极点
    的 lat（调用方应改用 zonal 谱）。

    带功率对纬度「行序翻转」是恒等的
    （P_lm(cos(pi-theta)) = (-1)^(l+m) P_lm(cos(theta))，模平方不变），
    所以不限制纬度升序还是降序。
    """
    lat = np.asarray(lat, dtype=np.float64)
    if lat.ndim != 1 or lat.size < 3 or not np.all(np.isfinite(lat)):
        raise ValueError("lat 必须是长度>=3 的一维有限数组")
    if np.max(np.abs(lat)) > 90.0 + atol:
        raise ValueError(
            "lat 必须是「度」且 |lat| <= 90；当前 max|lat|=%.6f（可能传了弧度？）"
            % np.max(np.abs(lat)))
    d = np.diff(lat)
    if d.size < 2 or not np.allclose(d, d.mean(), rtol=1e-6, atol=atol):
        raise ValueError("lat 必须是等距网格（dlat 不是常数）")
    global_ends = (
        (abs(lat[0] + 90.0) <= atol and abs(lat[-1] - 90.0) <= atol)
        or (abs(lat[0] - 90.0) <= atol and abs(lat[-1] + 90.0) <= atol)
    )
    if not global_ends:
        raise ValueError(
            "球谐带功率要求全球含极网格（-90..+90）；当前 lat=[%.4f, %.4f]"
            "（n=%d）。区域/bbox 检验请只用 zonal 谱。"
            % (lat[0], lat[-1], lat.size))


def spherical_band_power(field, lat=None, bands=DEFAULT_SPHERICAL_BANDS, block: int = 16):
    """球谐带功率（spherical band power），返回 (..., nband)。

    与 zonal_spectrum 不同：不是逐纬圈 1D FFT，而是球谐展开的带功率。
    公式与老仓 ``vfc/metrics/spectrum.py::spherical_band_power``（即参考
    xmetai ``SphericalBandPowerLoss.band_power``）逐位一致：去全球面积加权
    平均、经向 rFFT、associated Legendre basis 展开、按球谐总阶数区间求和。

    ``lat``（度）必传：既做全球网格校验，也防止把区域场喂进球谐展开。
    """
    arr = np.asarray(field)
    if arr.ndim < 2:
        raise ValueError("field must have at least 2 dimensions")
    if lat is None:
        raise ValueError(
            "spherical_band_power 需要 lat（度）来校验全球网格；请传 lat=file.lat")
    lat_arr = np.asarray(lat, dtype=np.float64)
    if lat_arr.size != arr.shape[-2]:
        raise ValueError(
            "lat 长度 %d != field 纬度维 %d" % (lat_arr.size, arr.shape[-2]))
    validate_global_lat(lat_arr)          # 区域/非全球 -> raise
    bands0 = tuple((int(lo), int(hi)) for lo, hi in bands)
    key = (int(arr.shape[-2]), int(arr.shape[-1]), bands0)
    obj = _SPH_CACHE.get(key)
    if obj is None:
        obj = _SphericalBandPower(key[0], key[1], bands0)
        _SPH_CACHE[key] = obj
    return obj.power(arr, block=block)


class SphericalBands(Metric):
    """球谐带功率（总波数分带）：预报/观测两侧 + 带内功率比。

    只对**全球含极网格**有定义（区域网格在装配期报错，不是静默算错——
    球谐基函数只在全球球面上正交）。带边界是总波数的闭区间，缺省沿用
    老仓分段 (1,4)/(5,20)/(21,40)/(41,64)/(65,128)。

    输出走长表：每带一组 ``spherical_pred / spherical_obs / spherical_ratio``
    行，带名（如 ``1_4``）落 ``group`` 列——对标老仓
    ``spherical_bands_<date>_<var>.csv`` 的
    ``spherical_{pred|obs|ratio}_{lo}_{hi}`` 列契约，透视即得。
    """

    PRODUCT_KIND = "specialized"

    def __init__(self, bands=None, block: int = 16, params=None):
        super().__init__(name="spherical_bands", version="1.0.0", params=params or {})
        self.bands = tuple(
            (int(lo), int(hi)) for lo, hi in (bands or DEFAULT_SPHERICAL_BANDS)
        )
        self.block = int(block)

    def requirements(self) -> MetricRequirements:
        return MetricRequirements(
            product_type=ProductType.DETERMINISTIC_FIELD,
            variables=["*"],
        )

    def _band_powers(self, batch: EvaluationBatch):
        forecast = _values(batch.forecast)
        observation = _values(batch.observation)
        if forecast.ndim < 2 or observation.ndim < 2:
            raise MetricError(
                f"球谐带功率需要 (..., lat, lon) 场，实际 ndim={forecast.ndim}",
                variable=self.name,
            )
        lat = _lat_from_batch(batch, self.name)
        f = forecast.reshape(-1, forecast.shape[-2], forecast.shape[-1])
        o = observation.reshape(-1, observation.shape[-2], observation.shape[-1])
        try:
            forecast_bands = spherical_band_power(
                f, lat=lat, bands=self.bands, block=self.block
            )
            observation_bands = spherical_band_power(
                o, lat=lat, bands=self.bands, block=self.block
            )
        except ValueError as exc:
            # 区域/非全球网格：配置错误，必须立刻暴露（老仓批量路径是降级
            # 只出 zonal，这里用户显式点名了球谐，静默降级等于丢结果）
            raise MetricError(f"球谐带功率：{exc}", variable=self.name) from exc
        return forecast_bands, observation_bands, int(f.shape[0])

    def accumulate(self, batch: EvaluationBatch) -> MetricState:
        forecast_bands, observation_bands, n_samples = self._band_powers(batch)
        return MetricState(
            metric_name=self.name,
            metric_version=self.version,
            data={
                "forecast_bands": np.asarray(forecast_bands.sum(axis=0), dtype="f8"),
                "observation_bands": np.asarray(observation_bands.sum(axis=0), dtype="f8"),
                "n_samples": n_samples,
            },
            n_accumulated=1,
        )

    def merge(self, states: List[MetricState]) -> MetricState:
        if not states:
            raise MetricError("Cannot merge empty states list")
        first = states[0]
        forecast_total = np.array(first.data["forecast_bands"], dtype="f8")
        observation_total = np.array(first.data["observation_bands"], dtype="f8")
        for state in states[1:]:
            if state.metric_name != first.metric_name:
                raise MetricError(
                    f"Cannot merge states from different metrics: "
                    f"{first.metric_name} vs {state.metric_name}",
                    variable=self.name,
                )
            forecast_total = forecast_total + state.data["forecast_bands"]
            observation_total = observation_total + state.data["observation_bands"]
        return MetricState(
            metric_name=self.name,
            metric_version=self.version,
            data={
                "forecast_bands": forecast_total,
                "observation_bands": observation_total,
                "n_samples": sum(state.data["n_samples"] for state in states),
            },
            n_accumulated=sum(state.n_accumulated for state in states),
        )

    def finalize(self, state: MetricState) -> MetricResult:
        count = max(int(state.data["n_samples"]), 1)
        forecast_bands = np.asarray(state.data["forecast_bands"], dtype="f8") / count
        observation_bands = np.asarray(state.data["observation_bands"], dtype="f8") / count

        value: Dict[str, Dict[str, float]] = {}
        with np.errstate(divide="ignore", invalid="ignore"):
            for (lo, hi), fp, op in zip(
                self.bands, forecast_bands, observation_bands
            ):
                ratio = float(fp / op) if op > 0 else float("nan")
                value[f"{lo}_{hi}"] = {
                    "spherical_pred": float(fp),
                    "spherical_obs": float(op),
                    "spherical_ratio": ratio,
                }

        return MetricResult(
            metric_name=self.name,
            metric_version=self.version,
            value=value,
            status=ResultStatus.SUCCESS,
            n_requested=count,
            n_valid=count,
            aggregation="spherical_band_power",
            unit="1",  # 进长表的 ratio 无量纲；pred/obs 的功率单位随变量走
            product_kind=self.PRODUCT_KIND,
        )
