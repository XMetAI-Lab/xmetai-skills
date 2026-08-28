from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills" / "xmetai-weather-modeling" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from evaluation import runner


def test_rmse_and_threshold_metrics_read_each_pair_once(monkeypatch) -> None:
    pairs = [
        (Path("pred_20230101.nc"), Path("obs_20230101.nc")),
        (Path("pred_20230102.nc"), Path("obs_20230102.nc")),
    ]
    calls = []

    def fake_load_pair(pred_path, obs_path):
        calls.append((pred_path, obs_path))
        offset = float(len(calls) - 1)
        obs = np.zeros((2, 1, 2, 2), dtype=np.float64)
        pred = np.full_like(obs, offset + 1.0)
        return pred, obs, ["tp"]

    monkeypatch.setattr(runner, "load_pair", fake_load_pair)
    report = runner.compute_all_metrics(
        pairs,
        metrics=["rmse", "ts", "pod", "far", "fb"],
        thresholds=[0.5],
        channel="tp",
        neighborhood=1,
    )

    assert len(calls) == len(pairs)
    assert report["n_leads"] == 2
    assert set(report["threshold_metrics"]["0.5"]) == {"ts", "pod", "far", "fb"}


def test_metric_functions_remain_exported_from_cli_facade() -> None:
    import evaluate_pred

    assert evaluate_pred.compute_rmse is not None
    assert evaluate_pred.compute_threshold_metrics is not None
    assert evaluate_pred.compute_tcc is not None
    assert evaluate_pred.compute_ps is not None
    assert evaluate_pred.compute_ips is not None


def test_split_curve_renderer_writes_rmse_figure(tmp_path: Path) -> None:
    from evaluation.visualization.curves import plot_metric_curves

    report = {
        "n_leads": 2,
        "levels": ["z500"],
        "rmse_per_level": {"z500": [1.0, 2.0]},
        "channel": "tp",
    }
    plot_metric_curves(report, ["rmse"], [], tmp_path, freq=24)
    assert (tmp_path / "rmse_curves.png").is_file()
