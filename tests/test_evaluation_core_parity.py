from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
CORE = Path(r"D:\xmetai-core")
if not CORE.exists():
    pytest.skip("local xmetai-core checkout is unavailable", allow_module_level=True)
sys.path.insert(0, str(CORE))

from xmetai.metrics.ips import IntegratedPatternScore
from xmetai.metrics.ps import ClimatePSScore


SCRIPT = ROOT / "skills" / "xmetai-weather-modeling" / "scripts" / "evaluate_pred.py"
SPEC = importlib.util.spec_from_file_location("evaluate_pred", SCRIPT)
assert SPEC and SPEC.loader
offline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(offline)


def test_ps_matches_core_counts_and_formula_including_zero() -> None:
    pred = np.array([0.0, 0.1, -0.3, 0.7, -0.8], dtype=np.float32)
    obs = np.array([0.2, 0.0, -0.4, -0.6, -0.9], dtype=np.float32)
    result = offline.compute_ps(
        [pred.reshape(1, 1, 1, -1)],
        [obs.reshape(1, 1, 1, -1)],
        ["x"],
    )
    n0, n1, n2, n = ClimatePSScore._accumulate_counts(
        torch.from_numpy(pred), torch.from_numpy(obs), 0.2, 0.5
    )
    expected = ClimatePSScore._ps_formula(n0, n1, n2, n)
    np.testing.assert_allclose(result["ps_per_level"]["x"], [expected])
    np.testing.assert_allclose(result["ps_overall"]["x"], expected)


def test_ips_matches_core_pentads_and_pattern_scores() -> None:
    rng = np.random.default_rng(7)
    pred = rng.normal(size=(42, 1, 2, 3)).astype(np.float32)
    obs = (pred * 0.7 + rng.normal(scale=0.4, size=pred.shape)).astype(np.float32)
    test_times = [24 * (idx + 1) for idx in range(42)]

    result = offline.compute_ips([pred], [obs], ["x"], freq=24, test_times=test_times)
    core = IntegratedPatternScore(["x"], test_times)
    core.reset()

    assert core.pentad_indices[3] == list(range(10, 15))
    assert result["pentad_labels"] == [f"P{p}" for p in core.pentads]
    for out_idx, pentad in enumerate(core.pentads):
        indices = core.pentad_indices[pentad]
        if not indices:
            assert result["ips_per_level"]["x"]["pcc"][out_idx] == 0.0
            assert result["ips_per_level"]["x"]["as"][out_idx] == 0.0
            assert result["ips_per_level"]["x"]["ips"][out_idx] == 0.0
            continue
        pred_pentad = torch.from_numpy(pred[indices, 0].mean(axis=0))
        obs_pentad = torch.from_numpy(obs[indices, 0].mean(axis=0))
        pcc, as_score, ips = core._pattern_scores(pred_pentad, obs_pentad, None)
        np.testing.assert_allclose(result["ips_per_level"]["x"]["pcc"][out_idx], pcc, rtol=1e-6)
        np.testing.assert_allclose(result["ips_per_level"]["x"]["as"][out_idx], as_score, rtol=1e-6)
        np.testing.assert_allclose(result["ips_per_level"]["x"]["ips"][out_idx], ips, rtol=1e-6)


def test_ips_identical_constant_field_is_perfect_like_core() -> None:
    pred = np.ones((15, 1, 2, 2), dtype=np.float32)
    result = offline.compute_ips([pred], [pred.copy()], ["x"], freq=24)
    assert result["ips_per_level"]["x"]["pcc"][0] == 1.0
    assert result["ips_per_level"]["x"]["ips"][0] == 100.0
