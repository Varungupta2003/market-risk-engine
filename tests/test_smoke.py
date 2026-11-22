"""End-to-end smoke test of run_all.py on a small SYNTHETIC dataset."""

from __future__ import annotations

import json

import pandas as pd
import pytest

import run_all


def test_run_all_synthetic(tmp_path) -> None:
    meta = run_all.main(
        ["--synthetic", "--synthetic-days", "400", "--results-dir", str(tmp_path),
         "--window", "150", "--n-sims", "500", "--refit-every", "20"]
    )
    for f in ["full_sample_var_es", "backtest_summary", "backtest_summary_full_period",
              "rolling_var_es", "vol_forecasts", "vol_metrics"]:
        assert (tmp_path / f"{f}.csv").stat().st_size > 0
    for f in ["var_backtest", "breaches", "vol_forecasts", "return_distribution"]:
        assert (tmp_path / "figures" / f"{f}.png").stat().st_size > 0
    saved = json.loads((tmp_path / "run_metadata.json").read_text())
    assert saved["synthetic"] is True and saved["data_source"] == "synthetic"
    assert meta["backtest_n_obs"] == 399 - (66 + 150)
    table = pd.read_csv(tmp_path / "backtest_summary.csv")
    methods = ["historical", "parametric", "monte_carlo", "garch", "ml", "har"]
    assert list(table["method"]) == methods


def test_synthetic_refuses_default_results_dir() -> None:
    with pytest.raises(SystemExit):
        run_all.main(["--synthetic"])
