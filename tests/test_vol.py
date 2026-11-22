"""Volatility models: no look-ahead, determinism, metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mre import vol

WINDOW, REFIT, K = 150, 10, 300


def _shuffle_after(r: pd.Series, k: int) -> pd.Series:
    out = r.copy()
    out.iloc[k:] = out.iloc[k:].to_numpy()[np.random.default_rng(0).permutation(len(r) - k)]
    return out


@pytest.mark.parametrize(
    "fn",
    [
        lambda r: vol.garch_forecasts(r, WINDOW, REFIT, start=vol.FEATURE_LAG + WINDOW),
        lambda r: vol.ml_forecasts(r, WINDOW, REFIT),
        lambda r: vol.har_forecasts(r, WINDOW, REFIT),
    ],
    ids=["garch", "ml", "har"],
)
def test_walk_forward_no_look_ahead(fn, synthetic_port) -> None:
    base = fn(synthetic_port)
    alt = fn(_shuffle_after(synthetic_port, K))
    cutoff = synthetic_port.index[K]  # forecast for day K uses data up to K-1
    pd.testing.assert_series_equal(base[:cutoff], alt[:cutoff])
    assert not base[base.index > cutoff].equals(alt[alt.index > cutoff])  # test has power


def test_ml_deterministic_and_positive(synthetic_port) -> None:
    a = vol.ml_forecasts(synthetic_port, WINDOW, REFIT, seed=1)
    b = vol.ml_forecasts(synthetic_port, WINDOW, REFIT, seed=1)
    pd.testing.assert_series_equal(a, b)
    assert (a > 0).all() and a.attrs["floor_hits"] == 0


def test_garch_start_validation(synthetic_port) -> None:
    with pytest.raises(ValueError):
        vol.garch_forecasts(synthetic_port, WINDOW, start=WINDOW - 1)
    with pytest.raises(ValueError):
        vol.ml_forecasts(synthetic_port, WINDOW, start=vol.FEATURE_LAG + 1)


def test_features_use_only_past() -> None:
    r = pd.Series(np.arange(1.0, 101.0))
    f = vol.make_vol_features(r)
    assert f["abs_lag1"].iloc[10] == 10.0  # r at index 9
    assert f["rv_5"].iloc[10] == pytest.approx(np.mean(np.arange(6.0, 11.0) ** 2))
    assert f.iloc[vol.FEATURE_LAG].notna().all()
    assert f.iloc[vol.FEATURE_LAG - 1].isna().any()


def test_vol_metrics_hand_values() -> None:
    idx = pd.RangeIndex(4)
    r = pd.Series([0.01, -0.01, 0.02, 0.0], index=idx)
    s = pd.Series(0.01, index=idx)  # h = 1 %^2; proxy = [1, 1, 4, 0]
    m = vol.vol_metrics(s, r)
    assert m["rmse_var_pct2"] == pytest.approx(np.sqrt((0 + 0 + 9 + 1) / 4))
    assert m["qlike"] == pytest.approx((1 + 1 + 4 + 0) / 4)  # log(1) = 0


def test_vol_var_es_matches_parametric() -> None:
    from mre.var_es import parametric_var_es

    out = vol.vol_var_es(pd.Series([0.02]))
    var, es = parametric_var_es(0.0, 0.02)
    assert out["var"].iloc[0] == pytest.approx(var)
    assert out["es"].iloc[0] == pytest.approx(es)
