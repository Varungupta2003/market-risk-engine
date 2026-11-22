"""Coverage tests and the rolling VaR backtest."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from mre.backtest import (
    backtest_table,
    christoffersen_independence,
    kupiec_pof,
    rolling_var_es,
)
from mre.returns import simple_returns


def test_kupiec_hand_case() -> None:
    # n=250, x=5, p=0.01, phat=0.02
    # ll0 = 245 ln .99 + 5 ln .01 = -25.488183 ; ll1 = 245 ln .98 + 5 ln .02 = -24.509778
    # LR = 2 * 0.978405 = 1.956810 ; p = P(chi2_1 > 1.956810) = 0.161855
    lr, p = kupiec_pof(5, 250, 0.99)
    assert lr == pytest.approx(1.956810, abs=1e-6)
    assert p == pytest.approx(0.161855, abs=1e-6)


def test_kupiec_edges() -> None:
    lr0, _ = kupiec_pof(0, 250, 0.99)
    assert lr0 == pytest.approx(-2 * 250 * math.log(0.99))
    lr_exact, p_exact = kupiec_pof(10, 1000, 0.99)  # phat == p
    assert lr_exact == 0.0 and math.copysign(1, lr_exact) == 1  # no -0.0
    assert p_exact == pytest.approx(1.0)
    assert np.isfinite(kupiec_pof(250, 250, 0.99)[0])
    with pytest.raises(ValueError):
        kupiec_pof(5, 0)
    with pytest.raises(ValueError):
        kupiec_pof(6, 5)


def test_christoffersen() -> None:
    assert christoffersen_independence(np.zeros(100)) == (0.0, 1.0)
    clustered = np.zeros(200, dtype=int)
    clustered[50:56] = 1
    spread = np.zeros(200, dtype=int)
    spread[::33] = 1
    assert christoffersen_independence(clustered)[1] < 0.01
    assert christoffersen_independence(spread)[1] > 0.5
    with pytest.raises(ValueError):
        christoffersen_independence(np.array([1]))


def test_backtest_table_counts() -> None:
    idx = pd.bdate_range("2024-01-01", periods=5)
    realised = pd.Series([-0.03, 0.01, -0.01, -0.05, 0.0], index=idx)
    var = pd.Series(0.02, index=idx)
    row = backtest_table({"x": var}, realised).iloc[0]
    assert row["breaches"] == 2 and row["n_obs"] == 5
    assert row["expected"] == pytest.approx(0.05)
    assert 0 <= row["cc_p"] <= 1


def test_rolling_var_no_look_ahead(synthetic_prices) -> None:
    rets = simple_returns(synthetic_prices)
    k = 300
    base = rolling_var_es(rets, window=200, n_sims=500, seed=3)
    shuffled = rets.copy()
    rng = np.random.default_rng(0)
    shuffled.iloc[k:] = shuffled.iloc[k:].to_numpy()[rng.permutation(len(rets) - k)]
    alt = rolling_var_es(shuffled, window=200, n_sims=500, seed=3)
    cols = [c for c in base.columns if c != "realised"]
    # forecasts for days <= k use data up to k-1 only
    pd.testing.assert_frame_equal(base[cols].iloc[: k - 200 + 1], alt[cols].iloc[: k - 200 + 1])
    assert not base[cols].iloc[k - 200 + 2 :].equals(alt[cols].iloc[k - 200 + 2 :])


def test_rolling_var_es_ge_var(synthetic_prices) -> None:
    out = rolling_var_es(simple_returns(synthetic_prices), window=200, n_sims=500)
    for m in ("historical", "parametric", "monte_carlo"):
        assert (out[f"es_{m}"] >= out[f"var_{m}"] - 1e-12).all()
    with pytest.raises(ValueError):
        rolling_var_es(simple_returns(synthetic_prices), window=10_000)
