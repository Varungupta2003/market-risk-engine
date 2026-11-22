"""Loader and validation tests (no network)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mre import data
from mre.returns import equal_weights, portfolio_returns, simple_returns


def _frame(values: list[list[float]]) -> pd.DataFrame:
    idx = pd.bdate_range("2024-01-01", periods=len(values))
    return pd.DataFrame(values, index=idx, columns=["A", "B"])


def test_validate_ok_sorts_and_casts() -> None:
    df = _frame([[1, 2], [3, 4], [5, 6]]).iloc[::-1]
    out = data.validate_prices(df, ["A", "B"])
    assert out.index.is_monotonic_increasing
    assert out.dtypes.eq(float).all()


@pytest.mark.parametrize(
    "bad, msg",
    [
        (pd.DataFrame({"A": [1.0], "B": [2.0]}), "DatetimeIndex"),
        (_frame([[1, 2], [3, 4]]).set_axis(pd.to_datetime(["2024-01-01"] * 2)), "duplicate"),
        (_frame([[1, 2], [-3, 4]]), "positive"),
        (_frame([[np.nan, 2], [3, 4]]), "at least"),
    ],
)
def test_validate_rejects(bad: pd.DataFrame, msg: str) -> None:
    with pytest.raises(ValueError, match=msg):
        data.validate_prices(bad, ["A", "B"])


def test_validate_missing_ticker() -> None:
    with pytest.raises(ValueError, match="missing tickers"):
        data.validate_prices(_frame([[1, 2], [3, 4]]), ["A", "C"])


def test_leading_nan_dropped_silently_interior_nan_warns() -> None:
    lead = _frame([[1, np.nan], [2, 3], [3, 4]])
    assert len(data.validate_prices(lead)) == 2
    interior = _frame([[1, 2], [2, np.nan], [3, 4]])
    with pytest.warns(UserWarning, match="interior"):
        assert len(data.validate_prices(interior)) == 2


def test_load_prices_uses_cache(tmp_path, synthetic_prices) -> None:
    path = tmp_path / "prices.csv"
    synthetic_prices.to_csv(path)
    out = data.load_prices(path, list(synthetic_prices.columns), start="2020-01-01")
    assert out.attrs["source"] == "cache"
    pd.testing.assert_frame_equal(out, synthetic_prices, check_freq=False, check_names=False)


def test_download_failure_raises_not_fabricates(monkeypatch, tmp_path) -> None:
    import yfinance

    monkeypatch.setattr(yfinance, "download", lambda *a, **k: pd.DataFrame())
    with pytest.raises(RuntimeError, match="no data"):
        data.load_prices(tmp_path / "none.csv", ["SPY"], refresh=True)
    assert not (tmp_path / "none.csv").exists()


def test_download_success_path_and_cache_write(monkeypatch, tmp_path) -> None:
    import yfinance

    idx = pd.DatetimeIndex(["2024-01-02", "2024-01-03"], tz="America/New_York")
    cols = pd.MultiIndex.from_product([["Close", "Open"], ["SPY", "TLT"]])
    raw = pd.DataFrame([[1.0, 2.0, 9.0, 9.0], [1.1, 2.2, 9.0, 9.0]], index=idx, columns=cols)
    monkeypatch.setattr(yfinance, "download", lambda *a, **k: raw)
    out = data.load_prices(tmp_path / "p.csv", ["SPY", "TLT"], refresh=True)
    assert out.attrs["source"] == "download" and out.index.tz is None
    assert list(out.columns) == ["SPY", "TLT"] and out.iloc[1, 1] == 2.2
    assert (tmp_path / "p.csv").exists()


def test_stale_cache_start_warns(tmp_path, synthetic_prices) -> None:
    path = tmp_path / "prices.csv"
    synthetic_prices.to_csv(path)
    with pytest.warns(UserWarning, match="cache starts"):
        data.load_prices(path, list(synthetic_prices.columns), start="2018-01-01")


def test_returns_and_portfolio() -> None:
    prices = _frame([[100, 50], [110, 45], [99, 45]])
    r = simple_returns(prices)
    np.testing.assert_allclose(r["A"], [0.10, -0.10])
    np.testing.assert_allclose(r["B"], [-0.10, 0.0])
    np.testing.assert_allclose(portfolio_returns(r), [0.0, -0.05], atol=1e-12)
    with pytest.raises(ValueError):
        portfolio_returns(r, np.array([0.7, 0.7]))
    with pytest.raises(ValueError):
        equal_weights(0)
