"""Asset and portfolio returns."""

from __future__ import annotations

import numpy as np
import pandas as pd


def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily simple returns ``P_t / P_{t-1} - 1`` (first row dropped).

    Args:
        prices: Validated wide price table.

    Returns:
        DataFrame of simple returns with no missing values.
    """
    rets = prices.pct_change().iloc[1:]
    if rets.isna().to_numpy().any():
        raise ValueError("returns contain NaN; validate prices first")
    return rets


def equal_weights(n_assets: int) -> np.ndarray:
    """Equal weights summing to one."""
    if n_assets <= 0:
        raise ValueError("n_assets must be positive")
    return np.full(n_assets, 1.0 / n_assets)


def portfolio_returns(asset_returns: pd.DataFrame, weights: np.ndarray | None = None) -> pd.Series:
    """Portfolio simple return with weights rebalanced daily.

    Args:
        asset_returns: Asset simple returns (dates x assets).
        weights: Portfolio weights; equal weights if None. Must sum to 1.

    Returns:
        Series of portfolio returns named ``portfolio``.
    """
    w = equal_weights(asset_returns.shape[1]) if weights is None else np.asarray(weights, float)
    if w.shape != (asset_returns.shape[1],):
        raise ValueError("weights length must match number of assets")
    if not np.isclose(w.sum(), 1.0):
        raise ValueError("weights must sum to 1")
    return pd.Series(asset_returns.to_numpy() @ w, index=asset_returns.index, name="portfolio")
