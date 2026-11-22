"""Walk-forward volatility forecasts: GARCH(1,1), gradient boosting, HAR-OLS baseline.

Every model forecasts the variance of day t using returns up to t-1 only.
Models are refit every ``refit_every`` days on the previous ``window`` days;
between refits GARCH is filtered forward with fixed parameters and the
feature-based models are applied to features that only use lagged data.

Internally returns are scaled to percent; outputs are in return units.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from arch import arch_model
from scipy.stats import norm

FEATURE_LAG: int = 66  # first row index with all features defined (66-day rolling, shifted 1)
MIN_TRAIN_ROWS: int = 100


def garch_forecasts(
    returns: pd.Series, window: int = 500, refit_every: int = 20, start: int | None = None
) -> pd.Series:
    """One-step-ahead GARCH(1,1) volatility forecasts (constant mean, normal).

    Args:
        returns: Portfolio simple returns.
        window: Rolling estimation window.
        refit_every: Days between parameter re-estimation.
        start: First forecast index (default ``window``); must be >= ``window``.

    Returns:
        Series of forecast volatilities (return units) from index ``start``.
        ``attrs["n_fits"]`` and ``attrs["n_not_converged"]`` count the refits.
    """
    r = returns.to_numpy(dtype=float) * 100.0
    start = window if start is None else start
    if start < window or start >= len(r):
        raise ValueError("start must satisfy window <= start < len(returns)")
    out = np.full(len(r), np.nan)
    n_fits = n_bad = 0
    for t0 in range(start, len(r), refit_every):
        train = r[t0 - window : t0]
        res = arch_model(train, mean="Constant", vol="GARCH", p=1, q=1, rescale=False).fit(
            disp="off"
        )
        n_fits += 1
        n_bad += int(res.convergence_flag != 0)
        mu, omega = res.params["mu"], res.params["omega"]
        a, b = res.params["alpha[1]"], res.params["beta[1]"]
        sig2 = float(res.conditional_volatility[-1] ** 2)  # variance of day t0-1
        eps = train[-1] - mu
        for t in range(t0, min(t0 + refit_every, len(r))):
            sig2 = omega + a * eps**2 + b * sig2  # uses data up to t-1 only
            out[t] = sig2
            eps = r[t] - mu  # revealed after forecasting day t
    series = pd.Series(np.sqrt(out[start:]) / 100.0, index=returns.index[start:], name="garch")
    series.attrs.update(n_fits=n_fits, n_not_converged=n_bad)
    return series


def make_vol_features(returns: pd.Series) -> pd.DataFrame:
    """Lagged features for forecasting day-t variance (all use data <= t-1).

    Features: |r| lags 1-5 and HAR-style realised variance (mean r^2) over
    1, 5, 22 and 66 days, all shifted by one day.

    Args:
        returns: Return series.

    Returns:
        Feature DataFrame aligned to ``returns`` (leading rows contain NaN).
    """
    a = returns.abs()
    r2 = returns**2
    feats = {f"abs_lag{k}": a.shift(k) for k in range(1, 6)}
    for k in (1, 5, 22, 66):
        feats[f"rv_{k}"] = r2.rolling(k).mean().shift(1)
    return pd.DataFrame(feats, index=returns.index)


def _make_model(seed: int) -> tuple[Any, str]:
    """LightGBM gamma regressor, or sklearn GradientBoosting if lightgbm is missing.

    The gamma objective (log link) keeps forecasts positive and its deviance
    is consistent with the QLIKE loss used for evaluation.
    """
    try:
        from lightgbm import LGBMRegressor

        model = LGBMRegressor(
            objective="gamma",
            n_estimators=200,
            learning_rate=0.03,
            num_leaves=7,
            min_child_samples=100,
            random_state=seed,
            n_jobs=1,
            deterministic=True,
            verbose=-1,
        )
        return model, "lightgbm"
    except ImportError:  # pragma: no cover - fallback path
        from sklearn.ensemble import HistGradientBoostingRegressor

        model = HistGradientBoostingRegressor(
            loss="gamma", max_iter=200, max_leaf_nodes=7, min_samples_leaf=100,
            learning_rate=0.03, random_state=seed,
        )
        return model, "sklearn_hgb"


class _HarOLS:
    """HAR-style OLS baseline: r_t^2 on rv_1, rv_5, rv_22 (plus intercept)."""

    cols = slice(5, 8)  # rv_1, rv_5, rv_22 columns of make_vol_features

    def fit(self, x: np.ndarray, y: np.ndarray) -> _HarOLS:
        design = np.column_stack([np.ones(len(x)), x[:, self.cols]])
        self.coef_ = np.linalg.lstsq(design, y, rcond=None)[0]
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.column_stack([np.ones(len(x)), x[:, self.cols]]) @ self.coef_


def _walk_forward(
    returns: pd.Series,
    make: Callable[[], Any],
    name: str,
    window: int,
    refit_every: int,
    start: int | None,
    target_eps: float = 0.0,
) -> pd.Series:
    """Refit ``make()`` every ``refit_every`` days on the last ``window`` rows; predict ahead.

    Training rows for a refit at t0 have targets r_{t}^2 with t < t0, so no
    future data is used. Predictions are floored at 1% of the training-window
    mean variance (a safety net; the number of days it binds is recorded in
    ``attrs["floor_hits"]``).
    """
    x = make_vol_features(returns * 100.0).to_numpy()
    y = (returns.to_numpy(dtype=float) * 100.0) ** 2
    start = FEATURE_LAG + window if start is None else start
    if start < FEATURE_LAG + MIN_TRAIN_ROWS or start >= len(y):
        raise ValueError(f"start must be in [FEATURE_LAG + {MIN_TRAIN_ROWS}, len(returns))")
    out = np.full(len(y), np.nan)
    floor_hits = 0
    for t0 in range(start, len(y), refit_every):
        lo = max(FEATURE_LAG, t0 - window)
        model = make().fit(x[lo:t0], y[lo:t0] + target_eps)
        hi = min(t0 + refit_every, len(y))
        pred = model.predict(x[t0:hi])
        floor = 0.01 * y[lo:t0].mean()
        floor_hits += int((pred < floor).sum())
        out[t0:hi] = np.maximum(pred, floor)
    series = pd.Series(np.sqrt(out[start:]) / 100.0, index=returns.index[start:], name=name)
    series.attrs["floor_hits"] = floor_hits
    return series


def ml_forecasts(
    returns: pd.Series,
    window: int = 500,
    refit_every: int = 20,
    start: int | None = None,
    seed: int = 42,
) -> pd.Series:
    """Walk-forward gradient-boosting volatility forecasts.

    Target is the squared return (percent^2) of day t, fit with a gamma
    objective, so the prediction is a positive conditional-variance forecast.

    Args:
        returns: Portfolio simple returns.
        window: Training window (rows).
        refit_every: Days between refits.
        start: First forecast index (default ``FEATURE_LAG + window``).
        seed: Model random seed.

    Returns:
        Series of forecast volatilities (return units) from index ``start``,
        with ``attrs["model"]`` naming the backend and ``attrs["floor_hits"]``.
    """
    backend = _make_model(seed)[1]
    # gamma needs strictly positive labels; 1e-6 %^2 is negligible vs typical ~1 %^2
    series = _walk_forward(returns, lambda: _make_model(seed)[0], "ml", window, refit_every,
                           start, target_eps=1e-6)
    series.attrs["model"] = backend
    return series


def har_forecasts(
    returns: pd.Series, window: int = 500, refit_every: int = 20, start: int | None = None
) -> pd.Series:
    """Walk-forward HAR-OLS variance baseline (same windows as :func:`ml_forecasts`).

    Returns:
        Series of forecast volatilities with ``attrs["floor_hits"]``.
    """
    return _walk_forward(returns, _HarOLS, "har", window, refit_every, start)


def vol_metrics(sigma: pd.Series, returns: pd.Series) -> dict[str, float]:
    """RMSE and QLIKE of variance forecasts against squared returns.

    Both are computed in percent units. Proxy: ``r_t^2`` (daily realised
    variance proxy). QLIKE = mean(log h + r^2 / h); both losses are robust to
    proxy noise in the sense of Patton (2011).

    Args:
        sigma: Forecast volatility (return units).
        returns: Realised returns aligned on the same index.

    Returns:
        Dict with ``n_obs``, ``rmse_var_pct2`` and ``qlike``.
    """
    s, r = sigma.align(returns, join="inner")
    h = (s.to_numpy() * 100.0) ** 2
    proxy = (r.to_numpy() * 100.0) ** 2
    return {
        "n_obs": float(len(h)),
        "rmse_var_pct2": float(np.sqrt(np.mean((h - proxy) ** 2))),
        "qlike": float(np.mean(np.log(h) + proxy / h)),
    }


def vol_var_es(sigma: pd.Series, alpha: float = 0.99) -> pd.DataFrame:
    """Zero-mean normal VaR/ES from a volatility forecast series.

    Args:
        sigma: Forecast volatility (return units).
        alpha: Confidence level.

    Returns:
        DataFrame with ``var`` and ``es`` columns (positive losses).
    """
    z = norm.ppf(alpha)
    return pd.DataFrame(
        {"var": sigma * z, "es": sigma * norm.pdf(z) / (1 - alpha)}, index=sigma.index
    )
