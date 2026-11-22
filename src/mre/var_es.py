"""Value-at-Risk and Expected Shortfall estimators.

Sign convention: VaR and ES are reported as POSITIVE loss numbers (fractions of
portfolio value). A breach on day t means ``return_t < -VaR_t``.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def _check_alpha(alpha: float) -> None:
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")


def historical_var_es(returns: np.ndarray, alpha: float = 0.99) -> tuple[float, float]:
    """Historical-simulation VaR and ES.

    VaR is the empirical ``alpha`` quantile of losses (``-returns``) using the
    "higher" order statistic, so with n observations the tail holds about
    ``n * (1 - alpha)`` losses. ES is the mean of losses at or beyond VaR.

    Args:
        returns: 1-D array of portfolio returns.
        alpha: Confidence level, e.g. 0.99.

    Returns:
        (VaR, ES) as positive loss numbers.
    """
    _check_alpha(alpha)
    r = np.asarray(returns, dtype=float)
    if r.ndim != 1 or r.size == 0:
        raise ValueError("returns must be a non-empty 1-D array")
    if not np.isfinite(r).all():
        raise ValueError("returns must be finite")
    losses = -r
    var = float(np.quantile(losses, alpha, method="higher"))
    es = float(losses[losses >= var].mean())
    return var, es


def parametric_var_es(mu: float, sigma: float, alpha: float = 0.99) -> tuple[float, float]:
    """Normal (variance-covariance) VaR and ES for returns ~ N(mu, sigma^2).

    ``VaR = -mu + sigma * z_alpha`` and ``ES = -mu + sigma * phi(z_alpha) / (1 - alpha)``.

    Args:
        mu: Mean return.
        sigma: Return standard deviation (>= 0).
        alpha: Confidence level.

    Returns:
        (VaR, ES) as positive loss numbers.
    """
    _check_alpha(alpha)
    if sigma < 0 or not np.isfinite(sigma):
        raise ValueError("sigma must be finite and non-negative")
    z = norm.ppf(alpha)
    return float(-mu + sigma * z), float(-mu + sigma * norm.pdf(z) / (1.0 - alpha))


def portfolio_moments(asset_returns: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    """Portfolio mean and standard deviation from asset sample moments.

    Args:
        asset_returns: 2-D array (obs x assets).
        weights: Portfolio weights.

    Returns:
        (mu_p, sigma_p) using the sample mean and ddof=1 covariance.
    """
    x = np.asarray(asset_returns, dtype=float)
    w = np.asarray(weights, dtype=float)
    cov = np.cov(x, rowvar=False, ddof=1)
    return float(w @ x.mean(axis=0)), float(np.sqrt(w @ cov @ w))


def monte_carlo_var_es(
    mean: np.ndarray,
    cov: np.ndarray,
    weights: np.ndarray,
    alpha: float = 0.99,
    n_sims: int = 10_000,
    seed: int | None = 42,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Monte Carlo VaR/ES from multivariate-normal asset returns.

    Simulates ``n_sims`` asset return vectors, forms portfolio returns and
    applies :func:`historical_var_es` to the simulated sample.

    Args:
        mean: Asset mean vector.
        cov: Asset covariance matrix.
        weights: Portfolio weights.
        alpha: Confidence level.
        n_sims: Number of simulated scenarios.
        seed: Seed used when ``rng`` is not given.
        rng: Optional generator (takes precedence over ``seed``).

    Returns:
        (VaR, ES) as positive loss numbers.
    """
    gen = rng if rng is not None else np.random.default_rng(seed)
    sims = gen.multivariate_normal(mean, cov, size=n_sims, method="cholesky")
    return historical_var_es(sims @ np.asarray(weights, dtype=float), alpha)
