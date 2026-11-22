"""Rolling-window VaR/ES forecasts and coverage tests (Kupiec, Christoffersen)."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy.special import xlogy
from scipy.stats import chi2

from .returns import equal_weights
from .var_es import historical_var_es, monte_carlo_var_es, parametric_var_es

METHODS: tuple[str, ...] = ("historical", "parametric", "monte_carlo")


def rolling_var_es(
    asset_returns: pd.DataFrame,
    weights: np.ndarray | None = None,
    window: int = 500,
    alpha: float = 0.99,
    n_sims: int = 10_000,
    seed: int = 42,
) -> pd.DataFrame:
    """One-day-ahead VaR/ES for each day t using only returns t-window .. t-1.

    Args:
        asset_returns: Asset simple returns (dates x assets).
        weights: Portfolio weights; equal if None.
        window: Estimation window length in days.
        alpha: Confidence level.
        n_sims: Monte Carlo scenarios per day.
        seed: Seed for the single Monte Carlo generator.

    Returns:
        DataFrame indexed by forecast date with ``var_<method>``, ``es_<method>``
        and ``realised`` (the portfolio return on that date).
    """
    x = asset_returns.to_numpy(dtype=float)
    if len(x) <= window:
        raise ValueError(f"need more than window={window} observations, got {len(x)}")
    w = equal_weights(x.shape[1]) if weights is None else np.asarray(weights, float)
    port = x @ w
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float]] = []
    for t in range(window, len(x)):
        hist = x[t - window : t]
        mu = hist.mean(axis=0)
        cov = np.cov(hist, rowvar=False, ddof=1)
        hv, he = historical_var_es(port[t - window : t], alpha)
        pv, pe = parametric_var_es(float(w @ mu), float(np.sqrt(w @ cov @ w)), alpha)
        mv, me = monte_carlo_var_es(mu, cov, w, alpha, n_sims, rng=rng)
        rows.append(
            {
                "var_historical": hv,
                "es_historical": he,
                "var_parametric": pv,
                "es_parametric": pe,
                "var_monte_carlo": mv,
                "es_monte_carlo": me,
            }
        )
    out = pd.DataFrame(rows, index=asset_returns.index[window:])
    out["realised"] = port[window:]
    return out


def kupiec_pof(n_breaches: int, n_obs: int, alpha: float = 0.99) -> tuple[float, float]:
    """Kupiec proportion-of-failures likelihood-ratio test.

    H0: breach probability equals ``1 - alpha``. LR ~ chi2(1) under H0.

    Args:
        n_breaches: Number of VaR breaches.
        n_obs: Number of forecasts.
        alpha: VaR confidence level.

    Returns:
        (LR statistic, p-value).
    """
    if n_obs <= 0 or not 0 <= n_breaches <= n_obs:
        raise ValueError("need n_obs > 0 and 0 <= n_breaches <= n_obs")
    p = 1.0 - alpha
    x, n = n_breaches, n_obs
    phat = x / n
    ll0 = xlogy(n - x, 1 - p) + xlogy(x, p)
    ll1 = xlogy(n - x, 1 - phat) + xlogy(x, phat)
    lr = max(0.0, float(-2.0 * (ll0 - ll1)))  # max(0.0, .) also avoids printing -0.0
    return lr, float(chi2.sf(lr, 1))


def christoffersen_independence(hits: np.ndarray) -> tuple[float, float]:
    """Christoffersen (1998) independence test for a breach (hit) sequence.

    H0: breaches are independent (first-order Markov with equal transition
    probabilities). LR ~ chi2(1) under H0.

    Args:
        hits: Boolean/0-1 sequence of breaches in time order.

    Returns:
        (LR statistic, p-value).
    """
    h = np.asarray(hits).astype(int)
    if h.ndim != 1 or h.size < 2:
        raise ValueError("hits must be a 1-D sequence of length >= 2")
    prev, curr = h[:-1], h[1:]
    n00 = int(np.sum((prev == 0) & (curr == 0)))
    n01 = int(np.sum((prev == 0) & (curr == 1)))
    n10 = int(np.sum((prev == 1) & (curr == 0)))
    n11 = int(np.sum((prev == 1) & (curr == 1)))
    pi01 = n01 / (n00 + n01) if n00 + n01 else 0.0
    pi11 = n11 / (n10 + n11) if n10 + n11 else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    ll0 = xlogy(n00 + n10, 1 - pi) + xlogy(n01 + n11, pi)
    ll1 = xlogy(n00, 1 - pi01) + xlogy(n01, pi01) + xlogy(n10, 1 - pi11) + xlogy(n11, pi11)
    lr = max(0.0, float(-2.0 * (ll0 - ll1)))  # max(0.0, .) also avoids printing -0.0
    return lr, float(chi2.sf(lr, 1))


def backtest_table(
    var_forecasts: Mapping[str, pd.Series], realised: pd.Series, alpha: float = 0.99
) -> pd.DataFrame:
    """Breach counts and coverage tests per method.

    Args:
        var_forecasts: Method name -> VaR series (positive losses), aligned to ``realised``.
        realised: Realised portfolio returns.
        alpha: VaR confidence level.

    Returns:
        One row per method with n_obs, breaches, expected, breach_rate,
        Kupiec, Christoffersen independence and conditional-coverage (cc)
        LR statistics with asymptotic chi2 p-values, mean VaR.
    """
    rows = []
    for name, var in var_forecasts.items():
        var, real = var.align(realised, join="inner")
        if var.isna().any():
            raise ValueError(f"VaR series for {name} contains NaN")
        hits = (real < -var).to_numpy()
        n, x = len(hits), int(hits.sum())
        k_lr, k_p = kupiec_pof(x, n, alpha)
        c_lr, c_p = christoffersen_independence(hits)
        # conditional coverage: POF on the same n-1 transitions + independence, chi2(2)
        cc_lr = kupiec_pof(int(hits[1:].sum()), n - 1, alpha)[0] + c_lr
        rows.append(
            {
                "method": name,
                "start": var.index[0].date().isoformat(),
                "end": var.index[-1].date().isoformat(),
                "n_obs": n,
                "breaches": x,
                "expected": n * (1 - alpha),
                "breach_rate": x / n,
                "kupiec_lr": k_lr,
                "kupiec_p": k_p,
                "christoffersen_lr": c_lr,
                "christoffersen_p": c_p,
                "cc_lr": cc_lr,
                "cc_p": float(chi2.sf(cc_lr, 2)),
                "mean_var": float(var.mean()),
            }
        )
    return pd.DataFrame(rows)
