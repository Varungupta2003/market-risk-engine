"""VaR/ES estimators against closed-form and hand-computed values."""

from __future__ import annotations

import numpy as np
import pytest

from mre.var_es import (
    historical_var_es,
    monte_carlo_var_es,
    parametric_var_es,
    portfolio_moments,
)

# Standard normal, alpha = 0.99: z = 2.326348, ES = phi(z)/0.01 = 2.665214
Z99, ES99 = 2.3263478740, 2.6652142203


def test_parametric_standard_normal_closed_form() -> None:
    var, es = parametric_var_es(0.0, 1.0, 0.99)
    assert var == pytest.approx(Z99, abs=1e-9)
    assert es == pytest.approx(ES99, abs=1e-9)


def test_parametric_location_scale() -> None:
    var, es = parametric_var_es(0.001, 0.02, 0.99)
    assert var == pytest.approx(-0.001 + 0.02 * Z99, abs=1e-12)
    assert es == pytest.approx(-0.001 + 0.02 * ES99, abs=1e-12)


def test_historical_hand_example() -> None:
    # losses sorted: -.06 -.05 -.04 -.03 -.02 -.01 0 .01 .03 .05
    # alpha=0.8, "higher" quantile index ceil(0.8*9)=8 -> VaR=0.03; ES=mean(.03,.05)=0.04
    r = np.array([-0.05, -0.03, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
    var, es = historical_var_es(r, 0.8)
    assert var == pytest.approx(0.03)
    assert es == pytest.approx(0.04)


def test_historical_tail_size_at_99() -> None:
    r = -np.arange(1, 501) / 1000.0  # losses 0.001..0.500
    var, es = historical_var_es(r, 0.99)
    assert var == pytest.approx(0.496)  # 5 losses at or beyond VaR = 500 * 1%
    assert es == pytest.approx(np.mean([0.496, 0.497, 0.498, 0.499, 0.500]))


@pytest.mark.parametrize("bad", [np.array([]), np.array([[0.1]]), np.array([0.1, np.nan])])
def test_historical_rejects_bad_input(bad: np.ndarray) -> None:
    with pytest.raises(ValueError):
        historical_var_es(bad)


@pytest.mark.parametrize("alpha", [0.0, 1.0, 1.5])
def test_alpha_validation(alpha: float) -> None:
    with pytest.raises(ValueError):
        parametric_var_es(0.0, 1.0, alpha)


def test_negative_sigma_rejected() -> None:
    with pytest.raises(ValueError):
        parametric_var_es(0.0, -1.0)


def test_monte_carlo_seed_reproducible() -> None:
    mean, cov, w = np.zeros(2), np.array([[1.0, 0.3], [0.3, 2.0]]), np.array([0.5, 0.5])
    a = monte_carlo_var_es(mean, cov, w, n_sims=5000, seed=7)
    b = monte_carlo_var_es(mean, cov, w, n_sims=5000, seed=7)
    c = monte_carlo_var_es(mean, cov, w, n_sims=5000, seed=8)
    assert a == b
    assert a != c


def test_monte_carlo_converges_to_parametric() -> None:
    mean, cov, w = np.array([0.001, 0.0]), np.array([[1e-4, 2e-5], [2e-5, 4e-4]]), np.ones(2) / 2
    sigma = float(np.sqrt(w @ cov @ w))
    pv, pe = parametric_var_es(float(w @ mean), sigma)
    mv, me = monte_carlo_var_es(mean, cov, w, n_sims=400_000, seed=0)
    assert mv == pytest.approx(pv, rel=0.01)
    assert me == pytest.approx(pe, rel=0.01)


def test_portfolio_moments() -> None:
    x = np.array([[0.01, 0.03], [0.03, -0.01], [-0.01, 0.01]])
    mu, sd = portfolio_moments(x, np.array([0.5, 0.5]))
    p = x @ np.array([0.5, 0.5])
    assert mu == pytest.approx(p.mean())
    assert sd == pytest.approx(p.std(ddof=1))
