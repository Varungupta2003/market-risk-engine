"""Shared fixtures. All data here is SYNTHETIC and used only for tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # so tests can import run_all

from mre.data import make_synthetic_prices  # noqa: E402
from mre.returns import portfolio_returns, simple_returns  # noqa: E402


@pytest.fixture(scope="session")
def synthetic_prices() -> pd.DataFrame:
    """SYNTHETIC 10-asset price panel."""
    return make_synthetic_prices(n_days=400, seed=1)


@pytest.fixture(scope="session")
def synthetic_port(synthetic_prices: pd.DataFrame) -> pd.Series:
    """SYNTHETIC equal-weight portfolio returns."""
    return portfolio_returns(simple_returns(synthetic_prices))
