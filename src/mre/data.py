"""Price data: download via yfinance, cache to CSV, validate.

Real data only comes from yfinance. If the download fails, a RuntimeError is
raised; nothing is fabricated. ``make_synthetic_prices`` exists solely for
unit tests and the smoke test and is labelled SYNTHETIC everywhere it is used.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

TICKERS: list[str] = ["SPY", "QQQ", "EFA", "EEM", "TLT", "GLD", "USO", "AAPL", "MSFT", "JPM"]
START_DATE: str = "2018-01-01"
DEFAULT_CACHE: Path = Path("data/prices.csv")


def validate_prices(
    prices: pd.DataFrame, tickers: Sequence[str] | None = None, min_rows: int = 2
) -> pd.DataFrame:
    """Validate a wide price table (dates x tickers) and return a clean copy.

    Checks: DatetimeIndex, no duplicate dates, requested tickers present,
    strictly positive prices, at least ``min_rows`` complete rows. Rows with any
    missing price (e.g. before an asset started trading) are dropped; a warning
    is issued if any dropped row lies after the first complete row.

    Args:
        prices: Adjusted close prices, one column per ticker.
        tickers: Columns that must be present (and are kept, in this order).
        min_rows: Minimum number of complete rows required.

    Returns:
        Sorted float DataFrame with no missing values.

    Raises:
        ValueError: If any check fails.
    """
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("prices must have a DatetimeIndex")
    if prices.index.has_duplicates:
        raise ValueError("prices index has duplicate dates")
    if tickers is not None:
        missing = [t for t in tickers if t not in prices.columns]
        if missing:
            raise ValueError(f"missing tickers in price data: {missing}")
        prices = prices[list(tickers)]
    prices = prices.sort_index()
    complete = prices.notna().all(axis=1)
    # rows missing after the first complete row would turn the next return into a multi-day one
    interior = int((~complete[complete.idxmax():]).sum()) if complete.any() else 0
    if interior:
        warnings.warn(f"dropping {interior} interior rows with missing prices", stacklevel=2)
    clean = prices[complete].astype(float)
    if (clean <= 0).to_numpy().any():
        raise ValueError("prices must be strictly positive")
    if len(clean) < min_rows:
        raise ValueError(f"need at least {min_rows} complete rows, got {len(clean)}")
    return clean


def download_prices(
    tickers: Sequence[str] = TICKERS, start: str = START_DATE, end: str | None = None
) -> pd.DataFrame:
    """Download daily adjusted close prices from yfinance.

    Args:
        tickers: Ticker symbols.
        start: First date (inclusive), ISO format.
        end: Last date (exclusive) or None for latest.

    Returns:
        Wide DataFrame of adjusted closes (dates x tickers).

    Raises:
        RuntimeError: If yfinance returns no data (e.g. no network). No data is
            fabricated in that case.
    """
    import yfinance as yf

    raw = yf.download(
        list(tickers), start=start, end=end, auto_adjust=True, progress=False, threads=False
    )
    if raw is None or raw.empty:
        raise RuntimeError(
            "yfinance returned no data (network down or API change). "
            "Refusing to continue: no data will be fabricated."
        )
    close = raw["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])
    close.index = pd.to_datetime(close.index).tz_localize(None)
    close.index.name = "Date"
    empty = [t for t in tickers if t not in close.columns or close[t].isna().all()]
    if empty:
        raise RuntimeError(f"yfinance returned no prices for: {empty}")
    return close[list(tickers)]


def load_prices(
    cache_path: Path | str = DEFAULT_CACHE,
    tickers: Sequence[str] = TICKERS,
    start: str = START_DATE,
    refresh: bool = False,
) -> pd.DataFrame:
    """Load prices from the CSV cache, downloading (and caching) if needed.

    Args:
        cache_path: CSV cache location.
        tickers: Ticker symbols required.
        start: Start date for a fresh download.
        refresh: Force a fresh download even if the cache exists.

    Returns:
        Validated price DataFrame; ``attrs["source"]`` is ``"cache"`` or ``"download"``.
    """
    path = Path(cache_path)
    if path.exists() and not refresh:
        prices = validate_prices(pd.read_csv(path, index_col=0, parse_dates=True), tickers)
        if prices.index[0] > pd.Timestamp(start) + pd.Timedelta(days=7):
            warnings.warn(f"cache starts {prices.index[0].date()}, after start={start}; "
                          "use refresh=True", stacklevel=2)
        prices.attrs["source"] = "cache"
        return prices
    prices = validate_prices(download_prices(tickers, start), tickers)
    path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(path)
    prices.attrs["source"] = "download"
    return prices


def make_synthetic_prices(
    n_days: int = 800, tickers: Sequence[str] = TICKERS, seed: int = 0
) -> pd.DataFrame:
    """SYNTHETIC prices for tests only: correlated GBM, business-day index.

    Never used for reported results.

    Args:
        n_days: Number of price rows.
        tickers: Column names.
        seed: RNG seed.

    Returns:
        Wide DataFrame of strictly positive synthetic prices.
    """
    rng = np.random.default_rng(seed)
    n = len(tickers)
    corr = np.full((n, n), 0.3) + 0.7 * np.eye(n)
    vols = np.linspace(0.008, 0.025, n)
    cov = corr * np.outer(vols, vols)
    rets = rng.multivariate_normal(np.full(n, 0.0003), cov, size=n_days - 1)
    levels = 100 * np.exp(np.vstack([np.zeros(n), np.cumsum(rets, axis=0)]))
    index = pd.bdate_range("2020-01-01", periods=n_days, name="Date")
    return pd.DataFrame(levels, index=index, columns=list(tickers))
