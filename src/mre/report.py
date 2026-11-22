"""Charts and CSV output."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

COLORS = {
    "historical": "#1f77b4",
    "parametric": "#ff7f0e",
    "monte_carlo": "#2ca02c",
    "garch": "#d62728",
    "ml": "#9467bd",
    "har": "#8c564b",
}


def save_csv(df: pd.DataFrame, path: Path, index: bool = True) -> Path:
    """Write a DataFrame to CSV, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index, float_format="%.6g")
    return path


def plot_var_backtest(
    var_frame: pd.DataFrame, realised: pd.Series, path: Path, alpha: float, window: int
) -> Path:
    """Portfolio returns against negative VaR lines, with breaches marked.

    Args:
        var_frame: Columns are method names, values are VaR (positive losses).
        realised: Realised portfolio returns.
        path: Output PNG path.
        alpha: VaR confidence level (for the title).
        window: Estimation window in days (for the title).
    """
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(realised.index, realised, color="0.6", lw=0.6, label="portfolio return")
    for name in var_frame.columns:
        ax.plot(var_frame.index, -var_frame[name], lw=1.0, color=COLORS.get(name), label=name)
    hist = var_frame.get("historical")
    if hist is not None:
        hits = realised.loc[hist.index] < -hist
        ax.scatter(hist.index[hits], realised.loc[hist.index][hits], color="k", s=12,
                   zorder=3, label="breach (historical)")
    ax.set_title(f"1-day {alpha:.0%} VaR (rolling {window}-day window) vs realised return")
    ax.set_ylabel("daily return")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), fontsize=8, ncol=5)
    return _save(fig, path)


def plot_breaches(table: pd.DataFrame, path: Path, alpha: float) -> Path:
    """Bar chart of breach counts per method with the expected count."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(table["method"], table["breaches"],
           color=[COLORS.get(m, "0.5") for m in table["method"]])
    ax.axhline(table["expected"].iloc[0], color="k", ls="--", lw=1,
               label=f"expected ({table['expected'].iloc[0]:.1f})")
    for i, (b, p) in enumerate(zip(table["breaches"], table["kupiec_p"], strict=True)):
        ax.text(i, b, f"{b}\np={p:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("VaR breaches")
    ax.set_title(f"{alpha:.0%} VaR breaches by method (Kupiec p-values)")
    ax.legend()
    ax.margins(y=0.2)
    return _save(fig, path)


def plot_vol_forecasts(vols: pd.DataFrame, realised: pd.Series, path: Path) -> Path:
    """Forecast volatility of each model against |realised return|."""
    fig, ax = plt.subplots(figsize=(11, 5))
    r = realised.loc[vols.index]
    ax.plot(r.index, r.abs(), color="0.75", lw=0.6, label="|return| (proxy)")
    for name in vols.columns:
        ax.plot(vols.index, vols[name], lw=1.0, color=COLORS.get(name), label=name)
    ax.set_title("1-day volatility forecasts by model")
    ax.set_ylabel("daily volatility")
    ax.legend(fontsize=8)
    return _save(fig, path)


def plot_return_distribution(
    returns: pd.Series, full_sample: pd.DataFrame, path: Path, alpha: float
) -> Path:
    """Histogram of portfolio returns with full-sample VaR lines."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(returns, bins=120, color="0.7")
    for _, row in full_sample.iterrows():
        ax.axvline(-row["var"], color=COLORS.get(row["method"]), lw=1.2,
                   label=f"{row['method']} VaR {row['var']:.4f}")
    ax.set_title(f"Portfolio daily returns and full-sample {alpha:.0%} VaR")
    ax.set_xlabel("daily return")
    ax.legend(fontsize=8)
    return _save(fig, path)


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
