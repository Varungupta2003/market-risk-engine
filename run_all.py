"""Run the full market-risk pipeline and write results/ (CSVs, charts, metadata).

Usage:
    python run_all.py                    # real data via yfinance (cached to data/prices.csv)
    python run_all.py --refresh          # force a fresh download
    python run_all.py --synthetic --results-dir tmp_synth   # SYNTHETIC smoke test only
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from mre import backtest, data, report, returns, var_es, vol  # noqa: E402

ALPHA = 0.99
SEED = 42


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command-line options."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--data", type=Path, default=data.DEFAULT_CACHE, help="price cache CSV")
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--refresh", action="store_true", help="re-download prices")
    p.add_argument("--synthetic", action="store_true",
                   help="use SYNTHETIC prices (tests only; never for reported results)")
    p.add_argument("--synthetic-days", type=int, default=800)
    p.add_argument("--window", type=int, default=500)
    p.add_argument("--n-sims", type=int, default=10_000)
    p.add_argument("--refit-every", type=int, default=20)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    """Run every step; return the metadata dict that is also written to disk."""
    args = parse_args(argv)
    if args.synthetic and args.results_dir.resolve() == Path("results").resolve():
        raise SystemExit("--synthetic requires a --results-dir other than results/")
    out = args.results_dir
    fig_dir = out / "figures"
    timings: dict[str, float] = {}
    t_all = time.perf_counter()

    def step(name: str, t0: float) -> None:
        timings[name] = round(time.perf_counter() - t0, 2)
        print(f"[{name}] done in {timings[name]:.1f}s", flush=True)

    # 1. data
    t0 = time.perf_counter()
    if args.synthetic:
        print("WARNING: using SYNTHETIC prices - results are not real market data", flush=True)
        prices = data.validate_prices(data.make_synthetic_prices(args.synthetic_days, seed=SEED))
    else:
        prices = data.load_prices(args.data, refresh=args.refresh)
    asset_rets = returns.simple_returns(prices)
    w = returns.equal_weights(asset_rets.shape[1])
    port = returns.portfolio_returns(asset_rets, w)
    step("data", t0)

    # 2. full-sample VaR/ES
    t0 = time.perf_counter()
    hv, he = var_es.historical_var_es(port.to_numpy(), ALPHA)
    mu_p, sig_p = var_es.portfolio_moments(asset_rets.to_numpy(), w)
    pv, pe = var_es.parametric_var_es(mu_p, sig_p, ALPHA)
    x = asset_rets.to_numpy()
    mv, me = var_es.monte_carlo_var_es(
        x.mean(axis=0), np.cov(x, rowvar=False), w, ALPHA, args.n_sims, seed=SEED
    )
    full = pd.DataFrame(
        {"method": ["historical", "parametric", "monte_carlo"], "var": [hv, pv, mv],
         "es": [he, pe, me]}
    )
    report.save_csv(full, out / "full_sample_var_es.csv", index=False)
    step("full_sample", t0)

    # 3. rolling VaR/ES (historical, parametric, Monte Carlo)
    t0 = time.perf_counter()
    rolling = backtest.rolling_var_es(asset_rets, w, args.window, ALPHA, args.n_sims, SEED)
    step("rolling_var", t0)

    # 4. volatility models (walk-forward), started where ML features exist
    t0 = time.perf_counter()
    start = vol.FEATURE_LAG + args.window
    garch = vol.garch_forecasts(port, args.window, args.refit_every, start=start)
    step("garch", t0)
    t0 = time.perf_counter()
    ml = vol.ml_forecasts(port, args.window, args.refit_every, start=start, seed=SEED)
    step("ml", t0)
    har = vol.har_forecasts(port, args.window, args.refit_every, start=start)
    vol_models = {"garch": garch, "ml": ml, "har": har}
    vols = pd.concat(vol_models.values(), axis=1)
    vols["realised"] = port.loc[vols.index]
    report.save_csv(vols, out / "vol_forecasts.csv")
    metrics = pd.DataFrame(
        [{"model": k, **vol.vol_metrics(v, port)} for k, v in vol_models.items()]
    )
    report.save_csv(metrics, out / "vol_metrics.csv", index=False)

    # 5. backtest all VaR forecasts on the common period
    common = vols.index
    vol_var = {k: vol.vol_var_es(v, ALPHA) for k, v in vol_models.items()}
    var_all = pd.DataFrame(
        {m: rolling.loc[common, f"var_{m}"] for m in backtest.METHODS}
        | {k: v["var"] for k, v in vol_var.items()}
    )
    es_all = pd.DataFrame(
        {m: rolling.loc[common, f"es_{m}"] for m in backtest.METHODS}
        | {k: v["es"] for k, v in vol_var.items()}
    )
    realised = port.loc[common]
    table = backtest.backtest_table({c: var_all[c] for c in var_all}, realised, ALPHA)
    report.save_csv(table, out / "backtest_summary.csv", index=False)
    # classical methods over their full forecast period (includes Feb-Mar 2020; the common
    # period above starts later because the vol models need FEATURE_LAG extra days)
    table_full = backtest.backtest_table(
        {m: rolling[f"var_{m}"] for m in backtest.METHODS}, rolling["realised"], ALPHA
    )
    report.save_csv(table_full, out / "backtest_summary_full_period.csv", index=False)
    report.save_csv(
        pd.concat([var_all.add_prefix("var_"), es_all.add_prefix("es_"),
                   realised.rename("realised")], axis=1),
        out / "rolling_var_es.csv",
    )

    # 6. charts
    t0 = time.perf_counter()
    report.plot_var_backtest(var_all[list(backtest.METHODS)], realised,
                             fig_dir / "var_backtest.png", ALPHA, args.window)
    report.plot_breaches(table, fig_dir / "breaches.png", ALPHA)
    report.plot_vol_forecasts(vols[list(vol_models)], port, fig_dir / "vol_forecasts.png")
    report.plot_return_distribution(port, full, fig_dir / "return_distribution.png", ALPHA)
    step("charts", t0)

    meta = {
        "synthetic": args.synthetic,
        "data_source": "synthetic" if args.synthetic else prices.attrs.get("source"),
        "tickers": list(prices.columns),
        "price_start": prices.index[0].date().isoformat(),
        "price_end": prices.index[-1].date().isoformat(),
        "n_return_obs": int(len(port)),
        "alpha": ALPHA,
        "window": args.window,
        "n_sims": args.n_sims,
        "refit_every": args.refit_every,
        "seed": SEED,
        "backtest_start": common[0].date().isoformat(),
        "backtest_end": common[-1].date().isoformat(),
        "backtest_n_obs": int(len(common)),
        "ml_backend": ml.attrs["model"],
        "ml_floor_hits": ml.attrs["floor_hits"],
        "har_floor_hits": har.attrs["floor_hits"],
        "garch_fits": garch.attrs["n_fits"],
        "garch_not_converged": garch.attrs["n_not_converged"],
        "timings_s": timings | {"total": round(time.perf_counter() - t_all, 2)},
        "python": platform.python_version(),
        "packages": {p: version(p) for p in
                     ["numpy", "pandas", "scipy", "arch", "lightgbm", "scikit-learn",
                      "yfinance", "matplotlib"]},
    }
    (out / "run_metadata.json").write_text(json.dumps(meta, indent=2))

    print("\nFull-sample 99% VaR/ES:\n", full.to_string(index=False))
    print("\nBacktest:\n", table.to_string(index=False))
    print("\nBacktest, classical methods, full period:\n", table_full.to_string(index=False))
    print("\nVol metrics:\n", metrics.to_string(index=False))
    print(f"\nTotal {meta['timings_s']['total']:.1f}s; results in {out.resolve()}")
    return meta


if __name__ == "__main__":
    main()
