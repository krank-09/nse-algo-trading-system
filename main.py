"""CLI entry point.

Modes (see README for the full workflow):
    fetch-data      download raw daily OHLCV from yfinance (--tickers for a subset)
    clean-data      validate raw → processed parquet + quality report
    build-features  processed → 25-column feature matrix
    reconcile       cross-check yfinance closes against NSE bhavcopy (--date)
    freeze          snapshot features into a checksummed walk-forward set
    backtest        run a strategy (--strategy) over the feature set
    paper-run       one end-of-day paper-trading step (run after 15:30 IST)
    paper-status    print paper portfolio, equity and trade count
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from data_engineering import cleaner, features, loader

STRATEGY_NAMES = {"sma_rsi": "SMA + RSI", "ml": "ML strategy"}

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def main() -> None:
    p = argparse.ArgumentParser(description="Algo trading system")
    p.add_argument(
        "--mode",
        required=True,
        choices=[
            "fetch-data", "clean-data", "build-features", "backtest",
            "paper-run", "paper-status", "reconcile", "freeze",
        ],
    )
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--tickers", nargs="*", help="override universe (for quick tests)")
    p.add_argument("--strategy", default="sma_rsi", help="strategy name (backtest mode)")
    p.add_argument("--capital", type=float, default=1_000_000)
    p.add_argument("--max-positions", type=int, default=10,
                   help="equal-weight slots (the ML strategy was validated with 4)")
    p.add_argument("--date", help="session date for --mode reconcile (YYYY-MM-DD)")
    p.add_argument("--tol", type=float, default=1.0, help="reconcile tolerance %%")
    args = p.parse_args()

    cfg = loader.load_config(args.config)
    if args.tickers:
        cfg["universe"] = args.tickers

    if args.mode == "fetch-data":
        saved = loader.fetch_universe(cfg)
        print(f"\nDone: {len(saved)} instruments saved to {cfg['data']['raw_dir']}/")
    elif args.mode == "clean-data":
        report = cleaner.process_all(cfg["data"]["raw_dir"], cfg["data"]["processed_dir"])
        print(report.to_string())
        print(f"\nDone: processed parquet in {cfg['data']['processed_dir']}/")
    elif args.mode == "reconcile":
        from data_engineering import bhavcopy

        d = args.date or pd.Timestamp.today().strftime("%Y-%m-%d")
        out = bhavcopy.reconcile(
            d, cfg["data"]["raw_dir"], tol_pct=args.tol,
            symbol_map=bhavcopy._load_symbol_map(args.config),
        )
        print(out.to_string())
        bad = out[out["status"] != "OK"]
        raise SystemExit(1 if len(bad) else 0)
    elif args.mode == "freeze":
        from pathlib import Path

        from data_engineering import freeze

        feat_dir = str(Path(cfg["data"]["processed_dir"]).parent / "features")
        manifest = freeze.freeze(feat_dir, source="yfinance(auto_adjust)")
        print(f"Frozen {manifest['n_files']} files → {freeze.FROZEN_DIR}/")
        print(f"split windows: {freeze.WALK_FORWARD}")
    elif args.mode == "build-features":
        from pathlib import Path

        proc = Path(cfg["data"]["processed_dir"])
        feat_dir = proc.parent / "features"
        feat_dir.mkdir(parents=True, exist_ok=True)
        files = list(proc.glob("*.parquet")) or list(proc.glob("*.clean.csv"))
        for f in files:
            df = (
                pd.read_parquet(f)
                if f.suffix == ".parquet"
                else pd.read_csv(f, index_col="date", parse_dates=True)
            )
            out = features.build_features(df)
            name = f.stem.replace(".clean", "")
            try:
                out.to_parquet(feat_dir / f"{name}.parquet")
            except ImportError:
                out.to_csv(feat_dir / f"{name}.csv")
        print(f"Done: features for {len(files)} instruments in {feat_dir}/")
    elif args.mode == "backtest":
        from pathlib import Path

        from backtesting import Backtester, save_report
        from strategies import get_strategy

        def read(f: Path) -> pd.DataFrame:
            return (
                pd.read_parquet(f)
                if f.suffix == ".parquet"
                else pd.read_csv(f, index_col="date", parse_dates=True)
            )

        feat_dir = Path(cfg["data"]["processed_dir"]).parent / "features"
        index_name = cfg["index"]["symbol"].replace("^", "IDX_")
        data, index_feat = {}, None
        for f in sorted(list(feat_dir.glob("*.parquet")) + list(feat_dir.glob("*.csv"))):
            if f.stem == index_name:
                index_feat = read(f)
            else:
                data[f.stem] = read(f)
        if not data:
            raise SystemExit(f"No feature files in {feat_dir}. Run --mode build-features first.")

        bt = Backtester(get_strategy(args.strategy), initial_capital=args.capital,
                        max_positions=args.max_positions)
        result = bt.run(data, index_feat)

        print(f"\n=== {args.strategy} | {len(data)} instruments ===")
        for k, v in result.metrics.items():
            print(f"{k:>24}: {v:,.2f}" if isinstance(v, float) else f"{k:>24}: {v}")

        out = save_report(
            result, f"{args.strategy}_full",
            label=f"{STRATEGY_NAMES.get(args.strategy, args.strategy)} · "
                  f"{result.metrics['start'][:4]}–{result.metrics['end'][:4]} in-sample",
            description=f"{len(data)} NIFTY 50 stocks, all available history (in-sample).",
        )
        print(f"\nSaved report: {out}/  (view with: streamlit run frontend/app.py)")
    elif args.mode == "paper-run":
        from data_engineering.features import build_features
        from execution import PaperTrader, StateStore
        from strategies import get_strategy

        # fetch fresh EOD data (last ~2y is enough for a 200-day SMA warmup)
        d = cfg["data"]
        start = (pd.Timestamp.today() - pd.Timedelta(days=730)).strftime("%Y-%m-%d")
        data, index_feat = {}, None
        for t in list(cfg["universe"]) + [cfg["index"]["symbol"]]:
            try:
                feat = build_features(loader.fetch_one(t, start, None))
            except Exception as e:  # noqa: BLE001
                logging.warning("skip %s: %s", t, e)
                continue
            if t == cfg["index"]["symbol"]:
                index_feat = feat
            else:
                data[t] = feat

        trader = PaperTrader(
            get_strategy(args.strategy),
            StateStore("paper_state.db", initial_capital=args.capital),
            max_positions=args.max_positions,
        )
        s = trader.run_daily(data, index_feat)
        print(f"\n=== paper run {s['date']} ===")
        print(f"equity: ₹{s['equity']:,.0f}   cash: ₹{s['cash']:,.0f}")
        print("filled today:", s["filled"] or "nothing")
        print("positions:", s["positions"] or "none")
        print("orders for tomorrow:", s["new_orders"] or "none")
    elif args.mode == "paper-status":
        from execution import StateStore

        store = StateStore("paper_state.db", initial_capital=args.capital)
        eq = store.equity_history()
        print(f"cash: ₹{store.cash:,.0f}")
        print("positions:")
        for t, p in store.positions().items():
            print(f"  {t}: {p['shares']} @ ₹{p['entry_price']:.2f} since {p['entry_date']}")
        if eq:
            first, last = eq[0], eq[-1]
            print(f"equity: ₹{last['value']:,.0f} ({last['date']}), "
                  f"return {last['value']/first['value']*100-100:+.2f}% since {first['date']}")
        print(f"closed trades: {len(store.trade_history())}")


if __name__ == "__main__":
    main()
