"""Backtest the trained ML strategy on the frozen validate window, vs
buy-and-hold NIFTY and vs the rule-based sma_rsi strategy.

This is the step that decides whether the model's statistical edge
(precision@0.6 ~1.6x lift) actually survives India transaction costs and
position-sizing frictions. A model can be 'right' and still lose money.

Usage:
    python -m ml.backtest_ml                       # validate 2022-2025
    python -m ml.backtest_ml --split holdout       # 2026+ — use ONCE, at the end
    python -m ml.backtest_ml --min-proba 0.55
"""

from __future__ import annotations

import argparse

import pandas as pd

from backtesting import Backtester
from data_engineering import freeze
from ml.models import load
from strategies import get_strategy
from strategies.ml_strategy import MLStrategy


def _run(strategy, data, index_feat, capital, label, *, max_positions=10):
    bt = Backtester(strategy, initial_capital=capital, max_positions=max_positions)
    res = bt.run(data, index_feat)
    m = res.metrics
    print(f"\n=== {label} ===")
    for k in ("start", "end", "final_equity", "total_return_pct", "cagr_pct",
              "sharpe", "max_drawdown_pct", "n_trades", "win_rate", "profit_factor",
              "benchmark_return_pct", "benchmark_cagr_pct", "benchmark_max_dd_pct"):
        if k in m:
            v = m[k]
            print(f"{k:>22}: {v:,.2f}" if isinstance(v, float) else f"{k:>22}: {v}")
    return res


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", default="validate",
                   choices=["tune", "validate", "holdout", "all",
                            "tune_train", "tune_eval"])
    p.add_argument("--model-path", default="ml/artifacts/model.pkl")
    p.add_argument("--capital", type=float, default=1_000_000)
    p.add_argument("--min-proba", type=float, default=0.60)
    p.add_argument("--exit-proba", type=float, default=0.45)
    p.add_argument("--max-positions", type=int, default=10)
    p.add_argument("--sweep", action="store_true",
                   help="grid over min_proba x max_positions, print a table")
    p.add_argument("--skip-rulebased", action="store_true")
    args = p.parse_args()

    data = freeze.load_frozen(args.split)
    idx = freeze.load_frozen(args.split, include_index=True)
    index_feat = idx.get("IDX_NSEI")
    span = f"{min(d.index.min() for d in data.values()).date()} → " \
           f"{max(d.index.max() for d in data.values()).date()}"
    print(f"split={args.split}  {len(data)} instruments  {span}")

    model = load(args.model_path)

    if args.sweep:
        print(f"\n{'min_proba':>10} {'max_pos':>8} {'ret%':>8} {'cagr%':>7} "
              f"{'sharpe':>7} {'maxDD%':>8} {'trades':>7} {'win%':>6}")
        for mp in (0.45, 0.50, 0.55, 0.60):
            for maxp in (4, 6, 10):
                bt = Backtester(MLStrategy(model=model, min_proba=mp,
                                          exit_proba=min(mp - 0.05, args.exit_proba)),
                                initial_capital=args.capital, max_positions=maxp)
                m = bt.run(data, index_feat).metrics
                print(f"{mp:>10.2f} {maxp:>8d} {m['total_return_pct']:>8.1f} "
                      f"{m['cagr_pct']:>7.2f} {m['sharpe']:>7.2f} "
                      f"{m['max_drawdown_pct']:>8.1f} {m['n_trades']:>7d} "
                      f"{m['win_rate']*100:>6.1f}")
        if index_feat is not None:
            b = index_feat["close"]
            print(f"\nNIFTY buy&hold: {(b.iloc[-1]/b.iloc[0]-1)*100:.1f}%  "
                  f"maxDD {(b/b.cummax()-1).min()*100:.1f}%")
        return

    ml_strat = MLStrategy(
        model=model, min_proba=args.min_proba, exit_proba=args.exit_proba,
    )
    _run(ml_strat, data, index_feat, args.capital,
         f"ML strategy (min_proba={args.min_proba}, max_positions={args.max_positions})",
         max_positions=args.max_positions)

    if not args.skip_rulebased:
        _run(get_strategy("sma_rsi"), data, index_feat, args.capital,
             "sma_rsi (rule-based reference)")


if __name__ == "__main__":
    main()
