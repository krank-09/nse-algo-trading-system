"""Render docs/equity_validate.png: ML strategy vs sma_rsi vs NIFTY on the validate window.

Usage:
    python -m scripts.make_readme_chart
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from backtesting import Backtester
from data_engineering import freeze
from ml.models import load
from strategies import get_strategy
from strategies.ml_strategy import MLStrategy

CAPITAL = 1_000_000


def main() -> None:
    data = freeze.load_frozen("validate")
    index_feat = freeze.load_frozen("validate", include_index=True).get("IDX_NSEI")
    model = load("ml/artifacts/model.pkl")

    ml = Backtester(MLStrategy(model=model, min_proba=0.50, exit_proba=0.45),
                    initial_capital=CAPITAL, max_positions=4).run(data, index_feat)
    rule = Backtester(get_strategy("sma_rsi"),
                      initial_capital=CAPITAL).run(data, index_feat)

    fig, ax = plt.subplots(figsize=(10, 5))
    for series, label, color in [
        (ml.equity, f"ML strategy ({ml.metrics['cagr_pct']:.1f}% CAGR)", "#1f77b4"),
        (ml.benchmark, f"NIFTY 50 buy & hold ({ml.metrics['benchmark_cagr_pct']:.1f}% CAGR)", "#7f7f7f"),
        (rule.equity, f"SMA+RSI rules ({rule.metrics['cagr_pct']:.1f}% CAGR)", "#d62728"),
    ]:
        ax.plot(series.index, series / CAPITAL, label=label, color=color, lw=1.6)
    ax.axhline(1, color="black", lw=0.6, ls=":")
    ax.set_title("Out-of-sample validate window (2022–2025), after Indian transaction costs")
    ax.set_ylabel("Growth of ₹1")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig("docs/equity_validate.png", dpi=150)
    print("saved docs/equity_validate.png")


if __name__ == "__main__":
    main()
