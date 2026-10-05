"""Performance metrics computed from a daily equity curve and trade log."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def cagr(equity: pd.Series) -> float:
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    if years <= 0:
        return 0.0
    return (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1


def sharpe(equity: pd.Series, risk_free_annual: float = 0.065) -> float:
    """Daily-return Sharpe, annualised. Default rf ≈ Indian T-bill yield."""
    rets = equity.pct_change().dropna()
    if rets.std() == 0:
        return 0.0
    rf_daily = (1 + risk_free_annual) ** (1 / TRADING_DAYS) - 1
    return (rets.mean() - rf_daily) / rets.std() * np.sqrt(TRADING_DAYS)


def max_drawdown(equity: pd.Series) -> float:
    """Worst peak-to-trough decline, as a negative fraction (e.g. -0.23)."""
    dd = equity / equity.cummax() - 1
    return float(dd.min())


def trade_stats(trades: pd.DataFrame) -> dict:
    """Win rate / profit factor from the trade log (needs 'pnl' column)."""
    if trades.empty:
        return {"n_trades": 0, "win_rate": np.nan, "profit_factor": np.nan, "avg_pnl": np.nan}
    wins = trades[trades["pnl"] > 0]["pnl"]
    losses = trades[trades["pnl"] <= 0]["pnl"]
    return {
        "n_trades": len(trades),
        "win_rate": len(wins) / len(trades),
        "profit_factor": float(wins.sum() / -losses.sum()) if losses.sum() < 0 else np.inf,
        "avg_pnl": float(trades["pnl"].mean()),
    }


def summarize(equity: pd.Series, trades: pd.DataFrame, benchmark: pd.Series | None = None) -> dict:
    out = {
        "start": str(equity.index[0].date()),
        "end": str(equity.index[-1].date()),
        "final_equity": float(equity.iloc[-1]),
        "total_return_pct": float((equity.iloc[-1] / equity.iloc[0] - 1) * 100),
        "cagr_pct": cagr(equity) * 100,
        "sharpe": sharpe(equity),
        "max_drawdown_pct": max_drawdown(equity) * 100,
        **trade_stats(trades),
    }
    if benchmark is not None:
        bench = benchmark.reindex(equity.index).ffill().bfill()
        out["benchmark_return_pct"] = float((bench.iloc[-1] / bench.iloc[0] - 1) * 100)
        out["benchmark_cagr_pct"] = cagr(bench) * 100
        out["benchmark_max_dd_pct"] = max_drawdown(bench) * 100
    return out
