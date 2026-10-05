"""Portfolio backtest engine.

Execution model (the part that keeps the backtest honest):
- Strategy signal for day t is computed from data up to t (strategy's job).
- The resulting order FILLS AT DAY t+1's OPEN, never at t's close. You cannot
  trade a close you needed to observe to generate the signal.
- Buys fill at open*(1+slippage), sells at open*(1-slippage).
- All India transaction costs applied via CostModel.

Position sizing (v1, deliberately simple):
- Equal-weight: each new position gets (current equity / max_positions).
- No pyramiding: one position per ticker.
Refinements (ATR sizing, sector caps) come in the risk module later.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from strategies.base import Strategy

from .costs import CostModel

log = logging.getLogger(__name__)


@dataclass
class Position:
    ticker: str
    shares: int
    entry_date: pd.Timestamp
    entry_price: float          # fill price incl. slippage
    entry_costs: float


@dataclass
class BacktestResult:
    equity: pd.Series
    trades: pd.DataFrame
    benchmark: pd.Series | None = None
    daily_positions: pd.DataFrame | None = None
    metrics: dict = field(default_factory=dict)


class Backtester:
    def __init__(
        self,
        strategy: Strategy,
        cost_model: CostModel | None = None,
        initial_capital: float = 1_000_000.0,
        max_positions: int = 10,
        slippage_bps: float = 5.0,
    ):
        self.strategy = strategy
        self.costs = cost_model or CostModel()
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.slip = slippage_bps / 10_000

    def run(
        self,
        data: dict[str, pd.DataFrame],
        index_feat: pd.DataFrame | None = None,
    ) -> BacktestResult:
        """data: {ticker: feature frame from build_features()} — clean daily OHLCV."""
        # 1. signals per ticker (vectorised; strategies are causal by contract)
        signals = {
            t: self.strategy.generate_signals(df, index_feat) for t, df in data.items()
        }

        # 2. unified trading calendar
        calendar = sorted(set().union(*(df.index for df in data.values())))

        cash = self.initial_capital
        positions: dict[str, Position] = {}
        equity_curve: list[float] = []
        trade_rows: list[dict] = []
        last_close: dict[str, float] = {}

        for i, day in enumerate(calendar):
            # -- A. execute orders decided on the PREVIOUS day, at today's open --
            if i > 0:
                prev = calendar[i - 1]
                # sells first: frees cash for same-day buys
                for t in list(positions):
                    df, sig = data[t], signals[t]
                    if prev in sig.index and sig.loc[prev] == 0 and day in df.index:
                        pos = positions.pop(t)
                        fill = df.at[day, "open"] * (1 - self.slip)
                        value = fill * pos.shares
                        sell_c = self.costs.sell_costs(value)
                        cash += value - sell_c
                        entry_value = pos.entry_price * pos.shares
                        trade_rows.append({
                            "ticker": t,
                            "entry_date": pos.entry_date, "exit_date": day,
                            "entry_price": pos.entry_price, "exit_price": fill,
                            "shares": pos.shares,
                            "costs": pos.entry_costs + sell_c,
                            "pnl": value - sell_c - entry_value - pos.entry_costs,
                            "return_pct": (value - sell_c) / (entry_value + pos.entry_costs) * 100 - 100,
                        })
                # buys
                for t, sig in signals.items():
                    if (
                        t not in positions
                        and len(positions) < self.max_positions
                        and prev in sig.index
                        and sig.loc[prev] == 1
                        and day in data[t].index
                    ):
                        equity_now = cash + sum(
                            last_close.get(p.ticker, p.entry_price) * p.shares
                            for p in positions.values()
                        )
                        alloc = min(equity_now / self.max_positions, cash)
                        fill = data[t].at[day, "open"] * (1 + self.slip)
                        shares = int(alloc / (fill * (1 + 0.002)))  # headroom for costs
                        if shares <= 0:
                            continue
                        value = fill * shares
                        buy_c = self.costs.buy_costs(value)
                        if value + buy_c > cash:
                            continue
                        cash -= value + buy_c
                        positions[t] = Position(t, shares, day, fill, buy_c)

            # -- B. mark to market at today's close --
            for t in data:
                if day in data[t].index:
                    last_close[t] = data[t].at[day, "close"]
            holdings = sum(last_close.get(p.ticker, p.entry_price) * p.shares
                           for p in positions.values())
            equity_curve.append(cash + holdings)

        equity = pd.Series(equity_curve, index=pd.DatetimeIndex(calendar), name="equity")
        trades = pd.DataFrame(trade_rows)

        benchmark = None
        if index_feat is not None:
            # bfill: the stock calendar can start a day before the index does;
            # a leading NaN would otherwise make the whole normalised series NaN
            b = index_feat["close"].reindex(equity.index).ffill().bfill()
            benchmark = b / b.iloc[0] * self.initial_capital

        from .metrics import summarize
        result = BacktestResult(equity=equity, trades=trades, benchmark=benchmark)
        result.metrics = summarize(equity, trades, benchmark)
        log.info("backtest done: %d trades, final equity %.0f",
                 len(trades), equity.iloc[-1])
        return result
