"""Paper trading engine — the daily cycle.

Run ONCE per trading day, after market close:

    1. FILL yesterday's pending orders at TODAY's open (+slippage, +costs)
       — identical execution model to the backtester, on purpose.
    2. MARK portfolio to market at today's close; record equity.
    3. SIGNAL: run the strategy on data up to today; diff desired state vs
       current positions → new pending orders for tomorrow.
    4. Everything persists in SQLite → safe across crashes/restarts.

The same strategy object, cost model, and sizing rule as the backtester are
used, so paper results are directly comparable to backtest expectations —
that comparison IS the go-live gate.
"""

from __future__ import annotations

import logging

import pandas as pd

from backtesting.costs import CostModel
from strategies.base import Strategy

from .state import StateStore

log = logging.getLogger(__name__)


class PaperTrader:
    def __init__(
        self,
        strategy: Strategy,
        store: StateStore,
        cost_model: CostModel | None = None,
        max_positions: int = 10,
        slippage_bps: float = 5.0,
    ):
        self.strategy = strategy
        self.store = store
        self.costs = cost_model or CostModel()
        self.max_positions = max_positions
        self.slip = slippage_bps / 10_000

    # ------------------------------------------------------------------ #
    def run_daily(
        self,
        data: dict[str, pd.DataFrame],
        index_feat: pd.DataFrame | None = None,
    ) -> dict:
        """data: {ticker: feature frame incl. today's candle}. Returns a summary."""
        today = max(df.index[-1] for df in data.values())
        today_s = str(today.date())
        summary = {"date": today_s, "filled": [], "new_orders": []}

        # -- 1. fill pending orders at today's open --------------------------
        positions = self.store.positions()
        pending = self.store.pending()
        sells = [o for o in pending if o["side"] == "SELL"]
        buys = [o for o in pending if o["side"] == "BUY"]

        for o in sells:  # sells first: free cash for buys
            t = o["ticker"]
            if t not in positions or t not in data or today not in data[t].index:
                continue
            fill = data[t].at[today, "open"] * (1 - self.slip)
            value = fill * positions[t]["shares"]
            sc = self.costs.sell_costs(value)
            trade = self.store.close_position(
                t, today_s, fill, sc, self.store.cash + value - sc
            )
            summary["filled"].append(f"SELL {t} {trade['shares']}@{fill:.2f} pnl={trade['pnl']:.0f}")
            log.info("SELL %s %d @ %.2f (pnl %.0f)", t, trade["shares"], fill, trade["pnl"])

        positions = self.store.positions()
        last_close = {
            t: df.at[today, "close"] for t, df in data.items() if today in df.index
        }
        for o in buys:
            t = o["ticker"]
            if t in positions or len(positions) >= self.max_positions:
                continue
            if t not in data or today not in data[t].index:
                continue
            holdings_val = sum(
                last_close.get(p["ticker"], p["entry_price"]) * p["shares"]
                for p in positions.values()
            )
            equity_now = self.store.cash + holdings_val
            alloc = min(equity_now / self.max_positions, self.store.cash)
            fill = data[t].at[today, "open"] * (1 + self.slip)
            shares = int(alloc / (fill * 1.002))
            if shares <= 0:
                continue
            value = fill * shares
            bc = self.costs.buy_costs(value)
            if value + bc > self.store.cash:
                continue
            self.store.open_position(t, shares, today_s, fill, bc, self.store.cash - value - bc)
            positions = self.store.positions()
            summary["filled"].append(f"BUY {t} {shares}@{fill:.2f}")
            log.info("BUY %s %d @ %.2f", t, shares, fill)

        # -- 2. mark to market ------------------------------------------------
        positions = self.store.positions()
        holdings_val = sum(
            last_close.get(p["ticker"], p["entry_price"]) * p["shares"]
            for p in positions.values()
        )
        equity = self.store.cash + holdings_val
        self.store.record_equity(today_s, equity)
        summary["equity"] = equity
        summary["cash"] = self.store.cash
        summary["positions"] = {t: p["shares"] for t, p in positions.items()}

        # -- 3. new signals → pending orders for tomorrow ---------------------
        orders: list[tuple[str, str, str]] = []
        for t, df in data.items():
            sig = self.strategy.generate_signals(df, index_feat)
            want_long = bool(sig.iloc[-1] == 1)
            have = t in positions
            if want_long and not have:
                orders.append((t, "BUY", today_s))
            elif not want_long and have:
                orders.append((t, "SELL", today_s))
        self.store.set_pending(orders)
        summary["new_orders"] = [f"{side} {t}" for t, side, _ in orders]
        log.info("day %s: equity %.0f, %d pending orders for tomorrow",
                 today_s, equity, len(orders))
        return summary
