"""Persistent portfolio state in SQLite.

Why SQLite and not memory/JSON: the paper (and later live) trader runs once a
day as a fresh process. If the machine crashes mid-run, state must survive and
be consistent. Every mutation here happens inside a transaction.

Tables:
    meta            key/value (cash, initial_capital)
    positions       one row per open position
    pending_orders  orders decided today, to fill at tomorrow's open
    trades          closed round-trips (the audit log)
    equity          daily equity curve
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value REAL);
CREATE TABLE IF NOT EXISTS positions (
    ticker TEXT PRIMARY KEY, shares INTEGER, entry_date TEXT,
    entry_price REAL, entry_costs REAL
);
CREATE TABLE IF NOT EXISTS pending_orders (
    ticker TEXT PRIMARY KEY, side TEXT CHECK(side IN ('BUY','SELL')),
    created_date TEXT
);
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT, entry_date TEXT, exit_date TEXT,
    entry_price REAL, exit_price REAL, shares INTEGER,
    costs REAL, pnl REAL, return_pct REAL
);
CREATE TABLE IF NOT EXISTS equity (date TEXT PRIMARY KEY, value REAL);
"""


class StateStore:
    def __init__(self, path: str | Path = "paper_state.db", initial_capital: float = 1_000_000):
        self.conn = sqlite3.connect(str(path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        if self.get_meta("cash") is None:  # first run
            with self.conn:
                self.conn.execute("INSERT INTO meta VALUES ('cash', ?)", (initial_capital,))
                self.conn.execute("INSERT INTO meta VALUES ('initial_capital', ?)", (initial_capital,))

    # -- meta ---------------------------------------------------------------
    def get_meta(self, key: str) -> float | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row else None

    def set_cash(self, value: float) -> None:
        with self.conn:
            self.conn.execute("UPDATE meta SET value=? WHERE key='cash'", (value,))

    @property
    def cash(self) -> float:
        return self.get_meta("cash")

    # -- positions ----------------------------------------------------------
    def positions(self) -> dict[str, sqlite3.Row]:
        rows = self.conn.execute("SELECT * FROM positions").fetchall()
        return {r["ticker"]: r for r in rows}

    def open_position(self, ticker, shares, entry_date, entry_price, entry_costs, new_cash) -> None:
        with self.conn:  # position + cash update must be atomic
            self.conn.execute(
                "INSERT INTO positions VALUES (?,?,?,?,?)",
                (ticker, shares, entry_date, entry_price, entry_costs),
            )
            self.conn.execute("UPDATE meta SET value=? WHERE key='cash'", (new_cash,))

    def close_position(self, ticker, exit_date, exit_price, sell_costs, new_cash) -> dict:
        pos = self.positions()[ticker]
        entry_value = pos["entry_price"] * pos["shares"]
        exit_value = exit_price * pos["shares"]
        pnl = exit_value - sell_costs - entry_value - pos["entry_costs"]
        trade = {
            "ticker": ticker, "entry_date": pos["entry_date"], "exit_date": exit_date,
            "entry_price": pos["entry_price"], "exit_price": exit_price,
            "shares": pos["shares"], "costs": pos["entry_costs"] + sell_costs,
            "pnl": pnl,
            "return_pct": (exit_value - sell_costs) / (entry_value + pos["entry_costs"]) * 100 - 100,
        }
        with self.conn:
            self.conn.execute("DELETE FROM positions WHERE ticker=?", (ticker,))
            self.conn.execute(
                "INSERT INTO trades (ticker,entry_date,exit_date,entry_price,exit_price,"
                "shares,costs,pnl,return_pct) VALUES (?,?,?,?,?,?,?,?,?)",
                tuple(trade.values()),
            )
            self.conn.execute("UPDATE meta SET value=? WHERE key='cash'", (new_cash,))
        return trade

    # -- pending orders -----------------------------------------------------
    def pending(self) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM pending_orders").fetchall()

    def set_pending(self, orders: list[tuple[str, str, str]]) -> None:
        """Replace pending orders with (ticker, side, created_date) list."""
        with self.conn:
            self.conn.execute("DELETE FROM pending_orders")
            self.conn.executemany("INSERT INTO pending_orders VALUES (?,?,?)", orders)

    # -- equity -------------------------------------------------------------
    def record_equity(self, date: str, value: float) -> None:
        with self.conn:
            self.conn.execute("INSERT OR REPLACE INTO equity VALUES (?,?)", (date, value))

    def equity_history(self) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM equity ORDER BY date").fetchall()

    def trade_history(self) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM trades ORDER BY exit_date").fetchall()
