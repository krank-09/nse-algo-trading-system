"""Streamlit dashboard.

Run:  streamlit run frontend/app.py
Reads: reports/ (backtest outputs) and paper_state.db (paper trading state).
Read-only by design — the dashboard observes the system, it never trades.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
PAPER_DB = ROOT / "paper_state.db"

st.set_page_config(page_title="Trading System", layout="wide")
st.title("Algo Trading Dashboard")

tab_bt, tab_paper = st.tabs(["Backtest", "Paper Trading"])

# --------------------------------------------------------------------------
with tab_bt:
    eq_file, tr_file = REPORTS / "equity_curve.csv", REPORTS / "trades.csv"
    if not eq_file.exists():
        st.info("No backtest results yet. Run:  python main.py --mode backtest")
    else:
        eq = pd.read_csv(eq_file, index_col=0, parse_dates=True)["equity"]
        ret = eq.iloc[-1] / eq.iloc[0] - 1
        dd = (eq / eq.cummax() - 1).min()

        c1, c2, c3 = st.columns(3)
        c1.metric("Final equity", f"₹{eq.iloc[-1]:,.0f}", f"{ret:+.1%}")
        c2.metric("Max drawdown", f"{dd:.1%}")

        trades = pd.read_csv(tr_file) if tr_file.exists() else pd.DataFrame()
        if not trades.empty:
            wr = (trades["pnl"] > 0).mean()
            c3.metric("Trades / win rate", f"{len(trades)} / {wr:.0%}")

        st.subheader("Equity curve")
        st.line_chart(eq)

        st.subheader("Drawdown")
        st.area_chart(eq / eq.cummax() - 1)

        if not trades.empty:
            st.subheader("Trades")
            st.dataframe(
                trades.sort_values("exit_date", ascending=False),
                width="stretch",
            )
            st.subheader("PnL by ticker")
            st.bar_chart(trades.groupby("ticker")["pnl"].sum())

# --------------------------------------------------------------------------
with tab_paper:
    if not PAPER_DB.exists():
        st.info("No paper trading yet. Run:  python main.py --mode paper-run")
    else:
        conn = sqlite3.connect(PAPER_DB)
        eq = pd.read_sql("SELECT * FROM equity ORDER BY date", conn,
                         index_col="date", parse_dates=["date"])
        pos = pd.read_sql("SELECT * FROM positions", conn)
        trades = pd.read_sql("SELECT * FROM trades ORDER BY exit_date DESC", conn)
        pending = pd.read_sql("SELECT * FROM pending_orders", conn)
        cash = conn.execute("SELECT value FROM meta WHERE key='cash'").fetchone()[0]
        initial = conn.execute(
            "SELECT value FROM meta WHERE key='initial_capital'"
        ).fetchone()[0]
        conn.close()

        c1, c2, c3, c4 = st.columns(4)
        if not eq.empty:
            latest = eq["value"].iloc[-1]
            c1.metric("Equity", f"₹{latest:,.0f}", f"{latest/initial-1:+.2%}")
        c2.metric("Cash", f"₹{cash:,.0f}")
        c3.metric("Open positions", len(pos))
        if not trades.empty:
            c4.metric("Closed trades / win rate",
                      f"{len(trades)} / {(trades['pnl'] > 0).mean():.0%}")

        if not eq.empty:
            st.subheader("Paper equity curve")
            st.line_chart(eq["value"])

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Open positions")
            st.dataframe(pos, width="stretch")
        with col_b:
            st.subheader("Pending orders (tomorrow's open)")
            st.dataframe(pending, width="stretch")

        if not trades.empty:
            st.subheader("Closed trades")
            st.dataframe(trades, width="stretch")
