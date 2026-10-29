"""Streamlit dashboard.

Run:  streamlit run frontend/app.py
Reads: reports/ml_validate/ (written by `python -m ml.backtest_ml --split validate`)
       and paper_state.db (paper trading state).
Read-only by design: the dashboard observes the system, it never trades.
"""

from __future__ import annotations

import base64
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from backtesting.costs import CostModel  # noqa: E402

RUN_DIR = ROOT / "reports" / "ml_validate"
PAPER_DB = ROOT / "paper_state.db"
REPO_URL = "https://github.com/krank-09/nse-algo-trading-system"

INK, MUTED, GRID = "#14151a", "#6b7080", "#eceef3"
VIOLET, BENCH = "#5b4ff5", "#a3a8b8"
GAIN, LOSS = "#16a37a", "#e5484d"
FONT = "Plus Jakarta Sans, Inter, system-ui, sans-serif"
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Lucide icon paths (MIT), drawn inside a 24×24 stroke icon
ICONS = {
    "up": '<polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/>',
    "down": '<polyline points="22 17 13.5 8.5 8.5 13.5 2 7"/><polyline points="16 17 22 17 22 11"/>',
    "pct": '<line x1="19" y1="5" x2="5" y2="19"/><circle cx="6.5" cy="6.5" r="2.5"/>'
           '<circle cx="17.5" cy="17.5" r="2.5"/>',
    "target": '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/>'
              '<circle cx="12" cy="12" r="2"/>',
    "pulse": '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>',
    "clock": '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>',
    "coins": '<circle cx="8" cy="8" r="6"/><path d="M18.09 10.37A6 6 0 1 1 10.34 18"/>'
             '<path d="M7 6h1v4"/>',
    "chart": '<path d="M3 3v18h18"/><path d="m19 9-5 5-4-4-3 3"/>',
    "github": '<path d="M15 22v-4a4.8 4.8 0 0 0-1-3.5c3 0 6-2 6-5.5.08-1.25-.27-2.48-1-3.5'
              '.28-1.15.28-2.35 0-3.5 0 0-1 0-3 1.5-2.64-.5-5.36-.5-8 0C6 2 5 2 5 2c-.3 1.15-.3 '
              '2.35 0 3.5A5.403 5.403 0 0 0 4 9c0 3.5 3 5.5 6 5.5-.39.49-.68 1.05-.85 1.65-.17.6'
              '-.22 1.23-.15 1.85v4"/><path d="M9 18c-4.51 2-5-2-7-2"/>',
}


def icon(name: str, size: int = 22, colour: str = "#ffffff") -> str:
    """Icon as a base64 <img>: st.html sanitises inline <svg> elements away."""
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
           f'viewBox="0 0 24 24" fill="none" stroke="{colour}" stroke-width="2" '
           f'stroke-linecap="round" stroke-linejoin="round">{ICONS[name]}</svg>')
    data = base64.b64encode(svg.encode()).decode()
    return f'<img src="data:image/svg+xml;base64,{data}" width="{size}" height="{size}" alt="">'



st.set_page_config(page_title="NSE Algo Trading", page_icon="📈", layout="wide",
                   initial_sidebar_state="collapsed")

st.html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
.stApp, .stApp p, .stApp li, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp label,
.stApp button, .stApp input {font-family: 'Plus Jakarta Sans', Inter, system-ui, sans-serif;}
.stApp {background: #f3f3f8;}
header[data-testid="stHeader"] {background: transparent;}
section[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] {display: none;}
.block-container {padding: 1.6rem 2.2rem 3rem; max-width: 1440px;}

/* ---- top bar ---- */
.topbar {display:flex; align-items:center; justify-content:space-between; gap:16px; flex-wrap:wrap;}
.brand {display:flex; align-items:center; gap:12px;}
.brand-logo {width:42px; height:42px; border-radius:13px; display:grid; place-items:center; color:#fff;
  background:linear-gradient(135deg,#7c6cff,#5b4ff5); box-shadow:0 8px 18px rgba(91,79,245,.35);}
.brand-name {font-weight:800; font-size:1.2rem; color:#14151a; letter-spacing:-.01em;}
.brand-sub {font-size:.78rem; color:#6b7080;}
.chips {display:flex; gap:8px; flex-wrap:wrap; align-items:center;}
.chip {background:#fff; border-radius:999px; padding:7px 14px; font-size:.8rem; font-weight:600;
  color:#3b3e4a; box-shadow:0 1px 2px rgba(16,24,40,.05);}
.chip-dark {background:#14151a; color:#fff; display:flex; gap:7px; align-items:center; text-decoration:none !important;}
.chip-dark:hover {background:#2a2c35;}

/* ---- pill tabs (Streamlit 1.65: react-aria tabs) ---- */
.stTabs [role="tablist"] {gap:4px; background:#fff; padding:5px; border-radius:999px; border:none;
  width:fit-content; box-shadow:0 1px 2px rgba(16,24,40,.05); margin:10px 0 18px;}
.stTabs [role="tablist"]::after, .stTabs [role="tablist"]::before {display:none;}
.stTabs [role="tab"] {display:flex; align-items:center; height:38px; padding:0 18px;
  border-radius:999px; background:transparent; cursor:pointer;}
.stTabs [role="tab"]:hover {background:#f1f1f6;}
.stTabs [role="tab"] p {font-weight:600; font-size:.9rem; color:#3b3e4a; margin:0;}
.stTabs [role="tab"][aria-selected="true"] {background:#14151a;}
.stTabs [role="tab"][aria-selected="true"] p {color:#fff;}
.stTabs .react-aria-SelectionIndicator {display:none;}
.stTabs > div:first-child {border:none; box-shadow:none;}
@media (max-width: 640px) {
  .block-container {padding: 1rem 1rem 2rem;}
  .stTabs [role="tablist"] {max-width:100%; overflow-x:auto; scrollbar-width:none;}
  .stTabs [role="tab"] {padding:0 13px; flex:none;}
}

/* ---- headings ---- */
.hello {font-size:1.6rem; font-weight:800; color:#14151a; letter-spacing:-.02em; margin:4px 0 0;}
.hello-sub {color:#6b7080; font-size:.92rem; margin-bottom:6px;}

/* ---- KPI cards ---- */
.kpi {display:flex; gap:14px; align-items:center; padding:18px; border-radius:22px; height:100%;
  box-shadow:0 1px 2px rgba(16,24,40,.04), 0 10px 26px rgba(16,24,40,.05);}
.kpi-tile {flex:none; width:52px; height:52px; border-radius:16px; display:grid; place-items:center; color:#fff;}
.kpi-label {font-size:.8rem; color:#5c6070; font-weight:600;}
.kpi-value {font-size:1.65rem; font-weight:800; color:#14151a; letter-spacing:-.02em; line-height:1.15;}
.kpi-note {font-size:.74rem; font-weight:600; margin-top:2px; color:#6b7080;}
.kpi-note.up {color:#16a37a;} .kpi-note.down {color:#e5484d;}
.t-violet {background:#efedff;} .t-violet .kpi-tile {background:linear-gradient(135deg,#8b7dff,#5b4ff5); box-shadow:0 8px 16px rgba(91,79,245,.3);}
.t-mint   {background:#e7f7f0;} .t-mint .kpi-tile   {background:linear-gradient(135deg,#3fd09f,#16a37a); box-shadow:0 8px 16px rgba(22,163,122,.28);}
.t-amber  {background:#fdf3e3;} .t-amber .kpi-tile  {background:linear-gradient(135deg,#fbbf4a,#f08c1b); box-shadow:0 8px 16px rgba(240,140,27,.28);}
.t-rose   {background:#fdecee;} .t-rose .kpi-tile   {background:linear-gradient(135deg,#fb7185,#e5484d); box-shadow:0 8px 16px rgba(229,72,77,.28);}
.t-sky    {background:#e8f3fd;} .t-sky .kpi-tile    {background:linear-gradient(135deg,#5ab4f5,#2f80ed); box-shadow:0 8px 16px rgba(47,128,237,.28);}

/* ---- white cards (st.container(key="card-...")) ---- */
div[class*="st-key-card"] {background:#fff; border-radius:24px; padding:20px 22px 12px;
  box-shadow:0 1px 2px rgba(16,24,40,.04), 0 10px 26px rgba(16,24,40,.05);}
.card-title {font-weight:700; font-size:1.02rem; color:#14151a;}
.card-sub {font-size:.8rem; color:#6b7080; margin-top:1px;}
.legend {display:flex; gap:16px; font-size:.78rem; color:#5c6070; font-weight:600; margin-top:6px;}
.dot {display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:6px;}

/* ---- dark balance card ---- */
.dark {background:radial-gradient(120% 120% at 100% 0%, #2c2e3a 0%, #14151a 55%); color:#fff;
  border-radius:24px; padding:22px; box-shadow:0 14px 30px rgba(20,21,26,.25);}
.dark-label {font-size:.82rem; color:#b9bccb; font-weight:600;}
.dark-value {font-size:2.1rem; font-weight:800; letter-spacing:-.02em; margin:2px 0 14px;}
.dark-grid {display:grid; grid-template-columns:repeat(3,1fr); gap:8px; background:rgba(255,255,255,.06);
  border-radius:16px; padding:12px;}
.dark-k {font-size:.7rem; color:#a6a9b8;} .dark-v {font-size:.95rem; font-weight:700; margin-top:2px;}

/* ---- gradient highlight card ---- */
.grad {background:linear-gradient(120deg,#f472b6 0%,#fb923c 100%); color:#fff; border-radius:24px;
  padding:20px 22px; margin-top:14px; box-shadow:0 14px 28px rgba(244,114,182,.28); position:relative; overflow:hidden;}
.grad-k {font-size:.82rem; opacity:.92; font-weight:600;}
.grad-v {font-size:1.9rem; font-weight:800; letter-spacing:-.02em;}
.grad-s {font-size:.8rem; opacity:.92;}
.grad-bars {position:absolute; right:18px; bottom:0; display:flex; gap:5px; align-items:flex-end; opacity:.45;}
.grad-bars span {width:5px; background:#fff; border-radius:3px 3px 0 0;}

/* ---- info tiles ---- */
.tiles {display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-top:14px;}
.tile {background:#fff; border-radius:18px; padding:14px 16px; box-shadow:0 1px 2px rgba(16,24,40,.04), 0 10px 26px rgba(16,24,40,.05);}
.tile-k {font-size:.74rem; color:#6b7080; font-weight:600;}
.tile-v {font-size:.98rem; font-weight:700; color:#14151a; margin-top:4px;}

/* ---- trades table ---- */
.tx-wrap {overflow-x:auto;}
table.tx {width:100%; border-collapse:collapse; font-size:.86rem;}
table.tx td, table.tx th {white-space:nowrap;}
table.tx th {text-align:left; font-weight:600; color:#6b7080; background:#f5f5f9; padding:10px 12px; font-size:.78rem;}
table.tx th:first-child {border-radius:12px 0 0 12px;} table.tx th:last-child {border-radius:0 12px 12px 0;}
table.tx td {padding:12px; border-bottom:1px solid #f0f1f5; color:#23252d;}
table.tx tr:last-child td {border-bottom:none;}
.tk {display:flex; align-items:center; gap:10px; font-weight:700;}
.tk-av {width:30px; height:30px; border-radius:10px; display:grid; place-items:center; font-size:.72rem;
  font-weight:800; background:#efedff; color:#5b4ff5;}
.pill {display:inline-block; padding:3px 11px; border-radius:999px; font-size:.76rem; font-weight:700; border:1px solid;}
.pill.win {color:#16a37a; border-color:#bfe8d8; background:#effaf5;}
.pill.loss {color:#e5484d; border-color:#f7c9cb; background:#fff3f3;}
.muted {color:#8a8e9c; font-size:.76rem;}
</style>
""")


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
@st.cache_data
def load_run() -> tuple[dict, pd.DataFrame, pd.DataFrame] | None:
    if not (RUN_DIR / "meta.json").exists():
        return None
    meta = json.loads((RUN_DIR / "meta.json").read_text())
    eq = pd.read_csv(RUN_DIR / "equity.csv", index_col="date", parse_dates=True)
    trades = pd.read_csv(RUN_DIR / "trades.csv", parse_dates=["entry_date", "exit_date"])
    if not trades.empty:
        trades["holding_days"] = (trades["exit_date"] - trades["entry_date"]).dt.days
        trades["ticker"] = trades["ticker"].str.replace("_", "&")  # M_M → M&M
    return meta, eq, trades


def load_paper() -> dict | None:
    if not PAPER_DB.exists():
        return None
    with sqlite3.connect(PAPER_DB) as conn:
        return {
            "equity": pd.read_sql("SELECT * FROM equity ORDER BY date", conn,
                                  index_col="date", parse_dates=["date"]),
            "positions": pd.read_sql("SELECT * FROM positions", conn),
            "pending": pd.read_sql("SELECT * FROM pending_orders", conn),
            "trades": pd.read_sql("SELECT * FROM trades ORDER BY exit_date DESC", conn),
            "cash": conn.execute("SELECT value FROM meta WHERE key='cash'").fetchone()[0],
            "initial": conn.execute(
                "SELECT value FROM meta WHERE key='initial_capital'").fetchone()[0],
        }


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def inr(x: float) -> str:
    """₹ in lakh/crore, the way Indian users read money."""
    if abs(x) >= 1e7:
        return f"₹{x / 1e7:,.2f} Cr"
    if abs(x) >= 1e5:
        return f"₹{x / 1e5:,.2f} L"
    return f"₹{x:,.0f}"


def kpi(col, tone: str, icon_name: str, label: str, value: str,
        note: str = "", direction: str = "") -> None:
    col.html(f"""
      <div class="kpi t-{tone}">
        <div class="kpi-tile">{icon(icon_name, 24)}</div>
        <div><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div>
        <div class="kpi-note {direction}">{note}</div></div>
      </div>""")


def card_header(title: str, sub: str = "", legend: list[tuple[str, str]] | None = None) -> None:
    items = "".join(f'<span><span class="dot" style="background:{c}"></span>{n}</span>'
                    for n, c in (legend or []))
    st.html(f'<div class="card-title">{title}</div>'
            + (f'<div class="card-sub">{sub}</div>' if sub else "")
            + (f'<div class="legend">{items}</div>' if items else ""))


def style(fig: go.Figure, height: int = 320, pct_axis: bool = False) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=52, r=12, t=8, b=36), showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=MUTED, size=12), hovermode="x unified",
        hoverlabel=dict(bgcolor="#fff", bordercolor=GRID, font=dict(family=FONT, color=INK)),
        barcornerradius=6,
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=GRID, automargin=True)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, griddash="dot", automargin=True)
    if pct_axis:
        fig.update_yaxes(tickformat=".0%")
    return fig


def chart(fig: go.Figure) -> None:
    st.plotly_chart(fig, width="stretch", theme=None, config={"displayModeBar": False})


def drawdown(s: pd.Series) -> pd.Series:
    return s / s.cummax() - 1


def period_returns(eq: pd.Series, rule: str) -> pd.Series:
    """Compounded return per calendar period; the first period starts from capital."""
    last = eq.resample(rule).last()
    prev = last.shift(1)
    prev.iloc[0] = eq.iloc[0]
    return last / prev - 1


# --------------------------------------------------------------------------
# top bar + tabs
# --------------------------------------------------------------------------
st.html(f"""
<div class="topbar">
  <div class="brand">
    <div class="brand-logo">{icon("chart", 22)}</div>
    <div><div class="brand-name">NSE Algo Trading</div>
    <div class="brand-sub">NIFTY 50 swing-trading research system</div></div>
  </div>
  <div class="chips">
    <span class="chip">Out-of-sample · 2022–25</span>
    <span class="chip">After Indian costs</span>
    <a class="chip chip-dark" href="{REPO_URL}" target="_blank">{icon("github", 15)} Source</a>
  </div>
</div>""")

tab_overview, tab_trades, tab_paper, tab_method = st.tabs(
    ["Overview", "Trades", "Paper trading", "Methodology"])

run = load_run()

# --------------------------------------------------------------------------
# Overview
# --------------------------------------------------------------------------
with tab_overview:
    if run is None:
        st.info("No results yet. Run `python -m ml.backtest_ml --split validate "
                "--min-proba 0.50 --max-positions 4`, then refresh.")
    else:
        meta, eq_df, trades = run
        m = meta["metrics"]
        eq, bench = eq_df["equity"], eq_df["benchmark"]
        start_cap = eq.iloc[0]

        st.html(f"""<div class="hello">Strategy performance</div>
        <div class="hello-sub">ML swing strategy on NIFTY 50 stocks · {m['start']} → {m['end']} ·
        never seen during training · educational project, not financial advice</div>""")

        k = st.columns(4, gap="medium")
        kpi(k[0], "violet", "up", "Total return", f"{m['total_return_pct']:+.0f}%",
            f"▲ {m['total_return_pct'] - m['benchmark_return_pct']:.0f} pp vs NIFTY 50", "up")
        kpi(k[1], "mint", "pct", "CAGR", f"{m['cagr_pct']:.1f}%",
            f"▲ {m['cagr_pct'] - m['benchmark_cagr_pct']:.1f} pp a year vs NIFTY 50", "up")
        kpi(k[2], "rose", "down", "Max drawdown", f"{m['max_drawdown_pct']:.1f}%",
            f"▼ {abs(m['max_drawdown_pct'] - m['benchmark_max_dd_pct']):.1f} pp deeper than NIFTY 50"
            if m["max_drawdown_pct"] < m["benchmark_max_dd_pct"] else "shallower than NIFTY 50",
            "down" if m["max_drawdown_pct"] < m["benchmark_max_dd_pct"] else "up")
        kpi(k[3], "amber", "target", "Win rate", f"{m['win_rate']:.0%}",
            f"{m['n_trades']} trades · profit factor {m['profit_factor']:.2f}")

        st.html('<div style="height:6px"></div>')
        main, side = st.columns([2.15, 1], gap="medium")
        with main, st.container(key="card-growth"):
            card_header("Growth of ₹1", "Strategy vs buying and holding the NIFTY 50 index",
                        [("Strategy", VIOLET), ("NIFTY 50", BENCH)])
            g, b = eq / start_cap, bench / bench.iloc[0]
            fig = go.Figure()
            fig.add_scatter(x=b.index, y=b, name="NIFTY 50", line=dict(color=BENCH, width=1.6),
                            hovertemplate="%{y:.2f}×")
            fig.add_scatter(x=g.index, y=g, name="Strategy", line=dict(color=VIOLET, width=2.6),
                            fill="tozeroy", hovertemplate="%{y:.2f}×",
                            fillgradient=dict(type="vertical", colorscale=[
                                [0, "rgba(91,79,245,0)"], [1, "rgba(91,79,245,0.22)"]]))
            lo = float(min(g.min(), b.min()))
            fig.update_yaxes(range=[lo - 0.05, float(max(g.max(), b.max())) + 0.08],
                             ticksuffix="×")
            chart(style(fig, 345))

        with side:
            costs_paid = trades["costs"].sum() if not trades.empty else 0.0
            st.html(f"""
            <div class="dark">
              <div class="dark-label">Portfolio value at the end</div>
              <div class="dark-value">{inr(eq.iloc[-1])}</div>
              <div class="dark-grid">
                <div><div class="dark-k">Started with</div><div class="dark-v">{inr(start_cap)}</div></div>
                <div><div class="dark-k">Net profit</div><div class="dark-v">{inr(eq.iloc[-1] - start_cap)}</div></div>
                <div><div class="dark-k">Costs paid</div><div class="dark-v">{inr(costs_paid)}</div></div>
              </div>
            </div>
            <div class="grad">
              <div class="grad-k">Same money in NIFTY 50 would be</div>
              <div class="grad-v">{inr(start_cap * bench.iloc[-1] / bench.iloc[0])}</div>
              <div class="grad-s">The strategy finished {inr(eq.iloc[-1] - start_cap * bench.iloc[-1] / bench.iloc[0])} ahead</div>
              <div class="grad-bars">{''.join(f'<span style="height:{h}px"></span>' for h in (14, 26, 18, 38, 22, 52, 30, 44, 20, 60, 34))}</div>
            </div>
            <div class="tiles">
              <div class="tile"><div class="tile-k">Model</div><div class="tile-v">Logistic regression</div></div>
              <div class="tile"><div class="tile-k">Predicts</div><div class="tile-v">+5% within 20 days</div></div>
              <div class="tile"><div class="tile-k">Buys when</div><div class="tile-v">P ≥ 0.50</div></div>
              <div class="tile"><div class="tile-k">Holds at most</div><div class="tile-v">4 stocks</div></div>
            </div>""")

        st.html('<div style="height:6px"></div>')
        c1, c2, c3 = st.columns([1.15, 1.15, 0.9], gap="medium")
        with c1, st.container(key="card-dd"):
            card_header("Drawdown", "How far below its previous peak",
                        [("Strategy", LOSS), ("NIFTY 50", BENCH)])
            fig = go.Figure()
            fig.add_scatter(x=bench.index, y=drawdown(bench), line=dict(color=BENCH, width=1.2),
                            name="NIFTY 50", hovertemplate="%{y:.1%}")
            fig.add_scatter(x=eq.index, y=drawdown(eq), name="Strategy", fill="tozeroy",
                            line=dict(color=LOSS, width=1.4), fillcolor="rgba(229,72,77,0.12)",
                            hovertemplate="%{y:.1%}")
            chart(style(fig, 250, pct_axis=True))
        with c2, st.container(key="card-years"):
            card_header("Calendar-year returns", "Each year on its own",
                        [("Strategy", VIOLET), ("NIFTY 50", BENCH)])
            yr, byr = period_returns(eq, "YE"), period_returns(bench, "YE")
            fig = go.Figure()
            fig.add_bar(x=yr.index.year, y=yr, name="Strategy", marker_color=VIOLET,
                        hovertemplate="%{y:.1%}")
            fig.add_bar(x=byr.index.year, y=byr, name="NIFTY 50", marker_color="#d6d8e3",
                        hovertemplate="%{y:.1%}")
            fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.08)
            fig.update_xaxes(type="category")
            chart(style(fig, 250, pct_axis=True))
        with c3, st.container(key="card-donut"):
            wins = int((trades["pnl"] > 0).sum())
            losses = int(len(trades) - wins)
            card_header("Winning vs losing trades", f"{len(trades)} closed trades")
            fig = go.Figure(go.Pie(
                values=[wins, losses], labels=["Winners", "Losers"], hole=0.68, sort=False,
                marker=dict(colors=[GAIN, "#f5a3a6"], line=dict(color="#fff", width=4)),
                textinfo="none", hovertemplate="%{label}: %{value}<extra></extra>"))
            fig.add_annotation(text=f"<b>{wins / len(trades):.0%}</b><br>"
                                    f"<span style='font-size:11px;color:{MUTED}'>win rate</span>",
                               showarrow=False, font=dict(size=22, color=INK, family=FONT))
            chart(style(fig, 190))
            avg_w = trades.loc[trades["pnl"] > 0, "return_pct"].mean()
            avg_l = trades.loc[trades["pnl"] <= 0, "return_pct"].mean()
            st.html(f"""<div style="display:flex;justify-content:space-between;font-size:.82rem;
                padding:0 4px 8px;color:#3b3e4a;font-weight:600">
                <span><span class="dot" style="background:{GAIN}"></span>{wins} winners · avg {avg_w:+.1f}%</span>
                <span><span class="dot" style="background:#f5a3a6"></span>{losses} losers · avg {avg_l:+.1f}%</span>
                </div>""")

        st.html('<div style="height:6px"></div>')
        left, right = st.columns([1.55, 1], gap="medium")
        with left, st.container(key="card-recent"):
            card_header("Recent trades", "Latest round trips, newest first, after costs")
            rows = ""
            for _, t in trades.sort_values("exit_date", ascending=False).head(7).iterrows():
                won = t["pnl"] > 0
                rows += f"""<tr>
                  <td><div class="tk"><div class="tk-av">{t['ticker'][:2]}</div>{t['ticker']}</div></td>
                  <td>{t['entry_date']:%d %b %Y}<div class="muted">{t['holding_days']} days held</div></td>
                  <td>₹{t['entry_price']:,.2f}<div class="muted">→ ₹{t['exit_price']:,.2f}</div></td>
                  <td style="font-weight:700;color:{GAIN if won else LOSS}">{t['return_pct']:+.2f}%</td>
                  <td><span class="pill {'win' if won else 'loss'}">{'Profit' if won else 'Loss'}
                      {inr(abs(t['pnl']))}</span></td></tr>"""
            st.html(f"""<div class="tx-wrap"><table class="tx"><thead><tr><th>Stock</th><th>Bought</th>
                <th>Price</th><th>Return</th><th>Result</th></tr></thead><tbody>{rows}</tbody></table></div>
                <div style="height:8px"></div>""")
        with right, st.container(key="card-heat"):
            card_header("Monthly returns", "Green months made money, red lost it")
            mr = period_returns(eq, "ME")
            grid = (pd.DataFrame({"y": mr.index.year, "m": mr.index.month, "r": mr.values})
                    .pivot(index="y", columns="m", values="r").reindex(columns=range(1, 13)))
            lim = float(np.nanmax(np.abs(grid.values))) or 0.01
            fig = go.Figure(go.Heatmap(
                z=grid.values, x=MONTHS, y=[str(y) for y in grid.index],
                zmin=-lim, zmax=lim, zmid=0, showscale=False, xgap=4, ygap=4,
                colorscale=[[0, LOSS], [0.5, "#f4f4f8"], [1, GAIN]],
                hovertemplate="%{x} %{y}: %{z:+.1%}<extra></extra>"))
            fig.update_yaxes(autorange="reversed", type="category", showgrid=False)
            fig.update_xaxes(type="category", tickangle=0, tickfont=dict(size=10))
            fig.update_layout(hovermode="closest")
            chart(style(fig, 70 + 52 * len(grid)))

# --------------------------------------------------------------------------
# Trades
# --------------------------------------------------------------------------
with tab_trades:
    if run is None:
        st.info("No results yet.")
    else:
        _, _, trades = run
        wins, losses = trades[trades["pnl"] > 0], trades[trades["pnl"] <= 0]
        k = st.columns(4, gap="medium")
        kpi(k[0], "violet", "pulse", "Average trade", f"{trades['return_pct'].mean():+.2f}%",
            "after all costs")
        kpi(k[1], "mint", "up", "Average winner", f"{wins['return_pct'].mean():+.2f}%",
            f"{len(wins)} trades", "up")
        kpi(k[2], "rose", "down", "Average loser", f"{losses['return_pct'].mean():+.2f}%",
            f"{len(losses)} trades", "down")
        kpi(k[3], "sky", "clock", "Typical holding", f"{trades['holding_days'].median():.0f} days",
            "median, entry to exit")

        st.html('<div style="height:6px"></div>')
        left, right = st.columns(2, gap="medium")
        with left, st.container(key="card-hist"):
            card_header("Return per trade", "How many trades landed in each 1% bucket",
                        [("Winners", GAIN), ("Losers", LOSS)])
            fig = go.Figure()
            for part, colour in ((losses, LOSS), (wins, GAIN)):
                fig.add_histogram(x=part["return_pct"], xbins=dict(size=1.0), marker_color=colour,
                                  opacity=0.9, hovertemplate="%{x}%: %{y} trades<extra></extra>")
            fig.update_layout(barmode="overlay", bargap=0.08, hovermode="closest")
            fig.update_xaxes(ticksuffix="%")
            chart(style(fig, 320))
        with right, st.container(key="card-stocks"):
            by = trades.groupby("ticker")["pnl"].sum().sort_values()
            show = pd.concat([by.head(7), by.tail(7)]) if len(by) > 14 else by
            show = show[~show.index.duplicated()]
            card_header("Profit by stock", "7 best and 7 worst, net of costs (₹ lakh)")
            fig = go.Figure(go.Bar(
                x=show.values / 1e5, y=show.index, orientation="h",
                marker_color=[GAIN if v > 0 else LOSS for v in show.values],
                hovertemplate="%{y}: ₹%{x:.2f} L<extra></extra>"))
            fig.update_xaxes(tickprefix="₹", ticksuffix=" L", gridcolor=GRID, showgrid=True,
                             griddash="dot")
            fig.update_yaxes(showgrid=False)
            fig.update_layout(hovermode="closest")
            chart(style(fig, 320))

        st.html('<div style="height:6px"></div>')
        with st.container(key="card-log"):
            card_header("All trades", "Filter by stock or outcome")
            f1, f2 = st.columns([3, 1])
            pick = f1.multiselect("Stocks", sorted(trades["ticker"].unique()),
                                  placeholder="All stocks", label_visibility="collapsed")
            outcome = f2.segmented_control("Outcome", ["All", "Winners", "Losers"],
                                           default="All", label_visibility="collapsed")
            view = trades
            if pick:
                view = view[view["ticker"].isin(pick)]
            if outcome == "Winners":
                view = view[view["pnl"] > 0]
            elif outcome == "Losers":
                view = view[view["pnl"] <= 0]
            view = view.round({"costs": 0, "pnl": 0}).sort_values("exit_date", ascending=False)[
                ["ticker", "entry_date", "exit_date", "holding_days", "shares",
                 "entry_price", "exit_price", "costs", "pnl", "return_pct"]]
            colour = lambda v: f"color: {GAIN if v > 0 else LOSS}; font-weight: 600"  # noqa: E731
            st.dataframe(
                view.style.map(colour, subset=["pnl", "return_pct"]),
                hide_index=True, width="stretch", height=400,
                column_config={
                    "ticker": "Stock",
                    "entry_date": st.column_config.DateColumn("Entry", format="DD MMM YYYY"),
                    "exit_date": st.column_config.DateColumn("Exit", format="DD MMM YYYY"),
                    "holding_days": st.column_config.NumberColumn("Days held"),
                    "shares": st.column_config.NumberColumn("Qty"),
                    "entry_price": st.column_config.NumberColumn("Entry ₹", format="%.2f"),
                    "exit_price": st.column_config.NumberColumn("Exit ₹", format="%.2f"),
                    "costs": st.column_config.NumberColumn("Costs ₹", format="localized"),
                    "pnl": st.column_config.NumberColumn("P&L ₹", format="localized"),
                    "return_pct": st.column_config.NumberColumn("Return", format="%.2f%%"),
                })
            st.caption(f"{len(view):,} of {len(trades):,} trades")

# --------------------------------------------------------------------------
# Paper trading
# --------------------------------------------------------------------------
with tab_paper:
    paper = load_paper()
    if paper is None:
        with st.container(key="card-paper-empty"):
            card_header("Paper trading hasn't started yet",
                        "Simulated money, real daily data. The last check before real money.")
            st.markdown(
                "Run it **once a day after market close (15:30 IST)**. It fills yesterday's "
                "orders at today's open, exactly like the backtest, and keeps its state in "
                "SQLite so nothing is lost on a crash. Run it for 2–3 months and compare with "
                "the backtest before trusting the strategy.")
            st.code("source .venv/bin/activate\n"
                    "python main.py --mode paper-run --strategy ml --max-positions 4   # once per trading day\n"
                    "python main.py --mode paper-status", language="bash")
    else:
        eqp, pt = paper["equity"]["value"], paper["trades"]
        k = st.columns(4, gap="medium")
        latest = eqp.iloc[-1] if len(eqp) else paper["initial"]
        chg = latest / paper["initial"] - 1
        kpi(k[0], "violet", "chart", "Paper equity", inr(latest),
            f"{chg:+.2%} since start", "up" if chg >= 0 else "down")
        kpi(k[1], "mint", "coins", "Cash", inr(paper["cash"]), "available to invest")
        kpi(k[2], "amber", "target", "Open positions", str(len(paper["positions"])),
            f"{len(paper['pending'])} orders for tomorrow")
        kpi(k[3], "sky", "pulse", "Closed trades", str(len(pt)),
            f"{(pt['pnl'] > 0).mean():.0%} win rate" if len(pt) else "none yet")
        st.html('<div style="height:6px"></div>')
        if len(eqp) > 1:
            with st.container(key="card-paper-eq"):
                card_header("Paper equity", "Value of the simulated portfolio each day")
                fig = go.Figure(go.Scatter(x=eqp.index, y=eqp, line=dict(color=VIOLET, width=2.4),
                                           hovertemplate="₹%{y:,.0f}<extra></extra>"))
                chart(style(fig, 280))
        money = st.column_config.NumberColumn(format="₹%.2f")
        left, right = st.columns(2, gap="medium")
        with left, st.container(key="card-paper-pos"):
            card_header("Open positions")
            if paper["positions"].empty:
                st.caption("No open positions.")
            else:
                st.dataframe(paper["positions"], hide_index=True, width="stretch",
                             column_config={"ticker": "Stock", "shares": "Qty",
                                            "entry_date": "Since", "entry_price": money,
                                            "entry_costs": money})
        with right, st.container(key="card-paper-orders"):
            card_header("Orders for tomorrow's open")
            if paper["pending"].empty:
                st.caption("No orders: the strategy wants no changes today.")
            else:
                st.dataframe(paper["pending"], hide_index=True, width="stretch",
                             column_config={"ticker": "Stock", "side": "Side",
                                            "created_date": "Decided on"})
        if len(pt):
            with st.container(key="card-paper-trades"):
                card_header("Closed trades")
                st.dataframe(pt.drop(columns=["id"]), hide_index=True, width="stretch")

# --------------------------------------------------------------------------
# Methodology
# --------------------------------------------------------------------------
with tab_method:
    left, right = st.columns([3, 2], gap="medium")
    with left, st.container(key="card-method"):
        card_header("How these numbers were produced")
        st.markdown(
            """
**The backtest can't see the future.** A signal computed from day *t*'s close is
filled at day *t+1*'s **open**, with 5 bps of slippage against us.

**Costs are real.** Every trade pays Indian delivery charges (see the breakdown).

**The model never saw these years.** It predicts whether a stock will gain
**more than 5% over the next 20 trading days**, was trained on 2015–2021 data, and
had the last 35 days of training purged so no label could peek into 2022.
Its settings were chosen on 2020–21, and the 2022–25 period was scored once.

**What could still flatter it.** The stock list is *today's* NIFTY 50
(survivorship bias). This is one market regime, and most gains came in
2022–early 2024. Slippage is a flat estimate with no market-impact model.
""")
    with right, st.container(key="card-costs"):
        card_header("Cost of one ₹1,00,000 round trip", "Buy, then sell, through a discount broker")
        cm, v = CostModel(), 100_000.0
        brokerage = min(cm.brokerage_cap, v * cm.brokerage_pct)
        exch_sebi = v * (cm.exchange_pct + cm.sebi_pct)
        rows = {
            "Brokerage (buy + sell)": 2 * brokerage,
            "STT (0.1% each side)": 2 * v * cm.stt_pct,
            "Exchange + SEBI fees": 2 * exch_sebi,
            "GST (18% on fees)": 2 * cm.gst_pct * (brokerage + exch_sebi),
            "Stamp duty (buy side)": v * cm.stamp_pct,
            "DP charge (sell side)": cm.dp_charge,
            "Slippage (5 bps each side)": 2 * v * 0.0005,
        }
        body = "".join(f"<tr><td>{k_}</td><td style='text-align:right;font-weight:600'>₹{v_:,.2f}</td></tr>"
                       for k_, v_ in rows.items())
        total = sum(rows.values())
        st.html(f"""<table class="tx"><thead><tr><th>Charge</th><th style="text-align:right">Amount</th>
            </tr></thead><tbody>{body}<tr><td style="font-weight:800">Total</td>
            <td style="text-align:right;font-weight:800">₹{total:,.0f}
            <span class="muted">({total / v:.2%})</span></td></tr></tbody></table>
            <div style="height:10px"></div>""")
