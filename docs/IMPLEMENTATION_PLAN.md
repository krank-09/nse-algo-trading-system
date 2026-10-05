# Algo Trading System — Implementation Plan

**Goal:** Swing/long-term algorithmic trading system for Indian equities (NIFTY 50 stocks + NIFTY index), paper-traded 2–3 months before any real money via Groww API.

## Locked decisions

| Decision | Choice |
|---|---|
| Market / universe | NSE — NIFTY 50 stocks + NIFTY index |
| Timeframe | Swing + long-term (daily candles) in v1; intraday added post-Phase G (needs streaming data, tighter slippage model, intraday cost structure) |
| Language | Python only for v1. C++/Rust deferred indefinitely (not needed for daily-candle latency) |
| Data (backtest) | Free: NSE bhavcopy archives + yfinance (deep history, pre-2020) |
| Data (live) + execution | Groww API (₹499+GST/mo) — subscribe only when paper trading starts (~Week 5) |
| Live gating | Paper trade 2–3 months; go live only if gate criteria pass |
| Frontend | Streamlit (no FastAPI in v1) |
| Live deployment | Mac at home — needs static IP from ISP + Mac awake during market hours (revisit before go-live) |

## SEBI compliance notes (April 2026 framework)

- Below 10 orders/sec → **no algo registration needed** (swing trading is far below this)
- Broker API requires: **whitelisted static IP**, OAuth 2FA, auto-logout before next pre-open
- Broker (Groww) is legally responsible for orders through its API — expect their onboarding checks

## Repo structure

```
trading-system/
├── data/                  # raw/ and processed/ (gitignored)
├── data_engineering/      # loader.py, cleaner.py, corporate_actions.py, features.py
├── strategies/            # base.py, sma_rsi.py, ...
├── backtesting/           # engine.py, costs.py, metrics.py
├── risk/                  # position_sizing.py, limits.py
├── execution/             # paper.py, groww_live.py (later)
├── ml/                    # (Phase F only)
├── frontend/              # app.py (Streamlit)
├── config.yaml
├── main.py                # CLI: python main.py --mode backtest --strategy sma_rsi
└── requirements.txt
```

## Phases (~3 hrs/day)

### Phase A — Data foundation (Week 1–2)
- Download daily OHLCV for NIFTY 50 stocks + NIFTY index (yfinance `.NS` tickers + NSE bhavcopy), 10+ years where available
- Clean: missing days, duplicate rows, timezone/date parsing
- **Corporate-action adjustment** (splits/bonuses) — verify against known events (e.g. IRCTC 2021 split)
- Note survivorship bias: record that today's NIFTY 50 list ≠ historical constituents; acceptable for v1, documented
- Indicators: SMA, EMA, RSI, ATR, returns, volatility
- **Output:** `data_engineering/`, a `make data` style CLI command

### Phase B — Strategy engine (Week 3)
- `Strategy` base class → emits signals (BUY/SELL/HOLD + confidence) from features only up to time t
- First strategies: SMA crossover + RSI filter + index-trend filter
- **Output:** `strategies/`

### Phase C — Backtester (Week 3–4) ⭐ most important
- Candle-by-candle event loop, orders execute at **next day's open** (no look-ahead)
- **India-realistic costs:** brokerage, STT (0.1% delivery), stamp duty, exchange charges, DP charge on sells, slippage
- Risk module: fixed-fraction position sizing, max positions, max drawdown kill-switch
- Metrics: CAGR, Sharpe, max drawdown, win rate, profit factor — vs buy-and-hold NIFTY benchmark
- Walk-forward split: tune on 2015–2021, validate on 2022–2025, never touch 2026
- **Output:** `backtesting/`, `risk/`

### Phase D — Paper trading (Week 5–6)
- Subscribe to Groww API; integrate `growwapi` SDK (live quotes, historical top-up from 2020)
- Paper engine: same strategy code, real live prices, simulated fills + costs, trade log to SQLite/CSV
- Daily run: fetch EOD data → generate signals → "place" paper orders next open
- **Output:** `execution/paper.py`

### Phase E — Dashboard (Week 6–7)
- Streamlit: equity curve, open positions, signal history, backtest vs paper comparison, logs
- **Output:** `frontend/app.py`

### Phase F — ML extension (Week 8+, optional)
- Only after rule-based system runs end-to-end
- Logistic regression / RandomForest on engineered features; walk-forward validation only
- ML strategy is just another `Strategy` subclass — architecture unchanged

### Phase H — IPO module (parallel, from Week 6+)
Two important constraints shape this module:
- **IPO applications cannot be automated.** Applying requires a UPI/ASBA mandate you approve manually — no broker API places IPO bids. So the bot *recommends*, you apply.
- **New listings have no price history**, so the backtester can't validate IPO-application decisions. This module is analysis + alerts, not backtested strategy.

Scope:
- `ipo/calendar.py` — track upcoming/open IPOs (NSE announcements, scraped or via free sources)
- `ipo/analyzer.py` — scorecard per IPO: subscription levels (QIB/NII/retail), grey-market premium, valuation vs listed peers, promoter dilution, anchor-book quality
- Alert: "IPO X closes tomorrow, score 7/10, reasons…" via dashboard/notification
- **Post-listing trading:** once a new stock has ~6 months of candles, it can enter the regular strategy universe — this part *is* backtestable and automated

### Phase G — Go-live gate (Month 3–4)
Go live with small capital **only if all pass** on 2–3 months of paper trading:
- [ ] Positive net PnL after all costs
- [ ] Max drawdown within pre-set limit (e.g. <10%)
- [ ] Paper results roughly match backtest expectations (no large live/backtest gap)
- [ ] Static IP arranged + Groww API auth flow (TOTP/OAuth) tested
- [ ] Kill-switch and max-loss-per-day limit implemented and tested

## Market-ready roadmap (layer in during paper-trading months, after A→D core loop)

### Reliability — the system can't silently fail
- Morning data-integrity checks: yesterday's candle arrived, cross-checked against a second source, alert on mismatch
- Crash recovery: positions persisted in SQLite; restart mid-day without losing state
- Order reconciliation: after every order, verify broker's fills/qty/avg-price match internal state
- Alerting (Telegram/email) on every trade, error, drawdown breach
- Audit log of every decision: signal, reason, data seen (SEBI-friendly)
- Kill switch: one command flattens all positions and halts

### Risk — where profits are protected (highest ROI section)
- ATR/volatility-scaled position sizing: same ₹ risk per trade regardless of stock
- Portfolio limits: max per stock, per sector, total exposure; correlation check across holdings
- Regime filter: cut exposure when index trend is down or volatility spikes
- Daily + monthly max-loss circuit breakers

### Edge — improves odds the strategy's edge is real (nothing guarantees profits)
- Multiple uncorrelated strategies (momentum + mean-reversion + breakout), capital split across them
- Walk-forward optimization on rolling windows (not one static parameter fit)
- Realistic execution model: volume-dependent slippage, partial fills
- Monte Carlo on trade sequences → realistic worst-case drawdown
- Fundamental/quality filters on universe (earnings, debt)
- Later: options hedging; expand universe past NIFTY 50 with proper survivorship-bias handling

## Cost summary

| Item | Cost |
|---|---|
| Weeks 1–4 | ₹0 (free data) |
| Week 5+ | ₹499+GST/mo (Groww API) |
| Go-live | possible ISP static-IP charge |

## Interview story
"I built a modular algo-trading system for Indian equities: bias-free backtesting with India-realistic transaction costs, walk-forward validation, risk management with drawdown kill-switch, paper trading against live Groww API data, and SEBI-compliant live-execution design."
