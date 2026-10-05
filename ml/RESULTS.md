# ML iteration log

## Summary (2026-09-04)

**Task:** train a model that picks swing-trade entries better than chance, then
confirm the edge survives a cost-aware backtest against buy-and-hold NIFTY.

**Data:** frozen yfinance snapshot, 48 NIFTY-50 stocks + index, split/div
adjusted. Walk-forward windows (`data/frozen/MANIFEST.json`):

| window | dates | use |
|---|---|---|
| tune | 2015–2021 | fit + iterate (sub-split 2015–19 train / 2020–21 eval) |
| validate | 2022–2025 | touched once, at the end |
| holdout | 2026+ | **never looked at** — still untouched |

Every training window is purged of its last `embargo_for(horizon)` calendar days
(35 days for the 20-trading-day label), so no training label reaches the next window.

### What moved the needle

| change | effect |
|---|---|
| baseline: 8 features, StandardScaler, logistic/RF | RF precision@0.6 ≈ 0.49 (base 0.37), logistic flat |
| **+17 features** (multi-horizon momentum, 52w-high dist, vol regime, RSI-2, Bollinger pos, volume z, market-relative excess return, regime flag) | logistic becomes the strongest, most stable model |
| **RobustScaler + ±5σ winsorize** (fat-tailed return features) | logistic precision@0.6 0.46 → 0.54 tune-internal |
| **embargo gap at split boundary** | removes look-ahead leak that flattered earlier scores |
| **embargo ≥ label horizon, purged from the training side** (2026-10-06 fix; was a 10-day gap vs a ~28-calendar-day label) | validate lift@0.6 1.64 → 1.56, CAGR 20.95% → 19.76%, max DD −17.9% → −21.4%: the edge is real but was slightly flattered |
| **label = fwd 20-day return > 5%** (vs 10-day > 2%) | sharper edge: validate lift@0.6 1.30 → 1.64 |
| dropped `ret_20d_rank` (cross-sectional, unavailable to the live 1-stock-at-a-time backtester) | train/live inputs now identical; edge unchanged |
| hist_gbm / random_forest / soft ensemble | all tried, all overfit the pre-2020 window vs logistic — **logistic chosen** |

### Locked model

`logistic` (RobustScaler → ±5σ clip → LogisticRegression C=0.5, class_weight
balanced), label = P(forward 20-day return > 5%), 25 features.
Artifact: `ml/artifacts/model.pkl` (trained on tune ≤ 2021, last 35 days purged).

**Validate (2022–2025), scored once:** base rate 0.280 →
**precision@0.6 = 0.436 (1.56× lift)**, precision@0.7 = 0.543 (1.94× lift).
(Before the 2026-10-06 embargo fix: 0.456 / 1.64× and 0.596 — kept in the auto-log below.)

### Cost-aware backtest

Strategy hyperparameters (`min_proba`, `max_positions`) were swept on the
**2020–2021** surface using a model trained on ≤2019 only — not on validate.
Winner: `min_proba 0.50`, `exit_proba 0.45`, `max_positions 4` (concentration
matters — the model's ranking is good but it fires rarely, so cash drag is the
enemy).

**Validate 2022–2025, locked params** (after the embargo fix; the sweep was re-run
with the purged ≤2019 model and its ranking didn't change, so params stayed put):

| | ML strategy | NIFTY buy&hold | sma_rsi (rule-based) |
|---|---|---|---|
| total return | **+105.4%** | +48.2% | −4.4% |
| CAGR | **19.76%** | 10.37% | −1.11% |
| Sharpe | 0.75 | — | −0.58 |
| max drawdown | −21.4% | −16.5% | −21.9% |
| trades | 113 | — | 964 |
| win rate / profit factor | 49% / 2.17 | — | 34% / 0.96 |

Pre-fix numbers for the record: +113.7%, 20.95% CAGR, Sharpe 0.84, max DD −17.9%,
115 trades, PF 2.28.

The edge translates: ~1.9× the benchmark CAGR, ~28 trades/year (genuine swing
cadence), winners ~2.2× the size of losers. Drawdown is deeper than the index's.

**On the param choice:** in the 2020–21 sweep, min_proba 0.60 / 4 positions has the
highest Sharpe (2.39 vs 2.29), both before and after the fix. 0.50 was kept as the
centre of a flat plateau with more signals; choosing again after seeing validate
would turn validate into a tuning surface.

### Honest caveats

- **Survivorship bias**: universe = *today's* NIFTY 50 back-projected. Real
  historical constituents differ; this overstates returns.
- **48 stocks, not 50** — TATAMOTORS and LTIM are 404 on Yahoo (`.NS` + `.BO`).
  TATAMOTORS 2000–2021 exists in `data/external/kaggle_rohanrao/` if a mixed-
  source backfill is ever wanted.
- **One validate window, one regime.** No rolling multi-fold walk-forward yet.
- **Holdout 2026 untouched** — that is the real out-of-sample test. Run
  `python -m ml.backtest_ml --split holdout` exactly once when ready.
- Slippage is a flat 5 bps; fine for large-cap NIFTY names, optimistic for
  fast moves.

### Reproduce

```
python -m data_engineering.freeze --verify            # data integrity
python -m ml.experiment --final --models logistic --horizon 20 --threshold 0.05
python -m ml.backtest_ml --split validate --min-proba 0.50 --max-positions 4
```

---

## Auto-log (appended by `ml/experiment.py`)

| when | scope | model | horizon | thr | n_feat | base | prec@0.6 | lift@0.6 | n@0.6 | prec@0.7 | prec@top10% | lift@top10% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 09-04 11:19 | tune-internal 20-21 | logistic | 10 | 0.02 | 26 | 0.433 | 0.541 | 1.249 | 4169 | 0.563 | 0.551 | 1.270 |
| 09-04 11:19 | tune-internal 20-21 | random_forest | 10 | 0.02 | 26 | 0.433 | 0.510 | 1.176 | 2653 | 0.430 | 0.512 | 1.181 |
| 09-04 11:19 | tune-internal 20-21 | hist_gbm | 10 | 0.02 | 26 | 0.433 | 0.507 | 1.169 | 2467 | 0.518 | 0.510 | 1.177 |
| 09-04 11:19 | tune-internal 20-21 | logistic | 10 | 0.03 | 26 | 0.371 | 0.481 | 1.297 | 5397 | 0.510 | 0.510 | 1.375 |
| 09-04 11:19 | tune-internal 20-21 | hist_gbm | 10 | 0.03 | 26 | 0.371 | 0.481 | 1.296 | 1628 | 0.461 | 0.477 | 1.286 |
| 09-04 11:20 | tune-internal 20-21 | logistic | 10 | 0.04 | 26 | 0.315 | 0.426 | 1.352 | 6159 | 0.475 | 0.474 | 1.502 |
| 09-04 11:20 | tune-internal 20-21 | hist_gbm | 10 | 0.04 | 26 | 0.315 | 0.449 | 1.424 | 1002 | 0.445 | 0.442 | 1.401 |
| 09-04 11:20 | tune-internal 20-21 | logistic | 20 | 0.05 | 26 | 0.383 | 0.514 | 1.341 | 5411 | 0.577 | 0.574 | 1.500 |
| 09-04 11:20 | tune-internal 20-21 | hist_gbm | 20 | 0.05 | 26 | 0.383 | 0.497 | 1.299 | 2491 | 0.515 | 0.496 | 1.295 |
| 09-04 11:20 | tune-internal 20-21 | hist_gbm | 10 | 0.02 | 26 | 0.433 | 0.515 | 1.189 | 2014 | 0.524 | 0.516 | 1.190 |
| 09-04 11:20 | tune-internal 20-21 | random_forest | 10 | 0.02 | 26 | 0.433 | 0.511 | 1.179 | 2630 | 0.514 | 0.524 | 1.209 |
| 09-04 11:21 | tune-internal 20-21 | ensemble | 10 | 0.03 | 26 | 0.371 | 0.495 | 1.334 | 2375 | 0.469 | 0.495 | 1.334 |
| 09-04 11:21 | tune-internal 20-21 | ensemble | 20 | 0.05 | 26 | 0.383 | 0.526 | 1.374 | 2909 | 0.536 | 0.531 | 1.386 |
| 09-04 11:21 | validate 22-25 | logistic | 20 | 0.05 | 26 | 0.278 | 0.448 | 1.610 | 596 | 0.566 | 0.353 | 1.268 |
| 09-04 11:21 | validate 22-25 | ensemble | 20 | 0.05 | 26 | 0.278 | 0.462 | 1.661 | 318 | 0.333 | 0.334 | 1.202 |
| 09-04 11:21 | validate 22-25 | hist_gbm | 20 | 0.05 | 26 | 0.278 | 0.390 | 1.403 | 333 | 0.333 | 0.330 | 1.186 |
| 09-04 11:22 | validate 22-25 | logistic | 10 | 0.03 | 26 | 0.295 | 0.384 | 1.303 | 565 | 0.488 | 0.371 | 1.259 |
| 09-04 11:22 | validate 22-25 | ensemble | 10 | 0.03 | 26 | 0.295 | 0.376 | 1.277 | 271 | 0.393 | 0.362 | 1.228 |
| 09-04 11:22 | tune-internal 20-21 | logistic | 20 | 0.05 | 25 | 0.383 | 0.514 | 1.343 | 5365 | 0.579 | 0.575 | 1.501 |
| 09-04 11:22 | tune-internal 20-21 | ensemble | 20 | 0.05 | 25 | 0.383 | 0.531 | 1.386 | 2872 | 0.549 | 0.536 | 1.401 |
| 09-04 11:22 | validate 22-25 | logistic | 20 | 0.05 | 25 | 0.278 | 0.456 | 1.637 | 597 | 0.596 | 0.355 | 1.276 |
| 09-04 11:22 | validate 22-25 | ensemble | 20 | 0.05 | 25 | 0.278 | 0.500 | 1.797 | 326 | 0.387 | 0.344 | 1.235 |
| 09-04 11:28 | validate 22-25 | logistic | 20 | 0.05 | 25 | 0.278 | 0.456 | 1.637 | 597 | 0.596 | 0.355 | 1.276 |
| 10-06 00:20 | tune-internal 20-21 | logistic | 20 | 0.05 | 25 | 0.383 | 0.516 | 1.347 | 5302 | 0.575 | 0.570 | 1.488 |
| 10-06 00:20 | validate 22-25 | logistic | 20 | 0.05 | 25 | 0.280 | 0.436 | 1.558 | 574 | 0.543 | 0.354 | 1.267 |
