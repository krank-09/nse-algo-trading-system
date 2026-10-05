"""ML dataset construction from the feature matrix.

Label: will the stock's forward N-day return beat a threshold?
    y[t] = 1 if close[t+N]/close[t] - 1 > threshold else 0
The label at t uses FUTURE prices — that's fine, labels are what we predict.
What must never happen: a FEATURE at t containing future info, or training
on data that overlaps the test period. Hence:

- Features at t come from build_features (causal by construction).
- The last N rows of each instrument are dropped (labels would peek past the
  data end).
- Splits are TIME-BASED, never random. Random splits on time series leak:
  adjacent, near-identical rows land in both train and test → fake 90% acc.
- An EMBARGO gap of `horizon` days is removed at every split boundary: the
  label for a row on the last train day looks `horizon` days into the future,
  i.e. straight into the test set. Without the gap the model is scored on
  windows it partly saw the answer to.
"""

from __future__ import annotations

import pandas as pd

# Scale-free features only — the model pools across ₹200 and ₹12,000 stocks.
FEATURE_COLS = [
    # trend
    "sma20_slope", "dist_ema20", "above_sma50", "above_sma200",
    "dist_sma20", "dist_sma50", "dist_sma200",
    # momentum
    "ret_1d", "ret_5d", "ret_20d", "ret_60d", "ret_120d", "mom_63_21",
    # oscillators
    "rsi_14", "rsi_2",
    # volatility
    "atr_pct", "vol_20", "vol_ratio",
    # price location
    "dist_high_252", "dist_low_252", "bb_pos",
    # volume
    "vol_z",
    # market-relative (index-derived, so MLStrategy can reproduce these
    # per-instrument at inference time — added in build_panel)
    "excess_ret_20d", "excess_ret_60d", "mkt_above_sma50",
]

# `ret_20d_rank` (cross-sectional percentile) is computed in build_panel for
# analysis but deliberately NOT in FEATURE_COLS: the live backtester calls a
# strategy one instrument at a time, so a same-day cross-section isn't
# available. Keeping it out means train and live see identical inputs.


def make_dataset(
    feat: pd.DataFrame,
    horizon: int = 10,
    threshold: float = 0.02,
) -> pd.DataFrame:
    """One instrument's feature frame → feature+label rows. Index = dates.

    Cross-sectional / market columns are NOT added here (they need the whole
    panel); build_panel fills them. Rows are returned with NaNs intact so the
    panel can compute ranks before the final dropna.
    """
    df = feat.copy()
    for w in (20, 50, 200):
        df[f"dist_sma{w}"] = df["close"] / df[f"sma_{w}"] - 1

    fwd_ret = df["close"].shift(-horizon) / df["close"] - 1
    df["y"] = (fwd_ret > threshold).astype("float64")
    df.loc[fwd_ret.isna(), "y"] = pd.NA  # label-peek rows at the tail

    keep = [c for c in FEATURE_COLS if c in df.columns] + ["y", "close"]
    return df[keep].iloc[:-horizon]


def build_panel(
    data: dict[str, pd.DataFrame],
    horizon: int = 10,
    threshold: float = 0.02,
    index_feat: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Stack all instruments into one panel with a `ticker` column.

    Adds market/cross-sectional features that need more than one series:
      excess_ret_20d/60d  stock return minus NIFTY return (relative strength)
      ret_20d_rank        cross-sectional percentile of 20d return that day
      mkt_above_sma50     1 if NIFTY closed above its 50d SMA (regime)
    """
    frames = []
    for t, feat in data.items():
        d = make_dataset(feat, horizon, threshold)
        d["ticker"] = t
        frames.append(d)
    panel = pd.concat(frames).sort_index()

    # --- market regime + relative strength from the index ---
    if index_feat is not None:
        idx_ret20 = index_feat["close"].pct_change(20)
        idx_ret60 = index_feat["close"].pct_change(60)
        mkt_regime = (index_feat["close"] > index_feat["sma_50"]).astype("float64")
        ts = panel.index
        panel["excess_ret_20d"] = panel["ret_20d"] - idx_ret20.reindex(ts).to_numpy()
        panel["excess_ret_60d"] = panel["ret_60d"] - idx_ret60.reindex(ts).to_numpy()
        panel["mkt_above_sma50"] = mkt_regime.reindex(ts).ffill().to_numpy()
    else:
        panel["excess_ret_20d"] = panel["ret_20d"]
        panel["excess_ret_60d"] = panel["ret_60d"]
        panel["mkt_above_sma50"] = 0.0

    # --- cross-sectional rank of 20d momentum, per day ---
    panel["ret_20d_rank"] = panel.groupby(level=0)["ret_20d"].rank(pct=True)

    feat_cols = [c for c in FEATURE_COLS if c in panel.columns]
    return panel.dropna(subset=feat_cols + ["y"])


def time_split(
    panel: pd.DataFrame,
    test_start: str,
    *,
    embargo_days: int = 10,
    test_end: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train strictly before (test_start - embargo); test in [test_start, test_end].

    The embargo drops the last `embargo_days` calendar days of would-be
    training data so no training label reaches into the test window.
    """
    cut = pd.Timestamp(test_start) - pd.Timedelta(days=embargo_days)
    train = panel[panel.index < cut]
    test = panel[panel.index >= test_start]
    if test_end is not None:
        test = test[test.index <= test_end]
    return train, test


def walk_forward_split(
    panel: pd.DataFrame,
    *,
    tune_end: str = "2021-12-31",
    validate_end: str = "2025-12-31",
    embargo_days: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a panel into (tune, validate, holdout) with embargo gaps.

    tune     ..            <= tune_end
    validate (tune_end + embargo) .. validate_end
    holdout  (validate_end + embargo) ..
    """
    emb = pd.Timedelta(days=embargo_days)
    t_end = pd.Timestamp(tune_end)
    v_end = pd.Timestamp(validate_end)
    tune = panel[panel.index <= t_end]
    validate = panel[(panel.index > t_end + emb) & (panel.index <= v_end)]
    holdout = panel[panel.index > v_end + emb]
    return tune, validate, holdout
