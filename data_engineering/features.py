"""Technical indicators / feature engineering.

Every function takes a clean OHLCV frame and returns a Series aligned to it.
Rule that matters more than any single indicator: a feature at row t may use
data up to and including t, NEVER after t. All rolling ops here respect that.

RSI uses Wilder's smoothing (the standard) — a plain rolling mean gives
different values than every charting platform and will confuse debugging.

The feature set grew during ML iteration (see ml/RESULTS.md). The extra
columns are all scale-free (ratios, z-scores, bounded oscillators) so a
single model can pool across stocks trading at ₹200 and ₹12,000 alike.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(close: pd.Series, window: int) -> pd.Series:
    return close.rolling(window).mean()


def ema(close: pd.Series, span: int) -> pd.Series:
    return close.ewm(span=span, adjust=False).mean()


def returns(close: pd.Series, periods: int = 1) -> pd.Series:
    """Simple pct returns over `periods` days."""
    return close.pct_change(periods)


def volatility(close: pd.Series, window: int = 20) -> pd.Series:
    """Annualised rolling volatility of daily returns (252 trading days)."""
    return close.pct_change().rolling(window).std() * np.sqrt(252)


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index, Wilder's smoothing. Range 0–100."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    # Wilder smoothing == EMA with alpha = 1/window
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss
    out = 100 - 100 / (1 + rs)
    # all-gain edge case: avg_loss == 0 → RSI 100
    out = out.where(avg_loss != 0, 100.0)
    out[avg_gain.isna() | avg_loss.isna()] = np.nan
    return out


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Average True Range (Wilder). Needs high/low/close columns."""
    prev_close = df["close"].shift()
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Standard feature set used by strategies. Input: clean OHLCV frame."""
    out = df.copy()
    c, v = out["close"], out["volume"]

    # --- trend: moving averages + their slope + regime flags ---
    out["sma_20"] = sma(c, 20)
    out["sma_50"] = sma(c, 50)
    out["sma_200"] = sma(c, 200)
    out["ema_20"] = ema(c, 20)
    out["sma20_slope"] = out["sma_20"].pct_change(5)          # is the trend turning?
    out["dist_ema20"] = c / out["ema_20"] - 1
    out["above_sma50"] = (c > out["sma_50"]).astype("float64")
    out["above_sma200"] = (c > out["sma_200"]).astype("float64")

    # --- momentum: multi-horizon, plus 12-1 style (skip last month) ---
    out["ret_1d"] = returns(c, 1)
    out["ret_5d"] = returns(c, 5)
    out["ret_20d"] = returns(c, 20)
    out["ret_60d"] = returns(c, 60)
    out["ret_120d"] = returns(c, 120)
    out["mom_63_21"] = c.shift(21) / c.shift(63) - 1          # 3m momentum, lagged 1m

    # --- oscillators ---
    out["rsi_14"] = rsi(c, 14)
    out["rsi_2"] = rsi(c, 2)                                  # Connors short-term MR

    # --- volatility: level, scale-free, and regime (short vs medium) ---
    out["atr_14"] = atr(out, 14)
    out["atr_pct"] = out["atr_14"] / c
    out["vol_20"] = volatility(c, 20)
    vol_60 = volatility(c, 60)
    out["vol_ratio"] = out["vol_20"] / vol_60                 # >1 = vol expanding

    # --- price location ---
    roll_max = c.rolling(252).max()
    roll_min = c.rolling(252).min()
    out["dist_high_252"] = c / roll_max - 1                   # 0 = at 52w high, <0 below
    out["dist_low_252"] = c / roll_min - 1
    std20 = c.rolling(20).std()
    out["bb_pos"] = (c - out["sma_20"]) / (2 * std20)         # ±1 = Bollinger band edge

    # --- volume ---
    out["vol_z"] = (v - v.rolling(60).mean()) / v.rolling(60).std()

    return out
