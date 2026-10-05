"""First rule-based strategy: trend-following with sanity filters.

LONG when ALL of:
  1. Trend:   close > SMA(trend_window)          — stock in an uptrend
  2. Entry:   RSI < rsi_max                      — not chasing an overbought spike
  3. Regime:  index close > index SMA(regime_window) — market itself is healthy

EXIT (go flat) when:
  close < SMA(exit_window)  OR  regime turns bearish

Entry and exit windows differ on purpose (hysteresis): entering and exiting
on the same line causes constant whipsaw trades that costs eat alive.
"""

from __future__ import annotations

import pandas as pd

from .base import Strategy, register


@register
class SmaRsiStrategy(Strategy):
    name = "sma_rsi"

    @classmethod
    def default_params(cls) -> dict:
        return {
            "trend_window": 50,     # entry trend line (must exist in features: sma_50)
            "exit_window": 20,      # faster exit line (sma_20)
            "rsi_max": 65,          # don't buy overbought
            "regime_window": 50,    # index health line
        }

    def generate_signals(
        self,
        feat: pd.DataFrame,
        index_feat: pd.DataFrame | None = None,
    ) -> pd.Series:
        p = self.params
        trend_col = f"sma_{p['trend_window']}"
        exit_col = f"sma_{p['exit_window']}"

        entry = (feat["close"] > feat[trend_col]) & (feat["rsi_14"] < p["rsi_max"])
        stay = feat["close"] > feat[exit_col]

        if index_feat is not None:
            regime_col = f"sma_{p['regime_window']}"
            regime_ok = (index_feat["close"] > index_feat[regime_col]).reindex(
                feat.index
            ).ffill().fillna(False)
            entry &= regime_ok
            stay &= regime_ok

        # state machine, vectorised: enter on `entry`, remain while `stay`
        # (a simple loop is clearer than clever tricks here, and 3000 rows is nothing)
        sig = pd.Series(0, index=feat.index, dtype=int)
        holding = False
        for i, (e, s) in enumerate(zip(entry.to_numpy(), stay.to_numpy())):
            if not holding and e:
                holding = True
            elif holding and not s:
                holding = False
            sig.iloc[i] = int(holding)

        # NaN warmup region (indicators not ready) → force flat
        warmup = feat[trend_col].isna() | feat["rsi_14"].isna()
        sig[warmup] = 0
        return sig
