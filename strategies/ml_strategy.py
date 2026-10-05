"""ML model as a Strategy — plugs into the same backtester/paper trader.

The model predicts P(forward return > threshold). We go long when confidence
clears `min_proba`, exit when it drops below `exit_proba` (hysteresis again).
The regime filter still applies — ML predicts stocks, not market crashes.

IMPORTANT: only backtest this on dates AFTER the model's training window.
Backtesting an ML strategy over its own training data is the classic way to
produce a beautiful, worthless equity curve.
"""

from __future__ import annotations

import pandas as pd

from ml.dataset import FEATURE_COLS

from .base import Strategy, register


@register
class MLStrategy(Strategy):
    name = "ml"

    def __init__(self, model=None, **params):
        super().__init__(**params)
        if model is None:
            from ml.models import load
            model = load(self.params["model_path"])
        self.model = model

    @classmethod
    def default_params(cls) -> dict:
        # min_proba / exit_proba locked on the 2020-2021 tuning surface with a
        # model trained on <=2019 only (see ml/RESULTS.md). max_positions=4 is
        # set on the Backtester/PaperTrader side.
        return {
            "model_path": "ml/artifacts/model.pkl",
            "min_proba": 0.50,      # confidence to enter
            "exit_proba": 0.45,     # confidence floor to stay in
            "regime_window": 50,
        }

    def generate_signals(
        self,
        feat: pd.DataFrame,
        index_feat: pd.DataFrame | None = None,
    ) -> pd.Series:
        df = feat.copy()
        for w in (20, 50, 200):
            df[f"dist_sma{w}"] = df["close"] / df[f"sma_{w}"] - 1

        # market-relative features — must match ml.dataset.build_panel exactly
        if index_feat is not None:
            idx_ret20 = index_feat["close"].pct_change(20).reindex(df.index).ffill()
            idx_ret60 = index_feat["close"].pct_change(60).reindex(df.index).ffill()
            mkt_regime = (
                (index_feat["close"] > index_feat["sma_50"]).astype("float64")
                .reindex(df.index).ffill().fillna(0.0)
            )
            df["excess_ret_20d"] = df["ret_20d"] - idx_ret20
            df["excess_ret_60d"] = df["ret_60d"] - idx_ret60
            df["mkt_above_sma50"] = mkt_regime
        else:
            df["excess_ret_20d"] = df["ret_20d"]
            df["excess_ret_60d"] = df["ret_60d"]
            df["mkt_above_sma50"] = 0.0

        valid = df[FEATURE_COLS].notna().all(axis=1)
        proba = pd.Series(0.0, index=df.index)
        if valid.any():
            proba[valid] = self.model.predict_proba(df.loc[valid, FEATURE_COLS])[:, 1]

        entry = proba >= self.params["min_proba"]
        stay = proba >= self.params["exit_proba"]

        if index_feat is not None:
            regime_col = f"sma_{self.params['regime_window']}"
            regime_ok = (index_feat["close"] > index_feat[regime_col]).reindex(
                df.index
            ).ffill().fillna(False)
            entry &= regime_ok
            stay &= regime_ok

        sig = pd.Series(0, index=df.index, dtype=int)
        holding = False
        for i, (e, s) in enumerate(zip(entry.to_numpy(), stay.to_numpy())):
            if not holding and e:
                holding = True
            elif holding and not s:
                holding = False
            sig.iloc[i] = int(holding)
        sig[~valid] = 0
        return sig
