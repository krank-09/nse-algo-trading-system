"""Features must be causal: values at t can't depend on data after t."""

import numpy as np
import pandas as pd

from data_engineering.features import build_features, rsi


def test_features_are_prefix_invariant(ohlcv):
    """Computing on data[:t] gives the same row t as computing on all data."""
    full = build_features(ohlcv)
    cut = 260
    partial = build_features(ohlcv.iloc[:cut])
    pd.testing.assert_frame_equal(partial, full.iloc[:cut])


def test_rsi_bounds(ohlcv):
    r = rsi(ohlcv["close"]).dropna()
    assert ((r >= 0) & (r <= 100)).all()


def test_rsi_all_gains_is_100():
    close = pd.Series(np.arange(1.0, 40.0))
    assert rsi(close).dropna().eq(100).all()
