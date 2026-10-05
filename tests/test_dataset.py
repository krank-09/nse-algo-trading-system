"""ML dataset: labels look forward, splits never overlap, embargo is respected."""

import pandas as pd

from data_engineering.features import build_features
from ml import dataset

from .conftest import make_ohlcv


def test_make_dataset_label_and_tail():
    feat = build_features(make_ohlcv(400))
    h, thr = 10, 0.02
    d = dataset.make_dataset(feat, horizon=h, threshold=thr)
    assert len(d) == len(feat) - h                      # tail rows that would peek are dropped
    t = d.index[300]
    fwd = feat["close"].shift(-h).loc[t] / feat["close"].loc[t] - 1
    assert d.loc[t, "y"] == float(fwd > thr)


def _panel():
    idx = pd.bdate_range("2015-01-01", "2026-06-30")
    return pd.DataFrame({"x": range(len(idx))}, index=idx)


def test_walk_forward_split_is_disjoint_and_ordered():
    tune, val, hold = dataset.walk_forward_split(_panel(), embargo_days=10)
    assert tune.index.max() < val.index.min() < val.index.max() < hold.index.min()
    assert (val.index.min() - tune.index.max()).days > 10
    assert (hold.index.min() - val.index.max()).days > 10


def test_time_split_embargo():
    train, test = dataset.time_split(_panel(), "2020-01-01", embargo_days=10)
    assert train.index.max() < pd.Timestamp("2020-01-01") - pd.Timedelta(days=10)
    assert test.index.min() >= pd.Timestamp("2020-01-01")
