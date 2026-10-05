"""ML dataset: labels look forward, splits never overlap, embargo is respected."""

import pandas as pd
import pytest

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
    tune, val, hold = dataset.walk_forward_split(_panel(), embargo_days=35)
    assert tune.index.max() < val.index.min() < val.index.max() < hold.index.min()
    # evaluation windows keep their first trading day; the gap is cut from training
    assert val.index.min() == pd.Timestamp("2022-01-03")
    assert hold.index.min() == pd.Timestamp("2026-01-01")


def _label_end(dates: pd.DatetimeIndex, calendar: pd.DatetimeIndex, h: int) -> pd.Timestamp:
    """Date the label of the LAST row in `dates` looks at (h trading days later)."""
    pos = calendar.get_loc(dates.max())
    return calendar[pos + h]


@pytest.mark.parametrize("h", [5, 10, 20, 40])
def test_no_training_label_reaches_eval_window(h):
    panel = _panel()
    emb = dataset.embargo_for(h)
    tune, val, _ = dataset.walk_forward_split(panel, embargo_days=emb)
    assert _label_end(tune.index, panel.index, h) < val.index.min()
    train, test = dataset.time_split(panel, "2020-01-01", embargo_days=emb)
    assert _label_end(train.index, panel.index, h) < test.index.min()


def test_old_10_day_embargo_leaked_for_20_day_labels():
    """Documents the bug the horizon-aware embargo fixed."""
    panel = _panel()
    tune, val, _ = dataset.walk_forward_split(panel, embargo_days=10)
    assert _label_end(tune.index, panel.index, 20) >= val.index.min()


def test_time_split_embargo():
    train, test = dataset.time_split(_panel(), "2020-01-01", embargo_days=10)
    assert train.index.max() < pd.Timestamp("2020-01-01") - pd.Timedelta(days=10)
    assert test.index.min() >= pd.Timestamp("2020-01-01")
