"""Backtester execution model: signal at t → fill at t+1 open, never t's close."""

import numpy as np
import pandas as pd
import pytest

from backtesting import Backtester, CostModel
from strategies.base import Strategy

from .conftest import make_ohlcv

FREE = CostModel(brokerage_pct=0, stt_pct=0, exchange_pct=0, sebi_pct=0,
                 stamp_pct=0, gst_pct=0, dp_charge=0)


class FixedSignal(Strategy):
    """Long between `enter` and `exit` (index positions); flat otherwise."""

    name = "fixed"

    def generate_signals(self, feat, index_feat=None):
        sig = pd.Series(0, index=feat.index)
        sig.iloc[self.params["enter"]:self.params["exit"]] = 1
        return sig


def _run(df, enter=10, exit=20, slippage_bps=0.0, costs=FREE, index_feat=None):
    bt = Backtester(FixedSignal(enter=enter, exit=exit), cost_model=costs,
                    initial_capital=100_000, max_positions=1, slippage_bps=slippage_bps)
    return bt.run({"AAA": df}, index_feat)


def test_entry_fills_at_next_day_open():
    df = make_ohlcv(60)
    res = _run(df, enter=10, exit=20)
    t = res.trades.iloc[0]
    assert t["entry_date"] == df.index[11]
    assert t["entry_price"] == pytest.approx(df["open"].iloc[11])
    assert t["entry_price"] != pytest.approx(df["close"].iloc[10])


def test_exit_fills_at_next_day_open():
    df = make_ohlcv(60)
    t = _run(df, enter=10, exit=20).trades.iloc[0]
    assert t["exit_date"] == df.index[21]
    assert t["exit_price"] == pytest.approx(df["open"].iloc[21])


def test_slippage_applied_against_us():
    df = make_ohlcv(60)
    t = _run(df, slippage_bps=10).trades.iloc[0]
    assert t["entry_price"] == pytest.approx(df["open"].iloc[11] * 1.001)
    assert t["exit_price"] == pytest.approx(df["open"].iloc[21] * 0.999)


def test_no_lookahead_future_prices_do_not_change_past_equity():
    df = make_ohlcv(60)
    shocked = df.copy()
    shocked.iloc[40:, :4] *= 3          # rewrite the future
    a, b = _run(df).equity, _run(shocked).equity
    pd.testing.assert_series_equal(a.iloc[:40], b.iloc[:40])


def test_costs_reduce_pnl():
    df = make_ohlcv(60)
    free = _run(df).trades.iloc[0]["pnl"]
    paid = _run(df, costs=CostModel()).trades.iloc[0]["pnl"]
    assert paid < free


def test_flat_strategy_keeps_cash():
    df = make_ohlcv(60)
    res = _run(df, enter=0, exit=0)
    assert res.trades.empty
    assert np.allclose(res.equity, 100_000)


def test_benchmark_not_nan_when_index_starts_late():
    df = make_ohlcv(60)
    idx = df.iloc[1:].copy()
    res = _run(df, index_feat=idx)
    assert res.benchmark.notna().all()
    assert res.benchmark.iloc[0] == pytest.approx(100_000)
