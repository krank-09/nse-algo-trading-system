import numpy as np
import pandas as pd
import pytest

from backtesting.metrics import cagr, max_drawdown, sharpe, summarize, trade_stats


def _series(values, start="2020-01-01", freq="D"):
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq=freq))


def test_cagr_doubling_over_two_years():
    eq = pd.Series([100.0, 200.0], index=pd.to_datetime(["2020-01-01", "2022-01-01"]))
    assert cagr(eq) == pytest.approx(2 ** (365.25 / 731) - 1, rel=1e-6)


def test_max_drawdown_peak_to_trough():
    assert max_drawdown(_series([100, 120, 90, 110, 130])) == pytest.approx(-0.25)


def test_sharpe_of_flat_curve_is_zero():
    assert sharpe(_series([100.0] * 10)) == 0.0


def test_trade_stats():
    trades = pd.DataFrame({"pnl": [100.0, -50.0, 200.0, -50.0]})
    s = trade_stats(trades)
    assert s["n_trades"] == 4
    assert s["win_rate"] == 0.5
    assert s["profit_factor"] == pytest.approx(3.0)


def test_trade_stats_empty():
    assert trade_stats(pd.DataFrame())["n_trades"] == 0


def test_benchmark_starting_a_day_late_is_not_nan():
    # regression: stock calendar began one day before the index → whole benchmark NaN
    eq = _series([100.0, 101.0, 102.0, 103.0], freq="B")
    bench = eq.iloc[1:] * 2
    out = summarize(eq, pd.DataFrame({"pnl": [1.0]}), bench)
    assert not np.isnan(out["benchmark_return_pct"])
