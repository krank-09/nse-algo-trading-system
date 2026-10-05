"""CostModel against hand-computed charges on a ₹1,00,000 delivery trade."""

import pytest

from backtesting.costs import CostModel

VALUE = 100_000.0
# brokerage = min(20, 0.1% * 1e5) = 20
# exch + sebi = 1e5 * (0.0000297 + 0.000001) = 3.07
# GST = 18% * (20 + 3.07) = 4.1526
BUY = 20 + VALUE * (0.001 + 0.0000297 + 0.000001 + 0.00015) + 4.1526   # + STT + stamp
SELL = 20 + VALUE * (0.001 + 0.0000297 + 0.000001) + 4.1526 + 16       # + STT + DP


def test_buy_costs_match_hand_calculation():
    assert CostModel().buy_costs(VALUE) == pytest.approx(BUY)


def test_sell_costs_match_hand_calculation():
    assert CostModel().sell_costs(VALUE) == pytest.approx(SELL)


def test_brokerage_is_capped_at_20():
    small, big = CostModel().buy_costs(10_000), CostModel().buy_costs(10_000_000)
    # 0.1% of ₹10k = ₹10 < cap; for ₹1cr the cap applies, so brokerage share shrinks
    assert small > 0 and big > 0
    m = CostModel(stt_pct=0, exchange_pct=0, sebi_pct=0, stamp_pct=0, gst_pct=0)
    assert m.buy_costs(10_000) == pytest.approx(10.0)
    assert m.buy_costs(10_000_000) == pytest.approx(20.0)


def test_round_trip_is_roughly_a_quarter_percent():
    # README claims ~0.285% round trip incl. 5bps slippage each way (0.1%)
    assert 0.2 < CostModel().round_trip_pct(VALUE) < 0.3
