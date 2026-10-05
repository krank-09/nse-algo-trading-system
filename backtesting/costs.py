"""India-realistic transaction costs for NSE equity DELIVERY trades.

Rates (mid-2026; verify against your broker's contract note and update):
  brokerage      min(₹20, 0.1% of value)  — Groww-style delivery brokerage
  STT            0.1% of value, BOTH sides (delivery)
  exchange txn   0.00297% (NSE)
  SEBI charges   0.0001%
  stamp duty     0.015%, BUY side only
  GST            18% on (brokerage + exchange txn + SEBI)
  DP charge      ~₹16 flat per scrip per sell day

Why this matters: a "profitable" strategy doing many small trades often turns
negative once these are applied. Costs are a feature of the backtest, not a detail.
All rates are constructor params so you can recalibrate from a real contract note.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CostModel:
    brokerage_pct: float = 0.001        # 0.1%
    brokerage_cap: float = 20.0         # ₹20 max per order
    stt_pct: float = 0.001              # 0.1% both sides
    exchange_pct: float = 0.0000297     # NSE 0.00297%
    sebi_pct: float = 0.000001          # 0.0001%
    stamp_pct: float = 0.00015          # 0.015% buy only
    gst_pct: float = 0.18               # on brokerage + exchange + sebi
    dp_charge: float = 16.0             # per sell (per scrip per day)

    def _common(self, value: float) -> tuple[float, float]:
        brokerage = min(self.brokerage_cap, value * self.brokerage_pct)
        taxable = brokerage + value * (self.exchange_pct + self.sebi_pct)
        gst = taxable * self.gst_pct
        return brokerage, gst

    def buy_costs(self, value: float) -> float:
        """Total charges on a buy of `value` rupees (excluding the value itself)."""
        brokerage, gst = self._common(value)
        return (
            brokerage
            + value * (self.stt_pct + self.exchange_pct + self.sebi_pct + self.stamp_pct)
            + gst
        )

    def sell_costs(self, value: float) -> float:
        """Total charges on a sell of `value` rupees."""
        brokerage, gst = self._common(value)
        return (
            brokerage
            + value * (self.stt_pct + self.exchange_pct + self.sebi_pct)
            + gst
            + self.dp_charge
        )

    def round_trip_pct(self, value: float) -> float:
        """Approx round-trip cost as % of position value — sanity metric."""
        return (self.buy_costs(value) + self.sell_costs(value)) / value * 100
