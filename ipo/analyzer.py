"""IPO scorecard — scores watchlist entries 0–10 and prints a recommendation.

This is decision support, NOT automation: SEBI/broker rules require you to
approve every IPO application manually via UPI/ASBA. And there's no price
history to backtest, so unlike the trading strategies, these weights are
judgment, not fitted parameters. Adjust them as you learn.

Scoring logic (max 10):
  QIB subscription       0–3   institutions do the most homework
  Retail+NII demand      0–1.5
  GMP vs issue price     0–1.5  sentiment only, capped low on purpose
  Valuation vs peers     0–2    cheaper than peers = room to rerate
  Issue structure        0–1    fresh capital > pure promoter exit
  Promoter skin post     0–0.5
  Anchor book quality    0–0.5
"""

from __future__ import annotations

from pathlib import Path

import yaml


def score_ipo(ipo: dict) -> tuple[float, list[str]]:
    s = 0.0
    notes: list[str] = []
    sub = ipo.get("subscription", {})

    # QIB (0-3)
    qib = sub.get("qib", 0)
    s += min(3.0, qib)
    if qib < 1:
        notes.append("⚠ QIB under-subscribed — institutions are passing")

    # retail + NII (0-1.5)
    s += min(1.5, (sub.get("retail", 0) + sub.get("nii", 0)) / 2 * 0.75)

    # GMP (0-1.5), relative to price
    band = ipo.get("price_band_high") or 0
    gmp_pct = (ipo.get("gmp", 0) / band * 100) if band else 0
    s += min(1.5, gmp_pct / 20 * 1.5)
    if gmp_pct > 60:
        notes.append("⚠ GMP >60% — froth risk, listing pop may be priced in")

    # valuation vs peers (0-2)
    pe, peers = ipo.get("pe_ipo"), ipo.get("pe_peers")
    if pe and peers:
        disc = (peers - pe) / peers  # + = cheaper than peers
        s += max(0.0, min(2.0, 1.0 + disc * 2.5))
        if disc < -0.3:
            notes.append("⚠ priced >30% richer than listed peers")

    # structure (0-1)
    fresh = ipo.get("fresh_issue_pct", 0)
    s += min(1.0, fresh / 100)
    if fresh < 25:
        notes.append("⚠ mostly OFS — proceeds go to sellers, not the company")

    # promoter skin (0-0.5)
    if ipo.get("promoter_holding_post", 0) >= 50:
        s += 0.5

    # anchors (0-0.5)
    s += {"good": 0.5, "average": 0.25}.get(ipo.get("anchor_quality", ""), 0.0)

    return round(min(10.0, s), 1), notes


def verdict(score: float) -> str:
    if score >= 7:
        return "APPLY — strong overall picture"
    if score >= 5:
        return "CONSIDER — decent, size the application small"
    return "SKIP — weak demand/valuation mix"


def main(path: str | Path = None) -> None:
    path = Path(path or Path(__file__).parent / "watchlist.yaml")
    entries = yaml.safe_load(path.read_text())["ipos"]
    for ipo in entries:
        score, notes = score_ipo(ipo)
        print(f"\n{'='*60}\n{ipo['name']}  (closes {ipo.get('close_date','?')})")
        print(f"score: {score}/10   →  {verdict(score)}")
        for n in notes:
            print(f"  {n}")
    print(f"\n{'='*60}\nReminder: apply manually via UPI/ASBA — this cannot be automated.")


if __name__ == "__main__":
    main()
