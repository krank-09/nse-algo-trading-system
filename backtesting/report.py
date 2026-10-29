"""Persist a BacktestResult as a named report the dashboard can load.

    reports/<name>/equity.csv    date, equity, benchmark (NIFTY rescaled to the same capital)
    reports/<name>/trades.csv    one row per closed round trip
    reports/<name>/meta.json     label, description, metrics, created time
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .engine import BacktestResult

REPORTS_DIR = Path("reports")


def _jsonable(v):
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if hasattr(v, "item"):  # numpy scalar
        return _jsonable(v.item())
    return v


def save_report(
    result: BacktestResult,
    name: str,
    *,
    label: str,
    description: str = "",
    reports_dir: str | Path = REPORTS_DIR,
) -> Path:
    out = Path(reports_dir) / name
    out.mkdir(parents=True, exist_ok=True)

    eq = pd.DataFrame({"equity": result.equity})
    if result.benchmark is not None:
        eq["benchmark"] = result.benchmark
    eq.to_csv(out / "equity.csv", index_label="date")
    result.trades.to_csv(out / "trades.csv", index=False)

    meta = {
        "label": label,
        "description": description,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "metrics": {k: _jsonable(v) for k, v in result.metrics.items()},
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return out
