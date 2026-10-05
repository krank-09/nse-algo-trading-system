"""Validate and clean raw OHLCV CSVs → processed parquet files.

Cleaning rules (deliberately conservative — never invent prices):
- drop exact duplicate dates
- drop rows where OHLC relationships are impossible (high < low, etc.)
- drop rows with non-positive prices
- volume NaN → 0 (indices have no volume); price NaN rows are dropped
- sort by date ascending
A per-ticker quality report is returned so bad data is visible, not silent.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


def clean_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    report = {"rows_in": len(df)}

    df = df[~df.index.duplicated(keep="first")].sort_index()
    report["dup_dates_dropped"] = report["rows_in"] - len(df)

    price_cols = ["open", "high", "low", "close"]
    before = len(df)
    df = df.dropna(subset=price_cols)
    report["nan_price_rows_dropped"] = before - len(df)

    df["volume"] = df["volume"].fillna(0)

    # impossible candles: high must be the max, low the min, prices positive
    valid = (
        (df["high"] >= df[price_cols].max(axis=1) - 1e-9)
        & (df["low"] <= df[price_cols].min(axis=1) + 1e-9)
        & (df[price_cols] > 0).all(axis=1)
    )
    report["invalid_candles_dropped"] = int((~valid).sum())
    df = df[valid]

    # gap detection (info only): trading-day gaps > 5 suggest suspension/missing data
    gaps = df.index.to_series().diff().dt.days
    report["max_gap_days"] = int(gaps.max()) if len(df) > 1 else 0
    report["rows_out"] = len(df)
    return df, report


def process_all(raw_dir: str | Path, processed_dir: str | Path) -> pd.DataFrame:
    """Clean every CSV in raw_dir, write parquet to processed_dir.

    Returns a quality-report DataFrame (one row per ticker).
    """
    raw, out = Path(raw_dir), Path(processed_dir)
    out.mkdir(parents=True, exist_ok=True)

    reports = []
    for csv in sorted(raw.glob("*.csv")):
        df = pd.read_csv(csv, index_col="date", parse_dates=True)
        cleaned, rep = clean_frame(df)
        rep["ticker"] = csv.stem
        reports.append(rep)
        try:
            cleaned.to_parquet(out / f"{csv.stem}.parquet")
        except ImportError:  # no pyarrow/fastparquet — fall back to CSV
            cleaned.to_csv(out / f"{csv.stem}.clean.csv")
        log.info("cleaned %s: %d → %d rows", csv.stem, rep["rows_in"], rep["rows_out"])

    report_df = pd.DataFrame(reports).set_index("ticker")
    report_df.to_csv(out / "_quality_report.csv")
    return report_df
