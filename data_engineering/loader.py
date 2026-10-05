"""Download daily OHLCV data for the configured universe via yfinance.

Key choice: auto_adjust=True → prices are adjusted for splits and dividends.
This is what makes backtests honest — an unadjusted chart of a stock that
split 1:5 shows an 80% "crash" that never happened.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yaml
import yfinance as yf

log = logging.getLogger(__name__)

EXPECTED_COLS = ["open", "high", "low", "close", "volume"]


def load_config(path: str | Path = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def to_yf_symbol(ticker: str) -> str:
    """NSE equity tickers need the .NS suffix; index symbols (^NSEI) don't."""
    return ticker if ticker.startswith("^") else f"{ticker}.NS"


def fetch_one(ticker: str, start: str, end: str | None, interval: str = "1d") -> pd.DataFrame:
    """Fetch one instrument. Returns a tidy OHLCV frame indexed by date."""
    df = yf.download(
        to_yf_symbol(ticker),
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,   # split/dividend adjusted — critical for backtesting
        progress=False,
        multi_level_index=False,
    )
    if df.empty:
        raise ValueError(f"No data returned for {ticker}")

    df = df.rename(columns=str.lower)[EXPECTED_COLS]
    df.index.name = "date"
    return df


def fetch_universe(config: dict, out_dir: str | Path | None = None) -> dict[str, Path]:
    """Fetch every ticker in the universe + the index. Saves one CSV per ticker.

    Returns {ticker: saved_path}. Failures are logged and skipped, not fatal —
    one delisted/renamed ticker shouldn't kill the whole pipeline.
    """
    d = config["data"]
    out = Path(out_dir or d["raw_dir"])
    out.mkdir(parents=True, exist_ok=True)

    tickers = list(config["universe"]) + [config["index"]["symbol"]]
    saved: dict[str, Path] = {}
    failed: list[str] = []

    for t in tickers:
        try:
            df = fetch_one(t, d["start_date"], d.get("end_date"), d.get("interval", "1d"))
            path = out / f"{t.replace('^', 'IDX_').replace('&', '_')}.csv"
            df.to_csv(path)
            saved[t] = path
            log.info("saved %s: %d rows (%s → %s)", t, len(df), df.index[0].date(), df.index[-1].date())
        except Exception as e:  # noqa: BLE001 — log and continue
            failed.append(t)
            log.warning("FAILED %s: %s", t, e)

    if failed:
        log.warning("%d/%d tickers failed: %s", len(failed), len(tickers), failed)
    return saved
