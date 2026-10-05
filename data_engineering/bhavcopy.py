"""NSE CM bhavcopy downloader — the reconciliation second source.

Why this exists: yfinance is the primary feed, but it is a single point of
failure (silent gaps, transient 404s on .NS symbols, opaque adjustment).
The reliability layer in docs/IMPLEMENTATION_PLAN.md wants a *second, independent*
source to cross-check every fresh candle against. NSE's own end-of-day
bhavcopy is that source — it is the exchange's official record.

Two archive formats, handled transparently:
  * UDiFF (current, ~July 2024 onward, but also serves many older dates):
      https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_YYYYMMDD_F_0000.csv.zip
  * Legacy (pre-UDiFF):
      https://archives.nseindia.com/content/historical/EQUITIES/YYYY/MMM/cmDDMMMYYYYbhav.csv.zip

NSE returns HTTP 403 to non-browser user agents, so every request carries a
browser UA and an nseindia.com Referer.

IMPORTANT — adjustment mismatch: bhavcopy prices are RAW (never adjusted for
splits/bonuses/dividends). yfinance data in this project is auto_adjust=True.
So a bhavcopy close only equals the yfinance close on dates with no corporate
action between that date and today (which is always true for very recent
dates). `reconcile()` is therefore meant for the *last few sessions*, exactly
the morning-data-integrity use case. Do not use it to validate deep history.
"""

from __future__ import annotations

import argparse
import io
import logging
import time
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger(__name__)

UDIFF_URL = (
    "https://nsearchives.nseindia.com/content/cm/"
    "BhavCopy_NSE_CM_0_0_0_{yyyymmdd}_F_0000.csv.zip"
)
LEGACY_URL = (
    "https://archives.nseindia.com/content/historical/EQUITIES/"
    "{year}/{mon}/cm{dd}{mon}{year}bhav.csv.zip"
)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/all-reports",
}

# Equity series worth keeping. EQ = rolling settlement equity, BE = trade-to-trade,
# BZ = trade-to-trade surveillance. SM/ST = SME. We keep the cash-equity ones.
EQUITY_SERIES = ("EQ", "BE", "BZ")

_OUT_COLS = ["open", "high", "low", "close", "volume"]


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(_HEADERS)
    # NSE sets cookies on the landing page; without them the archive host 403s.
    try:
        s.get("https://www.nseindia.com", timeout=10)
    except requests.RequestException:
        pass
    return s


def _urls_for(d: date) -> list[str]:
    mon = d.strftime("%b").upper()
    return [
        UDIFF_URL.format(yyyymmdd=d.strftime("%Y%m%d")),
        LEGACY_URL.format(year=d.strftime("%Y"), mon=mon, dd=d.strftime("%d")),
    ]


def download_bhavcopy(
    d: date,
    dest_dir: str | Path = "data/bhavcopy",
    *,
    overwrite: bool = False,
    session: requests.Session | None = None,
    retries: int = 3,
) -> Path | None:
    """Download one session's bhavcopy CSV. Returns the saved path, or None
    if NSE has no file for that date (weekend / holiday / not yet published)."""
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    out_path = dest / f"{d:%Y-%m-%d}.csv"
    if out_path.exists() and not overwrite:
        return out_path

    sess = session or _session()
    for url in _urls_for(d):
        for attempt in range(1, retries + 1):
            try:
                r = sess.get(url, timeout=30)
            except requests.RequestException as e:
                log.warning("‰ %s attempt %d: %s", url, attempt, e)
                time.sleep(1.5 * attempt)
                continue
            if r.status_code == 404:
                break  # this format has no file for this date; try next format
            if r.status_code == 403:
                time.sleep(1.5 * attempt)
                continue
            r.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
                name = zf.namelist()[0]
                raw = zf.read(name)
            out_path.write_bytes(raw)
            log.info("bhavcopy %s: %d bytes from %s", d, len(raw), url.split("/")[-1])
            return out_path
    log.info("no bhavcopy for %s (weekend/holiday/unpublished)", d)
    return None


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Map either archive schema to symbol-indexed open/high/low/close/volume."""
    cols = set(df.columns)
    if {"TckrSymb", "ClsPric"} <= cols:  # UDiFF
        df = df[df["FinInstrmTp"].isin(("STK", "EQ"))]
        df = df[df["SctySrs"].isin(EQUITY_SERIES)]
        out = df.rename(
            columns={
                "TckrSymb": "symbol", "OpnPric": "open", "HghPric": "high",
                "LwPric": "low", "ClsPric": "close", "TtlTradgVol": "volume",
            }
        )
    elif {"SYMBOL", "CLOSE"} <= cols:  # legacy
        df = df[df["SERIES"].str.strip().isin(EQUITY_SERIES)]
        out = df.rename(
            columns={
                "SYMBOL": "symbol", "OPEN": "open", "HIGH": "high", "LOW": "low",
                "CLOSE": "close", "TOTTRDQTY": "volume",
            }
        )
    else:
        raise ValueError(f"unrecognised bhavcopy schema, columns={sorted(cols)[:10]}")

    out = out[["symbol", *_OUT_COLS]].copy()
    out["symbol"] = out["symbol"].str.strip()
    for c in _OUT_COLS:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.dropna(subset=["close"]).set_index("symbol").sort_index()


def load_bhavcopy(
    source: str | Path | date,
    dest_dir: str | Path = "data/bhavcopy",
) -> pd.DataFrame:
    """Load a bhavcopy as a symbol-indexed OHLCV frame. `source` may be a
    path to a saved CSV, or a date (downloaded/cached on demand)."""
    if isinstance(source, (str, Path)) and Path(source).exists():
        path = Path(source)
    else:
        d = source if isinstance(source, date) else _parse_date(str(source))
        path = download_bhavcopy(d, dest_dir)
        if path is None:
            raise FileNotFoundError(f"no bhavcopy available for {source}")
    # UDiFF is comma-sep with a clean header; legacy has a trailing empty col.
    return _normalise(pd.read_csv(path))


def download_range(
    start: str | date,
    end: str | date,
    dest_dir: str | Path = "data/bhavcopy",
    *,
    pause: float = 0.4,
) -> list[Path]:
    """Download every weekday session between start and end (inclusive).
    Weekends are skipped up front; holidays fall out as 404 → None."""
    s = _parse_date(str(start)) if not isinstance(start, date) else start
    e = _parse_date(str(end)) if not isinstance(end, date) else end
    sess = _session()
    saved: list[Path] = []
    d = s
    while d <= e:
        if d.weekday() < 5:  # Mon–Fri
            p = download_bhavcopy(d, dest_dir, session=sess)
            if p:
                saved.append(p)
            time.sleep(pause)
        d += timedelta(days=1)
    log.info("downloaded %d bhavcopy files into %s", len(saved), dest_dir)
    return saved


def _nse_symbol(file_stem: str, symbol_map: dict[str, str] | None) -> str:
    """Undo loader.py's filename sanitising (`&`→`_`) to get the NSE ticker."""
    if symbol_map and file_stem in symbol_map:
        return symbol_map[file_stem]
    return file_stem.replace("_", "&") if "_" in file_stem else file_stem


def reconcile(
    d: str | date,
    yf_raw_dir: str | Path = "data/raw",
    *,
    tickers: list[str] | None = None,
    tol_pct: float = 1.0,
    dest_dir: str | Path = "data/bhavcopy",
    symbol_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Cross-check yfinance close vs NSE bhavcopy close for one session.

    Only meaningful for recent dates (see module docstring re: adjustment).
    Returns one row per ticker with both closes, the % difference, and a
    status of OK / MISMATCH / MISSING_BHAV / MISSING_YF.

    symbol_map: optional {file_stem: nse_symbol} to resolve sanitised names
    like `M_M` → `M&M`; if omitted a best-effort `_`→`&` guess is used.
    """
    day = _parse_date(str(d)) if not isinstance(d, date) else d
    bhav = load_bhavcopy(day, dest_dir)
    raw = Path(yf_raw_dir)

    files = sorted(raw.glob("*.csv"))
    rows = []
    for f in files:
        sym = f.stem
        if sym.startswith(("IDX_", "TEST")):
            continue
        if tickers and sym not in tickers:
            continue
        nse_sym = _nse_symbol(sym, symbol_map)
        yf_df = pd.read_csv(f, index_col="date", parse_dates=True)
        ts = pd.Timestamp(day)
        yf_close = float(yf_df.at[ts, "close"]) if ts in yf_df.index else None
        bhav_close = float(bhav.at[nse_sym, "close"]) if nse_sym in bhav.index else None

        if yf_close is None:
            status, diff = "MISSING_YF", None
        elif bhav_close is None:
            status, diff = "MISSING_BHAV", None
        else:
            diff = (yf_close - bhav_close) / bhav_close * 100
            status = "OK" if abs(diff) <= tol_pct else "MISMATCH"
        rows.append(
            {"symbol": sym, "yf_close": yf_close, "bhav_close": bhav_close,
             "diff_pct": diff, "status": status}
        )

    out = pd.DataFrame(rows).set_index("symbol").sort_values(
        "diff_pct", key=lambda s: s.abs(), ascending=False, na_position="last"
    )
    n_bad = (out["status"] == "MISMATCH").sum()
    log.info("reconcile %s: %d tickers, %d mismatch (>%.2f%%)",
             day, len(out), n_bad, tol_pct)
    return out


def backfill_from_bhavcopy(
    dates: list[str | date],
    yf_raw_dir: str | Path = "data/raw",
    *,
    symbol_map: dict[str, str] | None = None,
    dest_dir: str | Path = "data/bhavcopy",
) -> pd.DataFrame:
    """Patch missing recent rows in the raw yfinance CSVs using bhavcopy.

    Safe ONLY for very recent dates (no corporate action since → raw ≈ adjusted).
    Rows that already exist are left untouched. Returns a summary of what was
    inserted, per ticker.
    """
    raw = Path(yf_raw_dir)
    days = [(_parse_date(str(x)) if not isinstance(x, date) else x) for x in dates]
    bhavs = {}
    for dd in days:
        try:
            bhavs[dd] = load_bhavcopy(dd, dest_dir)
        except FileNotFoundError:
            log.warning("backfill: no bhavcopy for %s, skipping", dd)

    summary = []
    for f in sorted(raw.glob("*.csv")):
        if f.stem.startswith(("IDX_", "TEST")):
            continue
        nse_sym = _nse_symbol(f.stem, symbol_map)
        df = pd.read_csv(f, index_col="date", parse_dates=True)
        added = []
        for dd, bhav in bhavs.items():
            ts = pd.Timestamp(dd)
            if ts in df.index or nse_sym not in bhav.index:
                continue
            r = bhav.loc[nse_sym]
            df.loc[ts] = {c: r[c] for c in _OUT_COLS}
            added.append(str(dd))
        if added:
            df.sort_index().to_csv(f)
            summary.append({"ticker": f.stem, "dates_added": ",".join(added)})
            log.info("backfilled %s: %s", f.stem, added)

    return pd.DataFrame(summary)


def _load_symbol_map(config_path: str | Path = "config.yaml") -> dict[str, str]:
    """{sanitised_stem: nse_symbol} for the configured universe."""
    import yaml

    try:
        with open(config_path) as fh:
            cfg = yaml.safe_load(fh)
    except FileNotFoundError:
        return {}
    out = {}
    for t in cfg.get("universe", []):
        stem = t.replace("^", "IDX_").replace("&", "_")
        out[stem] = t
    return out


def _parse_date(s: str) -> date:
    for fmt in ("%Y-%m-%d", "%Y%m%d", "%d-%m-%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"unparseable date: {s!r}")


def _cli() -> None:
    p = argparse.ArgumentParser(description="NSE bhavcopy downloader / reconciler")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("get", help="download one date or a range")
    g.add_argument("start")
    g.add_argument("end", nargs="?", help="omit for a single date")
    g.add_argument("--dest", default="data/bhavcopy")

    r = sub.add_parser("reconcile", help="compare yfinance vs bhavcopy for a date")
    r.add_argument("date")
    r.add_argument("--raw-dir", default="data/raw")
    r.add_argument("--tol", type=float, default=1.0)

    b = sub.add_parser("backfill", help="fill missing recent yfinance rows from bhavcopy")
    b.add_argument("dates", nargs="+", help="one or more YYYY-MM-DD")
    b.add_argument("--raw-dir", default="data/raw")

    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.cmd == "get":
        if args.end:
            download_range(args.start, args.end, args.dest)
        else:
            path = download_bhavcopy(_parse_date(args.start), args.dest)
            print(path or "no file for that date")
    elif args.cmd == "reconcile":
        out = reconcile(
            args.date, args.raw_dir, tol_pct=args.tol, symbol_map=_load_symbol_map()
        )
        print(out.to_string())
        bad = out[out["status"].isin(("MISMATCH", "MISSING_BHAV", "MISSING_YF"))]
        if len(bad):
            print(f"\n⚠ {len(bad)} tickers need attention "
                  f"(small negative diff_pct on a recent day usually = ex-dividend)")
    elif args.cmd == "backfill":
        out = backfill_from_bhavcopy(
            args.dates, args.raw_dir, symbol_map=_load_symbol_map()
        )
        print(out.to_string(index=False) if len(out) else "nothing to backfill")


if __name__ == "__main__":
    _cli()
