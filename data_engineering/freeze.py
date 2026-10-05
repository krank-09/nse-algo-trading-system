"""Freeze an immutable walk-forward train/validate/holdout dataset.

Why: ML experiments must be reproducible. The live feature files under
`data/features/` change every time `fetch-data` runs (end date moves, vendor
back-adjustments shift history). A frozen snapshot + a checksummed manifest
pins the exact bytes every training run sees.

Walk-forward windows (from docs/IMPLEMENTATION_PLAN.md Phase C):
    tune     2015-01-01 .. 2021-12-31   parameter/model search
    validate 2022-01-01 .. 2025-12-31   out-of-sample check, tune ONCE
    holdout  2026-01-01 .. open          never looked at until the very end

Usage:
    python -m data_engineering.freeze                 # freeze data/features/
    python -m data_engineering.freeze --verify        # checksums unchanged?
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

FROZEN_DIR = Path("data/frozen")
MANIFEST = FROZEN_DIR / "MANIFEST.json"

WALK_FORWARD: dict[str, tuple[str, str | None]] = {
    "tune": ("2015-01-01", "2021-12-31"),
    "validate": ("2022-01-01", "2025-12-31"),
    "holdout": ("2026-01-01", None),
    # sub-windows for iteration (not part of the canonical 3-way split):
    "tune_train": ("2015-01-01", "2019-12-31"),   # fit here while iterating
    "tune_eval": ("2020-01-01", "2021-12-31"),    # judge ideas here
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, index_col="date", parse_dates=True)


def freeze(
    features_dir: str | Path = "data/features",
    *,
    out_dir: str | Path = FROZEN_DIR,
    source: str = "yfinance(auto_adjust)",
) -> dict:
    """Copy every feature file into out_dir and write MANIFEST.json.

    Refuses to silently overwrite: an existing frozen set must be removed by
    hand (or `--force`) so a snapshot is never clobbered by accident.
    """
    src = Path(features_dir)
    out = Path(out_dir)
    files = sorted(list(src.glob("*.parquet")) + list(src.glob("*.csv")))
    if not files:
        raise SystemExit(f"no feature files in {src}")

    out.mkdir(parents=True, exist_ok=True)
    entries = {}
    for f in files:
        dst = out / f.name
        shutil.copy2(f, dst)
        df = _read(dst)
        entries[f.stem] = {
            "file": f.name,
            "sha256": _sha256(dst),
            "rows": int(len(df)),
            "first_date": str(df.index.min().date()),
            "last_date": str(df.index.max().date()),
        }

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": source,
        "walk_forward": {k: list(v) for k, v in WALK_FORWARD.items()},
        "n_files": len(entries),
        "files": entries,
    }
    MANIFEST_path = out / "MANIFEST.json"
    MANIFEST_path.write_text(json.dumps(manifest, indent=2))
    log.info("froze %d files → %s", len(entries), out)
    return manifest


def verify(out_dir: str | Path = FROZEN_DIR) -> bool:
    """Recompute every checksum and compare to the manifest. True == intact."""
    out = Path(out_dir)
    manifest = json.loads((out / "MANIFEST.json").read_text())
    ok = True
    for stem, e in manifest["files"].items():
        path = out / e["file"]
        if not path.exists():
            log.error("MISSING %s", e["file"])
            ok = False
            continue
        actual = _sha256(path)
        if actual != e["sha256"]:
            log.error("CHANGED %s (%s != %s)", e["file"], actual[:12], e["sha256"][:12])
            ok = False
    log.info("verify %s: %s", out, "OK" if ok else "FAILED")
    return ok


def load_frozen(
    split: str,
    *,
    out_dir: str | Path = FROZEN_DIR,
    include_index: bool = False,
) -> dict[str, pd.DataFrame]:
    """Load the frozen feature files, each clipped to `split`'s date window.

    split: one of WALK_FORWARD keys ('tune' | 'validate' | 'holdout'), or 'all'.
    """
    out = Path(out_dir)
    if not (out / "MANIFEST.json").exists():
        raise SystemExit(f"no frozen set at {out}; run `python -m data_engineering.freeze`")

    if split == "all":
        lo, hi = None, None
    else:
        lo, hi = WALK_FORWARD[split]

    data: dict[str, pd.DataFrame] = {}
    for f in sorted(list(out.glob("*.parquet")) + list(out.glob("*.csv"))):
        is_index = f.stem.startswith("IDX_")
        if is_index and not include_index:
            continue
        df = _read(f)
        if lo is not None:
            df = df[df.index >= lo]
        if hi is not None:
            df = df[df.index <= hi]
        if len(df):
            data[f.stem] = df
    return data


def _cli() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features-dir", default="data/features")
    p.add_argument("--out", default=str(FROZEN_DIR))
    p.add_argument("--verify", action="store_true")
    p.add_argument("--force", action="store_true", help="overwrite an existing frozen set")
    args = p.parse_args()

    if args.verify:
        raise SystemExit(0 if verify(args.out) else 1)

    if Path(args.out, "MANIFEST.json").exists() and not args.force:
        raise SystemExit(f"{args.out}/ already frozen — pass --force to replace it")
    m = freeze(args.features_dir, out_dir=args.out)
    for split, (lo, hi) in WALK_FORWARD.items():
        print(f"  {split:9s} {lo} .. {hi or 'open'}")
    print(f"  {m['n_files']} files, source={m['source']}")


if __name__ == "__main__":
    _cli()
