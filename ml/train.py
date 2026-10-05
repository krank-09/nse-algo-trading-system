"""Train + evaluate an ML model on the feature files.

Usage:
    python -m ml.train --test-start 2024-01-01 --model logistic
Saves model to ml/artifacts/model.pkl and prints precision@confidence table.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import dataset, models


def read(f: Path) -> pd.DataFrame:
    return (
        pd.read_parquet(f)
        if f.suffix == ".parquet"
        else pd.read_csv(f, index_col="date", parse_dates=True)
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--features-dir", default="data/features")
    p.add_argument("--model", default="logistic", choices=list(models.MODELS))
    p.add_argument("--test-start", required=True, help="e.g. 2024-01-01")
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--threshold", type=float, default=0.02)
    p.add_argument("--out", default="ml/artifacts/model.pkl")
    args = p.parse_args()

    feat_dir = Path(args.features_dir)
    files = list(feat_dir.glob("*.parquet")) + list(feat_dir.glob("*.csv"))
    data = {f.stem: read(f) for f in files if not f.stem.startswith("IDX_")}
    if not data:
        raise SystemExit(f"no feature files in {feat_dir}")

    panel = dataset.build_panel(data, args.horizon, args.threshold)
    train_p, test_p = dataset.time_split(panel, args.test_start)
    print(f"train rows: {len(train_p)}  test rows: {len(test_p)}")

    model = models.train(train_p, args.model)
    report = models.evaluate(model, test_p)
    for k, v in report.items():
        print(f"{k:>20}: {v:.3f}" if isinstance(v, float) else f"{k:>20}: {v}")
    print("\nRead: precision@0.6 should beat base_rate meaningfully, else no edge.")

    models.save(model, args.out)
    print(f"saved → {args.out}")


if __name__ == "__main__":
    main()
