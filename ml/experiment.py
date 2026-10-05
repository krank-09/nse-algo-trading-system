"""Walk-forward experiment harness for model/feature/label iteration.

Discipline this enforces:
  * Iterate on the TUNE window only (2015-2021). Inside it we sub-split:
        train  <= tune_train_end (default 2019-12-31)
        eval   2020-01-01 .. 2021-12-31        (with a 10d embargo gap)
    Every idea is judged here.
  * VALIDATE (2022-2025) is touched with `--final` ONLY, once an idea is
    locked. It is not a tuning surface.
  * HOLDOUT (2026+) is never read by this script at all.

Every run appends a row to ml/RESULTS.md so the iteration history is visible.

Usage:
    python -m ml.experiment --models logistic random_forest hist_gbm
    python -m ml.experiment --models hist_gbm --horizon 10 --threshold 0.03
    python -m ml.experiment --models hist_gbm --final     # score on validate
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from data_engineering import freeze

from . import dataset, models

RESULTS_MD = Path("ml/RESULTS.md")


def _load_panel(horizon: int, threshold: float) -> pd.DataFrame:
    data = freeze.load_frozen("all")
    idx = freeze.load_frozen("all", include_index=True)
    index_feat = idx.get("IDX_NSEI")
    return dataset.build_panel(data, horizon, threshold, index_feat=index_feat)


def _fmt(report: dict) -> str:
    keys = ["base_rate", "precision@0.6", "lift@0.6", "n_signals@0.6",
            "precision@0.7", "n_signals@0.7", "precision@top10pct", "lift@top10pct"]
    return "  ".join(
        f"{k.split('@')[-1] if '@' in k else k}="
        f"{report[k]:.3f}" if isinstance(report[k], float) else f"{k}={report[k]}"
        for k in keys
    )


def _append_results(rows: list[dict]) -> None:
    RESULTS_MD.parent.mkdir(parents=True, exist_ok=True)
    new = not RESULTS_MD.exists()
    with open(RESULTS_MD, "a") as f:
        if new:
            f.write("# ML iteration log\n\n")
            f.write("Auto-appended by `ml/experiment.py`. TUNE-internal eval unless "
                    "marked `[FINAL]` (scored on the 2022-2025 validate window).\n\n")
            f.write("| when | scope | model | horizon | thr | n_feat | base | "
                    "prec@0.6 | lift@0.6 | n@0.6 | prec@0.7 | prec@top10% | lift@top10% |\n")
            f.write("|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(
                f"| {r['when']} | {r['scope']} | {r['model']} | {r['horizon']} | "
                f"{r['threshold']} | {r['n_feat']} | {r['base_rate']:.3f} | "
                f"{r['precision@0.6']:.3f} | {r['lift@0.6']:.3f} | {r['n_signals@0.6']} | "
                f"{r['precision@0.7']:.3f} | {r['precision@top10pct']:.3f} | "
                f"{r['lift@top10pct']:.3f} |\n"
            )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=["logistic", "random_forest", "hist_gbm"],
                   choices=list(models.MODELS))
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--threshold", type=float, default=0.02)
    p.add_argument("--tune-train-end", default="2019-12-31")
    p.add_argument("--embargo", type=int, default=10)
    p.add_argument("--final", action="store_true",
                   help="score on the validate window (2022-2025) instead of tune-internal")
    p.add_argument("--recency-weight", action="store_true",
                   help="exponentially weight recent rows (tree models only)")
    p.add_argument("--save", help="pickle the last trained model here")
    args = p.parse_args()

    panel = _load_panel(args.horizon, args.threshold)
    n_feat = len(models._feat_cols(panel))

    if args.final:
        train_p, eval_p, _ = dataset.walk_forward_split(
            panel, tune_end="2021-12-31", validate_end="2025-12-31",
            embargo_days=args.embargo,
        )
        scope = "[FINAL] validate 22-25"
    else:
        train_p, eval_p = dataset.time_split(
            panel, "2020-01-01", embargo_days=args.embargo, test_end="2021-12-31",
        )
        scope = "tune-internal 20-21"

    sw = None
    if args.recency_weight:
        age_days = (train_p.index.max() - train_p.index).days.to_numpy()
        sw = 0.5 ** (age_days / 504)  # ~2-year half-life

    print(f"scope: {scope}   train={len(train_p):,}  eval={len(eval_p):,}  "
          f"n_feat={n_feat}  horizon={args.horizon}  thr={args.threshold}")
    when = datetime.now().strftime("%m-%d %H:%M")
    rows, last_model = [], None
    for name in args.models:
        model = models.train(train_p, name, sample_weight=sw)
        rep = models.evaluate(model, eval_p)
        print(f"  {name:14s} {_fmt(rep)}")
        rows.append({"when": when, "scope": scope.replace("[FINAL] ", ""),
                     "model": name, "horizon": args.horizon,
                     "threshold": args.threshold, "n_feat": n_feat, **rep})
        last_model = model

    _append_results(rows)
    print(f"\nlogged {len(rows)} row(s) → {RESULTS_MD}")
    if args.save and last_model is not None:
        models.save(last_model, args.save)
        print(f"saved {args.models[-1]} → {args.save}")


if __name__ == "__main__":
    main()
