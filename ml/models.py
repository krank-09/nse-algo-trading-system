"""Model wrappers: train, evaluate, persist.

Start simple on purpose (logistic regression, random forest). If these can't
find signal, a neural net won't either — it'll just overfit more convincingly.
Evaluation reports precision at high-confidence thresholds, not accuracy:
a trading model only acts on its most confident predictions, and base rates
make raw accuracy meaningless (predicting 'no move' is always ~70% accurate).
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from .dataset import FEATURE_COLS

MODELS = {
    "logistic": lambda: Pipeline(
        # RobustScaler + clip: return/vol features have fat tails that wreck a
        # plain StandardScaler's mean/variance and thus the logistic fit.
        [("scale", RobustScaler(quantile_range=(1, 99))),
         ("clip", _Clipper(5.0)),
         ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5))]
    ),
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=400, min_samples_leaf=50, max_features="sqrt",
        class_weight="balanced_subsample", n_jobs=-1, random_state=42,
    ),
    "hist_gbm": lambda: HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=400, max_leaf_nodes=31,
        min_samples_leaf=100, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.15, random_state=42,
    ),
}


class _SoftEnsemble:
    """Average calibrated-ish probabilities of a linear and a tree model.

    The two make uncorrelated errors — logistic captures smooth monotone
    effects, the GBM captures interactions/thresholds — so the mean is
    steadier across regimes than either alone.
    """

    def __init__(self, names=("logistic", "hist_gbm"), weights=None):
        self.names = list(names)
        self.weights = weights or [1.0] * len(self.names)
        self.members_ = []

    def fit(self, X, y, sample_weight=None):
        self.members_ = []
        for n in self.names:
            m = MODELS[n]()
            if sample_weight is not None and n != "logistic":
                m.fit(X, y, sample_weight=sample_weight)
            else:
                m.fit(X, y)
            self.members_.append(m)
        return self

    def predict_proba(self, X):
        w = np.asarray(self.weights) / np.sum(self.weights)
        p = sum(wi * m.predict_proba(X)[:, 1] for wi, m in zip(w, self.members_))
        return np.column_stack([1 - p, p])


MODELS["ensemble"] = _SoftEnsemble


class _Clipper:
    """Winsorise scaled features to ±z (sklearn-compatible, stateless)."""

    def __init__(self, z: float = 5.0):
        self.z = z

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype="float64"), -self.z, self.z)

    def get_params(self, deep=True):
        return {"z": self.z}

    def set_params(self, **p):
        self.z = p.get("z", self.z)
        return self


def _feat_cols(panel: pd.DataFrame) -> list[str]:
    return [c for c in FEATURE_COLS if c in panel.columns]


def train(train_panel: pd.DataFrame, model_name: str = "logistic", *, sample_weight=None):
    model = MODELS[model_name]()
    cols = _feat_cols(train_panel)
    X, y = train_panel[cols], train_panel["y"].astype(int)
    if sample_weight is not None and model_name != "logistic":
        model.fit(X, y, sample_weight=sample_weight)
    else:
        model.fit(X, y)
    return model


def evaluate(model, test_panel: pd.DataFrame) -> dict:
    cols = _feat_cols(test_panel)
    X, y = test_panel[cols], test_panel["y"].astype(int)
    proba = model.predict_proba(X)[:, 1]
    base = float(y.mean())
    out = {"n_test": int(len(y)), "base_rate": base}
    for thresh in (0.5, 0.6, 0.7):
        mask = proba >= thresh
        prec = float(y[mask].mean()) if mask.any() else float("nan")
        out[f"n_signals@{thresh}"] = int(mask.sum())
        out[f"precision@{thresh}"] = prec
        out[f"lift@{thresh}"] = prec / base if mask.any() else float("nan")
    # top-decile precision: the regime a trading model actually acts in
    k = max(1, int(0.10 * len(proba)))
    top = np.argsort(proba)[-k:]
    out["precision@top10pct"] = float(y.iloc[top].mean())
    out["lift@top10pct"] = out["precision@top10pct"] / base
    return out


def save(model, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load(path: str | Path):
    with open(path, "rb") as f:
        return pickle.load(f)
