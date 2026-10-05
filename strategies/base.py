"""Strategy contract.

A strategy is a pure function of features → signals. It does NOT know about
money, positions, or orders — that's the backtester/execution layer's job.
This separation is what lets the exact same strategy code run in backtest,
paper trading, and live trading without modification.

Signal semantics (per date):
    1  = want to be LONG this instrument
    0  = want to be FLAT (no position)
Signals are DESIRED STATE, not orders. The engine diffs consecutive signals
to produce orders (0→1 = buy at next open, 1→0 = sell at next open).
This makes strategies stateless and trivially testable.

Look-ahead rule: the signal for date t may only use feature values up to
and including t. Execution at t+1's open is the engine's responsibility.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Strategy(ABC):
    """Base class. Subclasses define `name`, `params`, and `generate_signals`."""

    name: str = "base"

    def __init__(self, **params):
        # keep params as a dict so walk-forward optimization can sweep them
        self.params = {**self.default_params(), **params}

    @classmethod
    def default_params(cls) -> dict:
        return {}

    @abstractmethod
    def generate_signals(
        self,
        feat: pd.DataFrame,
        index_feat: pd.DataFrame | None = None,
    ) -> pd.Series:
        """Return a Series of {0, 1} aligned to feat.index.

        feat:       feature matrix for ONE instrument (from build_features)
        index_feat: feature matrix for the market index (regime filter), optional
        """

    def __repr__(self) -> str:
        return f"{self.name}({self.params})"


# ---------------------------------------------------------------------------
# registry: maps CLI names → classes.  strategies/__init__.py populates this.
# ---------------------------------------------------------------------------
REGISTRY: dict[str, type[Strategy]] = {}


def register(cls: type[Strategy]) -> type[Strategy]:
    REGISTRY[cls.name] = cls
    return cls


def get_strategy(name: str, **params) -> Strategy:
    if name not in REGISTRY:
        raise KeyError(f"Unknown strategy '{name}'. Available: {list(REGISTRY)}")
    return REGISTRY[name](**params)
