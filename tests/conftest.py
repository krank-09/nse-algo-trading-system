"""Shared synthetic-data fixtures. No network, no real market data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def make_ohlcv(n: int = 300, start: str = "2020-01-01", seed: int = 0) -> pd.DataFrame:
    """Random-walk daily OHLCV on business days."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n, name="date")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    high = np.maximum(open_, close) * 1.005
    low = np.minimum(open_, close) * 0.995
    volume = rng.integers(100_000, 1_000_000, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


@pytest.fixture
def ohlcv() -> pd.DataFrame:
    return make_ohlcv()
