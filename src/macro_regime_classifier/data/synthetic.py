"""Synthetic cross-asset panel for demos and tests."""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_cross_asset_panel(
    n_days: int = 800,
    seed: int = 42,
) -> pd.DataFrame:
    """Simulate daily returns for rates, equities, FX, credit, and vol."""
    rng = np.random.default_rng(seed)
    regimes = rng.integers(0, 4, size=n_days)
    mu = np.array([0.0004, -0.0002, 0.0003, -0.0001])
    sigma = np.array([0.008, 0.012, 0.010, 0.009])

    eq = mu[regimes] + sigma[regimes] * rng.standard_normal(n_days)
    rates = -0.3 * eq + 0.0001 * rng.standard_normal(n_days)
    fx = 0.2 * eq + 0.0002 * rng.standard_normal(n_days)
    credit = -0.5 * eq + 0.0003 * rng.standard_normal(n_days)
    vol = np.abs(-2.0 * eq + 0.01 * rng.standard_normal(n_days))

    idx = pd.bdate_range("2022-01-03", periods=n_days)
    return pd.DataFrame(
        {
            "equity_ret": eq,
            "rates_ret": rates,
            "fx_ret": fx,
            "credit_ret": credit,
            "vol_change": vol,
            "regime": regimes,
        },
        index=idx,
    )
