"""Pseudo-labels from realized volatility and momentum."""

from __future__ import annotations

import numpy as np
import pandas as pd

from macro_regime_classifier.models.regimes import MacroRegime


def derive_pseudo_labels(features: pd.DataFrame) -> pd.Series:
    """Rule-based labels for supervised training when true regimes are unknown."""
    labels = np.full(len(features), MacroRegime.RISK_ON, dtype=int)
    vol = features["vol_z_63"].values
    eq = features["equity_mom_21"].values
    infl = features["eq_rates_spread"].values

    labels[(vol > 1.0) & (eq < 0)] = MacroRegime.RISK_OFF
    labels[(infl > 0.02) & (eq > 0)] = MacroRegime.REFLATION
    labels[(eq < -0.01) & (vol < 1.0)] = MacroRegime.SLOWDOWN
    return pd.Series(labels, index=features.index, name="regime")
