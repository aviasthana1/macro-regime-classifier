"""Cross-asset feature engineering."""

from __future__ import annotations

import pandas as pd


FEATURE_COLUMNS = [
    "equity_mom_21",
    "rates_mom_21",
    "fx_mom_21",
    "credit_mom_21",
    "vol_z_63",
    "eq_rates_spread",
    "risk_appetite",
]


def build_features(panel: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=panel.index)
    out["equity_mom_21"] = panel["equity_ret"].rolling(21).sum()
    out["rates_mom_21"] = panel["rates_ret"].rolling(21).sum()
    out["fx_mom_21"] = panel["fx_ret"].rolling(21).sum()
    out["credit_mom_21"] = panel["credit_ret"].rolling(21).sum()
    vol_roll = panel["vol_change"].rolling(63)
    out["vol_z_63"] = (panel["vol_change"] - vol_roll.mean()) / vol_roll.std()
    out["eq_rates_spread"] = out["equity_mom_21"] - out["rates_mom_21"]
    out["risk_appetite"] = out["equity_mom_21"] - out["credit_mom_21"]
    return out.dropna()
