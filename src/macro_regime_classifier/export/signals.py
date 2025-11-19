"""Export regime signals for downstream systems."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def export_signals(signals: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = signals.copy()
    out.index.name = "date"
    out.to_csv(path)
    return path
