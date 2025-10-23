"""Load panels from CSV or synthetic fallback."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from macro_regime_classifier.data.synthetic import generate_cross_asset_panel


def load_panel(path: Path | None = None) -> pd.DataFrame:
    if path is not None and path.exists():
        frame = pd.read_csv(path, parse_dates=["date"], index_col="date")
        return frame.sort_index()
    return generate_cross_asset_panel()
