"""Walk-forward regime classification evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.metrics import accuracy_score

from macro_regime_classifier.ensemble.predictor import RegimePredictor
from macro_regime_classifier.features.engineering import build_features
from macro_regime_classifier.features.labels import derive_pseudo_labels


@dataclass
class WalkForwardResult:
    fold: int
    train_start: str
    train_end: str
    test_end: str
    accuracy: float


def walk_forward_eval(
    panel: pd.DataFrame,
    train_days: int = 504,
    test_days: int = 63,
) -> list[WalkForwardResult]:
    features = build_features(panel)
    labels = derive_pseudo_labels(features)
    results: list[WalkForwardResult] = []
    fold = 0
    start = 0
    while start + train_days + test_days <= len(features):
        train_slice = features.iloc[start : start + train_days]
        test_slice = features.iloc[start + train_days : start + train_days + test_days]
        train_panel = panel.loc[train_slice.index]

        model = RegimePredictor().fit(train_panel)
        context_panel = panel.loc[: test_slice.index[-1]]
        try:
            preds = model.predict(context_panel)["ensemble_regime"].loc[test_slice.index]
        except ValueError:
            fold += 1
            start += test_days
            continue
        y_true = labels.loc[preds.index]
        acc = float(accuracy_score(y_true, preds))
        results.append(
            WalkForwardResult(
                fold=fold,
                train_start=str(train_slice.index[0].date()),
                train_end=str(train_slice.index[-1].date()),
                test_end=str(test_slice.index[-1].date()),
                accuracy=acc,
            )
        )
        fold += 1
        start += test_days
    return results
