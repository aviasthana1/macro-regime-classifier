"""Ensemble regime predictor combining HMM and GBM."""

from __future__ import annotations

import pandas as pd

from macro_regime_classifier.detectors.gbm import GBMRegimeClassifier
from macro_regime_classifier.detectors.hmm import HMMRegimeDetector
from macro_regime_classifier.features.engineering import build_features
from macro_regime_classifier.features.labels import derive_pseudo_labels


class RegimePredictor:
    def __init__(self, n_regimes: int = 4, random_state: int = 42) -> None:
        self.hmm = HMMRegimeDetector(n_components=n_regimes, random_state=random_state)
        self.gbm = GBMRegimeClassifier(random_state=random_state)

    def fit(self, panel: pd.DataFrame) -> "RegimePredictor":
        features = build_features(panel)
        labels = derive_pseudo_labels(features)
        self.hmm.fit(features)
        self.gbm.fit(features, labels)
        return self

    def predict(self, panel: pd.DataFrame) -> pd.DataFrame:
        features = build_features(panel)
        hmm = self.hmm.predict(features)
        gbm = self.gbm.predict(features)
        proba = self.gbm.predict_proba(features)
        ensemble = ((hmm + gbm) / 2).round().astype(int)
        return pd.DataFrame(
            {"hmm_regime": hmm, "gbm_regime": gbm, "ensemble_regime": ensemble, **proba.to_dict("series")},
            index=features.index,
        )
