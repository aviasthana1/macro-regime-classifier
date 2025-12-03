"""Gradient boosting macro regime classifier."""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler


class GBMRegimeClassifier:
    def __init__(self, random_state: int = 42) -> None:
        self.scaler = StandardScaler()
        self.clf = GradientBoostingClassifier(random_state=random_state)
        self._fitted = False

    def fit(self, features: pd.DataFrame, labels: pd.Series) -> "GBMRegimeClassifier":
        x = self.scaler.fit_transform(features.values)
        self.clf.fit(x, labels.values)
        self._fitted = True
        return self

    def predict(self, features: pd.DataFrame) -> pd.Series:
        if not self._fitted:
            raise RuntimeError("GBMRegimeClassifier must be fit before predict")
        x = self.scaler.transform(features.values)
        pred = self.clf.predict(x)
        return pd.Series(pred, index=features.index, name="gbm_regime")

    def predict_proba(self, features: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("GBMRegimeClassifier must be fit before predict_proba")
        x = self.scaler.transform(features.values)
        proba = self.clf.predict_proba(x)
        cols = [f"p_{i}" for i in range(proba.shape[1])]
        return pd.DataFrame(proba, index=features.index, columns=cols)
