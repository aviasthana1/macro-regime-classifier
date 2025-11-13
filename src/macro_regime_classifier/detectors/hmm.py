"""Hidden Markov Model regime detector."""

from __future__ import annotations

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


class HMMRegimeDetector:
    def __init__(self, n_components: int = 4, random_state: int = 42) -> None:
        self.model = GaussianHMM(
            n_components=n_components,
            covariance_type="diag",
            n_iter=500,
            random_state=random_state,
            min_covar=1e-3,
            tol=0.01,
        )
        self._fitted = False

    def fit(self, features: pd.DataFrame) -> "HMMRegimeDetector":
        x = features.values
        self.model.fit(x)
        self._fitted = True
        return self

    def predict(self, features: pd.DataFrame) -> pd.Series:
        if not self._fitted:
            raise RuntimeError("HMMRegimeDetector must be fit before predict")
        states = self.model.predict(features.values)
        return pd.Series(states, index=features.index, name="hmm_regime")

    def transition_matrix(self) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Model not fitted")
        return self.model.transmat_
