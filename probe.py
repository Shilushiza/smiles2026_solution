from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class HallucinationProbe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._scaler = StandardScaler()
        self._pca: PCA | None = None
        self._clf: LogisticRegression | None = None
        self._threshold: float = 0.5
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def _preprocess(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        if fit:
            X_scaled = self._scaler.fit_transform(X)
            n_comp = min(64, X_scaled.shape[0] - 1, X_scaled.shape[1])
            self._pca = PCA(n_components=n_comp, random_state=42)
            return self._pca.fit_transform(X_scaled)
        return self._pca.transform(self._scaler.transform(X))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        X_proc = self._preprocess(X, fit=True)
        self._clf = LogisticRegression(
            C=0.1,
            max_iter=2000,
            random_state=42,
            solver="lbfgs",
        )
        self._clf.fit(X_proc, y)
        return self

    def fit_hyperparameters(self, X_val: np.ndarray, y_val: np.ndarray) -> "HallucinationProbe":
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._clf.predict_proba(self._preprocess(X))
