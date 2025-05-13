from __future__ import annotations
import math
from collections.abc import Sequence
from typing import Literal
import numpy as np
import sklearn.preprocessing


class ScoreNormalizer:
    def __init__(self, type: Literal["sigmoid", "minmax"]) -> None:
        self.type = type
        if type == "minmax":
            self.scaler = sklearn.preprocessing.MinMaxScaler()
        else:
            self.scaler = None

    def fit(self, all_scores: np.ndarray | Iterable[float]) -> None:
        if type(all_scores) != np.ndarray:
            all_scores = np.array(all_scores)
        if self.type == "minmax":
            self.scaler.fit(all_scores.reshape(-1, 1))
        else:
            pass

    def normalize_score(self, score: float) -> float:
        """Normalize VespaG score to range."""
        return self.normalize_scores([score])[0]

    def normalize_scores(self, scores: np.ndarray | Sequence[float]) -> list[float]:
        """Normalize VespaG scores to range."""
        if self.type == "sigmoid":
            return [1 / (1 + math.exp(-score)) for score in scores]
        elif self.type == "minmax":
            if type(scores) != np.ndarray:
                scores = np.array(scores)
            return list(self.scaler.transform(scores.reshape(-1, 1)).reshape(-1))
