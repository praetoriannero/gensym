from __future__ import annotations
from abc import ABC

import numpy as np


class BaseTree(ABC):
    def __init__(self):
        pass

    def fit(self, inputs: np.ndarray, target: np.ndarray) -> BaseTree:
        pass

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        pass

    def fit_predict(self, inputs: np.ndarray, target: np.ndarray) -> np.ndarray:
        pass
