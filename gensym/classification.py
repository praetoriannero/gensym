from gensym.base_tree import BaseTree

import numpy as np


class Classifier(BaseTree):

    def fit(self, inputs: np.ndarray, target: np.ndarray) -> Classifier:
        pass

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        pass

    def fit_predict(self, inputs: np.ndarray, target: np.ndarray) -> np.ndarray:
        pass
