from __future__ import annotations
from abc import ABC

import numpy as np


class BaseTree(ABC):
    def __init__(
        self,
        min_depth: int = 1,
        max_depth: int = 4,
        crossover_prob: float = 0.5,
        branch_mutation_prob: float = 0.5,
        node_mutation_prob: float = 0.5,
        hoist_mutation_prob: float = 0.5,
        tree_simplify_prob: float = 0.5,
        optimize_const_prob: float = 0.5,
    ):
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.crossover_prob = crossover_prob
        self.branch_mutation_prob = branch_mutation_prob
        self.node_mutation_prob = node_mutation_prob
        self.hoist_mutation_prob = hoist_mutation_prob
        self.tree_simplify_prob = tree_simplify_prob
        self.optimize_const_prob = optimize_const_prob

    def fit(self, inputs: np.ndarray, target: np.ndarray) -> BaseTree:
        raise NotImplementedError

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def fit_predict(self, inputs: np.ndarray, target: np.ndarray) -> np.ndarray:
        raise NotImplementedError
