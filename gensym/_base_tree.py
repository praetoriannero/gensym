from __future__ import annotations
from abc import ABC
from typing import Callable

import numpy as np
from sklearn.metrics import mean_squared_error

from gensym.genalg import run


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
        keep_top: float = 0.1,
        prune_bottom: float = 0.1,
        fitness_func: Callable = mean_squared_error,
        population_size: int = 100,
        generations: int = 10,
        verbose: bool = False,
    ):
        self._min_depth = min_depth
        self._max_depth = max_depth
        self._crossover_prob = crossover_prob
        self._branch_mutation_prob = branch_mutation_prob
        self._node_mutation_prob = node_mutation_prob
        self._hoist_mutation_prob = hoist_mutation_prob
        self._tree_simplify_prob = tree_simplify_prob
        self._optimize_const_prob = optimize_const_prob
        self._keep_top = keep_top
        self._prune_bottom = prune_bottom
        self._fitness_func = fitness_func
        self._population_size = population_size
        self._generations = generations
        self._verbose = verbose

        self._losses = []

    def fit(self, inputs: np.ndarray, target: np.ndarray) -> BaseTree:
        tree, losses = run(
            data=inputs,
            target=target,
            crossover_prob=self._crossover_prob,
            branch_mutation_prob=self._branch_mutation_prob,
            node_mutation_prob=self._node_mutation_prob,
            hoist_mutation_prob=self._hoist_mutation_prob,
            tree_simplify_prob=self._tree_simplify_prob,
            optimize_const_prob=self._optimize_const_prob,
            pop_size=self._population_size,
            generations=self._generations,
            keep_top=self._keep_top,
            prune_bottom=self._prune_bottom,
            generations=self._generations,
        )

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def fit_predict(self, inputs: np.ndarray, target: np.ndarray) -> np.ndarray:
        raise NotImplementedError
