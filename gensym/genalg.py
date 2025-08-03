import multiprocessing as mp
import numpy as np
from tqdm import tqdm
from typing import Any

from gensym.exptree import ExpressionTree


def twin_sort(arr1: list, arr2: list) -> list:
    combined = zip(arr1, arr2)
    sorted_combined = sorted(combined, key=lambda x: x[0])
    return zip(*sorted_combined)


def mse(x: np.ndarray, y: np.ndarray) -> float:
    return np.mean(np.square(x.squeeze() - y.squeeze()))


def job(tree: ExpressionTree) -> float:
    return tree.compute()


class FitnessScore:
    def __init__(self, target: Any):
        self.target = target

    def get_score(self, tree: ExpressionTree) -> float:
        y_hat = tree.compute()
        return mse(y_hat, self.target) + (len(tree.gr

    def __call__(self, tree: ExpressionTree) -> float:
        return self.get_score(tree)


def run(
    data: np.ndarray,
    target: np.ndarray,
    crossover_prob: float = 0.5,
    branch_mutation_prob: float = 0.5,
    node_mutation_prob: float = 0.5,
    hoist_mutation_prob: float = 0.5,
    tree_simplify_prob: float = 0.5,
    optimize_const_prob: float = 0.5,
    pop_size: int = 100,
    generations: int = 100,
    return_top: int = 0,
    keep_top: int = 50,
    kill_bottom: int = 20,
) -> ExpressionTree | list[ExpressionTree]:
    """
    Executes the genetic algorithm simulation.
    """
    population = [
        ExpressionTree(
            data=data,
            branch_mutation_prob=branch_mutation_prob,
            crossover_prob=crossover_prob,
            node_mutation_prob=node_mutation_prob,
            hoist_mutation_prob=hoist_mutation_prob,
            tree_simplify_prob=tree_simplify_prob,
            optimize_const_prob=optimize_const_prob,
        )
        for _ in range(pop_size)
    ]

    for tree in population:
        tree.generate()

    losses = []
    top_tree = None
    best_score = None
    fitness_score = FitnessScore(target)
    for _ in tqdm(range(generations), disable=False):
        scores = None
        with mp.Pool(processes=mp.cpu_count() - 1) as pool:
            scores = pool.map(fitness_score.get_score, population)

        scores = [score if not np.isnan(score) else float("inf") for score in scores]
        scores, population = twin_sort(scores, population)
        scores = np.array(scores)

        # if losses:
        #     assert min(scores) <= losses[-1], (
        #         f"minimum loss must never increase; increase of {min(scores) - losses[-1]};"
        #         + f" {min(scores)} {losses[-1]}"
        #     )

        losses.append(min(scores))

        if top_tree is None:
            top_tree = population[0]
            best_score = scores[0]
        else:
            if scores[0] < best_score:
                best_score = scores[0]
                top_tree = population[0]

        if best_score == 0.0:
            break

        for tree in population:
            tree.tree_simplify()

        new_pop = list(population[:keep_top])

        for tree in population[keep_top:-kill_bottom]:
            tree.branch_mutate()
            tree.node_mutate()
            tree.hoist_mutate()
            tree.optimize_constants(target)
            new_pop.append(tree)

        # exit()
        # parent_group_a = population[::2]
        # parent_group_b = population[1::2]
        # for pa, pb in zip(parent_group_a, parent_group_b):
        #     pa.crossover(pb)
        #     new_pop.append(pa)
        #     new_pop.append(pb)

        while len(new_pop) < pop_size:
            tree = ExpressionTree(
                data=data,
                branch_mutation_prob=branch_mutation_prob,
                crossover_prob=crossover_prob,
                node_mutation_prob=node_mutation_prob,
                hoist_mutation_prob=hoist_mutation_prob,
                optimize_const_prob=optimize_const_prob,
            )
            tree.generate()
            new_pop.append(tree)

        population = new_pop

    return top_tree, losses


if __name__ == "__main__":
    run()
