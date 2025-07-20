import multiprocessing as mp
import numpy as np
from tqdm import tqdm
from typing import Any

from gensym.exptree import ExpressionTree


def percentile(arr: np.ndarray, val: float) -> float:
    return np.mean(arr <= val)


def twin_sort(arr1: list, arr2: list) -> list:
    combined = zip(arr1, arr2)
    sorted_combined = sorted(combined, key=lambda x: x[0])
    return zip(*sorted_combined)


def mse(x: np.ndarray, y: np.ndarray) -> float:
    return np.mean(np.square(x - y))


def job(tree: ExpressionTree) -> float:
    return tree.compute()


class FitnessScore:
    def __init__(self, target: Any):
        self.target = target

    def get_score(self, tree: ExpressionTree) -> float:
        y_hat = tree.compute()
        return mse(y_hat, self.target)


def run(
    data: np.ndarray,
    target: np.ndarray,
    mutation_rate: float = 0.5,
    crossover_rate: float = 0.5,
    pop_size: int = 100,
    generations: int = 100,
    return_top: int = 0,
    keep_top: int = 50,
) -> ExpressionTree | list[ExpressionTree]:
    """
    Executes the genetic algorithm simulation.
    """
    population = [
        ExpressionTree(data=data, mut_prob=mutation_rate, co_prob=crossover_rate)
        for _ in range(pop_size)
    ]

    for tree in population:
        tree.generate()

    # pool = mp.Pool(processes=mp.cpu_count() - 1)
    # pool.map(job, population)
    # pool.close()
    # pool.
    losses = []
    top_tree = None
    best_score = None
    fitness_score = FitnessScore(target)
    for _ in tqdm(range(generations), disable=True):
        scores = None
        with mp.Pool(processes=mp.cpu_count() - 1) as pool:
            scores = pool.map(fitness_score.get_score, population)

        scores, population = twin_sort(scores, population)
        losses.append(scores[0])

        if top_tree is None:
            top_tree = population[0]
            best_score = scores[0]
        else:
            if scores[0] < best_score:
                best_score = scores[0]
                top_tree = population[0]

        new_pop = []
        for fitness, tree in zip(scores, population):
            percent = percentile(scores, fitness)
            tree.mutate(mut_prob=(1.0 - percent))

        new_pop = list(population)[:keep_top]
        print(new_pop[0].to_string())
        print(losses[-1])
        # print(type(new_pop))
        # parent_group_a = population[::2]
        # parent_group_b = population[1::2]
        # for pa, pb in zip(parent_group_a, parent_group_b):
        #     pa.crossover(pb)
        #     new_pop.append(pa)
        #     new_pop.append(pb)

        while len(new_pop) < pop_size:
            tree = ExpressionTree(
                data=data, mut_prob=mutation_rate, co_prob=crossover_rate
            )
            tree.generate()
            new_pop.append(tree)

        population = new_pop

    return top_tree, losses


if __name__ == "__main__":
    run()
