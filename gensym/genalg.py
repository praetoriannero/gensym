import numpy as np
from tqdm import tqdm

from gensym.exptree import ExpressionTree


def run(
    data: np.ndarray,
    target: np.ndarray,
    score_func: str = "mse",
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
    if score_func == "mse":
        score = lambda x, y: np.sum(((x - y) ** 2)) / y.shape[0]
    else:
        raise ValueError

    population = [
        ExpressionTree(data=data, mut_rate=mutation_rate, co_rate=crossover_rate)
        for _ in range(pop_size)
    ]

    for tree in population:
        tree.generate()

    losses = []
    for _ in tqdm(range(generations)):
        scores = []
        for tree_idx in range(pop_size):
            tree = population[tree_idx]
            tree.mutate()
            y_hat = tree.compute()
            scores.append((score(y_hat, target), tree))

        scores.sort(key=lambda x: x[0])
        losses.append(scores[0][0])
        top_scores = scores[:keep_top]
        parent_group_a = top_scores[::2]
        parent_group_b = top_scores[1::2]
        new_pop = []
        for (_score_a, pa), (_score_b, pb) in zip(parent_group_a, parent_group_b):
            pa.crossover(pb)
            new_pop.append(pa)

        while len(new_pop) < pop_size:
            tree = ExpressionTree(
                data=data, mut_rate=mutation_rate, co_rate=crossover_rate
            )
            tree.generate()
            new_pop.append(tree)

        population = new_pop

    return losses


if __name__ == "__main__":
    run()
