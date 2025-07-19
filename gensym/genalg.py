import numpy as np

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
) -> ExpressionTree | list[ExpressionTree]:
    """
    Executes the genetic algorithm simulation.
    """
    if score_func == "mse":
        score = lambda x, y: ((x - y) ** 2) / y.shape[0]
    else:
        raise ValueError

    for _ in range(generations):
        for _ in range(pop_size):
            tree = ExpressionTree(data, mut_rate=mutation_rate)
            tree.generate()
            y_hat = tree.compute()
            if not isinstance(y_hat, np.ndarray):
                y_hat = np.array([y_hat] * data.shape[0])

            loss = score(y_hat, target)
