import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_regression

from gensym.exptree import ExpressionTree
from gensym.genalg import run


def main():
    # data, targets = make_regression(n_features=3, shuffle=False)
    data = np.arange(100).reshape(-1, 1) / 10
    targets = np.sin(data)
    losses = run(
        data,
        targets,
        crossover_rate=0.5,
        mutation_rate=0.5,
        generations=100,
        keep_top=500,
        pop_size=1000,
    )
    plt.plot(losses)
    plt.show()


if __name__ == "__main__":
    main()
