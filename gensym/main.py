import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_regression

from gensym.exptree import ExpressionTree
from gensym.genalg import run


def main():
    import random
    # seed = 42
    # np.random.seed(seed)
    # random.seed(seed)
    # data, targets = make_regression(n_features=3, shuffle=False)
    data = np.arange(100).reshape(-1, 1) / 10
    targets = np.sin(data)
    # exit()
    tree, losses = run(
        data,
        targets,
        crossover_rate=0.5,
        mutation_rate=0.5,
        generations=100,
        keep_top=20,
        pop_size=100,
    )
    # plt.plot(losses)
    print()
    print(tree.to_string())
    # print(losses[-1])
    plt.plot(data, targets, label="Ground Truth")
    plt.plot(data, tree.compute(), label="Best Fit")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
