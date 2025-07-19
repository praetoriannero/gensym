import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_regression

from gensym.exptree import ExpressionTree
from gensym.genalg import run


def main():
    data, targets = make_regression(n_features=3, shuffle=False)
    plt.scatter(data[:, 0], targets)
    for _ in range(10):
        print()
        tree_a = ExpressionTree(data, mut_rate=1.0, co_rate=1.0)
        tree_b = ExpressionTree(data, mut_rate=1.0)
        tree_a.generate()
        tree_b.generate()

        print("Tree A")
        print(tree_a.to_string())
        print("Tree B")
        print(tree_b.to_string())

        tree_a.crossover(tree_b)
        print("Tree A: Crossover")
        print(tree_a.to_string())
        print("Tree B: Crossover")
        print(tree_b.to_string())

        tree_a.mutate()
        tree_b.mutate()

        y_hat = tree_a.compute()
        if not isinstance(y_hat, np.ndarray):
            y_hat = np.array([y_hat] * data.shape[0])

        plt.plot(data[:, 0], y_hat)

    plt.show()


if __name__ == "__main__":
    main()
