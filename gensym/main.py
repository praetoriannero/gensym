import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_regression

from gensym.exptree import ExpressionTree
from gensym.genalg import run


def main():
    data, targets = make_regression(n_features=3, shuffle=False)
    plt.scatter(data[:, 0], targets)
    for _ in range(10):
        tree = ExpressionTree(data, mut_rate=1.0)
        tree.generate()
        print()
        print(tree.to_string())
        tree.mutate()
        print(tree.to_string())
        y_hat = tree.compute()
        if not isinstance(y_hat, np.ndarray):
            y_hat = np.array([y_hat] * data.shape[0])

        plt.plot(data[:, 0], y_hat)

    plt.show()


if __name__ == "__main__":
    main()
