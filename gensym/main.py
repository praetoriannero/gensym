import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.preprocessing import MinMaxScaler

from gensym.exptree import ExpressionTree


def main():
    data, targets = make_regression(n_features=1, shuffle=False)
    plt.scatter(data[:, 0], targets)
    for _ in range(10):
        tree = ExpressionTree(data)
        tree.generate()
        y_hat = tree.compute()
        if not isinstance(y_hat, np.ndarray):
            y_hat = np.array([y_hat] * data.shape[0])

        print(tree.to_string())
        plt.plot(data[:, 0], y_hat)

    plt.show()


if __name__ == "__main__":
    main()
