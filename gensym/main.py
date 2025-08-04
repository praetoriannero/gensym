import matplotlib.pyplot as plt
import numpy as np
import random

from gensym.exptree import ExpressionTree
from gensym.genalg import run, mse

SEED = 963
np.random.seed(SEED)
random.seed(SEED)


def main():
    regression_test = {
        "sin(x * x)": lambda x: np.sin(x * x),
        "cos(sin(inv(x + 1)))": lambda x: np.cos(np.sin(1 / (x + 1))),
        "x * (x + x)": lambda x: x * (x + x),
        "-1 * (x * x)": lambda x: -1 * (x * x),
        "-2 + inv(x)": lambda x: -2 + (1 / x),
        "inv(tan(x))": lambda x: 1 / np.tan(x),
    }
    data = (np.arange(100) + 1).reshape(-1, 1)

    _, axs = plt.subplots(len(regression_test), 2)
    for idx, (exp, func) in enumerate(regression_test.items()):
        target = func(data)
        tree, losses = run(
            data.reshape(-1, 1),
            target,
            crossover_prob=0.5,
            branch_mutation_prob=0.2,
            node_mutation_prob=0.1,
            hoist_mutation_prob=0.1,
            tree_simplify_prob=1.0,
            optimize_const_prob=0.01,
            generations=100,
            keep_top=1,
            kill_bottom=30,
            pop_size=1000,
        )

        y_hat = tree.compute(data)
        axs[idx][0].plot(data, target, label=exp)
        axs[idx][0].plot(data, y_hat, label="Best Fit", c="red")
        axs[idx][1].plot(losses, label="Loss")
        axs[idx][0].legend()
        axs[idx][1].legend()
        print(tree.to_string())

    # plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
