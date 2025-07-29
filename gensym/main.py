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
    }
    inputs = np.arange(100) + 1
    _, axs = plt.subplots(len(regression_test), 2)
    for idx, (exp, func) in enumerate(regression_test.items()):
        target = func(inputs)
        tree, losses = run(
            inputs.reshape(-1, 1),
            target,
            crossover_prob=0.5,
            branch_mutation_prob=0.2,
            node_mutation_prob=0.1,
            hoist_mutation_prob=0.1,
            tree_simplify_prob=0.2,
            optimize_const_prob=0.01,
            generations=10,
            keep_top=20,
            pop_size=1000,
        )
        # print(losses)
        y_hat = tree.compute()
        # print("best", tree.to_string())
        axs[idx][0].plot(inputs, target, label=exp)
        axs[idx][0].plot(inputs, y_hat, label="Best Fit", c="red")
        axs[idx][1].plot(losses, label="Loss")
        axs[idx][0].legend()
        axs[idx][1].legend()
        print(tree.to_string())

    # plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
