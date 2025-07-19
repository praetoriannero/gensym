from abc import ABC
import networkx as nx
import numpy as np
import random
from typing import Any


class Node(ABC):
    def __init__(self, mut_prob: float = 0.2, **kwargs):
        self.mut_prob = mut_prob

    def forward(self) -> Any:
        raise NotImplementedError

    def to_string(self, *args, **kwargs) -> str:
        raise NotImplementedError

    def mutate(self) -> None:
        pass


class Leaf(Node):
    pass


class Operator(Node):
    def __init__(self, op: str | None = None, **kwargs):
        super().__init__(**kwargs)
        if op is None:
            self._op = random.choice(list(self.dispatch))
        else:
            self._op = op

        self.op = self.dispatch[self._op]


class BinaryOperator(Operator):
    """
    BinaryOperator can represent one of the following:
    +, -, *, /
    """

    dispatch = {
        "+": lambda x, y: x + y,
        "-": lambda x, y: x - y,
        "*": lambda x, y: x * y,
        "/": lambda x, y: x / y,
    }
    child_count = 2

    def forward(self, x, y) -> float | int:
        return self.op(x, y)

    def to_string(self, graph: nx.DiGraph) -> str:
        child_a, child_b = list(graph.successors(self))
        return (
            str(child_a.to_string(graph))
            + f" {self._op} "
            + str(child_b.to_string(graph))
        )


class UnaryOperator(Operator):
    """
    UnaryOperator can represent one of the following:
    sin, cos, tan, inv
    """

    dispatch = {
        "sin": lambda x: np.sin(x),
        "cos": lambda x: np.cos(x),
        "tan": lambda x: np.tan(x),
        "inv": lambda x: 1 / x,
    }
    child_count = 1

    def forward(self, x) -> float:
        return self.op(x)

    def to_string(self, graph: nx.DiGraph) -> str:
        child = list(graph.successors(self))[0]
        return f"{self._op}({child.to_string(graph)})"


class Variable(Leaf):
    """
    Variable takes in a matrix and selects one of the columns in the matrix.
    """

    child_count = 0

    def __init__(self, data: np.ndarray, column: int | None = None, **kwargs):
        self.data = data
        self.col = (
            column if column is not None else np.random.randint(0, data.shape[-1])
        )
        super().__init__(**kwargs)

    def forward(self) -> np.ndarray:
        return self.data[:, self.col]

    def to_string(self, *args) -> str:
        return f"x_{self.col}"


class Value(Leaf):
    """
    Value represents some floating point number that can mutate.
    """

    child_count = 0

    def __init__(self, init_value: float | int | None = None, **kwargs):
        self.value = (np.random.rand() - 1) * 2.0 if init_value is None else init_value
        super().__init__(**kwargs)

    def forward(self) -> float | int:
        return self.value

    def to_string(self, *args) -> str:
        return f"{self.value:.3f}"


class ExpressionTree:
    """
    ExpressionTree represents the tree of expressions executed in topological ordering.
    """

    def __init__(self, data: np.ndarray, init_depth: int = 3, max_depth: int = 100):
        self.data = data
        self.init_depth = init_depth
        self.max_depth = max_depth
        self.root = BinaryOperator()
        self.graph = nx.DiGraph()
        self.graph.add_node(self.root)
        self.node_types = [Value, Variable, UnaryOperator, BinaryOperator]

    def _gather_sinks(self) -> list[Node]:
        return [node for node, out_degree in self.graph.out_degree() if not out_degree]

    def _grow(self, node: Node, depth: int) -> None:
        if not isinstance(node, Operator):
            return

        if depth == self.max_depth:
            node_types = [Value, Variable]
        else:
            node_types = self.node_types

        depth += 1
        child_nodes = [
            random.choice(node_types)(data=self.data) for _ in range(node.child_count)
        ]
        for child in child_nodes:
            self.graph.add_edge(node, child)
            self._grow(child, depth)

    def generate(self) -> None:
        self._grow(self.root, 0)
        self.sorted_graph = nx.topological_sort(nx.reverse(self.graph))

    def compute(self) -> float:
        ret_val = 0.0
        args = []
        for node in self.sorted_graph:
            if node.child_count == 0:
                args.append(node.forward())
            elif node.child_count == len(args):
                ret = node.forward(*args)
                args.clear()
                args.append(ret)

        ret_val = args[0]
        return ret_val

    def to_string(self) -> str:
        return f"f(x) = {self.root.to_string(self.graph)}"
