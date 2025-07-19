from __future__ import annotations
from abc import ABC
import networkx as nx
from networkx.algorithms.traversal.depth_first_search import dfs_tree
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
        raise NotImplementedError


class Leaf(Node):
    def __repr__(self):
        return self.to_string()


class Operator(Node):
    def __init__(self, op: str | None = None, **kwargs):
        super().__init__(**kwargs)
        if op is None:
            self._op = random.choice(list(self.dispatch))
        else:
            self._op = op

        self.op = self.dispatch[self._op]

    def __repr__(self):
        return self._op


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

    def to_string(
        self, graph: nx.DiGraph, *args, is_root: bool = False, **kwargs
    ) -> str:
        child_a, child_b = list(graph.successors(self))
        ret_str = (
            str(child_a.to_string(graph))
            + f" {self._op} "
            + str(child_b.to_string(graph))
        )
        if not is_root:
            ret_str = f"({ret_str})"

        return ret_str


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

    def to_string(self, graph: nx.DiGraph, *args, **kwargs) -> str:
        child = next(graph.successors(self))
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

    def mutate(self) -> None:
        self.col = np.random.randint(0, self.data.shape[-1])

    def forward(self) -> np.ndarray:
        return self.data[:, self.col]

    def to_string(self, *args, **kwargs) -> str:
        return f"x_{self.col}"


class Value(Leaf):
    """
    Value represents some floating point number that can mutate.
    """

    child_count = 0

    def __init__(self, init_value: float | int | None = None, **kwargs):
        self.value = (
            (np.random.rand() - 0.5) * 2.0 if init_value is None else init_value
        )
        super().__init__(**kwargs)

    def mutate(self) -> None:
        self.value = (np.random.rand() - 0.5) * 2.0

    def forward(self) -> float | int:
        return self.value

    def to_string(self, *args, **kwargs) -> str:
        return f"{self.value:.3f}"


class ExpressionTree:
    """
    ExpressionTree represents the tree of expressions executed in topological ordering.
    """

    def __init__(
        self,
        data: np.ndarray,
        init_depth: int = 3,
        max_depth: int = 10,
        mut_rate: float = 0.5,
        co_rate: float = 0.5,
    ):
        self.data = data
        self.init_depth = init_depth
        self.max_depth = max_depth
        self.mut_rate = mut_rate
        self.co_rate = co_rate

        self.root: BinaryOperator | None = None
        self.graph = nx.DiGraph()
        self.node_types = [Value, Variable, UnaryOperator, BinaryOperator]

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
        self.graph.clear()
        self.root = BinaryOperator()
        self.graph.add_node(self.root)
        self._grow(self.root, 0)
        self.sorted_graph = nx.topological_sort(nx.reverse(self.graph))

    def compute(self) -> float:
        args = []
        self.sorted_graph = nx.topological_sort(nx.reverse(self.graph))
        for node in self.sorted_graph:
            if not node.child_count:
                args.append(node.forward())
            elif node.child_count == len(args):
                ret = node.forward(*args)
                args.clear()
                args.append(ret)

        y_hat = args[0]
        if not isinstance(y_hat, np.ndarray):
            y_hat = np.array([y_hat] * self.data.shape[0])

        return y_hat

    def to_string(self) -> str:
        return f"f(x) = {self.root.to_string(self.graph, is_root=True)}"

    def _prune_branch(self, node) -> None:
        self.graph.remove_nodes_from(dfs_tree(self.graph, node))

    def mutate(self) -> None:
        if random.random() <= self.mut_rate:
            node = random.choice(list(self.graph.nodes()))
            if node is self.root:
                self.generate()
                return

            parent_node = next(self.graph.predecessors(node))
            self._prune_branch(node)

            new_node = random.choice(self.node_types)(data=self.data)
            self.graph.add_edge(parent_node, new_node)
            self._grow(new_node, 0)  # TODO need to get actual node depth here

            self.sorted_graph = nx.topological_sort(nx.reverse(self.graph))

    def get_random_subgraph(self) -> nx.DiGraph:
        node = random.choice(list(self.graph.nodes()))
        parent = next(self.graph.predecessors(node), None)
        subgraph = self.graph.subgraph(dfs_tree(self.graph, node)).copy()
        return parent, node, subgraph

    def insert_tree(self, parent, node, new_node, new_tree) -> None:
        if parent is not None:
            self.graph.add_edge(parent, new_node)
            self.graph.add_edges_from(list(new_tree.edges()))
            self._prune_branch(node)
        else:
            self.graph.clear()
            self.root = new_node
            self.graph = new_tree

        self.sorted_graph = nx.topological_sort(nx.reverse(self.graph))

    def crossover(self, other: ExpressionTree) -> None:
        if random.random() <= self.co_rate:
            this_parent, this_node, this_subgraph = self.get_random_subgraph()
            other_parent, other_node, other_subgraph = other.get_random_subgraph()
            self.insert_tree(this_parent, this_node, other_node, other_subgraph)
            other.insert_tree(other_parent, other_node, this_node, this_subgraph)
