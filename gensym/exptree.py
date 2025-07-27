from __future__ import annotations
from abc import ABC
import networkx as nx
from networkx.algorithms.traversal.depth_first_search import dfs_tree
import numpy as np
import random
from typing import Any


np.seterr(all="ignore")


class Node(ABC):
    def __init__(self, **kwargs):
        pass

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


def add(x: Any, y: Any) -> Any:
    return x + y


def sub(x: Any, y: Any) -> Any:
    return x - y


def mul(x: Any, y: Any) -> Any:
    return x * y


def div(x: Any, y: Any) -> Any:
    return x / y


def exp(x: Any, y: Any) -> Any:
    return x**y


class BinaryOperator(Operator):
    """
    BinaryOperator can represent one of the following:
    +, -, *, /
    """

    dispatch = {
        "+": add,
        "-": sub,
        "*": mul,
        "/": div,
        # "**": lambda x, y: x ** y,
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


def inv(x: Any) -> Any:
    return 1 / x


class UnaryOperator(Operator):
    """
    UnaryOperator can represent one of the following:
    sin, cos, tan, inv, sqrt, abs
    """

    dispatch = {
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "inv": inv,
        # "sqrt": np.sqrt,
        "abs": np.abs,
    }
    child_count = 1

    def forward(self, x) -> float:
        return self.op(x)

    def to_string(self, graph: nx.DiGraph, *args, **kwargs) -> str:
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

    def mutate(self) -> None:
        self.col = np.random.randint(0, self.data.shape[-1])

    def forward(self) -> np.ndarray:
        return self.data[:, self.col]

    def to_string(self, *args, **kwargs) -> str:
        return f"x_{self.col}"


class Constant(Leaf):
    """
    Constant represents some floating point number that can mutate.
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
        branch_mutation_prob: float = 0.5,
        crossover_prob: float = 0.5,
        node_mutation_prob: float = 0.5,
        hoist_mutation_prob: float = 0.5,
        tree_simplify_prob: float = 0.5,
    ):
        self.data = data
        self.init_depth = init_depth
        self.max_depth = max_depth
        self.branch_mutation_prob = branch_mutation_prob
        self.crossover_prob = crossover_prob
        self.node_mutation_prob = node_mutation_prob
        self.hoist_mutation_prob = hoist_mutation_prob
        self.tree_simplify_prob = tree_simplify_prob

        self.root: UnaryOperator | BinaryOperator | None = None
        self.graph = nx.DiGraph()
        self.node_types = [Constant, Variable, UnaryOperator, BinaryOperator]

    def _grow(self, node: Node, depth: int) -> None:
        if not isinstance(node, Operator):
            return

        if depth == self.max_depth:
            node_types = [Constant, Variable]
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
        self.root = random.choice([UnaryOperator(), BinaryOperator()])
        self.graph.add_node(self.root)
        self._grow(self.root, 0)
        self.sorted_reverse = list(nx.topological_sort(nx.reverse(self.graph)))

    def compute(self) -> float:
        cache = {}
        reverse_graph = nx.reverse(self.graph)
        self.sorted_reverse = list(nx.topological_sort(nx.reverse(self.graph)))
        for node in self.sorted_reverse:
            if isinstance(node, (Constant, Variable)):
                cache[node] = node.forward()
            elif isinstance(node, UnaryOperator):
                parent = next(reverse_graph.predecessors(node))
                cache[node] = node.forward(cache[parent])
            elif isinstance(node, BinaryOperator):
                parents = tuple(reverse_graph.predecessors(node))
                cache[node] = node.forward(cache[parents[0]], cache[parents[1]])

        result = cache[self.root]
        if not isinstance(result, np.ndarray):
            result = np.array([result] * len(self.data))

        return result.squeeze()

    def to_string(self) -> str:
        return f"f(x) = {self.root.to_string(self.graph, is_root=True)}"

    def _prune_branch(self, node) -> None:
        self.graph.remove_nodes_from(dfs_tree(self.graph, node))

    @staticmethod
    def sort_post(func) -> function:
        def wrapped_func(self) -> None:
            res = func(self)
            self.sorted_reverse = list(nx.topological_sort(nx.reverse(self.graph)))
            self.root = self.sorted_reverse[-1]
            return res

        return wrapped_func

    @sort_post
    def branch_mutate(self) -> None:
        if random.random() <= self.branch_mutation_prob:
            node = random.choice(list(self.graph.nodes()))

            if node is self.root:
                self.generate()
                return

            parent_node = next(self.graph.predecessors(node))
            self._prune_branch(node)

            new_node = random.choice(self.node_types)(data=self.data)
            self.graph.add_edge(parent_node, new_node)
            self._grow(new_node, 0)  # TODO need to get actual node depth here

    @sort_post
    def node_mutate(self) -> None:
        if random.random() <= self.node_mutation_prob:
            node = random.choice(list(self.graph.nodes()))
            new_node = type(node)(data=self.data)
            self.graph.add_node(new_node)

            for parent in self.graph.predecessors(node):
                self.graph.add_edge(parent, new_node)

            for neighbor in self.graph.neighbors(node):
                self.graph.add_edge(new_node, neighbor)

            self.graph.remove_node(node)

    @sort_post
    def hoist_mutate(self) -> None:
        if random.random() <= self.hoist_mutation_prob:
            node = random.choice(list(self.graph.nodes()))
            self.graph = dfs_tree(self.graph, node)

    @sort_post
    def tree_simplify(self) -> None:
        if random.random() <= self.tree_simplify_prob and len(self.graph) > 1:
            unary_ops = [
                node for node in self.graph.nodes() if isinstance(node, UnaryOperator)
            ]
            if not unary_ops:
                return

            node = random.choice(unary_ops)
            child = next(nx.neighbors(self.graph, node))
            if not isinstance(child, Constant):
                return

            if node is self.root:
                self.root = Constant(init_value=node.forward(child.value))
            else:
                new_const_node = Constant(init_value=node.forward(child.value))
                parent = next(self.graph.predecessors(node))
                self.graph.remove_nodes_from([node, child])
                self.graph.add_edge(parent, new_const_node)

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

        self.sorted_reverse = list(nx.topological_sort(nx.reverse(self.graph)))

    def crossover(self, other: ExpressionTree) -> None:
        if random.random() <= self.crossover_prob:
            this_parent, this_node, this_subgraph = self.get_random_subgraph()
            other_parent, other_node, other_subgraph = other.get_random_subgraph()
            self.insert_tree(this_parent, this_node, other_node, other_subgraph)
            other.insert_tree(other_parent, other_node, this_node, this_subgraph)
