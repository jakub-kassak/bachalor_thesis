from abc import ABC
from typing import Sequence, TypeVar, Tuple

import math
from math import floor, log2

from abstract_game.move import D, S
from mcts.mcts import Node, TreeMetric


class RootVisits(TreeMetric[int, S, D]):
    @classmethod
    def value(cls, root: Node[S, D]) -> int:
        return root.visits


class RChildVisits(TreeMetric[Tuple[int, int], S, D]):
    @classmethod
    def value(cls, root: Node[S, D]) -> Tuple[int, int]:
        x = [0, 0] + sorted(c.visits for c in root.children)
        return x[-1], x[-2]


class MaxDepth(TreeMetric[int, S, D]):
    @classmethod
    def value(cls, root: Node[S, D], depth: int = 0) -> int:
        if root.leaf:
            return depth
        return max(MaxDepth.value(child, depth + 1) for child in root.children)


class AvgDepth(TreeMetric[float, S, D]):
    def value(self, root: Node[S, D]) -> float:
        return self._sum_depth(root) / LeafCnt.value(root)

    def _sum_depth(self, root: Node[S, D], depth: int = 0) -> float:
        if root.leaf:
            return depth
        return sum(self._sum_depth(child, depth + 1) for child in root.children)


class LeafCnt(TreeMetric[int, S, D]):
    @classmethod
    def value(cls, root: Node[S, D]) -> int:
        if root.leaf:
            return 1
        return sum(map(LeafCnt.value, root.children))


class TerminalLeafCnt(TreeMetric[int, S, D]):
    def value(self, root: Node[S, D]) -> int:
        if root.leaf:
            return int(root.terminal)
        return sum(self.value(child) for child in root.children)


class Size(TreeMetric[int, S, D]):
    @classmethod
    def value(cls, root: Node[S, D]) -> int:
        if root.leaf:
            return 1
        return sum(map(Size.value, root.children)) + 1


class InnerSize(TreeMetric[int, S, D]):
    # nodes with out-degree 1 are not counted
    @classmethod
    def value(cls, root: Node[S, D]) -> int:
        if root.leaf:
            return 0
        if len(root.children) == 1:
            return InnerSize.value(root.children[0])
        return sum(map(InnerSize.value, root.children)) + 1


class MaxDegree(TreeMetric[int, S, D]):
    @classmethod
    def value(cls, root: Node[S, D]) -> int:
        if root.leaf:
            return 0
        return max(len(root.children), max(map(MaxDegree.value, root.children)))


class AvgDegree(TreeMetric[float, S, D]):
    @classmethod
    def value(cls, root: Node[S, D]) -> float:
        k = Size.value(root)
        n = LeafCnt.value(root)
        return (k - 1) / (k - n)


class RChildWins(TreeMetric[Sequence[int], S, D]):
    @classmethod
    def value(cls, root: Node[S, D]) -> Sequence[int]:
        children = sorted(root.children, key=lambda x: x.visits)
        return ([0, 0] + [c.wins for c in children])[:-3:-1]


class MinMaxIndex(TreeMetric[int | float, S, D], ABC):
    @classmethod
    def max(cls, root: Node[S, D]) -> int | float:
        raise NotImplementedError

    @classmethod
    def min(cls, root: Node[S, D]) -> int | float:
        raise NotImplementedError


I_co = TypeVar("I_co", covariant=True, bound=MinMaxIndex)


class NormalizedIndex(TreeMetric[float, S, D]):
    def __init__(self, index: I_co):
        super().__init__()
        self._index = index

    def value(self, root: Node[S, D]) -> float:
        min_si: int | float = self._index.min(root)
        max_si: int | float = self._index.max(root)
        if min_si == max_si:
            return 0
        # print(f"n={n}, m={m}, min={min_si:.2f}, max={max_si:.2f}, (max-min)={max_si - min_si}", flush=True)
        return (self._index.value(root) - min_si) / (max_si - min_si)

    def __repr__(self) -> str:
        return f"{self._index}"


class SackinIndex(MinMaxIndex[S, D]):
    # nodes with out-degree 1 are not counted
    @staticmethod
    def value(node: Node[S, D], depth: int = 0) -> int:
        if node.leaf:
            return depth
        if len(node.children) == 1:
            return SackinIndex.value(node.children[0], depth)
        return sum(SackinIndex.value(child, depth + 1) for child in node.children)

    @classmethod
    def max(cls, root: Node[S, D]) -> float:
        """n - number of leaves, m - number of inner vertices"""
        n = LeafCnt.value(root)
        m = InnerSize.value(root)
        return n * m - (m - 1) * m / 2

    @classmethod
    def min(cls, root: Node[S, D]) -> float:
        n = LeafCnt.value(root)
        m = InnerSize.value(root)
        if n == 1 and m == 0:
            return 0
        k = n - m + 1
        log2nk = floor(log2(n / k))
        value = log2nk * n + 3 * n - k * 2 ** (log2nk + 1)
        # print(f'n={n}, m={m}, k={k}, n/k={n/k}, floor(log2(n/k))={log2nk}, value={value}')
        return value


class CopheneticIndex(MinMaxIndex[S, D]):
    # nodes with out-degree 1 are not counted
    @classmethod
    def value(cls, root: Node[S, D]) -> int:
        if root.leaf:
            return 0
        if len(root.children) == 1:
            return cls.value(root.children[0])
        return sum(cls.value(c) + math.comb(len(c.children), 2) for c in root.children)

    @classmethod
    def min(cls, _) -> int:
        return 0

    @classmethod
    def max(cls, root: Node[S, D]) -> int:
        return math.comb(LeafCnt.value(root), 3)


class WinExpectancy(TreeMetric[float, S, D]):
    def value(self, root: Node[S, D]) -> float:
        return sum(child.wins for child in root.children) / sum(child.visits for child in root.children)
