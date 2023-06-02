from __future__ import annotations

from collections import deque
from random import choice as rand_choice, shuffle
from time import process_time
from typing import List, Optional, Deque, Callable, TypeVar, Generic, Any, Iterable, MutableSequence, Sequence

from math import sqrt, log

from abstract_game.game import Game
from abstract_game.move import S, Move, D

A = TypeVar('A')


def my_shuffle(seq: MutableSequence[A]) -> MutableSequence[A]:
    shuffle(seq)
    return seq


def identity(seq: MutableSequence[A]) -> MutableSequence[A]:
    return seq


class MCTSException(Exception):
    pass


class Node(Generic[S, D]):
    def __init__(self, game: Game[S, D], move: Optional[Move[S, D]], parent: Optional[Node[S, D]],
                 sort_moves: Callable[[MutableSequence[Move[S, D]]], MutableSequence[Move[S, D]]]):
        self._game = game
        self._move = move
        self._parent = parent
        self._children: List[Node[S, D]] = []
        self._visits: int = 0
        self._wins: int = 0
        self._untried_moves: Optional[MutableSequence[Move[S, D]]] = None
        self._sort_moves = sort_moves
        self.stamp = 0

    def fork_to_next_move(self) -> Node[S, D]:
        move: Move[S, D] = self.untried_moves.pop()
        return Node(self._game.fork(move), move, self, self._sort_moves)

    @property
    def children(self) -> List[Node[S, D]]:
        """Return list of children of the node."""
        return self._children

    @property
    def _previous_i(self) -> int:
        """Return the value of index in the parent of the node."""
        if self.parent:
            return self.parent._game.state.i
        return -1  # self is the root

    @property
    def expanded(self) -> bool:
        """Return whether all children of the node were constructed."""
        return len(self.untried_moves) == 0

    @property
    def leaf(self) -> bool:
        """Return whether this node is a leaf, i.e. whether this node has children."""
        return len(self.children) == 0

    @property
    def terminal(self) -> bool:
        """Return true if the game stored in the node is finished."""
        return self._game.finished()

    @property
    def parent(self) -> Optional[Node[S, D]]:
        """Return the parent of this node or None if the node does not have a parent."""
        return self._parent

    @property
    def move(self) -> Move[S, D]:
        """Return the move that led to this state, if such a move exists, otherwise return None"""
        if not self._move:
            raise MCTSException("No move leads to this node.")
        return self._move

    @property
    def visits(self) -> int:
        """Return the number of visits."""
        return self._visits

    @property
    def wins(self) -> int:
        """Return the number of wins."""
        return self._wins

    @property
    def state(self) -> S:
        """Return the state stored in this node."""
        return self._game.state

    @property
    def untried_moves(self) -> MutableSequence[Move[S, D]]:
        """Return moves from which a child was not constructed yet."""
        if self._untried_moves is None:
            self._untried_moves = self._sort_moves(self._game.legal_moves())
        return self._untried_moves

    def expand(self) -> Node[S, D]:
        """Hang another child under the node and return it."""
        assert not self.expanded
        self.children.append(self.fork_to_next_move())
        return self.children[-1]

    def update_score(self, result: List[int]) -> None:
        """Increment number of visits and update number of wins according to the result."""
        self._visits += 1
        self._wins += self._game.state.n - 1 - result[self._previous_i]

    def random_playout(self) -> List[int]:
        """Simulate random playout from state stored in this node and return the winners."""
        game: Game[S, D] = self._game.fork()
        while not game.finished():
            move: Move = rand_choice(game.legal_moves())
            game.apply_move(move)
        return game.winners()

    def delete_parent(self) -> None:
        """Delete the reference to the parent."""
        self._parent = None

    def __repr__(self, depth: int = 0) -> str:
        s: str = "\t"*depth + f"{self.visits}_{self.stamp}"
        for child in self.children:
            s += "\n" + child.__repr__(depth+1)
        return s


T = TypeVar('T')


class TreeMetric(Generic[T, S, D]):
    def __init__(self):
        self._results: List[T] = []

    @property
    def results(self) -> List[T]:
        """Return results saved from evaluations of the metric."""
        return self._results

    def evaluate(self, root: Node[S, D]) -> T:
        """Evaluate this metric for the tree with given root, save and return the result."""
        self.results.append(self.value(root))
        return self.results[-1]

    def value(self, root: Node[S, D]) -> T:
        """Compute the value of this metric for the given root."""
        raise NotImplementedError()

    def __repr__(self) -> str:
        return self.__class__.__name__

    def name(self) -> str:
        """Return string representation of the metric."""
        return self.__class__.__name__

    def reset(self) -> None:
        """Clear the results list of the metric."""
        self.results.clear()


class MCTS(Generic[S, D]):
    UCT_MAX = 2 ** 64

    def __init__(self, state: S, moves: Iterable[Move[S, D]],
                 game_factory: Callable[[S, Sequence[Move[S, D]], int], Game[S, D]], iterations: int, width: int,
                 prune_limit: int, expl_const: float,
                 heuristic_move_sort: Callable[[MutableSequence[Move[S, D]]], MutableSequence[Move[S, D]]] = my_shuffle,
                 heuristic_move_sort_description: str = "my_shuffle"):
        self._moves = tuple(moves)
        self._game_factory = game_factory
        self._move_sort = heuristic_move_sort
        self._ITERATIONS = iterations
        self._WIDTH = width
        self._PRUNE_LIMIT = prune_limit
        self._EXPL_CONST = expl_const
        self._root: Node[S, D] = Node(self._game_factory(state, self._moves, state.n), None, None, self._move_sort)
        self._last_pruned_level: List[Node[S, D]] = [self._root]
        self._metrics: List[TreeMetric[Any, S, D]] = []
        self.total_time: float = 0
        self.playout_time: float = 0
        self._move_sort_description = heuristic_move_sort_description
        self._stamp_cnt: int = 1

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(iterations: {self._ITERATIONS}, width={self._WIDTH}, ' \
               f'prune_limit: {self._PRUNE_LIMIT}, move_sort={self._move_sort_description})'

    @property
    def metrics(self) -> List[TreeMetric[Any, S, D]]:
        """Return the list of metrics."""
        return self._metrics

    def reset(self) -> None:
        """Reset playout time, total time and each metric."""
        self.playout_time = 0
        self.total_time = 0
        self._stamp_cnt = 1
        for m in self.metrics:
            m.reset()

    def search(self, state: S) -> Move[S, D]:
        """Search next promising move from the state."""
        self._set_root(state)
        start = process_time()
        for _ in range(self._ITERATIONS):
            if self._root.expanded and len(self._root.children) == 1:
                break
            node: Node[S, D] = self._select_next()
            if not node.terminal:
                node = node.expand()
            simulation_result: List[int] = self._random_playout(node)
            self._backpropagate(node, simulation_result)
            self._prune_tree()
        self.total_time += process_time() - start
        self._evaluate_metrics()
        return self._most_promising_move()

    def _evaluate_metrics(self) -> None:
        """Evaluate each metric for the constructed tree."""
        for m in self.metrics:
            m.evaluate(self._root)

    def _random_playout(self, node) -> List[int]:
        """Simulate random playout and measure the playout time."""
        playout_start = process_time()
        simulation_result = node.random_playout()
        self.playout_time += process_time() - playout_start
        return simulation_result

    def _select_next(self) -> Node[S, D]:
        """Select the next node the next node to expand. This is the selection phase of the MCTS algorithm."""
        node: Node[S, D] = self._root
        while node.expanded and not node.terminal:
            node = max(node.children, key=self._uct)
        return node

    def _find_node(self, state: S) -> Node[S, D]:
        """Find node with the state or if such node does not exist construct new node with the state."""
        queue: Deque[Node[S, D]] = deque()
        queue.append(self._root)
        while queue and queue[0].state.mc <= state.mc:
            node: Node[S, D] = queue.popleft()
            if node.state == state:
                node.delete_parent()
                return node
            queue.extend(node.children)
        return Node(self._game_factory(state, self._moves, state.n), None, None, self._move_sort)

    def _set_root(self, state: S) -> None:
        """Root the tree in a node that stores the state."""
        self._root = self._find_node(state)
        self._last_pruned_level = [self._root]

    @staticmethod
    def _backpropagate(node: Optional[Node[S, D]], result: List[int]) -> None:
        """Update score on the path from node to root. This is the backpropagation phase of the algorithm."""
        while node is not None:
            node.update_score(result)
            node = node.parent

    def _uct(self, node: Node[S, D]) -> float:
        """Return the uct value of the node."""
        if node.visits == 0:
            return self.UCT_MAX
        assert node.parent is not None
        return node.wins / node.visits + sqrt(self._EXPL_CONST * log(node.parent.visits) / node.visits)

    def _most_promising_move(self) -> Move[S, D]:
        """Return the most promising move."""
        assert self._root.children
        best_node: Node[S, D] = max(self._root.children, key=lambda node: node.visits)
        return best_node.move

    def _pruning_level_nodes(self) -> List[Node[S, D]]:
        """Return nodes at the pruning level."""
        return sum([node.children for node in self._last_pruned_level], [])

    def _visits_at_prune_level(self) -> int:
        """Return sum of visits in nodes at pruning the level."""
        return sum(node.visits for node in self._pruning_level_nodes())

    def _prune_tree(self) -> None:
        """Restricts the number of tree nodes at prune level to the maximum number given by the beam WIDTH."""
        if self._visits_at_prune_level() < self._PRUNE_LIMIT:
            return
        self._last_pruned_level = sorted(self._pruning_level_nodes(),
                                         key=lambda x: x.visits, reverse=True)[:self._WIDTH]
        self._stamp_ancestors()
        self._remove_unstamped()
        self._stamp_cnt += 1

    def _stamp_ancestors(self) -> None:
        """Stamp ancestors of nodes at last pruned level."""
        node: Optional[Node[S, D]]
        for node in self._last_pruned_level:
            while node is not None and node.stamp != self._stamp_cnt:
                node.stamp = self._stamp_cnt
                node = node.parent

    def _remove_unstamped(self) -> None:
        """Remove unstamped nodes from the tree up to last pruned level."""
        stack: Deque[Node[S, D]] = deque()
        stack.append(self._root)
        while stack:
            node: Node[S, D] = stack.pop()
            if node not in self._last_pruned_level:
                node.children[:] = (child for child in node.children if child.stamp == self._stamp_cnt)
                stack.extend(node.children)


class DebuggingMCTS(MCTS[S, D]):
    def search(self, state: S) -> Move[S, D]:
        print("SEARCH START", flush=True)
        move = super().search(state)
        print("SEARCH END", flush=True)
        return move

    def _prune_tree(self) -> None:
        print(f"\t\tbefore stamp_cnt: {self._stamp_cnt}\n", self._root.__repr__(3), flush=True)
        super()._prune_tree()
        print("\t\tafter\n", self._root.__repr__(3), flush=True)

    def _set_root(self, state: S) -> None:
        print(self._root.__repr__(3), flush=True)
