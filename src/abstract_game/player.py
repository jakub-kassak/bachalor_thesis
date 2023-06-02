import random
from time import process_time
from typing import Generic, MutableSequence, Callable, Sequence, Dict, Set

from abstract_game.game import Game
from abstract_game.move import Move, S, D


class Player(Generic[S, D]):
    def __init__(self, name: str):
        self._name = name
        self._spent_time: float = 0
        self._rounds: int = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def spent_time(self) -> float:
        return self._spent_time

    @property
    def rounds(self) -> int:
        return self._rounds

    def play(self, state: S, legal_moves: MutableSequence[Move[S, D]]) -> Move[S, D]:
        start = process_time()
        mv: Move[S, D] = self._play(state, legal_moves)
        self._spent_time += process_time() - start
        self._rounds += 1
        return mv

    def _play(self, state: S, legal_moves: MutableSequence[Move[S, D]]) -> Move[S, D]:
        raise NotImplementedError()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(name={self.name})'

    def reset(self) -> None:
        self._spent_time = 0
        self._rounds = 0
        self._reset()

    def _reset(self) -> None:
        pass


class RandomPlayer(Player[S, D]):
    def _play(self, state: S, legal_moves: MutableSequence[Move]) -> Move[S, D]:
        return random.choice(list(legal_moves))


class BackTrack(Player[S, D]):
    def __init__(self, name: str, moves: Sequence[Move[S, D]], move_sort: Callable, limit: int,
                 game_factory: Callable[[S, Sequence[Move[S, D]], int], Game[S, D]]):
        super().__init__(name)
        self._move_sort = move_sort
        self._moves: Sequence[Move[S, D]] = tuple(moves)
        self._limit = limit
        self._game_factory = game_factory

    def _play(self, state: S, legal_moves: MutableSequence[Move[S, D]]) -> Move[S, D]:
        assert state.n == 2
        game: Game[S, D] = self._game_factory(state, self._moves, 2)
        for mv in legal_moves:
            if not self._winning_strategy(game.fork(mv)):
                return mv
        return self._move_sort(legal_moves)[0]

    def _winning_strategy(self, game: Game[S, D], depth: int = 0) -> bool:
        if depth >= self._limit:
            return False
        if game.finished():
            return False
        return any(not self._winning_strategy(game.fork(mv), depth + 1) for mv in self._move_sort(game.legal_moves()))


class BackTrackMemo(BackTrack[S, D]):
    def __init__(self, name: str, moves: Sequence[Move[S, D]], move_sort: Callable, limit: int,
                 game_factory: Callable[[S, Sequence[Move[S, D]], int], Game[S, D]]):
        super().__init__(name, moves, move_sort, limit, game_factory)
        self._memo: Dict[str, bool] = {}

    def _reset(self) -> None:
        self._memo.clear()

    @staticmethod
    def mk_key(game) -> str:
        return f"{game.state.dp[-1]}{game.state.st}{game.state.hands}{game.state.lp_mc}{game.state.i})"

    def _winning_strategy(self, game: Game[S, D], depth: int = 0) -> bool:
        key = self.mk_key(game)
        if key in self._memo:
            return self._memo[key]
        result = super()._winning_strategy(game, depth)
        self._memo[key] = result
        return result
