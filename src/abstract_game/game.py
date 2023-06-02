from __future__ import annotations

from typing import Generic, List, Optional, TypeVar, cast, Sequence, MutableSequence

from abstract_game.move import S, Move, D

S_co = TypeVar('S_co', bound=Move, covariant=True)


class GameException(Exception):
    pass


class Game(Generic[S, D]):
    # Game should be a simple wrapper for GameState, which allows GameState manipulation according to rules.
    @property
    def state(self) -> S:
        raise NotImplementedError()

    @property
    def moves(self) -> Sequence[Move[S, D]]:
        raise NotImplementedError()

    def legal_moves(self) -> MutableSequence[Move[S, D]]:
        if self.finished():
            return []
        return [cast(Move[S, D], mv) for mv in self.moves if mv.test(self.state)]

    def finished(self) -> bool:
        raise NotImplementedError()

    def winners(self) -> List[int]:
        """Return a list of ints where player numbers are indices and values are players' ranks."""
        raise NotImplementedError()

    def apply_move(self, move: Move[S, D]) -> None:
        raise NotImplementedError()

    def fork(self, move: Optional[Move[S, D]] = None) -> Game[S, D]:
        raise NotImplementedError()
