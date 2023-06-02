from abc import ABC
from typing import TypeVar, Generic, Iterable, List

from abstract_game.state import State

S = TypeVar("S", bound=State)
D = TypeVar('D')


class Condition(Generic[S]):
    def test(self, state: S) -> bool:
        raise NotImplementedError

    def _description(self) -> str:
        raise NotImplementedError

    def __repr__(self):
        return f'{self.__class__.__name__}({self._description()})'


class Action(Generic[S]):
    def apply(self, state: S) -> None:
        raise NotImplementedError

    def _description(self) -> str:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self._description()})'


class MoveException(Exception):
    pass


class Move(Generic[S, D]):
    def test(self, state: S) -> bool:
        raise NotImplementedError()

    def apply(self, state: S) -> None:
        raise NotImplementedError()

    @property
    def data(self) -> D:
        raise NotImplementedError()


class AbstractMove(Move[S, D], ABC):
    def __init__(self, conditions: Iterable[Condition[S]], actions: Iterable[Action[S]]):
        self._conds: List[Condition[S]] = list(conditions)
        self._actions: List[Action[S]] = list(actions)

    def test(self, state: S) -> bool:
        return all(c.test(state) for c in self._conds)

    def apply(self, state: S) -> None:
        if not self.test(state):
            raise MoveException("Conditions are not fulfilled.")
        for a in self._actions:
            a.apply(state)
        return None

    def __repr__(self) -> str:
        return f'Move(conds={self._conds}, actions={self._actions})'
