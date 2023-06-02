from __future__ import annotations

from typing import Callable, Iterable, Optional, List, Generic, TypeVar, cast

from pyrsistent import v
from pyrsistent.typing import PVector

from abstract_game.move import AbstractMove, Condition, Action
from pharaoh.card import Card, Value, Suit
from pharaoh.state import PhState, Pile, Hand

T = TypeVar("T", bound=Value | Suit | int)
ConditionCallable = Callable[[Pile | List[Hand] | List[int] | int], bool]


class PhMove(AbstractMove[PhState, PVector[Card]]):
    def __init__(self, conditions: Iterable[Condition[PhState]], actions: Iterable[Action[PhState]],
                 play_cards: Optional[PlayCards]):
        super().__init__(conditions, actions)
        self._cards: PVector[Card]
        if play_cards:
            self._cards = play_cards.cards
            self._actions.insert(0, play_cards)
        else:
            self._cards = v()

    @property
    def data(self) -> PVector[Card]:
        return self._cards

    def apply(self, state: PhState) -> None:
        prev_i: int = state.i
        super().apply(state)
        if len(state.hands[prev_i]) == 0:
            state.lp_mc[prev_i] = state.mc
        while state.lp_mc[state.i] != -1:
            state.i = (state.i + 1) % state.n


class VariableCondition(Condition[PhState]):
    def __init__(self, variable: str, condition: ConditionCallable, description: str = 'unknown'):
        self._desc = description
        self._var = variable
        self._cond = condition

    def test(self, state: PhState) -> bool:
        return self._cond(state[self._var])

    def _description(self) -> str:
        return f'variable={self._var}, cond=<{self._desc}>'


class CardsInHand(Condition[PhState]):
    def __init__(self, cards: Iterable[Card]):
        self._cards = set(cards)

    def test(self, state: PhState) -> bool:
        hand: Hand = state.hands[state.i]
        return all(c in hand for c in self._cards)

    def _description(self) -> str:
        return repr(self._cards)


class PlayCards(Action[PhState]):
    def __init__(self, cards: PVector[Card]):
        self._cards = cards

    @property
    def cards(self) -> PVector[Card]:
        return self._cards

    def apply(self, state: PhState) -> None:
        state.hands[state.i].difference_update(self._cards)
        state.dp.extend(self._cards)

    def _description(self) -> str:
        return repr(', '.join(map(repr, self._cards)))


class ChangeVariable(Action[PhState], Generic[T]):
    def __init__(self, variable: str, action: Callable[[T], T], description: str = 'unknown'):
        self._action = action
        self._desc = description
        self._var = variable

    def apply(self, state: PhState) -> None:
        val: T = cast(T, state[self._var])
        state[self._var] = self._action(val)

    def _description(self) -> str:
        return f'variable={self._var}, action=<{self._desc}>'


class DrawCards(Action[PhState]):
    """Draws last card in stock (card with index -1)."""

    def apply(self, state: PhState) -> None:
        if not state.st:
            return
        cards = state.st.pop()
        state.hands[state.i].add(cards)

    def _description(self) -> str:
        return ''


play_heart_ix_x = PlayCards(v(Card(Suit.HEART, Value.IX), Card(Suit.HEART, Value.X)))
increase_player_index = ChangeVariable('i', lambda x: (x + 1) % 2, '(i + 1) % 2')
increase_mc = ChangeVariable('mc', lambda x: x + 1, 'mc + 1')
draw_card = DrawCards()

move2 = PhMove(v(VariableCondition('mc', lambda mc: mc == 1)), v(draw_card), None)
