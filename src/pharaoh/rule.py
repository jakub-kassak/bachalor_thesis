from itertools import permutations
from typing import List, Tuple, Callable, Dict, cast, Iterator, Optional, Iterable, Any

from pyrsistent import pvector
from pyrsistent.typing import PVector

from abstract_game.move import Action
from pharaoh.card import Card, Value, Suit, Deck
from pharaoh.move import PhMove, ChangeVariable, Condition, VariableCondition, CardsInHand, PlayCards, DrawCards, \
    ConditionCallable
from pharaoh.state import PhState


def raise_(e: Exception):
    raise e


def partial_permutations(cards: List[Card], size: int) -> Iterator[Tuple[Card]]:
    for i in range(size):
        for perm in permutations(cards, i + 1):
            yield cast(Tuple[Card], perm)


def increment_player_index(cards: Tuple[Card] | Tuple[()], player_count: int) -> List[Action[PhState]]:
    if len(cards) < 4:
        def f(i: int, pc=player_count) -> int:
            return (i + 1) % pc
        # f: Callable[[int], int] = lambda i, pc=player_count:
        return [ChangeVariable('i', f, f'(i + 1) % {player_count}')]
    return []


def increment_move_counter(_, __) -> List[Action[PhState]]:
    return [ChangeVariable('mc', lambda mc: mc + 1, "mc += 1")]


class PhRule:
    def __init__(self, move_factory: Callable[[Iterable[Condition[PhState]], Iterable[Action[PhState]],
                                               Optional[PlayCards]], PhMove] = PhMove):
        self._move_factory = move_factory

    def generate_moves(self, deck: Deck, player_count: int) -> List[PhMove]:
        raise NotImplementedError

    @staticmethod
    def _sort_deck(deck: Deck) -> Dict[Value, List[Card]]:
        v_dict: Dict[Value, List[Card]] = {val: [] for val in deck.values}
        for card in deck.cards:
            v_dict[card.value].append(card)
        return v_dict


class DrawRule(PhRule):
    def generate_moves(self, deck: Deck, player_count: int) -> List[PhMove]:
        empty_tuple: Tuple[()] = cast(Tuple[()], tuple())
        actions: List[Action[PhState]] = [
            DrawCards(),
            *increment_move_counter(empty_tuple, player_count),
            *increment_player_index(empty_tuple, player_count)
        ]
        return [self._move_factory([], actions, None)]


class MatchSuitRule(PhRule):
    def generate_moves(self, deck: Deck, player_count: int) -> List[PhMove]:
        v_dict = self._sort_deck(deck)
        moves: List[PhMove] = []
        for val in deck.values:
            for perm in partial_permutations(v_dict[val], len(v_dict[val])):
                suit: Suit = perm[0].suit
                conds: List[Condition] = [CardsInHand(perm), suit_conds[suit]]
                actions: List[Action[PhState]] = [
                    *increment_move_counter(perm, player_count),
                    *increment_player_index(perm, player_count)
                ]
                moves.append(self._move_factory(conds, actions, PlayCards(pvector(perm))))
        return moves


class MatchValueRule(PhRule):
    def generate_moves(self, deck: Deck, player_count: int) -> List[PhMove]:
        v_dict = self._sort_deck(deck)
        moves: List[PhMove] = []
        for val in deck.values:
            for perm in partial_permutations(v_dict[val], len(v_dict[val]) - 1):
                conds: List[Condition] = [CardsInHand(perm), val_conds[val]]
                actions: List[Action[PhState]] = [
                    *increment_move_counter(perm, player_count),
                    *increment_player_index(perm, player_count)
                ]
                moves.append(self._move_factory(conds, actions, PlayCards(pvector(perm))))
        return moves


def f_generator(arg: Suit | Value) -> ConditionCallable:
    def f(a1: Any, a2: Any = arg) -> bool:
        return a1 == a2
    return f


suit_conds: Dict[Suit, Condition] = {
    suit: VariableCondition('suit', f_generator(suit), f'suit=={suit}')
    for suit in Suit
}
val_conds: Dict[Value, Condition] = {
    val: VariableCondition('val', f_generator(val), f'val=={val}')
    for val in Value
}

SMALL_RULESET: PVector[PhRule] = pvector([
    MatchSuitRule(),
    MatchValueRule(),
    DrawRule()
])
