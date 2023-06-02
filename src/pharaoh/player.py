import itertools
from random import shuffle
from typing import Callable, Dict, List, Set, MutableSequence

from pyrsistent.typing import PVector

from abstract_game.move import Move
from abstract_game.player import Player
from pharaoh.card import Value, Card
from pharaoh.state import PhState


PharaohMv = Move[PhState, PVector[Card]]


def biggest_tuple(moves: MutableSequence[PharaohMv]) -> MutableSequence[PharaohMv]:
    return sorted(same_value_together(moves), key=lambda m: -len(m.data)) + find_draw(moves)


class BiggestTuplePlayer(Player[PhState, PVector[Card]]):
    def _play(self, state: PhState, legal_moves: MutableSequence[PharaohMv]) -> PharaohMv:
        return biggest_tuple(legal_moves)[0]


def same_value_together(moves: MutableSequence[PharaohMv]) -> List[PharaohMv]:
    d: Dict[Value, List[PharaohMv]] = {}
    for mv in (mv for mv in moves if len(mv.data)):
        value: Value = mv.data[0].value
        if value not in d or len(mv.data) > len(d[value][0].data):
            d[value] = [mv]
        elif len(mv.data) == len(d[value][0].data):
            d[value].append(mv)
    return sum(d.values(), [])


def find_draw(moves):
    return [mv for mv in moves if len(mv.data) == 0]


def smallest_tuple(moves: MutableSequence[PharaohMv]) -> MutableSequence[PharaohMv]:
    return sorted(same_value_together(moves), key=lambda x: len(x.data)) + find_draw(moves)


def draw_then_smallest_tuple(moves: MutableSequence[PharaohMv]) -> MutableSequence[PharaohMv]:
    moves = smallest_tuple(moves)
    if len(moves) > 1:
        moves[0], moves[-1] = moves[-1], moves[0]
    return moves


def randomized_complete_value(moves: MutableSequence[PharaohMv]) -> MutableSequence[PharaohMv]:
    moves = same_value_together(moves) + find_draw(moves)
    shuffle(moves)
    return moves


class SmallestTuplePlayer(Player[PhState, PVector[Card]]):
    def _play(self, state: PhState, legal_moves: MutableSequence[PharaohMv]) -> PharaohMv:
        return smallest_tuple(legal_moves)[0]


class AnnoyingPlayer(Player[PhState, PVector[Card]]):
    def __init__(self, player: Player[PhState, PVector[Card]]):
        super().__init__(player.name)
        self._player = player

    @staticmethod
    def suit_value_tester(hand: Set[Card]) -> Callable[[PVector[Card]], bool]:
        suits = frozenset(c.suit for c in hand)
        vals = frozenset(c.value for c in hand)

        def test(cards: PVector[Card], s=suits, v=vals) -> bool:
            if not cards:
                return False
            return cards[-1].suit not in s and cards[-1].value not in v

        return test

    def _play(self, state: PhState, legal_moves: MutableSequence[PharaohMv]) -> PharaohMv:
        next_i: int = (state.i + 1) % state.n
        test: Callable[[PVector[Card]], bool] = self.suit_value_tester(state.hands[next_i])
        moves = [m for m in legal_moves if test(m.data)]
        if moves:
            return self._player.play(state, moves)
        return self._player.play(state, legal_moves)


class HumanPlayer(Player[PhState, PVector[Card]]):
    def __init__(self, name: str, load_move: Callable[[PhState, int], PVector[Card]]):
        super().__init__(name)
        self._load_move = load_move

    def _play(self, state: PhState, legal_moves: MutableSequence[PharaohMv]) -> PharaohMv:
        for i in itertools.count():
            cards_list = self._load_move(state, i)
            moves2: List[PharaohMv] = list(mv for mv in legal_moves if mv.data == cards_list)
            if moves2:
                return moves2[0]
        raise Exception()
