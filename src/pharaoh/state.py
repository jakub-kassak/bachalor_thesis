from __future__ import annotations

from random import shuffle
from typing import List, Callable, cast, Set, Iterable

from abstract_game.state import State
from pharaoh.card import Suit, Card, Value, Deck

Pile = List[Card]
Hand = Set[Card]


class PhState(State):
    def __init__(self, dp: Pile, st: Pile, hands: Iterable[Hand], mc: int, lp_mc: Iterable[int], i: int, lck_cnt: int,
                 drw_cnt: int):
        self.hands = list(hands)
        super(PhState, self).__init__(i, len(self.hands), mc)
        self.dp = dp
        self.st = st
        self.lp_mc = list(lp_mc)
        self.lck_cnt = lck_cnt
        self.drw_cnt = drw_cnt

    @property
    def suit(self) -> Suit:
        return self.dp[-1].suit

    @property
    def val(self) -> Value:
        return self.dp[-1].value

    def __getitem__(self, name: str) -> Pile | List[Hand] | List[int] | int | Suit | Value:
        return self.__getattribute__(name)

    def __setitem__(self, key, value):
        super().__setattr__(key, value)

    def __eq__(self, other):
        if not super().__eq__(other):
            return False
        other: PhState = cast(PhState, other)
        return (self.dp == other.dp and self.st == other.st and (h1 == h2 for h1, h2 in zip(self.hands, other.hands))
                and self.lp_mc == other.lp_mc
                and self.i == other.i and self.mc == other.mc)

    @classmethod
    def init_state(cls, deck: Deck, player_cnt: int, init_cards: int,
                   mix_cards: Callable[[List[Card]], None] = shuffle) -> PhState:
        cards_list: List[Card] = list(deck.cards)
        mix_cards(cards_list)
        top: Card = cards_list.pop()
        hands: List[Hand] = [set() for _ in range(player_cnt)]
        for i in range(player_cnt):
            for _ in range(init_cards):
                if len(cards_list) == 0:
                    raise Exception("not enough cards for everyone")
                hands[i].add(cards_list.pop())
        return cls(
            dp=[top, ],
            st=cards_list,
            hands=hands,
            i=0,
            mc=0,
            lp_mc=[-1] * player_cnt,
            lck_cnt=0,
            drw_cnt=0
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(" \
               f"dp={self.dp}, " \
               f"st={self.st}, " \
               f"hands={self.hands}, " \
               f"mc={self.mc}, " \
               f"lp_mc={self.lp_mc}, " \
               f"i={self.i})"
