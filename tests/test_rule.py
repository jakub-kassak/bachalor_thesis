import unittest
from copy import deepcopy
from itertools import product
from typing import Iterable, List

from pyrsistent import PBag, pbag, pvector

from pharaoh.card import Deck, Card, Suit, Value, SUITS, VALUES
from pharaoh.move import PhMove
from pharaoh.state import PhState
from pharaoh.rule import MatchSuitRule, MatchValueRule, DrawRule


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.called = 0

    @staticmethod
    def create_deck(values: Iterable[Value], suits: Iterable[Suit]):
        cards: PBag[Card] = pbag(Card(suit, val) for suit, val in product(suits, values))
        return Deck(cards, pvector(suits), pvector(values))

    def assert_no_change(self, state1: PhState, state2: PhState, *args: str):
        for key in args:
            self.assertEqual(state1[key], state2[key])

    def assert_change(self, state: PhState, **kwargs):
        for key, value in kwargs.items():
            self.assertEqual(value, state[key])

    def assert_change_in_hands_dp_mc(self, state1: PhState, state2: PhState, move: PhMove):
        if move.data:
            self.assertEqual(len(state1.hands[state1.i]) - len(move.data), len(state2.hands[state1.i]))
        self.assertFalse(any(c in state2.hands[state1.i] for c in move.data))
        self.assertTrue(all(c in state2.hands[state1.i] for c in state1.hands[state1.i] if c not in move.data))
        self.assertEqual(state1.mc + 1, state2.mc)
        if self.called == 0:
            dp = deepcopy(state1.dp)
            dp.extend(move.data)
            self.assertEqual(dp, state2.dp)

    def assert_and_make_legal_moves(self, count: int, state: PhState, moves: List[PhMove]) -> List[PhMove]:
        legal_moves: List[PhMove] = [mv for mv in moves if mv.test(state)]
        self.assertEqual(count, len(legal_moves))
        return legal_moves

    def test_draw_rule(self):
        deck = Deck.create(SUITS[:2], VALUES[:3])
        cards_list = list(deck.cards)
        player_cnt = 2
        rule = DrawRule()
        moves = rule.generate_moves(deck, player_cnt)
        state = PhState(
            dp=cards_list[0:1],
            st=cards_list[1:2],
            hands=(set(cards_list[2:4]), set(cards_list[4:6])),
            i=0,
            mc=0,
            lp_mc=(-1, -1),
            lck_cnt=0,
            drw_cnt=0
        )
        # state = GameState.init_state(deck, player_cnt, 2, 0, self.fake_shuffle)
        self.assertEqual(1, len(moves))
        move = moves[0]
        self.assertTrue(move.test(state))
        state2 = deepcopy(state)
        move.apply(state2)
        self.assertEqual(len(state.st) - 1, len(state2.st))
        expected_hand = state.hands[state.i] | {state.st[0]}
        actual_hand = state2.hands[state.i]
        self.assertEqual(expected_hand, actual_hand)
        self.assertEqual(1, state2.i)

        self.assertTrue(move.test(state2))
        state3 = deepcopy(state2)
        move.apply(state3)
        self.assertEqual(state2.st, state3.st)
        expected_hand = state2.hands[state2.i]
        actual_hand = state3.hands[state2.i]
        self.assertEqual(expected_hand, actual_hand)
        self.assertEqual(0, state3.i)

    def test_match_suit_small(self):
        deck = Deck.create([Suit.HEART], range(Value.VII, Value.UNDER))
        cards_list = list(deck.cards)
        cards_list.sort()
        rule = MatchSuitRule()
        player_cnt = 2
        moves: List[PhMove] = rule.generate_moves(deck, player_cnt)
        self.assertEqual(4, len(moves))
        self.assertTrue(all(len(mv.data) == 1 for mv in moves))
        state1 = PhState(
            dp=cards_list[0:1],
            st=cards_list[1:2],
            hands=(set(cards_list[2:3]), set(cards_list[3:])),
            i=0,
            mc=0,
            lp_mc=(-1, -1),
            lck_cnt=0,
            drw_cnt=0
        )
        legal_moves: List[PhMove] = self.assert_and_make_legal_moves(1, state1, moves)

        next_mv = legal_moves[0]
        state2 = deepcopy(state1)
        next_mv.apply(state2)
        self.assert_no_change(state1, state2, 'st')
        self.assert_change_in_hands_dp_mc(state1, state2, next_mv)
        self.assert_change(
            state2,
            val=next_mv.data[0].value,
            i=1,
            lp_mc=[1, -1]
        )

    def test_match_suit_medium(self):
        deck = Deck.create((Suit.HEART, Suit.BELL), (Value.VII, Value.IX, Value.ACE))
        cards_list = list(deck.cards)
        cards_list.sort()
        rule = MatchSuitRule()
        moves = rule.generate_moves(deck, 2)
        self.assertEqual(2 * (3 + 3), len(moves))
        self.assertTrue(all(1 <= len(mv.data) <= 2 for mv in moves))

        top = cards_list[0]  # (HEART, VII)
        state1 = PhState(
            dp=[top],
            st=cards_list[1:2],  # (HEART, IX)
            hands=(set(cards_list[2:-1]),  # (HEART, ACE), (BELL, VII), (BELL, IX)
                   set(cards_list[-1:])),  # (BELL, ACE)
            i=0,
            mc=0,
            lp_mc=(-1, -1),
            lck_cnt=0,
            drw_cnt=0
        )
        legal_moves: List[PhMove] = self.assert_and_make_legal_moves(1, state1, moves)
        next_mv = legal_moves[0]  # PLAY (HEART, ACE)
        state2 = deepcopy(state1)
        next_mv.apply(state2)
        self.assert_no_change(state1, state2, 'st', 'suit', 'lp_mc')
        self.assert_change_in_hands_dp_mc(state1, state2, next_mv)
        self.assert_change(state2, i=1)
        self.assert_and_make_legal_moves(0, state2, moves)

        state1.hands[0] = set(cards_list[2:-2] + [top])  # (HEART, VII), (HEART, ACE), (BELL, VII)
        legal_moves: List[PhMove] = self.assert_and_make_legal_moves(3, state1, moves)
        next_mv = next(mv for mv in legal_moves if len(mv.data) > 1)  # PLAY (HEART, VII), (BELL, VII)
        state2 = deepcopy(state1)
        next_mv.apply(state2)
        self.assert_no_change(state1, state2, 'lp_mc')
        self.assert_change_in_hands_dp_mc(state1, state2, next_mv)
        self.assertEqual(next_mv.data[-1], state2.dp[-1])
        self.assert_change(state2, i=1)

    def test_match_suit_big(self):
        deck = Deck.create(SUITS, VALUES)
        cards_list = list(deck.cards)
        cards_list.sort()
        rule = MatchSuitRule()
        moves = rule.generate_moves(deck, 4)
        self.assertEqual(4 * (8 * 3 * 2 * 1 + 8 * 3 * 2 + 8 * 3 + 8), len(moves))
        top = Card(Suit.HEART, Value.VIII)
        state1 = PhState(
            dp=[top],
            st=[Card(Suit.HEART, Value.KING), Card(Suit.HEART, Value.ACE), Card(Suit.ACORN, Value.UNDER),
                Card(Suit.ACORN, Value.OVER), Card(Suit.ACORN, Value.KING), Card(Suit.ACORN, Value.ACE),
                Card(Suit.LEAF, Value.VII), Card(Suit.LEAF, Value.VIII), Card(Suit.LEAF, Value.X),
                Card(Suit.LEAF, Value.UNDER), Card(Suit.LEAF, Value.OVER), Card(Suit.BELL, Value.VIII),
                Card(Suit.BELL, Value.X), Card(Suit.BELL, Value.UNDER), Card(Suit.BELL, Value.OVER),
                Card(Suit.BELL, Value.KING), Card(Suit.BELL, Value.ACE), Card(Suit.HEART, Value.OVER),
                Card(Suit.ACORN, Value.VIII)],
            hands=({Card(Suit.HEART, Value.VII), Card(Suit.ACORN, Value.VII)},
                   {Card(Suit.LEAF, Value.KING), Card(Suit.LEAF, Value.ACE), Card(Suit.BELL, Value.VII)},
                   {},
                   {Card(Suit.HEART, Value.IX), Card(Suit.HEART, Value.X), Card(Suit.HEART, Value.UNDER),
                    Card(Suit.BELL, Value.IX), Card(Suit.LEAF, Value.IX), Card(Suit.ACORN, Value.IX),
                    Card(Suit.ACORN, Value.X)}),
            i=3,
            mc=11,
            lp_mc=(-1, -1, 4, -1),
            lck_cnt=0,
            drw_cnt=0
        )
        # 3 + 1*3*2*1 + 1*3*2 + 1*3 + 1*1 = 19
        legal_moves1: List[PhMove] = self.assert_and_make_legal_moves(19, state1, moves)
        next_mv1: PhMove = next(mv for mv in legal_moves1 if len(mv.data) == 4 and mv.data[-1].suit == Suit.ACORN)
        state2 = deepcopy(state1)
        next_mv1.apply(state2)
        self.assert_no_change(state1, state2, 'st', 'lp_mc')
        self.assert_change_in_hands_dp_mc(state1, state2, next_mv1)
        self.assert_change(
            state2,
            suit=Suit.ACORN,
            val=Value.IX
        )

        legal_moves2: List[PhMove] = self.assert_and_make_legal_moves(2, state2, moves)
        next_mv2: PhMove = next(mv for mv in legal_moves2 if len(mv.data) == 2)
        state3 = deepcopy(state2)
        next_mv2.apply(state3)
        self.assert_no_change(state2, state3, 'st', 'lp_mc')
        self.assert_change_in_hands_dp_mc(state2, state3, next_mv2)
        self.assert_change(
            state3,
            suit=Suit.HEART,
            val=Value.X,
            i=0
        )

        legal_moves3: List[PhMove] = self.assert_and_make_legal_moves(2, state3, moves)
        next_mv3: PhMove = next(mv for mv in legal_moves3 if len(mv.data) == 2)
        state4 = deepcopy(state3)
        next_mv3.apply(state4)
        self.assert_no_change(state3, state4, 'st')
        self.assert_change_in_hands_dp_mc(state3, state4, next_mv3)
        self.assert_change(
            state4,
            i=1,
            lp_mc=[14, -1, 4, -1]
        )
        self.assert_and_make_legal_moves(0, state4, moves)

    def test_match_value_small(self):
        deck = Deck.create(SUITS, [Value.VIII])
        cards_list = list(deck.cards)
        cards_list.sort()
        rule = MatchValueRule()
        moves = rule.generate_moves(deck, 2)
        # 4*3*2 + 4*3 + 4 = 64
        self.assertEqual(40, len(moves))
        top = Card(Suit.HEART, Value.VIII)
        state1 = PhState(
            dp=[top],
            st=[Card(Suit.BELL, Value.VIII)],
            hands=(({Card(Suit.LEAF, Value.VIII)}), ({Card(Suit.ACORN, Value.VIII)})),
            i=0,
            mc=0,
            lp_mc=(-1, -1),
            lck_cnt=0,
            drw_cnt=0
        )
        legal_moves: List[PhMove] = self.assert_and_make_legal_moves(1, state1, moves)
        next_mv = legal_moves[0]
        state2 = deepcopy(state1)
        next_mv.apply(state2)
        self.assert_no_change(state1, state2, 'st')
        dp = deepcopy(state1.dp)
        dp.extend(next_mv.data)
        self.assert_change(
            state2,
            dp=dp,
            i=1,
            mc=1,
            lp_mc=[1, -1]
        )

    def test_match_value_medium(self):
        deck = Deck.create(SUITS, [Value.VIII, Value.IX, Value.ACE])
        cards_list = list(deck.cards)
        cards_list.sort()
        rule = MatchValueRule()
        moves: List[PhMove] = rule.generate_moves(deck, 2)
        # 3 * (4*3*2 + 4*3 + 4) = 120
        self.assertEqual(120, len(moves))

        top = Card(Suit.HEART, Value.ACE)
        state1 = PhState(
            dp=[top],
            st=[Card(Suit.HEART, Value.VIII), Card(Suit.HEART, Value.IX), Card(Suit.BELL, Value.VIII),
                Card(Suit.LEAF, Value.VIII), Card(Suit.LEAF, Value.IX)],
            hands=({Card(Suit.LEAF, Value.ACE)},
                   {Card(Suit.BELL, Value.ACE), Card(Suit.ACORN, Value.ACE), Card(Suit.ACORN, Value.VIII),
                    Card(Suit.ACORN, Value.IX), Card(Suit.BELL, Value.IX)}),
            i=1,
            mc=0,
            lp_mc=(-1, -1),
            lck_cnt=0,
            drw_cnt=0
        )
        legal_moves1: List[PhMove] = self.assert_and_make_legal_moves(4, state1, moves)
        next_mv1 = next(mv for mv in legal_moves1 if len(mv.data) == 2 and mv.data[-1].suit == Suit.ACORN)
        state2 = deepcopy(state1)
        next_mv1.apply(state2)
        self.assert_no_change(state1, state2, 'st', 'lp_mc')
        self.assert_change_in_hands_dp_mc(state1, state2, next_mv1)
        self.assert_change(
            state2,
            i=0,
        )

        legal_moves2: List[PhMove] = self.assert_and_make_legal_moves(1, state2, moves)
        next_mv2: PhMove = legal_moves2[0]
        state3 = deepcopy(state2)
        next_mv2.apply(state3)
        self.assert_no_change(state2, state3, 'st')
        self.assert_change_in_hands_dp_mc(state1, state2, next_mv1)
        self.assert_change(
            state3,
            i=1,
            lp_mc=[2, -1]
        )
        self.assert_and_make_legal_moves(0, state3, moves)


if __name__ == '__main__':
    unittest.main()
