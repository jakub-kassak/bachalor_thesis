from typing import List, Callable, Dict

from abstract_game.move import Move, MoveException
from pharaoh.move import PhMove
from unittest import TestCase
from unittest.mock import Mock

from pharaoh.state import PhState


def mock_condition(ret_val: bool) -> Mock:
    cond: Mock = Mock()
    cond.test.return_value = ret_val
    return cond


def mock_action(apply: Callable = lambda _: None) -> Mock:
    action: Mock = Mock()
    action.apply = apply
    return action


def mock_hand(cnt: int) -> Mock:
    player: Mock = Mock()
    player.__len__ = Mock(return_value=cnt)
    return player


def mock_state(i: int, mc: int, hands: List) -> Mock:
    state: Mock = Mock()
    state.i = i
    state.hands = hands
    state.n = len(hands)
    state.lp_mc = [-1] * state.n
    state.mc = mc
    return state


class TestMove(TestCase):
    def setUp(self) -> None:
        self.conds = [mock_condition(True), mock_condition(True)]
        self.acts = [mock_action()]
        self.cards_action = mock_action()
        self.state = mock_state(0, 0, [mock_hand(1), mock_hand(1)])
        self.create_move()

    def create_move(self):
        self.move: Move[PhState] = PhMove(self.conds, self.acts, self.cards_action)

    def test_test(self):
        self.assertTrue(self.move.test(self.state))
        self.conds[0].test.assert_called_with(self.state)
        self.conds.append(mock_condition(False))
        self.create_move()
        self.assertFalse(self.move.test(self.state))

    def assert_values(self, d: Dict):
        for k, v in d.items():
            if k == "i":
                self.assertEqual(v, self.state.i)
            elif k == "hands":
                self.assertEqual(v, self.state.hands)
            elif k == "n":
                self.assertEqual(v, self.state.n)
            elif k == "lp_mc":
                self.assertEqual(v, self.state.lp_mc)
            elif k == "mc":
                self.assertEqual(v, self.state.mc)

    def test_apply(self):
        d = {
            "i": self.state.i,
            "hands": list(self.state.hands),
            "n": self.state.n,
            "lp_mc": self.state.lp_mc,
            "mc": self.state.mc,
        }
        self.move.apply(self.state)
        self.assert_values(d)

        def f(state) -> None:
            state.i = 1

        d["i"] = 1
        self.acts.append(mock_action(f))
        self.create_move()
        self.move.apply(self.state)
        self.assert_values(d)

        self.state = mock_state(0, 10, [mock_hand(0), mock_hand(0), mock_hand(1), mock_hand(1)])
        d = {
            "i": 2,
            "hands": list(self.state.hands),
            "n": self.state.n,
            "mc": self.state.mc,
            "lp_mc": [self.state.mc, 1, -1, -1]
        }
        self.state.lp_mc[1] = 1
        self.create_move()
        self.move.apply(self.state)
        self.assert_values(d)

        self.conds.append(mock_condition(False))
        self.create_move()
        with self.assertRaises(MoveException):
            self.move.apply(self.state)
