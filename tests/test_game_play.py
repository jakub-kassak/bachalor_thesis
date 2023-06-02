import contextlib
import os
import unittest
from copy import deepcopy

from config.data import load_configuration
from pharaoh.game_play import PhGamePlay
from pharaoh.card import Value


class TestGamePlay(unittest.TestCase):
    def prepare_test(self):
        gp = PhGamePlay(load_configuration("./test_config.yml"))
        with open(os.devnull, 'w', encoding='UTF-8') as devnull:
            with contextlib.redirect_stdout(devnull):
                gp.play()
                self.states = gp.game.state_history
                self.moves = gp.game.move_history
        self.first_state = self.states[0]

    def shuffle_replacement(self, *arg, **kwargs):
        pass

    @property
    def history_iterator(self):
        return zip(self.states[:-1], self.states[1:], self.moves)

    def assert_correct_move_saved(self):
        self.assertEqual(len(self.states), len(self.moves) + 1)
        state = deepcopy(self.states[0])
        for i in range(1, len(self.states)):
            self.moves[i-1].apply(state)
            self.assertEqual(self.states[i], state)

    def assert_cards_were_allowed(self):
        for prev_state, next_state, move in filter(lambda x: x[2].data, self.history_iterator):
            top = prev_state.dp[-1]
            new = move.data[0]
            if new.value == Value.OVER:
                pass
            elif top.value == Value.OVER:
                self.assertTrue(prev_state.suit == new.suit)
            else:
                self.assertTrue(top.value == new.value or top.suit == new.suit)
                self.assertTrue(top.suit == prev_state.suit)

    def assert_drawing(self):
        for prev_state, next_state, move in filter(lambda x: not x[2].data, self.history_iterator):
            i = prev_state.i
            self.assertTrue(len(prev_state.hands[i]) + 1 == len(next_state.hands[i])
                            or len(prev_state.st) == 0)

    def assert_cards_in_hand(self):
        for prev_state, next_state, move in filter(lambda x: x[2].data, self.history_iterator):
            i = prev_state.i
            self.assertTrue(all(c in prev_state.hands[i] for c in move.data))
            self.assertFalse(any(c in next_state.hands[i] for c in move.data))
            top_dp_cards = next_state.dp[-len(move.data):]
            self.assertTrue(move.data == top_dp_cards or len(next_state.dp) < len(move.data) and
                            all(c in top_dp_cards or c in next_state.st for c in move.data))

    def test_game(self):
        for i in range(10):
            with self.subTest(f"run {i}", i=i):
                self.prepare_test()
                self.assert_correct_move_saved()
                self.assert_drawing()
                self.assert_cards_in_hand()


if __name__ == '__main__':
    unittest.main()
