from typing import MutableSequence

from abstract_game.move import S, D, Move
from abstract_game.player import Player
from mcts.mcts import MCTS


class MCTSPlayer(Player[S, D]):
    def __init__(self, name: str, mcts: MCTS[S, D]):
        super().__init__(name)
        self._mcts = mcts

    def _play(self, state: S, legal_moves: MutableSequence[Move[S, D]]) -> Move[S, D]:
        return self._mcts.search(state)

    def _reset(self) -> None:
        self._mcts.reset()
