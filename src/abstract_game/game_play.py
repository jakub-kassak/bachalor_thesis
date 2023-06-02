from copy import deepcopy
from typing import Callable, List, Optional, Generic, Tuple

from pyrsistent import pvector
from pyrsistent.typing import PVector

from abstract_game.game import Game
from abstract_game.move import S, D
from config.data import ConfigData
from mcts.mcts import MCTS
from pharaoh.card import Card
from pharaoh.player import Player


class GamePlayException(Exception):
    pass


class GamePlay(Generic[S, D]):
    # GamePlay is meant as a simple way to run game - business logic.
    def __init__(self, config: ConfigData, observer: Callable[[], None] = lambda: None,
                 load_move: Callable[[S, int], PVector[Card]] = lambda _, __: pvector()):
        self._observer = observer
        self._mcts: Optional[MCTS[S, D]] = None
        self._players: List[Player[S, D]] = self._load_players(config, load_move)

    def _load_players(self, config_file: ConfigData, load_move: Callable[[S, int], PVector[Card]]) -> List[Player]:
        raise NotImplementedError()

    @property
    def _current_player(self) -> Player[S, D]:
        return self._players[self.game.state.i]

    @property
    def players(self):
        return self._players

    @property
    def mcts(self) -> Optional[MCTS[S, D]]:
        return self._mcts

    @property
    def game(self) -> Game[S, D]:
        raise NotImplementedError()

    def play(self) -> List[int]:
        while not self.game.finished():
            self.game.apply_move(self._current_player.play(deepcopy(self.game.state), self.game.legal_moves()))
            self._observer()
        return self.game.winners()

    def reset(self) -> None:
        raise NotImplementedError()

    def player_ranking(self) -> List[Tuple[Player, int]]:
        winners: List[int] = self.game.winners()
        result: List[Tuple[int, int]] = list((i, winners[i] + 1) for i in range(len(self.players)))
        result.sort(key=lambda x: x[1])
        return list((self.players[i], pos) for i, pos in result)
