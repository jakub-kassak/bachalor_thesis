from copy import deepcopy
from random import shuffle
from typing import List, Callable, MutableSequence, Tuple, Sequence, cast

from pyrsistent import pvector
from pyrsistent.typing import PVector

from abstract_game.game_play import GamePlay, GamePlayException
from abstract_game.move import Move, S, D
from abstract_game.player import Player, RandomPlayer, BackTrack, BackTrackMemo
from config.data import ConfigData, PlayerData, MCTSData
from mcts.mcts import MCTS, my_shuffle, identity
from mcts.player import MCTSPlayer
from pharaoh.card import Card
from pharaoh.game import FullGame, PhGame
from pharaoh.player import HumanPlayer, BiggestTuplePlayer, SmallestTuplePlayer, AnnoyingPlayer, biggest_tuple, \
    smallest_tuple
from pharaoh.state import PhState


class PhGamePlay(GamePlay[PhState, PVector[Card]]):
    def __init__(self, config: ConfigData, observer: Callable[[], None] = lambda: None,
                 load_move: Callable[[PhState, int], PVector[Card]] = lambda _, __: pvector()):
        self._game = FullGame(config)
        super().__init__(config, observer, load_move)

    def reset(self) -> None:
        self.game.reset()
        if self.game.n == 2:
            self.players[0], self.players[1] = self.players[1], self.players[0]
        else:
            shuffle(self.players)
        for player in self.players:
            player.reset()

    @property
    def game(self) -> FullGame:
        return self._game

    @staticmethod
    def _get_move_sort(player_data) -> Callable[[MutableSequence[Move[S, D]]], MutableSequence[Move[S, D]]]:
        heuristic: str = player_data['heuristic']
        if heuristic == "my_shuffle":
            return cast(Callable[[MutableSequence[Move[S, D]]], MutableSequence[Move[S, D]]], my_shuffle)
        elif heuristic == "identity":
            return cast(Callable[[MutableSequence[Move[S, D]]], MutableSequence[Move[S, D]]], identity)
        elif heuristic == 'biggest_tuple':
            return cast(Callable[[MutableSequence[Move[S, D]]], MutableSequence[Move[S, D]]], biggest_tuple)
        elif heuristic == "smallest_tuple":
            return cast(Callable[[MutableSequence[Move[S, D]]], MutableSequence[Move[S, D]]], smallest_tuple)
        else:
            raise GamePlayException("unknown heuristic")

    def _new_mcts_player(self, player: PlayerData) -> Player[PhState, PVector[Card]]:
        name: str = player['name']
        if not player['data']:
            raise GamePlayException('Player with type=mcts without MCTSData.')
        player_data: MCTSData = player['data']
        mcts: MCTS[PhState, PVector[Card]] = MCTS(
            deepcopy(self.game.state),
            self.game.moves,
            PhGame,
            player_data['iterations'],
            player_data['width'],
            player_data['limit'],
            player_data['expl_const'],
            self._get_move_sort(player_data),
            player_data['heuristic']
        )
        if not self._mcts:
            self._mcts = mcts
        return MCTSPlayer(name, mcts)

    def _back_track_args(self, player: PlayerData) -> Tuple[str, Sequence[Move], Callable, int, Callable]:
        name: str = player['name']
        if not player['data']:
            raise GamePlayException('Player with type=backtrack without MCTSData.')
        limit: int = player['data']['limit']
        return name, self.game.moves, self._get_move_sort(player['data']), limit, PhGame

    def _new_player(self, player: PlayerData, load_move: Callable[[PhState, int],
                    PVector[Card]]) -> Player[PhState, PVector[Card]]:
        type_, name = player['type'], player['name']
        if type_ == 'human':
            return HumanPlayer(name, load_move)
        if type_ == 'random':
            return RandomPlayer(name)
        if type_ == 'ai1':
            return BiggestTuplePlayer(name)
        if type_ == 'ai2':
            return SmallestTuplePlayer(name)
        if type_ == 'apr':
            return AnnoyingPlayer(RandomPlayer(name))
        if type_ == 'ap1':
            return AnnoyingPlayer(BiggestTuplePlayer(name))
        if type_ == 'ap2':
            return AnnoyingPlayer(SmallestTuplePlayer(name))
        if type_ == 'mcts':
            return self._new_mcts_player(player)
        if type_ == 'backtrack':
            return BackTrack(*self._back_track_args(player))
        if type_ == 'backtrack_memo':
            return BackTrackMemo(*self._back_track_args(player))
        raise GamePlayException(f"Unsupported player type '{type_}")

    def _load_players(self, config: ConfigData, load_move: Callable[[PhState, int], PVector[Card]]) -> List[Player]:
        if len(config['players']) < 2:
            raise GamePlayException("not enough players")
        return [self._new_player(player, load_move) for player in config["players"]]
