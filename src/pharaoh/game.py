from __future__ import annotations

from copy import deepcopy
from itertools import product
from typing import List, Iterable, Dict, Callable, Optional, Sequence

from pyrsistent import pvector, pbag
from pyrsistent.typing import PVector, PBag

from abstract_game.game import Game, GameException
from abstract_game.move import Move
from config.data import ConfigData
from pharaoh.card import Deck, Suit, Value, Card
from pharaoh.config_constants import *
from pharaoh.rule import PhRule, SMALL_RULESET
from pharaoh.state import PhState

MV = Move[PhState, PVector[Card]]


class PhGame(Game[PhState, PVector[Card]]):
    # Game should be a simple wrapper for GameState, which allows GameState manipulation according to rules.
    def __init__(self, state: PhState, moves: Sequence[MV], player_cnt: int):
        if player_cnt < 2:
            raise GameException('Not enough players!')
        self.n = player_cnt
        self._moves = moves
        self._state = state

    @property
    def state(self) -> PhState:
        return self._state

    @property
    def moves(self) -> Sequence[MV]:
        return self._moves

    def finished(self) -> bool:
        """Return true if only one player has cards or if there is livelock or if there is deadlock."""
        return self.state.lp_mc.count(-1) == 1 \
            or self.state.drw_cnt >= self.n * 3 \
            or self.state.lck_cnt == self.n

    def winners(self) -> List[int]:
        """Return a list of ints where player numbers are indices and values are players' ranks."""
        if not self.finished():
            raise GameException('Game has not finished yet, therefore there are no winners')
        lp_mc = list(self.state.lp_mc)
        for i in range(self.n):
            if lp_mc[i] == -1:
                lp_mc[i] = self.state.mc + self.n - i + 100 * (len(self.state.hands[i]) + 1)
                # If there is a tie, the player with the fewest cards wins.
                # If players have the same number of cards, the player with the higher index wins.
        order: List[int] = list(lp_mc)
        order.sort()
        mc_position: Dict[int, int] = {}
        for i in range(len(order)):
            mc_position[order[i]] = i
        positions: List[int] = [0] * self.n
        for i in range(self.n):
            positions[i] = mc_position[lp_mc[i]]
        return positions

    def apply_move(self, move: MV) -> None:
        if len(self.legal_moves()) == 1 and len(self.state.st) == 0:
            self.state.lck_cnt += 1
        else:
            self.state.lck_cnt = 0
        if len(move.data) == 0:
            self.state.drw_cnt += 1
        else:
            self.state.drw_cnt = 0
        move.apply(self.state)

    def fork(self, move: Optional[Move[PhState, PVector[Card]]] = None) -> Game[PhState, PVector[Card]]:
        game: Game[PhState, PVector[Card]] = PhGame(deepcopy(self.state), self._moves, self.n)
        if move:
            game.apply_move(move)
        return game


class FullGame(PhGame):
    def __init__(self, config: ConfigData, ruleset: Iterable[PhRule] = SMALL_RULESET,
                 game_state_factory: Callable = PhState.init_state):
        player_cnt: int = len(config['players'])
        self._deck: Deck = self._load_deck(config)
        self._game_state_factory = game_state_factory
        moves: PVector[MV] = self._create_moves(ruleset, self._deck, player_cnt)
        self._init_cards: int = config['init_cards']
        super().__init__(self._game_state_factory(self._deck, player_cnt, self._init_cards), moves, player_cnt)
        self.state_history: List[PhState] = [deepcopy(self.state)]
        self.move_history: List[MV] = []

    @staticmethod
    def _load_deck(config: Dict) -> Deck:
        suits: PVector[Suit] = pvector(Suit(entry[KW_NUMERIC]) for entry in config[KW_SUITS])
        values: PVector[Value] = pvector(Value(entry[KW_NUMERIC]) for entry in config[KW_VALUES])
        cards: PBag[Card] = pbag(Card(s, v) for s, v in product(suits, values))
        return Deck(cards, suits, values)

    @staticmethod
    def _create_moves(ruleset: Iterable[PhRule], deck: Deck, player_count: int) -> PVector[MV]:
        return pvector(mv for rule in ruleset for mv in rule.generate_moves(deck, player_count))

    def reset(self) -> None:
        self._state = self._game_state_factory(self._deck, self.n, self._init_cards)

    def apply_move(self, move: MV) -> None:
        super().apply_move(move)
        self.state_history.append(deepcopy(self.state))
        self.move_history.append(move)
