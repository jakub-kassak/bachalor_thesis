from typing import List, Dict, Tuple

from pyrsistent import pvector
from pyrsistent.typing import PVector

from abstract_game.move import Move
from config.data import load_configuration, ConfigData
from pharaoh.card import Card, symbols
from pharaoh.config_constants import KW_PREFIX, KW_DELIMITER, KW_SUFFIX
from pharaoh.game_play import PhGamePlay
from pharaoh.state import PhState

CURSOR: str = '\033[32m>>>\033[m '
EMPTY_STR: str = '""'


class EndGame(Exception):
    pass


class ConsolePharaoh:
    DEBUG_OUTPUT: bool = False

    def __init__(self, config_file: str):
        config: ConfigData = load_configuration(config_file)
        self._symbols: Dict[str, str] = self._load_symbols(config)
        self._gp: PhGamePlay = PhGamePlay(config, self._print_new_state, self._load_move_from_console)

    @staticmethod
    def _load_symbols(config: ConfigData) -> Dict[str, str]:
        symbols2: Dict[str, str] = symbols.copy()
        for d in config['suits']:
            symbols2[d['name']] = d['symbol']
        return symbols2

    @property
    def _state(self) -> PhState:
        return self._gp.game.state

    @property
    def _top_card(self):
        return self._state.dp[-1]

    def _print_cards(self, cards: List[Card]):
        print('Your cards are:')
        for j in range(len(cards)):
            print(f'\t[{j}] {self._card_to_str(cards[j])}')

    def _card_to_str(self, card: Card) -> str:
        pref: str = self._symbols[KW_PREFIX]
        delim: str = self._symbols[KW_DELIMITER]
        suf: str = self._symbols[KW_SUFFIX]
        suit: str = self._symbols[card.suit.name]
        val: str = self._symbols[card.value.name]
        return pref + suit + delim + val + suf

    def _print_top_card(self):
        print(f'top card: {self._top_card}')

    def _load_move_from_console(self, state: PhState, cnt: int) -> PVector[Card]:
        hand: List[Card] = list(state.hands[state.i])
        hand.sort(key=lambda card: (card.value, card.suit))
        if cnt == 0:
            print("It's your turn.")
        else:
            print("That was not a valid move. Please enter a your command.")
        self._print_cards(hand)
        while True:
            command = input(CURSOR).split()
            if len(command) == 0:
                continue
            elif command[0] == 'play' and len(command) == 1:
                print('please enter at least one number')
            elif command[0] == 'play':
                try:
                    card_nums: List[int] = [int(x) for x in command[1:]]
                except ValueError:
                    print('please enter numbers with base 10')
                    continue
                card_nums.sort()
                if card_nums[0] < 0 or card_nums[-1] > len(hand) - 1:
                    print(f'please enter numbers in range [0, {len(hand)})')
                    continue
                card_nums = [int(x) for x in command[1:]]
                return pvector(hand[i] for i in card_nums)
            elif command[0] == "draw" or command[0] == "skip":
                return pvector()
            elif command[0] == "status":
                player_stats = (f' {self._gp.players[j].name}: {len(state.hands[j])}'
                                for j in range(state.n))
                print(f'stock: {len(self._state.st)},' + ", ".join(player_stats))
            elif command[0] == "top":
                self._print_top_card()
            elif command[0] == "hand" or command[0] == "cards":
                self._print_cards(hand)
            elif command[0] == "state":
                print(self._state)
            elif command[0] == "help":
                print("enter 'play ' number for each card you want to play seperated by space (e.g. 'play 1 4 2')")
                print("enter 'draw' to draw cards")
                print("enter 'status' to see, how many cards each player has")
                print("enter 'top' to show the top card")
                print("enter 'hand' to see your cards")
                print("table of symbols:")
                for key, value in symbols.items():
                    if key not in ("__PREFIX__", "__SUFFIX__", "__DELIMITER__"):
                        print(f"\t{key} = {value if value else EMPTY_STR}")
            elif command[0] == 'end':
                print("the game was ended prematurely")
                raise EndGame("Player wants to end the game")
            else:
                print("unknown command")

    def _print_new_state(self):
        last_move: Move[PhState, PVector[Card]] = self._gp.game.move_history[-1]
        prev_i: int = self._gp.game.state_history[-2].i % self._state.n
        prev_pl = self._gp.players[prev_i]
        if len(last_move.data) == 0:
            if self._state.mc >= 1 and len(self._gp.game.state_history[-2].st) > 0:
                print(f'{prev_pl.name}:\tDRAW')
            else:
                print(f'{prev_pl.name}:\tSKIP')
        else:
            print(f'{prev_pl.name}:\tPLAY {", ".join(str(c) for c in last_move.data)}')
        if self._state.lp_mc[prev_i] != -1:
            print(f'{prev_pl.name}:\tfinished')
        if self.DEBUG_OUTPUT:
            print("\t", self._state)
            print("\t", self._gp.game.legal_moves())

    def play_game(self) -> None:
        print("Starting game of pharaoh...")
        if self.DEBUG_OUTPUT:
            print(self._state)
        self._print_top_card()
        winners: List[int] = self._gp.play()
        result: List[Tuple[int, int]] = list((i, winners[i]) for i in range(self._gp.game.n))
        result.sort(key=lambda x: x[1])
        print("Results:")
        for player, pos in result:
            print(f'\t{pos + 1}. {self._gp.players[player]}')


def main():
    cp: ConsolePharaoh = ConsolePharaoh("config/console_pharaoh_config.yml")
    cp.play_game()


if __name__ == '__main__':
    main()
