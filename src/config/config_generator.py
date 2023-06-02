from copy import deepcopy

import yaml

from analysis.config_const import CONFIG_FOLDER
from config.data import ConfigData, PlayerData

MY_SHUFFLE = 'my_shuffle'
SMALLEST_TUPLE = 'smallest_tuple'
BIGGEST_TUPLE = 'biggest_tuple'
HEURISTICS = (BIGGEST_TUPLE, SMALLEST_TUPLE)

P2_NAME = 'MCTS'
P4_NAME = 'MCTS_E'
P5_NAME = 'MCTS_H'
P6_NAME = 'BMCTS'
P8_METRICS = ["MaxDepth", "AvgDepth", "MaxDegree", "AvgDegree", "RootVisits", "RChildVisits", "RChildWins", "Size",
              "LeafCnt", "TerminalLeafCnt", "InnerSize", "SackinIndex", "CopheneticIndex", "WinExpectancy"]

INF = 100000000
DECK_S = {'suits': [{'name': 'HEART', 'symbol': '🧡', 'numeric': 1},
                    {'name': 'BELL', 'symbol': '🟡', 'numeric': 2},
                    {'name': 'ACORN', 'symbol': '♧', 'numeric': 3}],
          'values': [{'name': 'VII', 'symbol': '7', 'numeric': 7},
                     {'name': 'VIII', 'symbol': '8', 'numeric': 8},
                     {'name': 'IX', 'symbol': '9', 'numeric': 9},
                     {'name': 'X', 'symbol': '10', 'numeric': 10},
                     {'name': 'UNDER', 'symbol': '⛛', 'numeric': 11}],
          'init_cards': 6}
DECK_L = {'suits': [{'name': 'HEART', 'symbol': '🧡', 'numeric': 1},
                    {'name': 'BELL', 'symbol': '🟡', 'numeric': 2},
                    {'name': 'ACORN', 'symbol': '♧', 'numeric': 3},
                    {'name': 'LEAF', 'symbol': '🍂', 'numeric': 4}],
          'values': [{'name': 'VII', 'symbol': '7', 'numeric': 7},
                     {'name': 'VIII', 'symbol': '8', 'numeric': 8},
                     {'name': 'IX', 'symbol': '9', 'numeric': 9},
                     {'name': 'X', 'symbol': '10', 'numeric': 10},
                     {'name': 'UNDER', 'symbol': '⛛', 'numeric': 11},
                     {'name': 'OVER', 'symbol': '🔺', 'numeric': 12},
                     {'name': 'KING', 'symbol': '👑', 'numeric': 13},
                     {'name': 'ACE', 'symbol': 'A', 'numeric': 14}],
          'init_cards': 10}

BT_S: PlayerData = {'name': 'BT_SMALL',
                    'type': 'backtrack',
                    'data': {'limit': 13,
                             'heuristic': SMALLEST_TUPLE,
                             'width': 0,
                             'expl_const': 0,
                             'iterations': 0}}
BT_L: PlayerData = {'name': 'BT_LARGE',
                    'type': 'backtrack',
                    'data': {'limit': 9,
                             'heuristic': SMALLEST_TUPLE,
                             'width': 0,
                             'expl_const': 0,
                             'iterations': 0}}

ITER1 = 32
ITER2 = 243
ITERS = (ITER1, ITER2)
MCTS_VANILLA: PlayerData = {'name': f'VANILLA',
                            'type': 'mcts',
                            'data': {'iterations': ITER2,
                                     'width': INF,
                                     'limit': INF,
                                     'heuristic': MY_SHUFFLE,
                                     'expl_const': 2.}}

EXPL_S = 0.4
EXPL_L1 = 0.285714
EXPL_L2 = 0.571429
MCTS_EXPL_S = deepcopy(MCTS_VANILLA)
MCTS_EXPL_S['name'] = 'EXPL_S'
MCTS_EXPL_S['data']['expl_const'] = EXPL_S
MCTS_EXPL_L = deepcopy(MCTS_VANILLA)
MCTS_EXPL_L['name'] = 'EXPL_L'
MCTS_EXPL_L['data']['expl_const'] = EXPL_L2

EXPL_L2_ST = EXPL_L2 / 4

RANDOM_PLAYER = {"name": "RANDOM",
                 "type": "random",
                 "data": None}
BIGGEST_TUPLE_PLAYER = {"name": "BIGGEST_TUPLE",
                        "type": "ai1",
                        "data": None}
SMALLEST_TUPLE_PLAYER = {"name": "SMALLEST_TUPLE",
                         "type": "ai2",
                         "data": None}
ANNOYING_PLAYER = {"name": "ANNOYING",
                   "type": "ap2",
                   "data": None}

MCTS_HE_L = deepcopy(MCTS_EXPL_L)
MCTS_HE_L['name'] = 'HEUR_L'
MCTS_HE_L['data']['heuristic'] = SMALLEST_TUPLE
MCTS_HE_L['data']['expl_const'] = EXPL_L2_ST

MCTS_HE_L32 = deepcopy(MCTS_HE_L)
MCTS_HE_L32['name'] = 'HEUR_L32'
MCTS_HE_L32['data']['iterations'] = ITER1
MCTS_HE_L32['data']['expl_const'] = EXPL_L1


def set_parameters(parameters, deck, players=None, metrics=None) -> ConfigData:
    return {'players': players if players is not None else parameters[0],
            'symbols': {'__PREFIX__': '',
                        '__SUFFIX__': '',
                        '__DELIMITER__': ''},
            'metrics': metrics if metrics else None,
            'parameters': parameters} | deck


def save_config(conf, conf_name):
    class NoAliasDumper(yaml.SafeDumper):
        def ignore_aliases(self, data):
            return True

    with open(CONFIG_FOLDER + conf_name, 'w') as file:
        yaml.dump(conf, file, Dumper=yaml.SafeDumper)


def p2_iterations_small(name: str):
    parameters = [[BT_S,
                   {'name': f'{name}',
                    'type': 'mcts',
                    'data': {'iterations': int(i ** 2.5),
                             'width': INF,
                             'limit': INF,
                             'heuristic': MY_SHUFFLE,
                             'expl_const': 2}}] for i in range(1, 12)]

    conf: ConfigData = set_parameters(parameters, DECK_S)
    save_config(conf, "p2_iterations_small.yml")


def p2_iterations_big(name: str):
    parameters = [[BT_L,
                   {'name': f'{name}',
                    'type': 'mcts',
                    'data': {'iterations': int(i ** 2.5),
                             'width': INF,
                             'limit': INF,
                             'heuristic': MY_SHUFFLE,
                             'expl_const': 2}}] for i in range(2, 12)]

    conf: ConfigData = set_parameters(parameters, DECK_L)
    save_config(conf, "p2_iterations_big.yml")


def p4_expl_small(name: str):
    parameters = [[MCTS_VANILLA,
                   {'name': f'{name}',
                    'type': 'mcts',
                    'data': {'iterations': i,
                             'width': INF,
                             'limit': INF,
                             'heuristic': MY_SHUFFLE,
                             'expl_const': e / 10}}]
                  for i in ITERS
                  for e in range(0, 40, 2)]
    conf: ConfigData = set_parameters(parameters, DECK_S)
    save_config(conf, "p4_expl_small.yml")


def p4_expl_big(name: str):
    parameters = [[MCTS_VANILLA,
                   {'name': f'{name}',
                    'type': 'mcts',
                    'data': {'iterations': i,
                             'width': INF,
                             'limit': INF,
                             'heuristic': MY_SHUFFLE,
                             'expl_const': e / 7}}]
                  for i in ITERS
                  for e in range(0, 30, 2)]
    conf: ConfigData = set_parameters(parameters, DECK_L)
    save_config(conf, "p4_expl_big.yml")


def p5_heuristic_small(name: str):
    parameters = [[MCTS_EXPL_S,
                   {'name': f'{name}',
                    'type': 'mcts',
                    'data': {'iterations': i,
                             'width': INF,
                             'limit': INF,
                             'heuristic': h,
                             'expl_const': EXPL_S * d}}]
                  for i in ITERS
                  for d in (1 / 2, 1, 4)
                  for h in HEURISTICS]
    conf: ConfigData = set_parameters(parameters, DECK_S)
    save_config(conf, "p5_heuristic_small.yml")


def p5_heuristic_big(name: str):
    parameters = [[MCTS_EXPL_L,
                   {'name': f'{name}',
                    'type': 'mcts',
                    'data': {'iterations': i,
                             'width': INF,
                             'limit': INF,
                             'heuristic': h,
                             'expl_const': e * d}}]
                  for (i, e) in ((ITER1, EXPL_L1), (ITER2, EXPL_L2))
                  for d in (1 / 2, 1, 4)
                  for h in HEURISTICS]
    conf: ConfigData = set_parameters(parameters, DECK_L)
    save_config(conf, "p5_heuristic_big.yml")


def p6_limit_and_width_small(name: str):
    parameters = [[MCTS_EXPL_S,
                   {'name': f'{name}',
                    'type': 'mcts',
                    'data': {'iterations': ITER1,
                             'width': w,
                             'limit': lim,
                             'heuristic': SMALLEST_TUPLE,
                             'expl_const': EXPL_S}}]
                  for w in (1, 2, 4, 8)
                  for lim in (4, 8, 16) if w < lim] + [
                     [MCTS_EXPL_S,
                      {'name': f'{name}',
                       'type': 'mcts',
                       'data': {'iterations': ITER2,
                                'width': w,
                                'limit': lim,
                                'heuristic': SMALLEST_TUPLE,
                                'expl_const': EXPL_S}}]
                     for w in (2, 4, 8, 16)
                     for lim in (4, 8, 16, 32, 64, 128) if w < lim]
    conf: ConfigData = set_parameters(parameters, DECK_S)
    save_config(conf, "p6_bmcts_small.yml")


def p6_limit_and_width_big(name: str):
    parameters = [[MCTS_EXPL_L,
                   {'name': f'{name}',
                    'type': 'mcts',
                    'data': {'iterations': ITER1,
                             'width': w,
                             'limit': lim,
                             'heuristic': SMALLEST_TUPLE,
                             'expl_const': EXPL_L1}}]
                  for w in (1, 2, 4, 8)
                  for lim in (4, 8, 16) if w < lim] + [
                     [MCTS_EXPL_L,
                      {'name': f'{name}',
                       'type': 'mcts',
                       'data': {'iterations': ITER2,
                                'width': w,
                                'limit': lim,
                                'heuristic': SMALLEST_TUPLE,
                                'expl_const': EXPL_L2_ST}}]
                     for w in (2, 4, 8, 16)
                     for lim in (4, 8, 16, 32, 64, 128) if w < lim]
    conf: ConfigData = set_parameters(parameters, DECK_L)
    save_config(conf, "p6_bmcts_big.yml")


def p7_tournament_small():
    BT_S_SHUFFLE = deepcopy(BT_S)
    BT_S_SHUFFLE['name'] = 'BTSH_S'
    BT_S_SHUFFLE['data']['heuristic'] = MY_SHUFFLE
    BT_S_SHUFFLE['data']['limit'] = 9

    MCTS_HE_S = deepcopy(MCTS_EXPL_S)
    MCTS_HE_S['name'] = 'HEUR_S'
    MCTS_HE_S['data']['heuristic'] = SMALLEST_TUPLE

    MCTS_HE_S32 = deepcopy(MCTS_HE_S)
    MCTS_HE_S32['name'] = 'HEUR_S32'
    MCTS_HE_S32['data']['iterations'] = ITER1

    players = [
        RANDOM_PLAYER,
        BIGGEST_TUPLE_PLAYER,
        SMALLEST_TUPLE_PLAYER,
        ANNOYING_PLAYER,
        BT_S_SHUFFLE,
        BT_S,
        MCTS_VANILLA,
        MCTS_EXPL_S,
        MCTS_HE_S32,
        MCTS_HE_S
    ]
    n = len(players)
    parameters = [[players[i], players[j]] for i in range(n) for j in range(i + 1, n)]
    conf: ConfigData = set_parameters(parameters, DECK_S)
    conf['sim_id'] = 36
    save_config(conf, "p7_tournament_small.yml")


def p7_tournament_big(save=True):
    BT_L_SHUFFLE = deepcopy(BT_L)
    BT_L_SHUFFLE['name'] = 'BTSH_L'
    BT_L_SHUFFLE['data']['heuristic'] = MY_SHUFFLE
    BT_L_SHUFFLE['data']['limit'] = 7

    players = [
        RANDOM_PLAYER,
        BIGGEST_TUPLE_PLAYER,
        SMALLEST_TUPLE_PLAYER,
        ANNOYING_PLAYER,
        BT_L,
        BT_L_SHUFFLE,
        MCTS_VANILLA,
        MCTS_EXPL_L,
        MCTS_HE_L32,
        MCTS_HE_L
    ]
    n = len(players)
    parameters = [[players[i], players[j]] for i in range(n) for j in range(i + 1, n)]
    conf: ConfigData = set_parameters(parameters, DECK_L)
    save_config(conf, "p7_tournament_big.yml")


def p8_evolution(metrics):
    MCTS_VANILLA2 = MCTS_VANILLA.copy()
    MCTS_VANILLA2['name'] = 'MCTS_VANILLA2'
    parameters = [[MCTS_VANILLA, MCTS_VANILLA2],
                  [MCTS_EXPL_L, MCTS_VANILLA2],
                  [MCTS_HE_L, MCTS_VANILLA2]]
    conf = set_parameters(parameters, DECK_L, metrics=metrics)
    save_config(conf, "p8_evolution.yml")


def main():
    p2_iterations_small(P2_NAME)
    p2_iterations_big(P2_NAME)

    p4_expl_small(P4_NAME)
    p4_expl_big(P4_NAME)

    p5_heuristic_small(P5_NAME)
    p5_heuristic_big(P5_NAME)

    p6_limit_and_width_small(P6_NAME)
    p6_limit_and_width_big(P6_NAME)

    p7_tournament_small()
    p7_tournament_big()

    p8_evolution(P8_METRICS)


if __name__ == '__main__':
    main()
