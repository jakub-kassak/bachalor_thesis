from typing import TypedDict, List, Optional, Dict

import yaml


class MCTSData(TypedDict):
    iterations: int
    width: int
    limit: int
    heuristic: str
    expl_const: float


class PlayerData(TypedDict):
    name: str
    type: str
    data: Optional[MCTSData]


class SuitValueData(TypedDict):
    name: str
    symbol: str
    numeric: int


class SymbolsData(TypedDict):
    __PREFIX__: str
    __SUFFIX__: str
    __DELIMITER__: str


class AbstractConfigData(TypedDict):
    players: List[PlayerData]
    init_cards: int
    suits: List[SuitValueData]
    values: List[SuitValueData]
    symbols: SymbolsData
    parameters: List[List[PlayerData]]


class ConfigData(AbstractConfigData, total=False):
    sim_id: int
    metrics: List


def load_configuration(config_file: str) -> ConfigData:
    with open(config_file, "r", encoding="utf8") as file:
        return yaml.load(file.read(), Loader=yaml.Loader)


class SimulationRun(TypedDict):
    players: List[PlayerData]
    ratio_of_time: float
    player_wins: Dict[str, int]
    player_shuffling: Dict[str, int]
    player_spent_time: Dict[str, float]
    metric_results: List[Dict[str, List]]


class SimulationData(TypedDict):
    iterations: int
    config: ConfigData
    config_file: str
    metric_names: List[str]
    runs: List[SimulationRun]
