import pickle
from datetime import datetime
from multiprocessing import Process, cpu_count, Value, Array, Queue
from typing import List, Any, Dict, Optional, Callable, Sequence

import sqlalchemy.exc
import yaml

import database.crud
from analysis import results_from_multiple_runs
from analysis.config_const import *
from config import config_generator
from config.data import ConfigData, load_configuration, SimulationData, SimulationRun
from database import crud
from mcts import tree_metrics
from mcts.mcts import TreeMetric
from pharaoh.game_play import PhGamePlay

SequenceOfTreeMetricConstructors = Sequence[Callable[[], TreeMetric[Any, Any, Any]]]


class MyProcess(Process):
    def __init__(self, iterations: int, finished: Any, config: bytes, metrics: SequenceOfTreeMetricConstructors,
                 queue: Optional[Queue] = None):
        super().__init__()
        self._iterations = iterations
        self._finished = finished
        self.gp = PhGamePlay(pickle.loads(config))
        self._name_index: Dict[str, int] = {self.gp.players[i].name: i for i in range(self.gp.game.n)}
        self._players = list(self.gp.players)
        if self.gp.mcts:
            self.gp.mcts.metrics.extend(f() for f in metrics)
        self.wins: Any = Array('i', [0] * len(self.gp.players), lock=False)
        self.shuffling_stats: Any = Array('i', [0] * len(self.gp.players), lock=False)
        self.spent_time_stats: Any = Array('d', [0] * len(self.gp.players), lock=False)
        self.total_time: Any = Value('d', 0, lock=False)
        self.playout_time: Any = Value('d', 0, lock=False)
        self.queue = queue

    def send_metrics(self, placings: Dict[str, int]) -> None:
        if not self.queue:
            return
        d: Dict[str, Dict[str, int] | List] = {"placings": placings}
        if self.gp.mcts:
            for m in self.gp.mcts.metrics:
                d[repr(m)] = m.results
        self.queue.put(pickle.dumps(d))

    def run(self) -> None:
        n: int = len(self.gp.players)
        while True:
            cnt: int
            with self._finished.get_lock():
                if self._finished.value >= self._iterations:
                    break
                self._finished.value += 1
                cnt = self._finished.value
            if len(self._players) == 2 and self._players[cnt % 2] is self.gp.players[0]:
                self.gp.reset()
            result = self.get_result()
            placings = {}
            for i in range(n):
                j: int = self._name_index[self.gp.players[i].name]
                placings[self.gp.players[i].name] = result[i]
                self.wins[j] += result[i]
                self.shuffling_stats[j] += i
                if self.gp.players[i].rounds > 0:
                    self.spent_time_stats[j] += self.gp.players[i].spent_time / self.gp.players[i].rounds
            if self.gp.mcts:
                self.playout_time.value += self.gp.mcts.playout_time
                self.total_time.value += self.gp.mcts.total_time

            self.send_metrics(placings)
            # print(cnt, end=' ')
            self.gp.reset()

    def get_result(self):
        while True:
            try:
                result = self.gp.play()
                break
            except Exception as e:
                print(e)
                self.gp.reset()  # we reset the game twice to get the same order of players (in a game of two)
                self.gp.reset()
        return result


def run_simulation(iterations: int, config: ConfigData, process_count: int,
                   metrics: SequenceOfTreeMetricConstructors = ()) -> SimulationRun:
    finished: Any = Value('i', 0)
    processes: List[MyProcess] = []
    queue: Queue = Queue()
    for player in config['players']:
        if player['data']:
            data = player['data']
        else:
            data = '{}'
        print(f"\t{player['name']} {data}")
    data = pickle.dumps(config)
    for i in range(process_count):
        processes.append(MyProcess(iterations, finished, data, metrics, queue))
        processes[-1].start()

    metric_results: List[Dict[str, List]] = []
    for i in range(iterations):
        metric_results.append(pickle.loads(queue.get()))

    n: int = len(processes[-1].wins)
    wins: List[int] = [0] * n
    shuffling_stats: List[int] = [0] * n
    spent_time_stats: List[int] = [0] * n
    total_time: float = 0.0
    playout_time: float = 0.1
    for j, p in enumerate(processes):
        p.join()
        total_time += p.total_time.value
        playout_time += p.playout_time.value
        for i in range(n):
            wins[i] += p.wins[i]
            shuffling_stats[i] += p.shuffling_stats[i]
            spent_time_stats[i] += p.spent_time_stats[i]
    ratio_of_time: float = playout_time / total_time if total_time != 0 else float('nan')
    # print(f'\n\tratio of time spent in playouts: {ratio_of_time}')
    player_wins: Dict[str, int] = {config['players'][i]['name']: wins[i] for i in range(n)}
    player_shuffling: Dict[str, int] = {config['players'][i]['name']: shuffling_stats[i] for i in range(n)}
    player_spent_time: Dict[str, float] = {config['players'][i]['name']: spent_time_stats[i] for i in range(n)}
    for name in player_wins.keys():
        print('\t', name, player_wins[name] / iterations, "-", player_shuffling[name] / iterations)
    return SimulationRun(ratio_of_time=ratio_of_time, players=config['players'], player_wins=player_wins,
                         player_shuffling=player_shuffling, metric_results=metric_results,
                         player_spent_time=player_spent_time)


def load_metric_constructors(config: ConfigData) -> SequenceOfTreeMetricConstructors:
    if 'metrics' not in config:
        return []
    lst: List[Callable] = []
    for m in config['metrics']:
        if m == "MaxDepth":
            lst.append(tree_metrics.MaxDepth)
        elif m == "AvgDepth":
            lst.append(tree_metrics.AvgDepth)
        elif m == "MaxDegree":
            lst.append(tree_metrics.MaxDegree)
        elif m == "AvgDegree":
            lst.append(tree_metrics.AvgDegree)
        elif m == "RootVisits":
            lst.append(tree_metrics.RootVisits)
        elif m == "RChildVisits":
            lst.append(tree_metrics.RChildVisits)
        elif m == "RChildWins":
            lst.append(tree_metrics.RChildWins)
        elif m == "Size":
            lst.append(tree_metrics.Size)
        elif m == "LeafCnt":
            lst.append(tree_metrics.LeafCnt)
        elif m == "TerminalLeafCnt":
            lst.append(tree_metrics.TerminalLeafCnt)
        elif m == "InnerSize":
            lst.append(tree_metrics.InnerSize)
        elif m == "SackinIndex":
            lst.append(lambda: tree_metrics.NormalizedIndex(tree_metrics.SackinIndex()))
        elif m == "CopheneticIndex":
            lst.append(lambda: tree_metrics.NormalizedIndex(tree_metrics.CopheneticIndex()))
        elif m == "WinExpectancy":
            lst.append(tree_metrics.WinExpectancy)
    return lst


def main(iterations: int, config_file: str, process_count: int = cpu_count()) -> None:
    config: ConfigData = load_configuration(config_file)
    deck_size = len(config['values']) * len(config['suits'])
    print(f'process count: {process_count}\n'
          f'iterations: {iterations}\n'
          f'deck size: {deck_size}\n'
          f'initial card count: {config["init_cards"]}\n'
          f'config file: {config_file}')
    if 'sim_id' in config:
        sim_id = config['sim_id']
    else:
        sim_id = crud.save_simulation(config, datetime.now().strftime("%Y/%m/%d-%H:%M:%S"), iterations, config_file)
    print(f'simulation id: {sim_id}')

    metrics: SequenceOfTreeMetricConstructors = load_metric_constructors(config) if config['metrics'] else []

    i: int = 0
    n = len(config['parameters'])
    for players in config['parameters']:
        i += 1
        print(f'{i}/{n}')
        config['players'] = players
        sim_run: SimulationRun = run_simulation(iterations, config, process_count, metrics)
        save_run(sim_id, sim_run)


def save_run(sim_id, sim_run):
    try:
        crud.save_run(sim_id, sim_run)
    except sqlalchemy.exc.SQLAlchemyError:
        crud.my_engine = crud.make_engine()
        crud.save_run(sim_id, sim_run)


def write_results_from_simulation(data: SimulationData):
    file_name = SIM_RES_DIR + "/simulation-" + datetime.now().strftime("%Y-%m-%d_%H-%M") + ".yml"
    with open(file_name, "w") as file:
        yaml.dump(data, file)


if __name__ == '__main__':
    database.crud.initialise_db()
    config_generator.main()
    cpu_cnt = cpu_count()
    small_i = 20
    big_i = 10
    main(small_i, CONFIG_FOLDER + "p2_iterations_small.yml", cpu_cnt)
    main(small_i, CONFIG_FOLDER + "p4_expl_small.yml", cpu_cnt)
    main(big_i, CONFIG_FOLDER + "p2_iterations_big.yml", cpu_cnt)
    main(small_i, CONFIG_FOLDER + "p5_heuristic_small.yml", cpu_cnt)
    main(small_i, CONFIG_FOLDER + "p6_bmcts_small.yml", cpu_cnt)
    main(big_i, CONFIG_FOLDER + "p5_heuristic_big.yml", cpu_cnt)
    main(small_i, CONFIG_FOLDER + "p6_bmcts_small.yml", cpu_cnt)

    main(big_i, CONFIG_FOLDER + "p6_bmcts_big.yml", cpu_cnt)
    main(small_i, CONFIG_FOLDER + "p7_tournament_small.yml", cpu_cnt)
    main(100, CONFIG_FOLDER + "p7_tournament_big.yml", cpu_cnt)
    main(100, CONFIG_FOLDER + "p8_evolution.yml", cpu_cnt)
    results_from_multiple_runs.main()
