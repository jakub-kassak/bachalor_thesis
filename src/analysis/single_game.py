from typing import List, Sequence

import matplotlib.pyplot as plt
import pandas as pd

from abstract_game.move import D, S
from analysis.config_const import *
from config.data import load_configuration
from mcts.mcts import TreeMetric
from mcts.tree_metrics import LeafCnt, Size, InnerSize, RootVisits, RChildVisits, \
    RChildWins, MaxDegree, AvgDegree, MaxDepth, NormalizedIndex, SackinIndex, CopheneticIndex
from pharaoh.game_play import PhGamePlay


def run_single(config_file: str, metrics: List[TreeMetric], min_length: int) -> PhGamePlay:
    gp: PhGamePlay = PhGamePlay(load_configuration(config_file))
    if gp.mcts:
        gp.mcts.metrics.extend(metrics)
    while True:
        gp.play()
        if gp.game.state.mc >= min_length:
            return gp
        print(gp.player_ranking(), gp.game.state.mc)
        gp.reset()


def print_winners(gp: PhGamePlay) -> None:
    ranking = gp.player_ranking()
    print("Placings:")
    for player, rank in ranking:
        print(f'\t{rank}. {player}')


def process_metrics(metrics: List[TreeMetric]) -> pd.DataFrame:
    d = {}
    for m in metrics:
        d[m.name()] = m.results
    return pd.DataFrame(d)


def plot_single_metric(df: pd.DataFrame, metric_name: str, title: str) -> None:
    df.plot(y=metric_name, title=title)
    plt.show()


def plot_root_children_visit_and_win_ratio(df: pd.DataFrame) -> None:
    def f1(col: pd.Series) -> pd.Series:
        return pd.Series(col[::-1] + (0,) * (max_ - len(col)))

    def f2(col: pd.Series) -> pd.Series:
        return col / (df.RootVisits - 1) * 100
    max_ = df.RChildVisits.apply(len).max()
    df_vis = df.RChildVisits.apply(f1).apply(f2)
    df_vis = df_vis.assign(Ratio=df_vis[0] / df_vis[1])
    df_win = df.RChildWins.apply(f1).apply(f2)

    plot1 = df_vis.plot.bar(y="Ratio")
    plot1.axhline(y=1, color='green')
    plot1.set_title("Visits of first child / visits of second")
    del df_vis["Ratio"]
    plt2 = df_vis.plot.bar(stacked=True)
    plt2.set_title("Distribution of visits of root's children")
    df_win.plot.bar(stacked=True).set_title("Victory ratios of root's children")
    plt.show()


def plot_multiple_metrics(df: pd.DataFrame, names: Sequence[str], title: str) -> None:
    df[names].plot(title=title)


def main():
    si: NormalizedIndex[float, S, D] = NormalizedIndex(SackinIndex())
    ci: NormalizedIndex[float, S, D] = NormalizedIndex(CopheneticIndex())
    metrics: List[TreeMetric] = [LeafCnt(), InnerSize(), Size(), RootVisits(), RChildVisits(), RChildWins(),
                                 MaxDegree(), AvgDegree(), MaxDepth(), si, ci]
    gp = run_single(CONFIG_XL, metrics, 10)
    print("Total moves:", gp.game.state.mc)
    print_winners(gp)
    df = process_metrics(metrics)
    print(df.to_string())

    plt.rcParams["figure.figsize"] = [16, 9]
    plot_single_metric(df, si.name(), "Normalised Sackin Index in a single game")
    plot_single_metric(df, ci.name(), "Normalised Cophenetic Index in a single game")
    plot_single_metric(pd.DataFrame({"simple_states": df.Size - df.InnerSize - df.LeafCnt}), "simple_states",
                       "Number of states with only one allowed move")
    plot_single_metric(df, "RootVisits", "Root Visits in a single game")
    plot_root_children_visit_and_win_ratio(df)
    plot_multiple_metrics(df.assign(NormalizedSize=df.InnerSize + df.LeafCnt),
                          ["Size", "NormalizedSize", "LeafCnt", "InnerSize"],
                          "Tree sizes in a single game")
    plot_multiple_metrics(df, ["MaxDegree", "AvgDegree"], "Branching during a game")
    plot_single_metric(df, "MaxDepth", "Maximal leaf depth in a single game")


if __name__ == '__main__':
    main()
