from os import listdir
from os.path import join, isfile
from typing import List, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from math import nan, ceil
from sqlalchemy.orm import Session

from config.config_generator import P2_NAME, P4_NAME, P5_NAME, P6_NAME
from config.data import SimulationData
from database import crud

TITLES = True
SAVE_FIGURES = False


def load_sim_data_from_file(file_name: str) -> SimulationData:
    with open(file_name, 'r') as file:
        return yaml.safe_load(file)


def run_dataframe_time_ratio_score(data: SimulationData):
    a = []
    for run in data['runs']:
        b = {'time': run['players'][0]['data']['time'],
             'ratio_of_time': run['ratio_of_time']}
        for name, score in run['player_wins'].items():
            b[name] = score
        a.append(b)
    return pd.DataFrame(a)


def load_file_names(mypath) -> List[str]:
    return [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]


def graph_metric(name: str, results: List[Dict[str, List[float]]], run_name: str) -> None:
    data = {}
    max_ = max(len(row[name]) for row in results)
    for row, i in zip(results, range(1000)):
        data[i] = row[name] + [nan] * (max_ - len(row[name]))
    df = pd.DataFrame(data)
    # df.plot(title=f"{name} in {run_name}")
    # plt.show()
    df2 = pd.DataFrame({"mean": df.mean(axis=1), "median": df.median(axis=1)})
    # df2 = pd.DataFrame({"min": df.min(), "mean": df.mean(), "max": df.max()})
    df2.plot(title=f"Mean and median for {name} in {run_name}")
    plt.show()


def graph_wins_over_time(data, run_name):
    df: pd.DataFrame = run_dataframe_time_ratio_score(data)
    if len(data['config']['players']) == 2:
        df = df[['time', 'MCTS', 'ratio_of_time']]
        df.plot(xticks=df.time, x='time', secondary_y='ratio_of_time',
                title="Wins over time " + run_name)
    else:
        del df["ratio_of_time"]
        df.plot(xticks=df.time, x='time', title="Wins over time " + run_name)
    plt.show()


def plot(df: pd.DataFrame, title: str, ylabel: str = '', xlabel: str = '', kind: str = 'line',
         hline: Optional[float] = 0.5, yticks=(np.arange(10) / 10), bottom: float = 0.1) -> None:
    ax = df.plot(kind=kind, title=title if TITLES else '')
    if yticks is not None:
        ax.set_yticks(yticks)
    if hline is not None:
        ax.axhline(y=hline, color='black', ls="--", lw=1)
    # ax.set_xlim(0, len(df) - 1)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    # plt.subplots_adjust(left=0.05, bottom=bottom, right=0.95, top=0.95)
    plt.subplots_adjust(left=0.02, bottom=0.08, right=0.99, top=0.98)
    file_name = f'generated_images/{title}.png'
    if SAVE_FIGURES:
        plt.savefig(file_name)
    plt.show()


def normalize_wins(df, iterations) -> None:
    df['wins'] = 1 - df.wins / iterations


def get_conf_file_name(session: Session, sim_id: int) -> str:
    conf_file = crud.query_config_file(session, sim_id).split('/')[-1].split('.')[0]
    return conf_file


# def plot_p2_iterations(sim_id: int, session: Session) -> None:
#     iterations = crud.query_sim_iterations(session, sim_id)
#     conf_file = get_conf_file_name(session, sim_id)
#     df = crud.query_to_df(crud.query_of_wins_and_iterations(session, sim_id, P2_NAME))
#     normalize_wins(df, iterations)
#     print(df.to_string())
#     title = f'Wins over iterations of {conf_file}'
#     # plot_wins_over_iterations(df, title)
#     df.set_index('iterations', inplace=True)
#     plot(df, title)

def plot_p2_iterations(sim_s_id: int, sim_l_id: int, session: Session):
    # conf_file = get_conf_file_name(session, sim_s_id)
    df1 = crud.query_to_df(crud.query_of_wins_and_iterations(session, sim_s_id, P2_NAME))
    df2 = crud.query_to_df(crud.query_of_wins_and_iterations(session, sim_l_id, P2_NAME))
    normalize_wins(df1, crud.query_sim_iterations(session, sim_s_id))
    normalize_wins(df2, crud.query_sim_iterations(session, sim_l_id))
    print(df1.to_string())
    print(df2.to_string())
    df1.set_index('iterations', inplace=True)
    df2.set_index('iterations', inplace=True)
    df = pd.DataFrame({"small deck": df1.wins, "large deck": df2.wins})
    print(df.to_string())
    title = f'p2_wins_and_iterations'
    plot(df, title, xlabel='iterations')


def plot_p4_expl(sim_id: int, session: Session) -> None:
    iterations = crud.query_sim_iterations(session, sim_id)
    conf_file = get_conf_file_name(session, sim_id)
    df = pd.DataFrame()
    for i in crud.query_mcts_iterations(session, sim_id, P4_NAME):
        df2 = crud.query_to_df(crud.query_of_wins_and_expl_const(session, sim_id, i, P4_NAME))
        normalize_wins(df2, iterations)
        if 'expl_const' not in df:
            df['expl_const'] = df2['expl_const']
        df[f'{i} iterations'] = df2['wins']
    print(df.to_string())
    df.set_index('expl_const', inplace=True)
    plot(df, f'Wins over expl of {conf_file}')


def plot_p5_heuristic(sim_id: int, session: Session) -> None:
    iterations = crud.query_sim_iterations(session, sim_id)
    conf_file = crud.query_config_file(session, sim_id).split('/')[-1].split('.')[0]
    df = pd.DataFrame()
    for i in crud.query_mcts_iterations(session, sim_id, P5_NAME):
        for e in crud.query_mcts_expl(session, sim_id, P5_NAME):
            df2 = crud.query_to_df(crud.query_of_wins_and_heuristic(session, sim_id, i, e, P5_NAME))
            if df2.wins.isna().all():
                continue
            normalize_wins(df2, iterations)
            if 'heuristic' not in df:
                df['heuristic'] = df2['heuristic']
            df[f'({e:.2f}:{i})'] = df2['wins']
    title = f'Wins for different heuristics of {conf_file}'
    df.set_index("heuristic", inplace=True)
    df = df.transpose()
    print(df)
    plot(df, title, kind='bar', bottom=0.2, ylabel='wins', xlabel='(exploration constant:iterations)')


def plot_3d(df, title) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.bar3d(
        x=df.width,
        y=df.limit,
        z=0,
        dx=df.width.max() / 30,
        dy=df.limit.max() / 30,
        dz=df.wins,
        color=plt.cm.jet(df.wins/float(df.wins.max())),
        shade=True
    )
    ax.set_xlabel("width")
    ax.set_ylabel('limit')
    ax.set_zlabel("wins")
    if TITLES:
        ax.set_title(title)
    plt.xticks(df.width.unique())
    plt.yticks(df.limit.unique())
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
    plt.show()


def plot_p6_bmcts(sim_id: int, session: Session) -> None:
    sim_iterations = crud.query_sim_iterations(session, sim_id)
    conf_file = crud.query_config_file(session, sim_id).split('/')[-1].split('.')[0]
    for mcts_iterations in crud.query_mcts_iterations(session, sim_id, P6_NAME):
        df = crud.query_to_df(crud.query_of_wins_width_limit(session, sim_id, mcts_iterations, P6_NAME))
        normalize_wins(df, sim_iterations)
        print(df.to_string())
        title = f'Wins with different width and limit of {conf_file} for {mcts_iterations} iterations'
        plot_3d(df, title)


def add_sum_mean_and_round(df: pd.DataFrame) -> pd.DataFrame:
    df['mean'] = df.mean(axis=1)
    df['sum'] = df.sum(axis=1)
    print(df)
    return df.round(2)


def table_p7_tournament(sim_id: int, session: Session) -> Dict[str, pd.DataFrame]:
    names = crud.query_names(session, sim_id)
    wins, time = {}, {}
    for name in names:
        df = crud.query_to_df(crud.query_of_wins_spent_time(session, sim_id, name))
        df.set_index('name', inplace=True)
        wins[name] = df.wins
        time[name] = df.spent_time
    return {"wins": add_sum_mean_and_round(pd.DataFrame(wins)),
            "time": add_sum_mean_and_round(pd.DataFrame(time))}


def plot_strip(df: pd.DataFrame, title=''):
    data = df.values

    flat_data = np.concatenate(data)
    categories = np.repeat(np.arange(len(data)), [len(arr) for arr in data])

    plt.scatter(categories, flat_data, marker='|', color='blue', s=4)
    if TITLES:
        plt.title(title)
    # plt.xlabel('Values')
    # plt.ylabel('Categories')
    plt.show()


def plot_lines_median(df: pd.DataFrame, title='', ylim=None, cross='', vertical=True):
    ax = df.plot(legend=False, title=title if TITLES else '')
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_xlim(0, len(df) - 1)
    for line in ax.lines:
        line.set_linewidth(.2)
    df_med = df.median(axis=1)
    df_med.plot(legend=True, linewidth=3, label='median', color='red')
    make_cross(cross, df_med, vertical)
    plt.subplots_adjust(left=0.04, bottom=0.08, right=0.99, top=0.98)
    file_name = f'generated_images/{title}.png'
    if SAVE_FIGURES:
        plt.savefig(file_name)
    plt.show()


def make_cross(cross, df_med, vertical):
    if cross == 'max':
        y = df_med.max()
        x = df_med.idxmax()
    elif cross == 'median':
        y = df_med.median()
        x = df_med.sub(y).abs().idxmin()
    else:
        return
    print('max:', y)
    plt.axhline(y=y, color='b', linestyle='--', linewidth=1)
    if vertical:
        plt.axvline(x=x, color='b', ls='--', linewidth=1)


def remove_extremal(row):
    threshold = np.percentile(row, [10, 90])  # Define threshold values (10% and 90%)
    return row[(row >= threshold[0]) & (row <= threshold[1])]  # Filter values within threshold


def clean_data(df):
    df_filtered = df.apply(remove_extremal, axis=1)
    return df_filtered.interpolate(axis=1)


def extend_length(x, n):
    # m = 2 * len(x) // 3
    # x = x[:m]
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x)


def plot_p8_evolution(sim_id: int, session: Session):
    metrics:  List[Tuple[str, bool, str, bool, List[str]]] = [
        ("AvgDepth", False, '', True, ['lines_median']),

        ("AvgDegree", False, 'median', False, ['lines_median']),
        ("SackinIndex", False, '', True, ['median', ]),
        ("CopheneticIndex", False, '', True, ['median', ]),
        ("WinExpectancy", True, '', True, ['lines_median']),
    ]

    avg_length = crud.query_avg_length_game(session, sim_id)
    print(f'average length of a game: {avg_length:.2f}')
    avg_length = ceil(avg_length)

    for metric, clean, cross, vertical, tps in metrics:
        dfs = []
        for i, run_id in enumerate(crud.query_run_ids(session, sim_id)):
            result: List[List[float]] = [row[0] for row in crud.query_of_metric(session, run_id, metric).all()]
            data = np.column_stack([extend_length(arr, avg_length) for arr in result])
            df = pd.DataFrame(data)

            title = f'p8_{metric}_{i:02}_{crud.query_player_name(session, run_id)}'
            if clean:
                title += "_clean"
                df = clean_data(df)
            dfs.append((df, title))
        if 'median' in tps:
            plot(pd.DataFrame({"median " + "_".join(title.split('_')[3:]): df.median(axis=1) for df, title in dfs}),
                 title=f'p8_{metric}_median', yticks=None, hline=None)
        if 'variance' in tps:
            plot(pd.DataFrame({"variance " + "_".join(title.split('_')[3:]): df.var(axis=1) for df, title in dfs}),
                 title=f'p8_{metric}_variance', yticks=None, hline=None)
        if 'lines_median' in tps:
            for df, title in dfs:
                plot_lines_median(df, title=title, ylim=(0, max(df.max().max() for df, _ in dfs)), cross=cross, vertical=vertical)


def main():
    with crud.SessionLocal() as session:
        plt.rcParams['figure.figsize'] = [10, 7]
        plot_p2_iterations(2, 8, session)  # small
        # plot_p2_iterations(8, session)  # large

        plot_p4_expl(7, session)  # small
        plot_p4_expl(9, session)  # large

        plot_p5_heuristic(13, session)  # small
        plot_p5_heuristic(14, session)  # large

        plot_p6_bmcts(18, session)  # small
        plot_p6_bmcts(23, session)  # large

        plt.rcParams['figure.figsize'] = [8, 3.3]
        plot_p8_evolution(66, session)

        return {'small': table_p7_tournament(36, session), 'large': table_p7_tournament(38, session)}


if __name__ == '__main__':
    TITLES = False
    # SAVE_FIGURES = True
    main()
