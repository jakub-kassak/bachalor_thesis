from typing import List

import pandas as pd
import config.data
from sqlalchemy import create_engine, func, desc, select, label, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from config.data import ConfigData
from database import models
from database.models import Base, Run, Simulation, MCTSPlayer, Player

# DB_URL = "postgresql://user:password@host.docker.internal:5432/card_games"
DB_URL = "postgresql://user:password@database:5432/card_games"


def make_engine():
    return create_engine(DB_URL)


my_engine: Engine = make_engine()
my_session: Session = Session(my_engine)
SessionLocal = sessionmaker(bind=my_engine)


def initialise_db(engine: Engine = my_engine, session: Session = my_session):
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(bind=engine)
    session.commit()


def save_simulation(config: ConfigData, name: str, iterations: int, config_file) -> int:
    session = SessionLocal()
    sim = models.Simulation(name=name, iterations=iterations, config_file=config_file, init_cards=config['init_cards'])
    session.add(sim)
    session.flush()
    for card in config['suits']:
        session.add(models.Suit(sim_id=sim.id, name=card['name'], numeric=card['numeric']))
    for card in config['values']:
        session.add(models.Value(sim_id=sim.id, name=card['name'], numeric=card['numeric']))
    session.commit()
    sim_id = sim.id
    session.close()
    return sim_id


def save_run(sim_id: int, sim_run: "config.data.SimulationRun") -> int:
    session = SessionLocal()
    run = Run(sim_id=sim_id, metrics_results=sim_run['metric_results'])
    session.add(run)
    session.flush()
    for player in sim_run['players']:
        type_ = player['type']
        name = player['name']
        d = dict(
            run_id=run.id,
            name=name,
            type=type_,
            shuffling=sim_run['player_shuffling'][name],
            wins=sim_run['player_wins'][name],
            spent_time=sim_run['player_spent_time'][name],
        )
        if type_ == 'mcts':
            session.add(models.MCTSPlayer(
                **d,
                heuristic=player['data']['heuristic'],
                limit=player['data']['limit'],
                iterations=player['data']['iterations'],
                width=player['data']['width'],
                expl_const=player['data']['expl_const']
            ))
        elif type_ == 'backtrack':
            session.add(models.BTPlayer(
                **d,
                heuristic=player['data']['heuristic'],
                limit=player['data']['limit'],
            ))
        else:
            session.add(models.Player(**d))
    session.commit()
    run_id = run.id
    session.close()
    return run_id


def query_sim_iterations(session: Session, sim_id: int) -> int:
    return session.query(Simulation.iterations).filter(Simulation.id == sim_id).first().iterations


def query_run_ids(session: Session, sim_id: int) -> List[int]:
    return [row.id for row in session.query(Run.id).filter(Run.sim_id == sim_id).all()]


def query_config_file(session: Session, sim_id: int) -> str:
    return session.query(Simulation.config_file).filter(Simulation.id == sim_id).first().config_file


def query_mcts_iterations(session: Session, sim_id: int, name: str):
    return [row.iterations for row in session.query(MCTSPlayer.iterations).join(Run)
            .filter(Run.sim_id == sim_id)
            .filter(MCTSPlayer.name == name).distinct().all()]


def query_mcts_expl(session: Session, sim_id: int, name: str):
    return [row.expl_const for row in session.query(MCTSPlayer.expl_const).join(Run)
            .filter(Run.sim_id == sim_id)
            .filter(MCTSPlayer.name == name).distinct().all()]


def query_mcts_heuristic(session: Session, sim_id: int) -> List[str]:
    return [row.heuristic for row in
            session.query(MCTSPlayer.heuristic).join(Run).filter(Run.sim_id == sim_id).distinct().all()]


def query_names(session: Session, sim_id: int) -> List[str]:
    return [row.name for row in
            session.query(Player.name).join(Run).filter(Run.sim_id == sim_id).distinct().order_by(Player.name).all()]


def query_max_value_for_metric(session: Session, sim_id: int, metric: str) -> float:
    subquery = session.query(text(f"jsonb_array_elements(data->'{metric}')::float8 AS value")) \
        .select_from(Run, func.jsonb_array_elements(Run.metrics_results).alias('data')) \
        .filter(Run.sim_id == sim_id) \
        .subquery()
    return session.query(func.max(text("value"))).select_from(subquery).scalar()


def query_avg_length_game(session: Session, sim_id: int) -> float:
    return session.query(func.avg(text("jsonb_array_length(data->'Size')")))\
        .select_from(Run, func.jsonb_array_elements(Run.metrics_results).alias('data'))\
        .filter(Run.sim_id == sim_id).scalar()


def query_of_wins_and_iterations(session: Session, sim_id: int, name: str):
    return session.query(MCTSPlayer.wins, MCTSPlayer.iterations) \
        .join(Run) \
        .filter(MCTSPlayer.name == name) \
        .filter(Run.sim_id == sim_id) \
        .order_by(MCTSPlayer.name, MCTSPlayer.iterations)


def query_of_wins_and_expl_const(session: Session, sim_id: int, iterations: int, name: str):
    return session.query(MCTSPlayer.wins, MCTSPlayer.expl_const).join(Run) \
        .filter(Run.sim_id == sim_id) \
        .filter(MCTSPlayer.name == name) \
        .filter(MCTSPlayer.iterations == iterations) \
        .order_by(MCTSPlayer.expl_const)


def query_of_wins_and_heuristic(session: Session, sim_id: int, iterations: int, expl_const: float, name: str):
    query_h = session.query(MCTSPlayer.heuristic).join(Run) \
        .filter(Run.sim_id == sim_id) \
        .distinct().subquery()
    query_w = session.query(MCTSPlayer.heuristic, MCTSPlayer.wins).join(Run) \
        .filter(Run.sim_id == sim_id) \
        .filter(MCTSPlayer.name == name) \
        .filter(MCTSPlayer.iterations == iterations) \
        .filter(MCTSPlayer.expl_const == expl_const).subquery()
    # noinspection PyTypeChecker
    return session.query(query_h.c.heuristic, query_w.c.wins) \
        .outerjoin(query_h, query_w.c.heuristic == query_h.c.heuristic, full=True) \
        .order_by(desc(func.length(query_h.c.heuristic)))


def query_of_wins_width_limit(session: Session, sim_id: int, iterations: int, name: str, ):
    return session.query(MCTSPlayer.wins, MCTSPlayer.width, MCTSPlayer.limit).join(Run) \
            .filter(Run.sim_id == sim_id) \
            .filter(MCTSPlayer.name == name) \
            .filter(MCTSPlayer.iterations == iterations) \
            .order_by(MCTSPlayer.width, MCTSPlayer.limit)


def query_of_wins_spent_time(session: Session, sim_id: int, name: str):
    iterations = query_sim_iterations(session, sim_id)
    sub_query = select(Run.id).join(Player)\
        .filter(Run.sim_id == sim_id)\
        .filter(Player.name == name)\
        .order_by(Run.id).distinct()
    return session.query(Player.name,
                    label("wins", 1 - Player.wins / iterations),
                    label("spent_time", Player.spent_time / iterations))\
        .join(Run)\
        .filter(Run.sim_id == sim_id)\
        .filter(Player.name != name)\
        .filter(Run.id.in_(sub_query)).order_by(Player.name)


def query_of_metric(session: Session, run_id: int, metric: str):
    return session.query(text("data -> '" + metric + "'")).\
            select_from(Run, func.jsonb_array_elements(Run.metrics_results).alias('data')).\
            filter(Run.id == run_id)


def query_to_df(query, engine=my_engine):
    return pd.read_sql_query(query.statement, engine.connect())


def query_player_name(session: Session, run_id: int) -> str:
    return session.query(Player.name)\
        .filter(Player.run_id == run_id)\
        .filter(Player.name != 'MCTS_VANILLA2').first().name


if __name__ == '__main__':
    initialise_db()
