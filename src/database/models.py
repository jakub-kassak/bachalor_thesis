from sqlalchemy import Column, Integer, String, ForeignKey, Double
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import DeclarativeMeta
from sqlalchemy.dialects.postgresql import JSONB

Base: DeclarativeMeta = declarative_base()


def _repr(obj, **kwargs) -> str:
    attrs = ", ".join(f"{k}={v}" for k, v in kwargs.items())
    return f"{type(obj).__name__}({attrs})"


class Simulation(Base):
    __tablename__ = "simulation"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, unique=True, nullable=False)
    init_cards = Column(Integer, nullable=False)
    config_file = Column(String, nullable=False)
    iterations = Column(Integer, nullable=False)

    def __repr__(self) -> str:
        return _repr(self, id=self.id, name=self.name)


class Suit(Base):
    __tablename__ = 'suit'
    id = Column(Integer, primary_key=True, autoincrement=True)
    sim_id = Column(Integer, ForeignKey('simulation.id', ondelete='CASCADE'), nullable=False)
    name = Column(String, nullable=False)
    numeric = Column(Integer, nullable=False)

    def __repr__(self) -> str:
        return _repr(self, suit_id=self.id, sim_id=self.sim_id, name=self.name, numeric=self.numeric)


class Value(Base):
    __tablename__ = 'value'
    id = Column(Integer, primary_key=True, autoincrement=True)
    sim_id = Column(Integer, ForeignKey('simulation.id', ondelete='CASCADE'), nullable=False)
    name = Column(String, nullable=False)
    numeric = Column(Integer, nullable=False)

    def __repr__(self) -> str:
        return _repr(self, value_id=self.id, sim_id=self.sim_id, name=self.name, numeric=self.numeric)


class Run(Base):
    __tablename__ = 'run'
    id = Column(Integer, primary_key=True, autoincrement=True)
    sim_id = Column(Integer, ForeignKey('simulation.id', ondelete='CASCADE'), nullable=False)
    metrics_results = Column(JSONB, nullable=True)

    def __repr__(self) -> str:
        return _repr(self, id=self.id, sim_id=self.sim_id, metric_results=self.metrics_results)


class Player(Base):
    __tablename__ = 'player'
    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey('run.id', ondelete='CASCADE'), nullable=False)
    name = Column(String, nullable=False)
    type = Column(String, nullable=False)
    shuffling = Column(Integer, nullable=False)
    wins = Column(Integer, nullable=False)
    spent_time = Column(Double, nullable=False)

    def _attrs(self):
        return dict(player_id=self.id, run_id=self.run_id, name=self.name, type=self.type,
                    shuffling=self.shuffling, wins=self.wins, spent_time=self.spent_time)

    def __repr__(self):
        return _repr(self, **self._attrs())


class MCTSPlayer(Player):
    __tablename__ = "mcts_player"
    id = Column(Integer, ForeignKey('player.id', ondelete='CASCADE'), primary_key=True, autoincrement=True)
    heuristic = Column(String, nullable=False)
    limit = Column(Integer, nullable=False)
    iterations = Column(Integer, nullable=False)
    width = Column(Integer, nullable=False)
    expl_const = Column(Double, nullable=False)

    def _attrs(self):
        d = super()._attrs()
        d.update(heuristic=self.heuristic, limit=self.limit, iterations=self.iterations, width=self.width,
                 expl_const=self.expl_const)
        return d


class BTPlayer(Player):
    __tablename__ = 'bt_player'
    id = Column(Integer, ForeignKey('player.id', ondelete='CASCADE'), primary_key=True, autoincrement=True)
    heuristic = Column(String, nullable=False)
    limit = Column(Integer, nullable=False)

    def _attrs(self):
        d = super()._attrs()
        d.update(heuristic=self.heuristic, limit=self.limit)
        return d
