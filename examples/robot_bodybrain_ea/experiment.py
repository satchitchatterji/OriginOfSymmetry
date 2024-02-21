"""Experiment class."""

import sqlalchemy.orm as orm
from base import Base
from revolve2.experimentation.database import HasId
import sqlalchemy
from parameters import ExperimentParameters


class Experiment(Base, HasId):
    """Experiment description."""

    __tablename__ = "experiment"

    # The seed for the rng.
    rng_seed: orm.Mapped[int] = orm.mapped_column(nullable=False)
    
    steer: orm.Mapped[bool] = orm.mapped_column(nullable=False)

    
    parameters_id: orm.Mapped[int] = orm.mapped_column(
        sqlalchemy.ForeignKey("parameters.id"), nullable=False, init=False
    )

    parameters: orm.Mapped[ExperimentParameters] = orm.relationship()