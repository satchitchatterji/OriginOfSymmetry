"""Individual class."""

from dataclasses import dataclass

from base import Base
from genotype import Genotype
from revolve2.experimentation.optimization.ea import Individual as GenericIndividual


@dataclass
class Individual(Base, GenericIndividual[Genotype], population_table="population"):
    """An individual in a population."""

    __tablename__ = "individual"

    # id: int
    # genotype: Genotype
    # fitness: float
    # population_table = "population"