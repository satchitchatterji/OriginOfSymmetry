import sys
sys.path.append("..")

import config
import pandas
from experiment import Experiment
from generation import Generation
from individual import Individual
from population import Population
from revolve2.experimentation.database import OpenMethod, open_database_sqlite
from sqlalchemy import select


def open_experiment_table(dbengine=None):
    if dbengine is None:
        dbengine = open_database_sqlite(
            "../"+config.DATABASE_FILE, open_method=OpenMethod.OPEN_IF_EXISTS
        )

    df = pandas.read_sql(
        select(
            Experiment.id.label("experiment_id"),
            Generation.generation_index,
            Individual.fitness,
            Individual.symmetry,
            Individual.xy_positions
        )
        .join_from(Experiment, Generation, Experiment.id == Generation.experiment_id)
        .join_from(Generation, Population, Generation.population_id == Population.id)
        .join_from(Population, Individual, Population.id == Individual.population_id),
        dbengine,
    )

    return df


if __name__ == "__main__":
    df = open_experiment_table()
    print(df)