import sys
import os
import inspect
#sys.path.append("..")


# get path of the current file
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# get the path of two directories above
parentdir = os.path.dirname(currentdir)
# add the parent directory to the sys path
sys.path.insert(0, parentdir)



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