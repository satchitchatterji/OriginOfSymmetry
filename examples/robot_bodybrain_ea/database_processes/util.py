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
from parameters import ExperimentParameters
from revolve2.experimentation.database import OpenMethod, open_database_sqlite
from sqlalchemy import select

def create_engine(filepath=None):
    if filepath is None:
        filepath = "../"+config.DATABASE_FILE

    return open_database_sqlite(
        filepath, open_method=OpenMethod.OPEN_IF_EXISTS
    )

def open_experiment_table(filepath=None, dbengine=None):
    if filepath is None:
        filepath = "../"+config.DATABASE_FILE

    if dbengine is None:
        dbengine = open_database_sqlite(
            filepath, open_method=OpenMethod.OPEN_IF_EXISTS
        )

    # get the parameters table
    df_params = pandas.read_sql(
        select(ExperimentParameters.id, ExperimentParameters.body_multineat_parameters, ExperimentParameters.brain_multineat_parameters, ExperimentParameters.evolution_parameters),
        dbengine,
    )

    df = pandas.read_sql(
        select(
            Experiment.id.label("experiment_id"),
            Experiment.parameters_id,
            Generation.generation_index,
            Individual.fitness,
            Individual.symmetry,
            Individual.xy_positions,
            Individual.population_index,

        )
        .join_from(Experiment, Generation, Experiment.id == Generation.experiment_id)
        .join_from(Generation, Population, Generation.population_id == Population.id)
        .join_from(Population, Individual, Population.id == Individual.population_id),
        dbengine,
    )

    

    return df, df_params


if __name__ == "__main__":
    df = open_experiment_table()
    print(df)