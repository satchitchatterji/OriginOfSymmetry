"""
Run the example.

A modular robot body and brain will be optimized using a simple evolutionary algorithm.
The genotypes for both body and brain are CPPNWIN.
"""

"""
Set up an experiment that optimizes the body and brain of a robot using a simple evolutionary algorithm.

The genotypes for both body and brain are CPPNWIN.

Before starting this tutorial, it is useful to look at the 'experiment_setup', 'evaluate_multiple_isolated_robots', and 'simple_ea_xor' examples.
It is also nice to understand the concept of a cpg brain and CPPN, although not really needed.

You learn:
- How to optimize the body and brain of a robot using an EA.
"""

import logging
import pickle

from revolve2.modular_robot import ModularRobot


import config
import multineat
import numpy as np
import numpy.typing as npt

from base import Base
from evaluator import Evaluator
from experiment import Experiment
from generation import Generation
from genotype import Genotype
from individual import Individual
from population import Population
from revolve2.ci_group.logging import setup_logging
from revolve2.ci_group.rng import make_rng, seed_from_time
from revolve2.experimentation.optimization.ea import population_management, selection
from revolve2.experimentation.database import OpenMethod, open_database_sqlite
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from statistics import mean 
from numpy import *
import math
import matplotlib.pyplot as plt

import datetime
from revolve2.simulation.running import RecordSettings


def select_parents(
    rng: np.random.Generator,
    population: list[Individual],
    offspring_size: int,
) -> npt.NDArray[np.float_]:
    """
    Select pairs of parents using a tournament.

    :param rng: Random number generator.
    :param population: The population to select from.
    :param offspring_size: The number of parent pairs to select.
    :returns: Pairs of indices of selected parents. offspring_size x 2 ints.
    """
    return np.array(
        [
            selection.multiple_unique(
                2,
                [individual.genotype for individual in population.individuals],
                [individual.fitness for individual in population.individuals],
                lambda _, fitnesses: selection.tournament(rng, fitnesses, k=1),
            )
            for _ in range(offspring_size)
        ],
    )


def select_survivors(
    rng: np.random.Generator,
    original_population: list[Individual],
    offspring_population: list[Individual],
) -> list[Individual]:
    """
    Select survivors using a tournament.

    :param rng: Random number generator.
    :param original_population: The population the parents come from.
    :param offspring_population: The offspring.
    :returns: A newly created population.
    """
    original_survivors, offspring_survivors = population_management.steady_state(
        [i.genotype for i in original_population.individuals],
        [i.fitness for i in original_population.individuals],
        [i.genotype for i in offspring_population.individuals],
        [i.fitness for i in offspring_population.individuals],
        lambda n, genotypes, fitnesses: selection.multiple_unique(
            n,
            genotypes,
            fitnesses,
            lambda _, fitnesses: selection.tournament(rng, fitnesses, k=2),
        ),
    )

    return Population(
        [
            Individual(
                original_population.individuals[i].genotype,
                original_population.individuals[i].fitness,
                original_population.individuals[i].symmetry,
                original_population.individuals[i].xy_positions,
            )
            for i in original_survivors
        ]
        + [
            Individual(
                offspring_population.individuals[i].genotype,
                offspring_population.individuals[i].fitness,
                offspring_population.individuals[i].symmetry,
                offspring_population.individuals[i].xy_positions,
            )
            for i in offspring_survivors
        ]
    )

def find_best_robot(
    current_best: Individual | None, population: Population
) -> Individual:
    """
    Return the best robot between the population and the current best individual.

    :param current_best: The current best individual.
    :param population: The population.
    :returns: The best individual.
    """
    return max(
        population.individuals + [] if current_best is None else population.individuals + [current_best],
        key=lambda x: x.fitness,
    )

def find_mean_fitness(
        population: Population
) -> Individual:
    """
    Return the mean fitness of the population.

    :param population: The population.
    :returns: The mean.
    """
    fitnesses = [individual.fitness for individual in population.individuals]
    return mean(fitnesses)

def plot_fitnesses(max_fitness_values, mean_fitness_values):
    # plot the fitness values
    plt.plot(max_fitness_values, label='max fitness')
    plt.plot(mean_fitness_values, label='mean fitness')
    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    plt.legend()
    plt.title('Fitness over generations')
    
    # save plot as png file with name of the current time
    now = datetime.datetime.now()

    date_time = now.strftime("%m-%d-%Y_%H-%M-%S")
    file_name = 'graph experiment '+date_time+'.png'
    
    plt.savefig(file_name)
    print("graph saved in "+ file_name)
    plt.close()

def run_experiment(dbengine: Engine, exp_num: int, steer = False, record_settings = RecordSettings()) -> Individual:
    """
    Run the experiment to optimize the body and brain of a robot using an evolutionary algorithm.

    :param dbengine: The database engine.
    :param exp_num: The experiment number.
    :param steer: Whether to enable steering.
    :param record_settings: The record settings for the simulation.
    :returns: The best individual found during the experiment.
    """
    logging.info("----------------")
    logging.info("Start experiment")

    # Set up the random number generater.
    rng_seed = seed_from_time()
    rng = make_rng(rng_seed)

    # Create and save the experiment instance.
    experiment = Experiment(rng_seed=rng_seed, steer = steer)
    logging.info("Saving experiment configuration.")
    with Session(dbengine) as session:
        session.add(experiment)
        session.commit()

    # Intialize the evaluator that will be used to evaluate robots. TODO fitness function as parameter

     #TODO add fps value and size of view usage


    evaluator = Evaluator(headless=True, num_simulators=config.NUM_SIMULATORS, record_settings=record_settings)

    # CPPN innovation databases.
    # If you don't understand CPPN, just know that a single database is shared in the whole evolutionary process.
    # One for body, and one for brain.
    innov_db_body = multineat.InnovationDatabase()
    innov_db_brain = multineat.InnovationDatabase()

    # Create an initial population.
    logging.info("Generating initial population.")
    initial_genotypes = [
        Genotype.random(
            innov_db_body=innov_db_body,
            innov_db_brain=innov_db_brain,
            rng=rng,
        )
        for _ in range(config.POPULATION_SIZE)
    ]

    # Evaluate the initial population.
    logging.info("Evaluating initial population.")
    initial_fitnesses, initial_sym, initial_xy = evaluator.evaluate(
        [genotype.develop() for genotype in initial_genotypes], generation_index=0,
        steer=steer
    )

    # Create a population of individuals, combining genotype with fitness.
    population = Population(
        [
            Individual(genotype, fitness, sym, xy)
            for genotype, fitness, sym, xy in zip(
                initial_genotypes, initial_fitnesses, initial_sym, initial_xy, strict=True
            )
        ]
    )

    # Finish the zeroth generation and save it to the database.
    generation = Generation(
        experiment=experiment, generation_index=0, population=population
    )
    logging.info("Saving generation.")
    with Session(dbengine, expire_on_commit=False) as session:
        session.add(generation)
        session.commit()

    # # Save the best robot
    best_robot = find_best_robot(None, population)

    # Set the current generation to 1.
    generation_index = 1

    # list to store the fitness values
    max_fitness_values = []
    mean_fitness_values = []

    # Start the actual optimization process.
    logging.info("Start optimization process.")
    while generation_index < config.NUM_GENERATIONS:
        logging.info(f"Generation {generation_index } / {config.NUM_GENERATIONS}.")

        # Create offspring.
        parents = select_parents(rng, population, config.OFFSPRING_SIZE)
        offspring_genotypes = [
            Genotype.crossover(
                population.individuals[parent1_i].genotype,
                population.individuals[parent2_i].genotype,
                rng,
            ).mutate(innov_db_body, innov_db_brain, rng)
            for parent1_i, parent2_i in parents
        ]

        # Evaluate the offspring.
        offspring_fitnesses, offspring_symmetries, offspring_xy = evaluator.evaluate(
            [genotype.develop() for genotype in offspring_genotypes], generation_index=generation_index
        )

        # Make an intermediate offspring population.
        offspring_population = Population(
            [
                Individual(genotype, fitness, sym, xy)
                for genotype, fitness, sym, xy in zip(offspring_genotypes, 
                                                      offspring_fitnesses, 
                                                      offspring_symmetries, 
                                                      offspring_xy
                                                      )
            ]
        )

        # Create the next population by selecting survivors.
        population = select_survivors(
            rng,
            population,
            offspring_population,
        )

        # Make it all into a generation and save it to the database.
        generation = Generation(
            experiment=experiment,
            generation_index=generation.generation_index + 1,
            population=population,
        )
        logging.info("Saving generation.")
        with Session(dbengine, expire_on_commit=False) as session:
            session.add(generation)
            session.commit()


        
        mean_fitness_values.append(find_mean_fitness(population))

        # Find the new best robot
        best_robot = find_best_robot(best_robot, population)

        max_fitness_values.append(best_robot.fitness)

        # logging.info(f"Best robot until now: {best_robot.fitness}")
        logging.info(f"Genotype pickle: {pickle.dumps(best_robot)!r}")

        # Increase the generation index counter.
        generation_index += 1
    
    plot_fitnesses(max_fitness_values, mean_fitness_values)
    return best_robot


def main(steer: bool, best_videos_dir = 'best_robots_videos',  exp_rc = RecordSettings(save_robot_view=False)) -> None:
    """
    Run multiple experiments and save the video of the best robot for each one of them.

    The videos with the steering functionality will be saved in a subfolder named "generation_1" while the video without it will be saved in a subfolder named "generation_0".

    :param steer: Whether to enable steering.
    :param best_videos_dir: The directory to save the best robot videos.
    :param exp_rc: The record settings for the experiment.
    """
    # Set up standard logging.
    setup_logging(file_name="log.txt")



    # Open the database, only if it does not already exists.
    # dbengine = open_database_sqlite(
    #     config.DATABASE_FILE, open_method=OpenMethod.NOT_EXISTS_AND_CREATE
    # )
    # dbengine = open_database_sqlite(
    #     config.DATABASE_FILE, open_method='TRUNCATE_AND_CREATE'
    # )
    # Changed because the above was not using the correct argument type for open_method
    dbengine = open_database_sqlite(
        config.DATABASE_FILE, open_method=OpenMethod.OPEN_OR_CREATE
    )
    # Create the structure of the database.
    Base.metadata.create_all(dbengine)

    # Run the experiment several times.
    best_robots = []
    for rep in range(config.NUM_REPETITIONS):
        best_robot = run_experiment(dbengine, rep, steer = steer, record_settings=exp_rc)
        best_robots.append(best_robot)

    # GETTING THE BEST ROBOT FROM DATABASE, not needed if we get it from run_experiment
        
    # get the best robot from the last experiment (the last experiment is the one with the highest id)
    # with Session(dbengine) as session:
    #     last_experiment = session.query(Experiment).order_by(Experiment.id.desc()).first()
    #     last_generation = session.query(Generation).filter_by(experiment_id=last_experiment.id).order_by(Generation.generation_index.desc()).first()
    #     # get the id of the population from last generation
    #     last_population_id = last_generation.population_id
    #     # get the best individual with this population id
    #     best_individual = session.query(Individual).filter_by(population_id=last_population_id).order_by(Individual.fitness.desc()).first()
    #     # get the genotype id of the best individual
    #     best_genotype_id = best_individual.genotype_id
    #     # get the genotype with this id
    #     best_genotype = session.query(Genotype).filter_by(id=best_genotype_id).first()
    #     # get the phenotype of this genotype
    #     best_robot = best_genotype.develop()

    # get all the files from the the best robots videos directory
    

    for best_robot in best_robots:
        video_name = f"{best_robot.id}_{'steer' if steer else 'nosteer'}"
        developed_robot = best_robot.genotype.develop()
        record_settings = RecordSettings( video_directory=best_videos_dir, generation_step=1, save_robot_view=True, video_name=video_name, fps=24, delete_at_init=False)
        evaluator = Evaluator(headless= True, num_simulators=1, record_settings=record_settings)

        fitness = evaluator.evaluate([developed_robot], generation_index= int(steer), steer=steer)[0]




if __name__ == "__main__":
    main(steer=True, best_videos_dir = 'best_robots_videos')
    main(steer=False, best_videos_dir = 'best_robots_videos')
