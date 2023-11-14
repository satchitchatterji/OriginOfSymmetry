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
                [individual.genotype for individual in population],
                [individual.fitness for individual in population],
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
        [i.genotype for i in original_population],
        [i.fitness for i in original_population],
        [i.genotype for i in offspring_population],
        [i.fitness for i in offspring_population],
        lambda n, genotypes, fitnesses: selection.multiple_unique(
            n,
            genotypes,
            fitnesses,
            lambda _, fitnesses: selection.tournament(rng, fitnesses, k=2),
        ),
    )

    return [
        Individual(
            original_population[i].genotype,
            original_population[i].fitness,
        )
        for i in original_survivors
    ] + [
        Individual(
            offspring_population[i].genotype,
            offspring_population[i].fitness,
        )
        for i in offspring_survivors
    ]


def find_best_robot(
    current_best: Individual | None, population: list[Individual]
) -> Individual:
    """
    Return the best robot between the population and the current best individual.

    :param current_best: The current best individual.
    :param population: The population.
    :returns: The best individual.
    """
    return max(
        population + [] if current_best is None else [current_best] + population,
        key=lambda x: x.fitness,
    )

def find_mean_fitness(
        population: list[Individual]
) -> Individual:
    """
    Return the mean fitness of the population.

    :param population: The population.
    :returns: The mean.
    """
    fitnesses = [individual.fitness for individual in population]
    return mean(fitnesses)

    def plot_fitnesses(max_fitness_values, mean_fitness_values, experiment_num):
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
        file_name = f'graph experiment {experiment_num} '+date_time+'.png'
        
        plt.savefig(file_name)
        print("graph saved in "+ file_name)
        plt.close()

def run_experiment(dbengine: Engine, exp_num: int) -> None:
    logging.info("----------------")
    logging.info("Start experiment")

    # Set up the random number generater.
    rng_seed = seed_from_time()
    rng = make_rng(rng_seed)

    # Create and save the experiment instance.
    experiment = Experiment(rng_seed=rng_seed)
    logging.info("Saving experiment configuration.")
    with Session(dbengine) as session:
        session.add(experiment)
        session.commit()

    # Intialize the evaluator that will be used to evaluate robots. TODO fitness function as parameter
    evaluator = Evaluator(headless=True, num_simulators=config.NUM_SIMULATORS)

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
    initial_fitnesses = evaluator.evaluate(
        [genotype.develop() for genotype in initial_genotypes]
    )

    # Create a population of individuals, combining genotype with fitness.
    population = Population(
        [
            Individual(genotype, fitness)
            for genotype, fitness in zip(
                initial_genotypes, initial_fitnesses, strict=True
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

    # Save the best robot
    best_robot = find_best_robot(None, population)

    # Set the current generation to 0.
    generation_index = 0

    # list to store the fitness values
    max_fitness_values = []
    mean_fitness_values = []

    # Start the actual optimization process.
    logging.info("Start optimization process.")
    while generation_index < config.NUM_GENERATIONS:
        logging.info(f"Generation {generation_index + 1} / {config.NUM_GENERATIONS}.")

        # Create offspring.
        parents = select_parents(rng, population, config.OFFSPRING_SIZE)
        offspring_genotypes = [
            Genotype.crossover(
                population[parent1_i].genotype,
                population[parent2_i].genotype,
                rng,
            ).mutate(innov_db_body, innov_db_brain, rng)
            for parent1_i, parent2_i in parents
        ]

        # Evaluate the offspring.
        offspring_fitnesses = evaluator.evaluate(
            [genotype.develop() for genotype in offspring_genotypes]
        )

        # Make an intermediate offspring population.
        offspring_population = Population(
            [
                Individual(genotype, fitness)
                for genotype, fitness in zip(offspring_genotypes, offspring_fitnesses)
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


        # Find the new best robot
        best_robot = find_best_robot(best_robot, population)

        max_fitness_values.append(best_robot.fitness)
        mean_fitness_values.append(find_mean_fitness(population))

        logging.info(f"Best robot until now: {best_robot.fitness}")
        logging.info(f"Genotype pickle: {pickle.dumps(best_robot)!r}")

        # Increase the generation index counter.
        generation_index += 1
    
    plot_fitnesses(max_fitness_values, mean_fitness_values, experiment_num)


def main() -> None:
    """Run the program."""
    # Set up standard logging.
    setup_logging(file_name="log.txt")

    # Open the database, only if it does not already exists.
    # dbengine = open_database_sqlite(
    #     config.DATABASE_FILE, open_method=OpenMethod.NOT_EXISTS_AND_CREATE
    # )
    dbengine = open_database_sqlite(
        config.DATABASE_FILE, open_method=OpenMethod.NOT_EXISTS_AND_CREATE
    )
    # Create the structure of the database.
    Base.metadata.create_all(dbengine)

    # Run the experiment several times.
    for rep in range(config.NUM_REPETITIONS):
        run_experiment(dbengine, rep)

if __name__ == "__main__":
    main()
