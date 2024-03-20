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
import itertools


from revolve2.modular_robot import ModularRobot


#import config 
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

from parameters import ExperimentParameters, EvolutionParameters, make_multineat_params

import wandb
import time
CUR_TIME = time.strftime("%Y-%m-%dT%H:%M:%S")

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
    tournament_size: int = 2,
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
            lambda _, fitnesses: selection.tournament(rng, fitnesses, k=tournament_size),
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

def plot_fitnesses(max_fitness_values, mean_fitness_values, exp_name = ''):
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
    file_name =f"fitnesses_{exp_name}_{date_time}.png"
    
    plt.savefig(file_name)
    print("graph saved in "+ file_name)
    plt.close()

def run_experiment(session, exp_num: int, experiment_parameters: ExperimentParameters, steer = False, record_settings = RecordSettings()) -> Individual:
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

    experiment = Experiment(rng_seed=rng_seed, steer = steer, parameters=experiment_parameters)

    logging.info("Saving experiment configuration.")
    
    session.add(experiment)
    session.commit()

    # Intialize the evaluator that will be used to evaluate robots. TODO fitness function as parameter

     #TODO add fps value and size of view usage


    evaluator = Evaluator(headless=True, num_simulators=experiment_parameters.evolution_parameters.num_simulators, record_settings=record_settings)

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
            body_params=experiment_parameters.body_multineat_parameters,
            brain_params=experiment_parameters.brain_multineat_parameters,

        )
        for _ in range(experiment_parameters.evolution_parameters.population_size)
    ]

    # Evaluate the initial population.
    logging.info("Evaluating initial population.")
    initial_fitnesses, initial_sym, initial_xy = evaluator.evaluate(
        [genotype.develop() for genotype in initial_genotypes], generation_index=0,
        steer=steer,
        simulation_time=experiment_parameters.evolution_parameters.sim_time
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
    while generation_index < experiment_parameters.evolution_parameters.num_generations:
        logging.info(f"Generation {generation_index } / {experiment_parameters.evolution_parameters.num_generations}.")

        # Create offspring.
        parents = select_parents(rng, population, experiment_parameters.evolution_parameters.offspring_size)
        offspring_genotypes = [
            Genotype.crossover(
                population.individuals[parent1_i].genotype,
                population.individuals[parent2_i].genotype,
                rng,
                experiment_parameters.body_multineat_parameters,
                experiment_parameters.brain_multineat_parameters,
            ).mutate(innov_db_body, innov_db_brain, rng, experiment_parameters.body_multineat_parameters, experiment_parameters.brain_multineat_parameters)
            for parent1_i, parent2_i in parents
        ]

        # Evaluate the offspring.
        offspring_fitnesses, offspring_symmetries, offspring_xy = evaluator.evaluate(
            [genotype.develop() for genotype in offspring_genotypes], generation_index=generation_index,
            steer=steer,
            simulation_time=experiment_parameters.evolution_parameters.sim_time

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
            experiment_parameters.evolution_parameters.tournament_size,
            
        )

        # Make it all into a generation and save it to the database.
        generation = Generation(
            experiment=experiment,
            generation_index=generation.generation_index + 1,
            population=population,
        )
        logging.info("Saving generation.")
        session.add(generation)
        session.commit()


        
        mean_fitness_values.append(find_mean_fitness(population))

        # Find the new best robot
        best_robot = find_best_robot(best_robot, population)

        max_fitness_values.append(best_robot.fitness)

        wandb.log({"max_fitness": max_fitness_values[-1], "mean_fitness": mean_fitness_values[-1]})
        # logging.info(f"Best robot until now: {best_robot.fitness}")
        #logging.info(f"Genotype pickle: {pickle.dumps(best_robot)!r}")

        # Increase the generation index counter.
        generation_index += 1
    
    plot_fitnesses(max_fitness_values, mean_fitness_values, "steer" if steer else "nosteer")
    return best_robot


def main(exp_parameters_array: list[ExperimentParameters] , repeated_params = False, best_videos_dir = 'best_robots_videos',  exp_rs = RecordSettings(save_robot_view=False)) -> None:
    """
    Run multiple experiments and save the video of the best robot for each one of them.

    The videos with the steering functionality will be saved in a subfolder named "generation_1" while the video without it will be saved in a subfolder named "generation_0".

    :param steer: Whether to enable steering.
    :param best_videos_dir: The directory to save the best robot videos.
    :param exp_rc: The record settings for the experiment.
    """

    for experiment_parameters in exp_parameters_array:
        dbengine = open_database_sqlite(
            experiment_parameters.evolution_parameters.database_file, open_method=OpenMethod.OPEN_OR_CREATE
        )
        # Create the structure of the database.
        Base.metadata.create_all(dbengine)

        # Save the experiment parameters to the database.
        with Session(dbengine) as session:
            # check if the same instanse of ExperimentParameters is already in the database
            experiment_parameters_copy = session.query(ExperimentParameters).filter_by(evolution_parameters=experiment_parameters.evolution_parameters, brain_multineat_parameters=experiment_parameters.brain_multineat_parameters, body_multineat_parameters=experiment_parameters.body_multineat_parameters).first()

            if experiment_parameters_copy:
                if repeated_params:
                    experiment_parameters = experiment_parameters_copy
                else:
                    print("Experiment with the same parameters already exists in the database. If you want to run the experiment with the same parameters again, set repeated_params to True")
                    continue
            else:

                session.add(experiment_parameters)
                session.commit()

            

            # Set up standard logging.
            setup_logging(file_name="log.txt")
    
    

            # Run the experiment several times.
            best_robots = []
            for rep in range(experiment_parameters.evolution_parameters.num_repetitions):
                config_dict = vars(experiment_parameters.evolution_parameters)
                config_dict.update({"brain_overall":experiment_parameters.brain_multineat_parameters.OverallMutationRate, "body_overall":experiment_parameters.body_multineat_parameters.OverallMutationRate})                
                wandb.init(project="integration test", config=config_dict, name=f"{CUR_TIME}_rep_{rep}")
                
                best_robot = run_experiment(session, rep, steer = experiment_parameters.evolution_parameters.steer, record_settings=exp_rs, experiment_parameters=experiment_parameters)
                best_robots.append(best_robot)

                wandb.finish()
        

            for best_robot in best_robots:
                # get the experiment id
                population_id = best_robot.population_id
                experiment_id = session.query(Generation).filter_by(population_id=population_id).first().experiment_id

                video_name = f"{experiment_id}_{best_robot.id}_{'steer' if experiment_parameters.evolution_parameters.steer else 'nosteer'}"
                developed_robot = best_robot.genotype.develop()
                record_settings = RecordSettings(video_directory=best_videos_dir, generation_step=1, save_robot_view=True, video_name=video_name, fps=24, delete_at_init=False)
                evaluator = Evaluator(headless= True, num_simulators=1, record_settings=record_settings)

                fitness = evaluator.evaluate([developed_robot], generation_index= int(experiment_parameters.evolution_parameters.steer), steer=experiment_parameters.evolution_parameters.steer, simulation_time=experiment_parameters.evolution_parameters.sim_time)[0]




if __name__ == "__main__":
    #main(steer=True, best_videos_dir = 'best_robots_videos', exp_rs=RecordSettings(save_robot_view=True, generation_step=1, delete_at_init=True ))

    same_for_brain_and_body = False
    
    
    parameters_to_test = {
        "brain_multineat_parameters": {
            "OverallMutationRate": [0.09],
        },
        "body_multineat_parameters": {
            "OverallMutationRate": [0.09],
        },
        "evolution_parameters": {
            "steer" : [False],
            "population_size": 20, 
            "num_generations": 10,
            "offspring_size": 20,
            "tournament_size": [3],
            "database_file" : "./database_sym.sqlite"
        }
        
    }
    # parameters_to_test = {
    #     "brain_multineat_parameters": {
    #         "OverallMutationRate": 0.15,
    #     },
    #     "body_multineat_parameters": {
    #         "OverallMutationRate": 0.09,
    #     },
    #     "evolution_parameters": {
    #         "steer" : [True, False],
    #         "population_size": 8,
    #         "num_generations": 2,
    #         "offspring_size": 8,
    #         "tournament_size": [3,6],
    #         "database_file" : "./database_sym.sqlite"
    #     }
    # }


    if same_for_brain_and_body:
        parameters_to_test["brain_multineat_parameters"] = parameters_to_test["body_multineat_parameters"]

    # create an array of ExperimentParameters all the possible combinations of the parameters to test
    exp_parameters_array = []

    # create dynamically all the possible combinations of the parameters to test

    # Extracting the keys and values, separating lists from single values
    # Reformatting approach to ensure consistency in parameter formatting
    keys, values = [], []
    for param_group, params in parameters_to_test.items():
        for param, value in params.items():
            

            keys.append((param_group, param)) 
            if isinstance(value, list):
                values.append(value)
            else:
                values.append([value])
    # Generate all combinations of variable parameters
    combinations = list(itertools.product(*tuple(values)))
    

    # Creating an ExperimentParameters instance for each combination
    for combination in combinations:
        exp_params = ExperimentParameters()
        for (param_group, param), value in zip(keys, combination):
            if param_group == "evolution_parameters" and param == "population_size":
                exp_params.body_multineat_parameters.PopulationSize = value
                exp_params.brain_multineat_parameters.PopulationSize = value
            elif param_group == "evolution_parameters" and param == "tournament_size":
                exp_params.body_multineat_parameters.TournamentSize = value
                exp_params.brain_multineat_parameters.TournamentSize = value
            
            setattr(getattr(exp_params, param_group), param, value)
        
        exp_parameters_array.append(exp_params)

    

    #exp_parameters_array = [ExperimentParameters()]

    
    #main(steer=True, best_videos_dir = 'best_robots_videos', experiment_parameters = experiment_parameters)
    main(exp_parameters_array=exp_parameters_array, best_videos_dir = 'best_robots_videos', repeated_params=True)




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