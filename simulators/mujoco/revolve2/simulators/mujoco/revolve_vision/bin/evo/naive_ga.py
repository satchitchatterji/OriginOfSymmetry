#!/usr/bin/env python3

import argparse
import logging
import sys
import time
from datetime import timedelta
from typing import Dict

import humanize

import abrain
from src.default_experiment.evaluator import Evaluator
from src.default_experiment.scenario import Scenario
from src.evolution.naive_ga import Evolver, NaiveTournamentSelection
from src.misc.config import Config
from src.misc.genome import RVGenome


# from genotype import random as random_genotype
# from optimizer import Optimizer

class Options:
    def __init__(self):
        self.id: int = None
        self.base_folder: str = "./tmp/"
        self.overwrite: bool = False

        self.seed: int = None
        self.pop_size: int = 5
        self.offsprings: int = 5
        self.generations: int = 3
        self.threads: int = 1

        self.items: int = 10

        self.brain: str = abrain.ANN.__name__

        self.reproduction: Dict[Evolver.Reproduction, float] = {
            Evolver.Reproduction.CLONE.name: 1
        }

        # number of initial mutations for body and brain CPPNWIN networks
        self.initial_mutations: int = 10

    @staticmethod
    def populate(parser: argparse.ArgumentParser):
        group = parser.add_argument_group("Storage",
                                          "Defines run name, location...")
        group.add_argument('--id', dest="id", metavar='ID', type=str,
                           help="Name of the run (use an int if no seed is provided)")
        group.add_argument('--data-folder', dest="base_folder", metavar='DIR',
                           type=str,
                           help="(relative) path under which to store data")
        group.add_argument('--overwrite', dest="overwrite", action='store_true',
                           help="[DANGER] Clear data folder before populating")

        group = parser.add_argument_group("Evolution",
                                          "Defines evolution parameters")
        group.add_argument('--seed', dest="seed", metavar='S', type=int,
                           help="Seed for the Random Number Generator")
        group.add_argument('--threads', dest="threads", metavar='N', type=int,
                           help="Number of concurrent evaluations")
        group.add_argument('--pop-size', dest="pop_size", metavar='N', type=int,
                           help="Number of individuals per generation")
        group.add_argument('--generations', dest="generations", metavar='N',
                           type=int, help="Number of generations")
        group.add_argument('--offsprings', dest="offsprings", metavar='N',
                           type=int, help="Redundant?")

        group = parser.add_argument_group("Init",
                                          "Initial population parameters")
        group.add_argument('--initial-mutations', dest="initial_mutations",
                           metavar='N', type=int,
                           help="Mutations for the initial population")

        group = parser.add_argument_group("Environment",
                                          "Environmental parameters")
        group.add_argument('--env-items', dest="items", metavar='N', type=int,
                           help="Number of items in the environment")

        group = parser.add_argument_group("Robot",
                                          "Robot parameters")
        group.add_argument('--brain-type', dest="brain", metavar='B', type=str,
                           help="Brain type to use", choices=Config.brain_types)


def main():
    tee.register()

    start = time.perf_counter()
    args = Options()
    parser = argparse.ArgumentParser(description="Toy optimizer")
    Options.populate(parser)
    parser.parse_args(namespace=args)
    # parser.print_help()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s]|[%(levelname)s]|[%(module)s] %(message)s",
        stream=sys.stdout
    )

    Config.brain_type = args.brain
    evolver = Evolver(args, algorithm=NaiveTournamentSelection())

    evolver.set_callbacks(
        random=RVGenome.random,
        mutate=RVGenome.mutated,
        crossover=None,
        evaluate=Evaluator.evaluate_evo,
    )

    Scenario.generate_initial_items(evolver.rng, args.items)
    def save_env(): Scenario.serialize(evolver.run_folder.joinpath("last/env.json"))
    evolver.add_misc_callback(Evolver.CallbackType.BEFORE_NEW_POP, save_env)

    logging.info("Starting evolution")
    evolver.run(None)

    duration = humanize.precisedelta(timedelta(seconds=time.perf_counter() - start))
    logging.info(f"Completed evolution in {duration}")

    tee.teardown()


if __name__ == "__main__":
    main()
