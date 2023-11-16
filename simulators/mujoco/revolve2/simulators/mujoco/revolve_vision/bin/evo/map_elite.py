#!/usr/bin/env python3

import argparse
import json
import logging
import multiprocessing
import pprint
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Optional

import humanize
from qdpy.base import ParallelismManager

from abrain.core.genome import logger as genome_logger
from src.default_experiment.evaluator import Evaluator
from src.default_experiment.scenario import Scenario
from src.evolution.common import Tee
from src.evolution.map_elite import Grid, Algorithm, Logger, QDIndividual
from src.misc.config import Config
from src.misc.genome import RVGenome


class Options:
    def __init__(self):
        self.id: Optional[int] = None
        self.base_folder: str = "./tmp/qdpy/toy-revolve"
        self.run_folder: str = None  # Automatically filled in
        self.snapshots: int = 10
        self.overwrite: bool = False

        self.verbosity: int = 1

        self.seed: Optional[int] = None
        self.batch_size: int = 10
        self.budget: int = 100
        self.tournament: int = 5
        self.threads: int = 1

        # number of initial mutations for abrain's genome
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
        group.add_argument('--snapshots', dest="snapshots", type=int,
                           help="Number of snapshots to keep")
        group.add_argument('--overwrite', dest="overwrite", action='store_true',
                           help="[DANGER] Clear data folder before populating")
        group.add_argument('--verbosity', dest="verbosity",
                           help="Specify the output verbosity level")

        group = parser.add_argument_group("Evolution",
                                          "Defines evolution parameters")
        group.add_argument('--seed', dest="seed", metavar='S', type=int,
                           help="Seed for the Random Number Generator")
        group.add_argument('--threads', dest="threads", metavar='N', type=int,
                           help="Number of concurrent evaluations")
        group.add_argument('--batch-size', dest="batch_size", metavar='N', type=int,
                           help="Number of individuals per batch")
        group.add_argument('--tournament-size', dest="tournament", metavar='N', type=int,
                           help="Number of individuals competing in curiosity-based selection")
        group.add_argument('--budget', dest="budget", metavar='N',
                           type=int, help="Number of evaluations")

        group = parser.add_argument_group("Init",
                                          "Initial population parameters")
        group.add_argument('--initial-mutations', dest="initial_mutations",
                           metavar='N', type=int,
                           help="Mutations for the initial population")

        # group = parser.add_argument_group("Environment", "Environmental parameters")

        # group = parser.add_argument_group("Robot", "Robot parameters")

        Config.argparse_setup(parser)


def eval_mujoco(ind: QDIndividual):
    assert isinstance(ind, QDIndividual)
    assert isinstance(ind.genome, RVGenome)
    assert ind.id() is not None, "ID-less individual"
    r: Evaluator.Result = Evaluator.evaluate_evo(ind.genome)
    ind.update(r)
    # print(ind, r)
    return ind


def main():
    start = time.perf_counter()

    # Parse command-line arguments
    args = Options()
    parser = argparse.ArgumentParser(description="QDPy-based map elite optimizer")
    Options.populate(parser)
    parser.parse_args(namespace=args)

    # =========================================================================

    # Log everything to file except for the progress bar
    tee = Tee(filter_out=lambda msg: "\r" in msg or "\x1b" in msg)
    tee.register()  # Start capturing now (including logging's reference to stdout)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s]|[%(levelname)s]|[%(module)s] %(message)s",
        stream=sys.stdout
    )
    genome_logger.setLevel(logging.INFO)

    for m in ['matplotlib', 'OpenGL.arrays.arraydatatype', 'OpenGL.acceleratesupport']:
        logger = logging.getLogger(m)
        logger.setLevel(logging.WARNING)
        logging.info(f"Muting {logger}")

    logging.captureWarnings(True)

    # =========================================================================

    grid = Grid(shape=(16, 16),
                max_items_per_bin=1,
                fitness_domain=Scenario.fitness_bounds(),
                features_domain=Scenario.descriptor_bounds())

    logging.info(f"Grid size: {grid.shape}")
    logging.info(f"   bounds: {grid.features_domain}")
    logging.info(f"     bins: "
                 f"{[(d[1]-d[0]) / s for d, s in zip(grid.features_domain, grid.shape)]}")

    algo = Algorithm(grid, args, labels=[Scenario.fitness_name(), *Scenario.descriptor_names()])
    run_folder = Path(args.run_folder)

    # Prepare (and store) configuration
    Config.argparse_process(args)
    Config.evolution = args.__dict__
    Config._evolving = True

    config_path = run_folder.joinpath("config.json")
    Config.write_json(config_path)
    logging.info(f"Stored configuration in {config_path.absolute()}")

    # Create a logger to pretty-print everything and generate output data files
    save_every = round(args.budget / (args.batch_size * args.snapshots))
    logger = Logger(algo,
                    save_period=save_every,
                    log_base_path=args.run_folder)

    tee.set_log_path(run_folder.joinpath("log"))

    with ParallelismManager(max_workers=args.threads) as mgr:
        mgr.executor._mp_context = multiprocessing.get_context("fork")  # TODO: Very brittle
        logging.info("Starting illumination!")
        best = algo.optimise(evaluate=eval_mujoco, executor=mgr.executor, batch_mode=True)

    if best is not None:
        with open(run_folder.joinpath("best.json"), 'w') as f:
            data = {
                "id": best.id(), "parents": best.genome.parents(),
                "fitnesses": best.fitnesses,
                "descriptors": best.descriptors,
                "stats": best.stats,
                "genome": best.genome.to_json()
            }
            logging.info(f"best:\n{pprint.pformat(data)}")
            json.dump(data, f)
    else:
        logging.warning("No best individual found")

    # Print results info
    logging.info(algo.summary())

    # Plot the results
    logger.summary_plots()

    logging.info(f"All results are available under {logger.log_base_path}")
    logging.info(f"Unified storage file is {logger.log_base_path}/{logger.final_filename}")

    duration = humanize.precisedelta(timedelta(seconds=time.perf_counter() - start))
    logging.info(f"Completed evolution in {duration}")


if __name__ == '__main__':
    main()
