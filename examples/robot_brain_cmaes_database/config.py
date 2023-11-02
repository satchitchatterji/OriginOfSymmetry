"""Configuration parameters for this example."""

from revolve2.ci_group.modular_robots import gecko

DATABASE_FILE = "database.sqlite"
NUM_REPETITIONS = 1
NUM_SIMULATORS = 1
INITIAL_STD = 0.5
NUM_GENERATIONS = 1
BODY = gecko()
