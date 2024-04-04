"""Evaluator class."""

import asyncio

from revolve2.ci_group import fitness_functions, terrains
from revolve2.ci_group.simulation import create_batch_multiple_isolated_robots_standard
from revolve2.modular_robot import (
    ModularRobot,
    get_body_states_multiple_isolated_robots,
    get_body_states_multiple_isolated_robots_intermediate,
    MorphologicalMeasures
)
from revolve2.simulation import Terrain
from revolve2.simulation.running import Runner
from revolve2.simulators.mujoco import LocalRunner

# vision
from revolve2.simulation.running import RecordSettings
# / vision

# class Evaluator:
#     """Provides evaluation of robots."""

#     _runner: Runner
#     _terrain: Terrain

#     def __init__(
#         self,
#         headless: bool,
#         num_simulators: int,
#     ) -> None:
#         """
#         Initialize this object.

#         :param headless: `headless` parameter for the physics runner.
#         :param num_simulators: `num_simulators` parameter for the physics runner.
#         """
#         self._runner = LocalRunner(headless=headless, num_simulators=num_simulators)
#         self._terrain = terrains.flat()

#     def evaluate(
#         self,
#         robots: list[ModularRobot],
#     ) -> list[float]:
#         """
#         Evaluate multiple robots.

#         Fitness is the distance traveled on the xy plane.

#         :param robots: The robots to simulate.
#         :returns: Fitnesses of the robots.
#         """
#         # Simulate the robots and process the results.
#         batch = create_batch_multiple_isolated_robots_standard(
#             robots, [self._terrain for _ in robots]
#         )
#         results = asyncio.run(self._runner.run_batch(batch))

#         body_states = get_body_states_multiple_isolated_robots(
#             [robot.body for robot in robots], results
#         )
#         xy_displacements = [
#             fitness_functions.xy_displacement(body_state_begin, body_state_end)
#             for body_state_begin, body_state_end in body_states
#         ]
#         return xy_displacements

# evaluaror class with intermediate states and height penality

import config

class Evaluator:
    """Provides evaluation of robots."""

    _runner: Runner
    _terrain: Terrain

    def __init__(
        self,
        headless: bool,
        num_simulators: int,
        record_settings: RecordSettings | None = None
    ) -> None:
        """
        Initialize this object.

        :param headless: `headless` parameter for the physics runner.
        :param num_simulators: `num_simulators` parameter for the physics runner.
        """
        self._runner = LocalRunner(headless=headless, num_simulators=num_simulators, record_settings=record_settings)
        print(f"Evaluator: Using {headless=}, {num_simulators=}")
        self._terrain = terrains.flat()
        self._record_settings = record_settings

    def evaluate(
        self,
        robots: list[ModularRobot],
        generation_index: int,
        steer: bool = False,
        simulation_time: float = 30,
        target_list: list[tuple[float, float]] = [(5,5)]
    ) -> list[float]:
        """
        Evaluate multiple robots.

        Fitness is the distance traveled on the xy plane.

        :param robots: The robots to simulate.
        :returns: Fitnesses of the robots.
        """

        target_for_generation = target_list[generation_index % len(target_list)]


        targets = [target_for_generation for  _ in robots]
        # TODO repeat k target for each generation
        

        # Simulate the robots and process the results.
        batch = create_batch_multiple_isolated_robots_standard(
            robots, [self._terrain for _ in robots], targets, simulation_time=simulation_time,  sampling_frequency = 1, steer = steer
        )

        results = asyncio.run(self._runner.run_batch(batch, record_settings=self._record_settings, generation_index=generation_index, steer = steer))

        # get the intermediate states
        body_states = get_body_states_multiple_isolated_robots(
            [robot.body for robot in robots], results
        )

        # calculate the fitnesses
        fitnesses = [
            #fitness_functions.xy_displacement_with_height_penality(body_states_robot)
            #fitness_functions.xy_displacement(body_state_begin, body_state_end)
            #for body_state_begin, body_state_end in body_states
            fitness_functions.distance_to_target(body_state_end, target) for (_, body_state_end), target in zip(body_states, targets)

        ]

        symmetries = [
            MorphologicalMeasures(robot.body).symmetry
            for robot in robots
        ]

        xy_positions = get_body_states_multiple_isolated_robots_intermediate(
                        [robot.body for robot in robots], results
                       )

        xy_positions = [repr([state for state in robot_state]) for robot_state in xy_positions]
        #print(xy_positions)

        return fitnesses, symmetries, xy_positions


        