from revolve2.simulation import Terrain, create_environment_single_actor
from revolve2.simulation.running import Batch

from ._modular_robot import ModularRobot


def create_batch_multiple_isolated_robots(
    robots: list[ModularRobot],
    terrains: list[Terrain],
    target_points: list[tuple[float]],
    steer: bool,
    simulation_time: int | None,
    sampling_frequency: float,
    simulation_timestep: float,
    control_frequency: float
    
) -> Batch:
    """
    Create a batch for simulating multiple robots that do not interact.

    :param robots: The robots to simulate.
    :param terrains: The terrains to simulate the robots in.
    :param simulation_time: See `Batch` class.
    :param sampling_frequency: See `Batch` class.
    :param simulation_timestep: See `Batch` class.
    :param control_frequency: See `Batch` class.
    :returns: The created batch.
    """
    actor_controllers = [robot.make_actor_and_controller() for robot in robots]
    envs = [
        create_environment_single_actor(actor, controller, terrain, steer, target_point)
        for (actor, controller), terrain, target_point in zip(actor_controllers, terrains, target_points)
    ]

    batch = Batch(
        simulation_time=simulation_time,
        sampling_frequency=sampling_frequency,
        simulation_timestep=simulation_timestep,
        control_frequency=control_frequency
    )

    batch.environments.extend(envs)

    return batch
