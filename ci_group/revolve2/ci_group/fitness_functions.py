"""Standard fitness functions for modular robots."""

import math

from revolve2.modular_robot import BodyState


def xy_displacement(begin_state: BodyState, end_state: BodyState) -> float:
    """
    Calculate the distance traveled on the xy-plane by a single modular robot.

    :param begin_state: Begin state of the robot.
    :param end_state: End state of the robot.
    :returns: The calculated fitness.
    """
    return math.sqrt(
        (begin_state.core_position[0] - end_state.core_position[0]) ** 2
        + ((begin_state.core_position[1] - end_state.core_position[1]) ** 2)
    )

def forward_travel(begin_state: BodyState, end_state: BodyState) -> float:
    """
    Calculate the distance traveled on the x coordinate in the positive direction.

    :param begin_state: Begin state of the robot.
    :param end_state: End state of the robot.
    :returns: The calculated fitness.
    """

    return begin_state.core_position[0] - end_state.core_position[0]

def distance_to_target(end_state: BodyState, target: tuple[float, float]) -> float:
    """
    Calculate the distance traveled on the x coordinate in the positive direction.

    :param begin_state: Begin state of the robot.
    :param end_state: End state of the robot.
    :param target: Target position of the robot.
    :returns: The calculated fitness.
    """

    return -math.sqrt((
        (end_state.core_position[0] - target[0]) ** 2
        + (end_state.core_position[1] - target[1]) ** 2)
    )# removed sqrt for higher push towards target

# same as xy_displacement but with a penality if a certain height is exceeded
def xy_displacement_with_height_penality(intermediate_states: list[BodyState]) -> float:
    """
    Calculate the distance traveled on the xy-plane by a single modular robot. If the height exceeds 0.5 a penality is applied.

    :param intermediate_states: Intermediate states of the robot.
    :returns: The calculated fitness.
    """
    # get the fitness without penality
    fitness = xy_displacement(intermediate_states[0], intermediate_states[-1])

    # get the height penality
    height_penality = 0
    for state in intermediate_states:
        if state.core_position[2] > 0.5:
            height_penality += 1

    # return the fitness with penality
    return fitness - height_penality
    