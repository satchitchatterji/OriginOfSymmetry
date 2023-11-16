from revolve2.simulation.running import BatchResults

from ._body import Body
from ._body_state import BodyState


def get_body_states_multiple_isolated_robots(
    bodies: list[Body], batch_results: BatchResults
) -> list[tuple[BodyState, BodyState]]:
    """
    Get the first and last body state of a robot from a simulation simulating only a single robot.

    :param bodies: The bodies of the robots.
    :param batch_results: The simulation results.
    :returns: The first and last body state for each robot body.
    """
    return [
        (
            body.body_state_from_actor_state(
                environment_results.environment_states[0].actor_states[0]
            ),
            body.body_state_from_actor_state(
                environment_results.environment_states[-1].actor_states[0]
            ),
        )
        for body, environment_results in zip(bodies, batch_results.environment_results)
    ]

# function to get also intermediate body states (max 100)
def get_body_states_multiple_isolated_robots_intermediate(
    bodies: list[Body], batch_results: BatchResults
) -> list[tuple[BodyState, BodyState]]:
    """
    Get the the intermediate states of multiple robots from the simulation with a max of 100. If the has more are returned 100 states evenly distributed over the simulation.

    :param bodies: The bodies of the robots.
    :param batch_results: The simulation results.
    :returns: The body states
    """
    # get the number of simulation steps
    num_steps = len(batch_results.environment_results[0].environment_states)
    print("num_steps: ", num_steps)

    # get the number of intermediate states to be returned
    num_intermediate = min(100, num_steps)
    # get the step size
    step_size = num_steps // num_intermediate
    print("step_size: ", step_size)

    # get the intermediate states
    intermediate_states = []
    for body, environment_results in zip(bodies, batch_results.environment_results):
        single_intermediate_states = []
        for i in range(0, num_steps, step_size):
            print("i: ", i)
            single_intermediate_states.append(
                (
                    body.body_state_from_actor_state(
                        environment_results.environment_states[i].actor_states[0]
                    ),
                    body.body_state_from_actor_state(
                        environment_results.environment_states[i].actor_states[0]
                    ),
                )
            )
        intermediate_states.append(tuple(single_intermediate_states))
    return intermediate_states
    
