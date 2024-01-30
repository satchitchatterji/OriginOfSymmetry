from pyrr import Quaternion, Vector3
from revolve2.actor_controller import ActorController

# from ._environment_actor_controller import EnvironmentActorController
# from ._environment_steering_controller import EnvironmentActorController
from ._terrain import Terrain
from .actor import Actor
from .running import Environment, PosedActor

"""Contains EnvironmentActorController, an environment controller for an environment with a single actor that uses a provided ActorController."""

from revolve2.actor_controller import ActorController
from revolve2.simulation.running import ActorControl, EnvironmentController
import numpy as np
from typing import List, Tuple
import numpy.typing as npt
from matplotlib import pyplot as plt


# from revolve2.simulators.mujoco import LocalRunner


class EnvironmentActorController(EnvironmentController):
    """An environment controller for an environment with a single actor that uses a provided ActorController."""

    actor_controller: ActorController
    steer: bool
    n: int
    is_left: List[bool]
    is_right: List[bool]
    x_pos: float
    picture_w: int

    def __init__(
        self,
        actor_controller: ActorController,
        target_points: List[Tuple[float]] = [(0.0, 0.0)], #TODO remove?
        steer: bool = False,
    ) -> None:
        """
        Initialize this object.

        :param actor_controller: The actor controller to use for the single actor in the environment.
        :param target_points: Target points the agent have to reach.
        :param steer: if True the agent is controlled using a steering policy.
        """
        self.actor_controller = actor_controller
        self.steer = steer
        if steer:
            self.n = 7

    def control(
        self,
        dt: float,
        actor_control: ActorControl,
        vision_img: npt.ArrayLike | None,
        joint_positions=None,
        current_pos=None,
        save_pos=False,
    ) -> None:
        """
        Control the single actor in the environment using an ActorController.

        :param dt: Time since last call to this function.
        :param actor_control: Object used to interface with the environment.
        :param coordinates: current coordinates of each joint
        :param current_pos: current position of the agent
        """

        # vision
        # vision_img = np.flip(vision_img) # flip the image because its axis are inverted #we are already doing it before passing the image to the controller
        if vision_img is None:
            self.actor_controller.step(dt)
            targets = self.actor_controller.get_dof_targets()
            actor_control.set_dof_targets(0, targets)
            return

        self.picture_w = vision_img.shape[1]
        # / vision

        # we use color filters to find the next target point
        green_filter = vision_img[:, :, 1] < 100
        red_filter = vision_img[:, :, 0] > 100
        blu_filter = vision_img[:, :, 2] < 100
        coords = np.where(green_filter & red_filter & blu_filter)

        target_in_sight = True  # vision
        if coords[1].shape[0] > 0:
            #print("sphere in sight")
            self.x_pos = np.mean(coords[1])
        else:
            target_in_sight = False

        self.actor_controller.step(dt)
        targets = self.actor_controller.get_dof_targets()

        assert len(targets) == len(joint_positions)

        # print("Test print")
        # print("targets: ", targets)
        if self.steer and target_in_sight and save_pos:
            core_position = current_pos[:2]

            self.is_left = []
            self.is_right = []

            
            
            for joint_pos in joint_positions[
                1:
            ]:  # TODO why are we skipping the first joint?
                self.is_left.append(joint_pos[0] > 0.0)
                self.is_right.append(joint_pos[0] < 0.0)

            # check if joints are on the left or right
            joint_positions = [c[:2] for c in joint_positions]

            # compute steering angle and parameters
            # theta = (self.picture_w - self.x_pos) - (self.picture_w / 2)
            theta = self.picture_w / 2 - self.x_pos
            g = (((self.picture_w / 2) - abs(theta)) / (self.picture_w / 2)) ** self.n

            old_steering = False

            if old_steering:
                # apply steering factor TODO shouldn't we apply the steering factor only to the joints that are not already in the right position?
                for i, (left, right) in enumerate(zip(self.is_left, self.is_right)):
                    if (
                        left and theta < 0
                    ):  # if theta is negative the target is on the right
                        #print("sphere on the right with theta ", theta)
                        #print("multiplying dof by ", g)
                        targets[i] *= g

                    elif (
                        right and theta >= 0
                    ):  # if theta is positive the target is on the left
                        #print("sphere on the left with theta ", theta)
                        #print("multiplying dof by ", g)
                        targets[i] *= g
            else:

                for i, (left, right) in enumerate(zip(self.is_left, self.is_right)):
                    if theta < 0:  # if theta is negative the target is on the right
                        #print("sphere on the right with theta ", theta)

                        # if left:  # left joints are sped up DIV BY ZERO could cause problems
                        #     targets[i] = targets[i] / g 

                        if right:  # right joints are slowed down
                            targets[i] *= g

                    elif theta >= 0:  # if theta is positive the target is on the left
                        #print("sphere on the left with theta ", theta)

                        if left:  # left joints are slowed down
                            targets[i] *= g

                        # if right:  # right joints are sped up DIV BY ZERO could cause problems
                        #     targets[i] = targets[i] / g

        actor_control.set_dof_targets(0, targets)

    def set_picture_w(self, width: int):
        self.picture_w = width
        self.x_pos = width / 2


def create_environment_single_actor(
    actor: Actor, controller: ActorController, terrain: Terrain, target_point: tuple[float, float] = (0.0, 0.0)
) -> Environment:
    """
    Create an environment for simulating a single actor.

    :param actor: The actor to simulate.
    :param controller: The controller for the actor.
    :param terrain: The terrain to simulate the actor in.
    :returns: The created environment.
    """
    bounding_box = actor.calc_aabb()
    # vision
    # env = Environment(EnvironmentActorController(controller, steer=True, target_points=[(LocalRunner.sphere_pos.x, LocalRunner.sphere_pos.y)]))
    env = Environment(EnvironmentActorController(controller, steer=True))
    # / vision
    env.static_geometries.extend(terrain.static_geometry)
    env.actors.append(
        PosedActor(
            actor,
            Vector3(
                [
                    0.0,
                    0.0,
                    bounding_box.size.z / 2.0 - bounding_box.offset.z,
                ]
            ),
            Quaternion(),
            [0.0 for _ in controller.get_dof_targets()],
        )
    )
    env.target_point = target_point
    return env
