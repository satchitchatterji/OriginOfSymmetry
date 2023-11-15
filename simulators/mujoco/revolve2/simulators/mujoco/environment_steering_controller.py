"""Contains EnvironmentActorController, an environment controller for an environment with a single actor that uses a provided ActorController."""

from revolve2.actor_controller import ActorController
from revolve2.core.physics.running import ActorControl, EnvironmentController
import numpy as np
from typing import List, Tuple
import numpy.typing as npt
from matplotlib import pyplot as plt


class EnvironmentActorController(EnvironmentController):
    """An environment controller for an environment with a single actor that uses a provided ActorController."""

    actor_controller: ActorController
    steer: bool
    n: int
    is_left: List[bool]
    is_right: List[bool]
    x_pos: float
    picture_w: int

    def __init__(self, actor_controller: ActorController,
                target_points: List[Tuple[float]] = [(0.0,0.0)],
                steer: bool = False) -> None:
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
            self.is_left = []
            self.is_right = []

    def control(self, dt: float, actor_control: ActorControl, vision_img: npt.ArrayLike, joint_positions=None, current_pos=None, save_pos=False) -> None:
        """
        Control the single actor in the environment using an ActorController.

        :param dt: Time since last call to this function.
        :param actor_control: Object used to interface with the environment.
        :param coordinates: current coordinates of each joint
        :param current_pos: current position of the agent
        """
        vision_img = np.flip(vision_img) # flip the image because its axis are inverted

        # we use color filters to find the next target point 
        green_filter = vision_img[:,:,1] < 100
        red_filter = vision_img[:,:,0] > 100
        blu_filter = vision_img[:,:,2] < 100
        coords = np.where(green_filter & red_filter & blu_filter)
        if coords[1].shape[0] > 0:
            self.x_pos = np.mean(coords[1])

        self.actor_controller.step(dt)
        targets = self.actor_controller.get_dof_targets()

        if self.steer:

            core_position = current_pos[:2]

            if save_pos:
                for joint_pos in joint_positions[1:]:
                    self.is_left.append(joint_pos[0] > 0.)
                    self.is_right.append(joint_pos[0] < 0.)

            # check if joints are on the left or right
            joint_positions = [c[:2] for c in joint_positions]

            # compute steering angle and parameters
            theta = (self.picture_w - self.x_pos) - (self.picture_w / 2)
            g = (((self.picture_w / 2) - abs(theta)) / (self.picture_w / 2)) ** self.n

            # apply steering factor
            for i, (left, right) in enumerate(zip(self.is_left, self.is_right)):
                if left:
                    if theta < 0:
                        targets[i] *= g
                elif right:
                    if theta >= 0:
                        targets[i] *= g
            
        actor_control.set_dof_targets(0, targets)

    def set_picture_w(self, width: int):
        self.picture_w = width
        self.x_pos = width / 2