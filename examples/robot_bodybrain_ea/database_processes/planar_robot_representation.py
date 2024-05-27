# Note from satchit: copied from https://github.com/ci-group/revolve2/blob/c81f3610004c1f33388c8208bacc52d0bc6ff394/ci_group/revolve2/ci_group/planar_robot_representation.py

"""Draw 2D representations of Modular Robots. Based on Karine Miras` Method."""
import os
import time
from typing import Any

import cairo
import numpy as np
from numpy.typing import NDArray
from pyrr import Vector3, Quaternion

from revolve2.modular_robot import ModularRobot
from revolve2.modular_robot import Module
from revolve2.modular_robot import ActiveHinge, Body, Brick, Core

from _attachement_point import AttachmentPoint
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def __mk_path() -> str:
    path = f"planar_robot_representations_{time.time()}"
    print(f"Saving images to: {path}")
    if not os.path.exists(path):
        os.mkdir(path)
    return path


def draw_robots(
    robots: list[ModularRobot] | list[Body], scale: int = 100, path: str | None = None
) -> None:
    """
    Draw multiple robots at once.

    How to use:
     >>> robots: list[revolve2.modular_robot.ModularRobot] | list[revolve2.modular_robot.body.base.Body]
     >>> draw_robots(robots, path="<your desired path to save the image to>")

    :param robots: The robots.
    :param scale: The scale for the robots to be drawn.
    :param path: The path for the output files.
    """
    if not path:
        path = __mk_path()

    for robot in robots:
        draw_robot(robot, scale, path)


def draw_robot(
    robot: ModularRobot | Body, scale: int = 100, path: str | None = None
) -> None:
    """
    Draw a 2D representation for a modular robots body.

    How to use:
     >>> robot: revolve2.modular_robot.ModularRobot | revolve2.modular_robot.body.base.Body
     >>> draw_robot(robot, path="<your desired path to save the image to>")

    :param robot: Supply the robot as a ModularRobot object, or the body directly as a Body object.
    :param scale: Allows to set the size of the drawing.
    :param path: The path to save images to.
    """
    if not path:
        path = __mk_path()

    body = robot if isinstance(robot, Body) else robot.body
    tpl: tuple[NDArray[Any], Vector3[np.int_]] = body.to_grid()
    body_grid, core_position = tpl
    x, y, _ = np.array(body_grid).shape

    image = cairo.ImageSurface(cairo.FORMAT_ARGB32, x * scale, y * scale)
    context = cairo.Context(image)
    context.scale(scale, scale)

    cx, cy, _ = tuple(core_position)
    _draw_module(
        module=body.core,
        position=(cx, cy),
        previous_position=(cx, cy),
        orientation=_make_rot_mat(0),
        context=context,
        print_id=True,
    )
    _save_png(image, path)
    img = mpimg.imread(f"{path}/robot_2d_{str(hash(image))}.png")
    plt.imshow(img)
    plt.show()
    

def get_attachment_points():
    # Front right back left 
    child_offset=0.4#https://github.com/ci-group/revolve2/blob/master/modular_robot/revolve2/modular_robot/body/v1/_core_v1.py
    attachment_points = {
        0: AttachmentPoint(
            offset=Vector3([child_offset, 0.0, 0.0]),
            orientation=Quaternion.from_eulers([0.0, 0.0, 0.0]),
        ),
        1: AttachmentPoint(
            offset=Vector3([child_offset, 0.0, 0.0]),
            orientation=Quaternion.from_eulers([0.0, 0.0, math.pi / 2.0 * 3]),
        ),
        2: AttachmentPoint(
            offset=Vector3([child_offset, 0.0, 0.0]),
            orientation=Quaternion.from_eulers([0.0, 0.0, math.pi]),
        ),
        3: AttachmentPoint(
            offset=Vector3([child_offset, 0.0, 0.0]),
            orientation=Quaternion.from_eulers([0.0, 0.0, math.pi / 2.0]),
        ),
    }
    return attachment_points

def _draw_module(
    module: Module,
    position: tuple[int, int],
    previous_position: tuple[int, int],
    orientation: NDArray[np.int_],
    context: "cairo.Context[cairo.ImageSurface]",
    print_id: bool = False,
    indent: int = 0,
) -> None:
    """
    Draw a module onto the canvas.

    :param module: The module.
    :param position: The position on the canvas.
    :param previous_position: The position of the previous module.
    :param orientation: The orientation to draw in.
    :param context: The context to draw it on.
    :param print_id: If the modules id should be drawn as well.
    :raises Exception: If the module cant be drawn.
    """
    x, y = position
    context.rectangle(x, y, 1, 1)  # draw module object

    match module:
        case Core():
            context.set_source_rgb(255, 255, 0)  # Yellow
        case ActiveHinge():
            context.set_source_rgb(1, 0, 0)  # Red
            if module.rotation == 0:
                context.set_source_rgb(1.0, 0.4, 0.4)  # Flesh Color
        case Brick():
            context.set_source_rgb(0, 0, 1)  # Blue
        case _:
            raise Exception(
                f"Module of type {type(module)} has no defined structure for drawing."
            )

    # default operation for every module
    context.fill_preserve()
    context.set_line_width(0.01)
    context.stroke()
    context.set_source_rgb(0, 0, 0)

    if type(module) == Core:
        # draw the connection to the parent module
        x_offset, y_offset = (
            previous_position[0] - position[0],
            previous_position[1] - position[1],
        )

        circ_x = (
            x + 0.5
            if x_offset == 0
            else x + (x_offset if x_offset > 0 else abs(x_offset) - 1)
        )
        circ_y = (
            y + 0.5
            if y_offset == 0
            else y + (y_offset if y_offset > 0 else abs(y_offset) - 1)
        )

        context.arc(circ_x, circ_y, 0.1, 0, np.pi * 2)
        context.fill_preserve()
        context.stroke()

    if print_id:
        # print module id onto canvas
        context.set_font_size(0.2)
        context.move_to(x, y + 0.4)
        context.show_text(module.__class__.__name__)
        context.stroke()

    print("\t"*indent, module.__class__.__name__, [child.__class__.__name__ for child in module.children])
    for key, child in enumerate(module.children):
        if str(type(child)) == "<class 'NoneType'>":
            break


        angle = get_attachment_points()[key].orientation.angle
        mapo = _make_rot_mat(angle)
        target_orientation = orientation @ mapo

        x, y = target_orientation.dot(np.array([1, 0]))

        new_pos = position[0] + x, position[1] + y


        _draw_module(
            module=child,
            position=new_pos,
            previous_position=position,
            context=context,
            orientation=target_orientation,
            print_id=print_id,
            indent=indent+1,
        )


def _make_rot_mat(theta: float) -> NDArray[np.int_]:
    """
    Make a rotation matrix from angle in 2D.

    This function casts angles to iterations of 90°, since we plot on a grid.

    :param theta: The angle.
    :return: The matrix.
    """
    c, s = int(round(np.cos(theta))), int(round(np.sin(theta)))
    rotation = np.array(((c, -s), (s, c)))
    return rotation


def _save_png(image: cairo.ImageSurface, path: str) -> None:
    """
    Save the image representation of a robot as png.

    :param image: The image.
    :param path: The path to save the image to.
    """
    image.write_to_png(f"{path}/robot_2d_{str(hash(image))}.png")