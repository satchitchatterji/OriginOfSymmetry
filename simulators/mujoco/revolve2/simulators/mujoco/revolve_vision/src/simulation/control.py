import logging
import pprint
from pathlib import Path
from random import Random
from typing import List, Tuple, Optional

import numpy as np
from mujoco import MjModel, MjData
from revolve2.serialization import StaticData, Serializable

import abrain
from abrain import Point, CPPN
from revolve2.actor_controller import ActorController
from revolve2.core.modular_robot import Brain, Body, ActiveHinge
from revolve2.core.modular_robot.brains import BrainCpgNetworkNeighbour
from .vision import OpenGLVision
from ..misc.config import Config
from ..misc.genome import RVGenome
from ..simulation.runner import DefaultActorControl, ANNDataLogging


# ==============================================================================
# Basic control data: (built-in) sensors
# ==============================================================================

class SensorControlData(DefaultActorControl):
    def __init__(self, mj_model: MjModel, mj_data: MjData):
        DefaultActorControl.__init__(self, mj_model, mj_data)
        self.sensors = mj_data.sensordata
        self.model, self.data = mj_model, mj_data


# ==============================================================================
# ANN (abrain) controller (also handles the camera)
# ==============================================================================

def retina_mapper(rng):
    rc = Config.RetinaConfiguration
    def debug_point(i, j, k, x, y, z):
        p = Point(x, y, z)
        print(f"[kgd-debug] {i=} {j=} {k=}, {x=:+5.3f} {y=:+5.3f} {z=:+5.3f}")
        return p
    lut = {
        rc.R: lambda *_:
            Point(rng.uniform(-1, 1),
                  -1 + .1,
                  rng.uniform(-1, 1)),
        rc.X: lambda i, j, k, w, h:
            Point(2 * (w * k + i) / (3 * w - 1) - 1,
                  -1 + .1,
                  2 * j / (h-1) - 1),
        rc.Y: lambda i, j, k, w, h:
            Point(2 * i / (w-1) - 1,
                  -1 + (k+1) * .1,
                  2 * j / (h-1) - 1),
        rc.Z: lambda i, j, k, w, h:
            Point(2 * i / (w - 1) - 1,
                  -1 + .1,
                  2 * (h * k + j) / (3 * h - 1) - 1),
    }
    return lut[Config.retina_configuration]


class ANNControl:
    class Controller(ActorController):
        def __init__(self, genome: abrain.Genome, inputs: List[Point], outputs: List[Point]):
            self.brain = abrain.ANN.build(inputs, outputs, genome)
            self.i_buffer, self.o_buffer = self.brain.buffers()

            # Vision is initialized by caller
            self.vision: Optional[OpenGLVision] = None

            self._step = 0

        def get_dof_targets(self) -> List[float]:
            return [self.o_buffer[i] for i in range(len(self.o_buffer))]

        def step(self, dt: float, data: 'ControlData') -> None:
            off = len(data.sensors)
            self.i_buffer[:off] = [pos for pos in data.sensors]

            if self.vision is not None:
                img = self.vision.process(data.model, data.data)
                self.i_buffer[off:] = [x / 255 for x in img.flat]

                # cv2.imwrite(f'vision_{self._step:010d}.png',
                #             cv2.cvtColor(np.flipud(img), cv2.COLOR_RGBA2BGR))
                # if self._step == 0:
                #     print(img)
                # header = f"step {self._step}"
                # print(f"== {header} {'=' * (80 - len(header) - 4)}\n{img}")

            self._step += 1

            self.brain.__call__(self.i_buffer, self.o_buffer)

            # pprint.pprint([n.value for n in self.brain.neurons()])

        def start_log_ann_data(self,
                               level: Optional[ANNDataLogging] = None,
                               filepath: Optional[Path] = None):
            self._ann_log_level = level
            self._ann_log_file = open(filepath, 'w')

            ann_type = abrain.ANN.Neuron.Type
            log_type = ANNDataLogging
            self._valid_types_flag = [
                ann_t for ann_t, log_t in [
                    (ann_type.I, log_type.INPUTS),
                    (ann_type.H, log_type.HIDDEN),
                    (ann_type.O, log_type.OUTPUTS),
                ] if level & log_t]

            self._ann_log_file.write("Step")
            for n in self.brain.neurons():
                if n.type in self._valid_types_flag:
                    self._ann_log_file.write(f" {n.pos}:{self.labels.get(n.pos)}")
            self._ann_log_file.write("\n")

        def log_ann_data(self):
            self._ann_log_file.write(f"{self._step}")
            for n in self.brain.neurons():
                if n.type in self._valid_types_flag:
                    self._ann_log_file.write(f" {n.value}")
            self._ann_log_file.write("\n")

        def stop_log_ann_data(self):
            logging.info(f"Generated {self._ann_log_file.name}")
            self._ann_log_file.close()

        @classmethod
        def deserialize(cls, data: StaticData) -> Serializable:
            raise NotImplementedError

        def serialize(self) -> StaticData:
            raise NotImplementedError

    class Brain(Brain):
        def __init__(self, brain_dna: RVGenome, with_labels=False):
            self.brain_dna = brain_dna
            self.with_labels = with_labels

        def make_controller(self, body: Body, dof_ids: List[int]) -> ActorController:
            parsed_coords = body.to_tree_coordinates()

            bounds = np.zeros((2, 3), dtype=int)
            np.quantile([c[1].tolist() for c in parsed_coords], [0, 1], axis=0, out=bounds)

            if bounds[0][2] != bounds[1][2]:
                raise NotImplementedError("Can only handle planar robots (with z=0 for all modules)")
            x_min, x_max = bounds[0][0], bounds[1][0]
            xrange = max(x_max-x_min, 1)
            y_min, y_max = bounds[0][1], bounds[1][1]
            yrange = max(y_max-y_min, 1)

            hinges_map = {
                c[0].id: (
                    2 * (c[1].x - x_min) / xrange - 1 if xrange > 1 else 0,
                    2 * (c[1].y - y_min) / yrange - 1 if yrange > 1 else 0)
                for c in parsed_coords if isinstance(c[0], ActiveHinge)
            }

            inputs, outputs = [], []
            if self.with_labels:
                labels = {}

            for i, did in enumerate(dof_ids):
                p = hinges_map[did]
                ip = Point(p[1], -1, p[0])
                inputs.append(ip)
                op = Point(p[1], 1, p[0])
                outputs.append(op)

                if self.with_labels:
                    labels[ip] = f"P{i}"
                    labels[op] = f"M{i}"

            if self.brain_dna.with_vision():
                mapper = retina_mapper(Random(0))
                w, h = self.brain_dna.vision
                for j in reversed(range(h)):
                    for i in range(w):
                        for k, c in enumerate("BGR"):
                            p = mapper(i, j, k, w, h)
                            inputs.append(p)

                            if self.with_labels:
                                labels[p] = f"{c}[{i},{j}]"

            # Ensure no duplicates
            try:
                # assert all(len(set(io_)) == len(io_) for io_ in [inputs, outputs])
                assert len({p for io in [inputs, outputs] for p in io}) \
                       == (len(inputs)+len(outputs))
            except AssertionError as e:
                duplicates = {}
                for io in [inputs, outputs]:
                    for p in io:
                        duplicates[p] = duplicates.get(p, 0) + 1
                duplicates = {k: v for k, v in duplicates.items() if v > 1}
                raise ValueError(f"Found duplicates: {pprint.pformat(duplicates)}") from e

            c = ANNControl.Controller(self.brain_dna.brain, inputs, outputs)

            if self.with_labels:
                c.labels = labels

            return c


# ==============================================================================
# CPG Controller (for compatibility)
# ==============================================================================

class CPGControl:
    class Brain(BrainCpgNetworkNeighbour):
        def __init__(self, brain_dna):
            self.genome = brain_dna

        def _make_weights(
                self,
                active_hinges: List[ActiveHinge],
                connections: List[Tuple[ActiveHinge, ActiveHinge]],
                body: Body,
        ) -> Tuple[List[float], List[float]]:
            cppn = abrain.CPPN(self.genome)

            internal_weights = [ # TODO Check the bounds
                cppn(Point(*pos), Point(*pos), CPPN.Output.Weight)
                for pos in [
                    body.grid_position(active_hinge) for active_hinge in active_hinges
                ]
            ]

            external_weights = [
                cppn(Point(*pos1), Point(*pos2), CPPN.Output.Weight)
                for (pos1, pos2) in [
                    (body.grid_position(active_hinge1), body.grid_position(active_hinge2))
                    for (active_hinge1, active_hinge2) in connections
                ]
            ]

            return internal_weights, external_weights
