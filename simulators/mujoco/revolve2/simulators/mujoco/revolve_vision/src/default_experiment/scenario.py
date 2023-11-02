import logging
import logging
import math
import pprint
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from random import Random
from typing import Optional, List, Dict

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mujoco import MjData, MjModel
from pyrr import Vector3
from revolve2.core.modular_robot import Body, Brick, ActiveHinge, ModularRobot, Module

from ..misc.config import Config
from ..simulation.control import ANNControl, SensorControlData
from ..simulation.runner import Runner, RunnerOptions


# ==============================================================================
# Robots
# ==============================================================================

def default_body() -> Body:
    BT = Config.BodyType
    b_type = Config.body_type  #BT.SNAKE

    def add_arms(m: Module):
        m.left = ActiveHinge(math.pi / 2.0)
        m.left.attachment = ActiveHinge(math.pi / 2.0)
        m.left.attachment.attachment = Brick(0.0)
        m.right = ActiveHinge(math.pi / 2.0)
        m.right.attachment = ActiveHinge(math.pi / 2.0)
        m.right.attachment.attachment = Brick(0.0)

    body = Body()
    if b_type == BT.GECKO:
        body.core.front = Brick(0.0)
        body.core.back = Brick(0.0)
        for side in ['front', 'back']:
            brick = Brick(0.0)
            setattr(body.core, side, brick)
            add_arms(brick)

    elif b_type == BT.TORSO:
        for segment in [body.core]:
            add_arms(segment)
        body.core.back = Brick(0.0)

    elif b_type == BT.SPIDER:
        body.core.front = Brick(0.0)
        body.core.back = Brick(0.0)
        for side in ['front', 'right', 'back', 'left']:
            hinge = ActiveHinge(math.pi / 2.0)
            setattr(body.core, side, hinge)
            hinge.attachment = ActiveHinge(math.pi / 2.0)
            hinge.attachment.attachment = Brick(0.0)

    elif b_type == BT.SNAKE:
        body.core.back = ActiveHinge(math.pi / 2.0)
        current = body.core.back
        for i in range(9):
            current.attachment = ActiveHinge(0)#math.pi / 2.0)
            current = current.attachment
    else:
        raise ValueError(f"Unknown body type {b_type}")

    body.finalize()
    return body


Runner.actorController_t = SensorControlData


# Runner.environmentActorController_t = ActorControl


def build_robot(brain_dna, with_labels):
    return ModularRobot(default_body(), ANNControl.Brain(brain_dna, with_labels))


# ==============================================================================
# Items
# ==============================================================================


class CollectibleType(Enum):
    Apple = 0
    Pepper = 1

    @staticmethod
    def random(rng: Random):
        return rng.choice([t for t in CollectibleType])

    def _v(self, separation):
        return .5 * (1 + (2 * self.value - 1) * separation)

    def radius(self, separation: float):
        v = 1-.5*self._v(separation)
        return [1, v, v]

    def color(self, separation: float):
        v = self._v(separation)
        return [v, 1-v, 0, 1]


@dataclass
class CollectibleObject:
    x: float = 0
    y: float = 0
    type: CollectibleType = CollectibleType.Apple
    lvl: float = 0

    def __repr__(self): return f"{self.type.name:6s} {self.x:+.2f} {self.y:+.2f} {self.lvl}"


# ==============================================================================
# Scenario
# ==============================================================================

class Scenario:
    _rewards = {
        str(CollectibleType.Apple): 1,
        str(CollectibleType.Pepper): -1,
    }

    _items: Optional[List[CollectibleObject]] = None

    def __init__(self, runner: Runner, run_id: Optional[int] = None):
        self.runner = runner
        self.id = run_id
        self.collected = []
        self._initial_position = self.subject_position()
        self._prev_position = self._initial_position
        self._steps = 0
        self._speed = 0

        if runner.options.log_path:
            self.path_log_path = runner.options.save_folder.joinpath(runner.options.log_path)
            logging.info(f"Logging path to {self.path_log_path}")
            self.path_logger = open(self.path_log_path, 'w')
            self.path_logger.write(f"X Y F\n")

    def finalize(self):
        if self.runner.options.log_path:
            self.path_logger.close()
            logging.info(f"Generated {self.path_log_path}")

            df = pd.read_csv(self.path_log_path, sep=' ')

            fig, ax = plt.subplots()
            ax: Axes = ax
            ax.plot(df.X, df.Y, zorder=-1)

            i_alpha = .25
            idf = pd.DataFrame(columns=list("XYCL"))
            model = self.runner.model
            srm_names: bytes = model.names
            for i in range(model.nbody):
                addr = model.name_bodyadr[i]
                ln = srm_names.find(b'\0', addr)
                name = srm_names[addr:ln].decode('ascii')
                tokens = name.split("#")
                if not tokens[0].startswith(CollectibleType.__name__):
                    continue
                i_type = tokens[0].split('.')[1]
                level = int(10*float(tokens[2]))
                b_pos = Vector3(self.runner.data.mocap_pos[int(tokens[1])])
                idf.loc[len(idf)] = [
                    b_pos.x, b_pos.y,
                    (float(i_type == CollectibleType.Pepper.name),
                     float(i_type == CollectibleType.Apple.name),
                     0,
                     float(b_pos.z < 0)*(1-i_alpha)+i_alpha),
                    level]

            idf_gbl = idf.groupby("L")
            for gk in idf_gbl.groups:
                gg = idf_gbl.get_group(gk)
                ax.scatter(x=gg.X, y=gg.Y, c=gg.C, marker=(3+gk, 0, 0),
                           zorder=1)
            # print(i_l)

            margin = .1
            lim = max(*np.abs(np.quantile(idf.X, [0, 1])),
                      *np.abs(np.quantile(idf.Y, [0, 1])))
            lim = math.ceil((1 + margin) * lim)
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)

            fig: Figure = fig
            fig.suptitle(",".join(f"{k}: {v}"
                                  for k, v in self.fitness().items()))

            img_file = self.path_log_path.with_suffix('.png')
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            fig.tight_layout()
            fig.savefig(img_file)
            logging.info(f"Generated {img_file}")

    def subject_position(self):
        return self.runner.get_actor_state(0).position

    def pre_control_step(self, dt: float, mj_model: MjModel, mj_data: MjData):
        pass

    def post_control_step(self, dt: float, mj_model: MjModel, mj_data: MjData):
        collected = set()
        for a, b in {(mj_data.geom(i), mj_data.geom(j)) for i, j in
                     zip(mj_data.contact.geom1, mj_data.contact.geom2)}:
            if "robot" in a.name and CollectibleType.__name__ in b.name:
                collected.add(b)
            if "robot" in b.name and CollectibleType.__name__ in a.name:
                collected.add(a)

        if len(collected) > 0:
            for geom in collected:
                tokens = geom.name.split('#')
                t_id = tokens[0]
                b_id = int(tokens[1])
                mj_data.mocap_pos[b_id][2] = -.25
                self.collected.append(self._fitness(t_id, float(tokens[2])))

        p0 = self._prev_position
        p1 = self.subject_position()
        self._speed += dt * math.sqrt((p0[0] - p1[0]) ** 2 + ((p0[1] - p1[1]) ** 2))
        self._prev_position = p1

        if self.runner.options.log_path:
            self.path_logger.write(f"{p1.x} {p1.y} {list(self.fitness(instantaneous=True).values())[0]}\n")

        self._steps += 1

    def process_video_frame(self, frame: np.ndarray):
        if (v := self.runner.controller.actor_controller.vision) is not None:
            ratio = .25
            w, h, _ = frame.shape
            raw_vision = v.img
            vision_ratio = raw_vision.shape[0] / raw_vision.shape[1]
            iw, ih = int(ratio * w), int(ratio * h * vision_ratio)
            scaled_vision = cv2.resize(
                cv2.cvtColor(np.flipud(raw_vision), cv2.COLOR_RGBA2BGR),
                (iw, ih),
                interpolation=cv2.INTER_NEAREST
            )
            frame[h-ih:h, w-iw:w] = scaled_vision

    # ==========================================================================

    @staticmethod
    def initial_position():
        # return [0, 0, 0]
        return [-2, 0, 0]

    # ==========================================================================

    @staticmethod
    def fitness_name():
        return "collect"

    @classmethod
    def _fitness(cls, c_type: str, c_level: float):
        return cls._rewards[c_type] * (1 + c_level)

    def fitness(self, instantaneous: bool = False) -> Dict[str, float]:
        score = sum(self.collected)

        if not instantaneous:  # Look for closest favorable target
            s_pos: Vector3 = self.subject_position()
            dists = []
            model = self.runner.model
            srm_names: bytes = model.names
            for i in range(model.nbody):
                addr = model.name_bodyadr[i]
                ln = srm_names.find(b'\0', addr)
                name = srm_names[addr:ln].decode('ascii')
                if CollectibleType.Apple.name not in name:
                    continue
                b_pos = Vector3(model.body(name).pos)
                if b_pos.z < 0:  # Already collected
                    continue
                dists.append((s_pos - b_pos).length)

            if len(dists) > 0:  # Still more to collect
                score += .1 * (1-min(dists) / math.sqrt(2*Config.ground_size))

        return {self.fitness_name(): score}

    @classmethod
    @lru_cache(maxsize=1)
    def fitness_bounds(cls):
        items = cls._generate_items()
        min_max = [0, 0]
        for item in items:
            f = cls._fitness(str(item.type), item.lvl)
            min_max[int(f > 0)] += f
        logging.debug(f"Computed fitness range as {min_max}")
        assert min_max[0] < min_max[1]

        return [tuple(min_max)]

    # ==========================================================================

    # @staticmethod
    # def fitness_name():
    #     return "speed"
    #
    # def fitness(self) -> Dict[str, float]:
    #     score = 0
    #     if self._steps > 0:
    #         score += 100 * self._speed / self._steps
    #     # score += sum([t for t in self.collected.values()])
    #     return {self.fitness_name(): score}
    #
    # @classmethod
    # def fitness_bounds(cls):
    #     return [(0, 2)]

    # ==========================================================================

    @staticmethod
    def descriptor_names():
        return ["depth", "vision"]

    def descriptors(self) -> Dict[str, float]:
        controller = self.runner.controller.actor_controller
        v = math.sqrt(controller.vision.width * controller.vision.height)
        y = np.clip(self.subject_position()[1], *self.descriptor_bounds()[0])
        return {"y": y, "vision": v}

    @classmethod
    @lru_cache(maxsize=1)
    def descriptor_bounds(cls):
        ys = [item.y for item in cls._generate_items()]
        min_max = [min(ys), max(ys)]
        logging.debug(f"Computed y range as {min_max}")
        assert min_max[0] < min_max[1]
        ubound = max([math.ceil(math.fabs(x)) for x in ys])
        bounds = (-ubound, ubound)
        logging.debug(f"Using y range: {bounds}")

        return [bounds, (2, 10)]

    # ==========================================================================
    #
    # @staticmethod
    # def descriptor_names():
    #     return ["x", "y"]
    #
    # def descriptors(self) -> Dict[str, float]:
    #     # random = Random()
    #     # return [random.uniform(-2, 2) for _ in range(2)]
    #     return {"x": self.subject_position()[0],
    #             "y": self.subject_position()[1]}
    #
    # @staticmethod
    # def descriptor_bounds():
    #     w = Config.ground_size / 2
    #     return [(-w, w), (-w, w)]

    # ==========================================================================
    #
    # @staticmethod
    # def descriptor_names(): return ["height", "angle"]
    #
    # def descriptors(self) -> List[float]:
    #     return [v / self._steps if self._steps > 0 else 0
    #             for v in [max(0, self._altitude), self._gait]]
    #
    # @staticmethod
    # def descriptor_bounds(): return [(0.0, .5), (-90.0, 90.0)]
    #
    # ==========================================================================

    @staticmethod
    def sunflower(n: int, r_range) -> np.ndarray:
        phase = np.pi#np.pi/4
        # Generate the angles. The factor k_theta corresponds to 2*pi/phi^2.
        k_theta = np.pi * (3 - np.sqrt(5))
        angles = phase + np.linspace(k_theta, k_theta * n, n)

        r_min, r_max = r_range
        radii = np.sqrt(np.linspace(0, 1, n)) * (r_max - r_min) + r_min

        # Return Cartesian coordinates from polar ones.
        return (radii * np.stack((np.cos(angles), np.sin(angles)))).T

    @classmethod
    @lru_cache(maxsize=1)
    def _generate_items(cls):
        items = []
        count = Config.item_count
        r_range = (0, 2)
        dx, dy = .5, 0

        items_dict = {k: [] for k in CollectibleType}
        if count > 0:
            coordinates = Scenario.sunflower(count, r_range)
            for i, (ix, iy) in enumerate(coordinates):
                item = CollectibleObject(ix+dx, iy+dy, list(CollectibleType)[i % 2])
                items.append(item)
                items_dict[item.type].append(item)

        checker = dict()
        p0 = cls.initial_position()
        levels = Config.item_levels
        for ct in CollectibleType:
            ct_items = sorted(items_dict[ct], key=lambda itm: math.sqrt((p0[0] - itm.x)**2 + (p0[1] - itm.y)**2))
            for i, item in enumerate(ct_items):
                item.lvl = levels[int(len(levels) * i / len(ct_items))]

                key = (item.type, item.lvl)
                checker[key] = checker.get(key, 0) + 1

        balanced = (len(set(checker.values())) == 1)
        if not balanced:
            dct = {v: [] for v in checker.values()}
            for k, v in checker.items():
                dct[v].append(k)
            assert balanced, f"Unbalanced items generation:\n{pprint.pformat(dct)}"

        logging.debug(f"Generated items: {pprint.pformat(items)}")

        return items

    # ==========================================================================

    @staticmethod
    def amend(xml, options: RunnerOptions):
        robots = [r for r in xml.worldbody.body]

        xml.visual.map.znear = ".001"

        # Reference to the ground
        ground = next(item for item in xml.worldbody.geom if item.name == "ground")
        g_size = Config.ground_size
        gh_size = .5 * g_size

        i_size = Config.item_size
        i_radius = .5 * i_size

        # Remove default ground
        ground.remove()

        # Add texture and material for the ground
        xml.asset.add('texture', name="grid", type="2d", builtin="checker",
                      width="512", height="512",
                      rgb1=".1 .2 .3", rgb2=".2 .3 .4")
        xml.asset.add('material', name="grid", texture="grid",
                      texrepeat="1 1", texuniform="true", reflectance="0")
        xml.worldbody.add('geom', name="floor",
                          size=[gh_size, gh_size, .05],
                          type="plane", material="grid", condim=3)

        xml.asset.add('texture', name="sky", type="skybox", builtin="flat",
                      rgb1="0 0 0", rgb2="0 0 0",
                      width=512, height=512,
                      mark="random", markrgb="1 1 1")

        # different textures for the objects
        materials = {t: {} for t in CollectibleType}
        levels = Config.item_levels
        for lvl in levels:
            ts, rnd = 8, lvl
            for t, rgb_a, rgb_b in [(CollectibleType.Apple, "0 1 0", "1 0 0"),
                                    (CollectibleType.Pepper, "1 0 0", "0 1 0")]:
                xml.asset.add('texture', name=f"txt{t.name}-lvl{lvl}",
                              type="cube", builtin="flat",
                              rgb1=rgb_a, rgb2=rgb_a, width=ts, height=ts,
                              mark="random", markrgb=rgb_b, random=rnd)
                m = xml.asset.add('material', name=f"mat{t.name}-lvl{lvl}",
                                  texture=f"txt{t.name}-lvl{lvl}",
                                  texrepeat="1 1", texuniform="true", reflectance="0")
                materials[t][lvl] = m

        # Toy textures for the robots (not working)
        # xml.compiler.texturedir = "/home/kgd/work/code/vu/revolve_vision/"
        # xml.asset.add('texture', name="creeper", type="cube",
        #               file="creeper_texture.png", width=64, height=64)
        # creeper_material = \
        #     xml.asset.add('material', name="creeper", texture="creeper")

        gh_width = .025
        for i, x, y in [(0, 0, 1), (1, 1, 0), (2, 0, -1), (3, -1, 0)]:
            b_height = g_size / 100
            xml.worldbody.add('geom', name=f"border#{i}",
                              pos=[x * (gh_size + gh_width),
                                   y * (gh_size + gh_width), b_height],
                              rgba=[1, 1, 1, 1],
                              euler=[0, 0, i * math.pi / 2],
                              type="box", size=[gh_size, gh_width, b_height])

        flip = -1 if options.flipped_items else 1
        items = Scenario._generate_items()
        if items is not None:
            for i, item in enumerate(items):
                s = item.lvl
                name = f"{item.type}#{i}#{s}"

                body = xml.worldbody.add('body', name=name,
                                         pos=[item.x, flip * item.y, 0],
                                         mocap=True)
                body.add('geom', name=name,
                         # rgba=item.type.color(item_distinction),
                         type="sphere", size=f"{i_radius} 0 0",
                         material=materials[item.type][s].name)

        if options is not None and options.view is not None and options.view.mark_start:
            for robot in robots:
                xml.worldbody.add('site',
                                  name=robot.full_identifier[:-1] + "_start",
                                  pos=robot.pos * [1, 1, 0], rgba=[0, 0, 1, 1],
                                  type="ellipsoid", size=[0.05, 0.05, 0.0001])

        for r in robots:
            for g in r.find_all('geom'):
                if math.isclose(g.size[0], g.size[1]):
                    g.rgba = ".3 0 .3 1"
                else:
                    g.rgba = "0 .3 0 1"
                    # g.material = creeper_material

        for hinge in filter(lambda j: j.tag == 'joint', xml.find_all('joint')):
            xml.sensor.add('jointpos', name=f"{hinge.full_identifier}_sensor".replace('/', '_'),
                           joint=hinge.full_identifier)
