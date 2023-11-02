import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import abrain
from abrain.core.ann import plotly_render
from .scenario import build_robot, Scenario
from ..evolution.common import EvaluationResult
from ..misc.genome import RVGenome
from ..simulation.control import OpenGLVision
from ..simulation.runner import Runner, RunnerOptions, CallbackType, \
    ANNDataLogging

logger = logging.getLogger(__name__)


@dataclass
class EvalOptions:
    runner: RunnerOptions = RunnerOptions()

    ann_save_path: Optional[Path] = None


class Evaluator:
    options = EvalOptions()

    @classmethod
    def set_runner_options(cls, options: RunnerOptions):
        cls.options.runner = options

    @dataclass
    class Result(EvaluationResult):
        # fitnesses: Individual.DataCollection = field(default_factory=dict)
        # descriptors: Individual.DataCollection = field(default_factory=dict)
        # stats: Individual.DataCollection = field(default_factory=dict)
        reject: bool = False

    @staticmethod
    def _clip(values: Dict[str, float], bounds: List[Tuple[float]], name: str):
        for i, b in zip(values.items(), bounds):
            k, v, lower, upper = *i, *b
            if not lower <= v <= upper:
                values[k] = max(lower, min(v, upper))
                logger.warning(f"Out-of-bounds value {name},{k}: {v} not in [{lower},{upper}]")
        return values

    @classmethod
    def _evaluate(cls, genome: RVGenome, options: EvalOptions, rerun: bool):
        r = cls.Result()
        viewer = None

        with_labels = (options.ann_save_path is not None
                       or options.runner.ann_data_logging != ANNDataLogging.NONE)

        robot = build_robot(genome, with_labels)

        runner = Runner(robot, options.runner, Scenario.amend,
                        position=Scenario.initial_position())
        scenario = Scenario(runner, genome.id())

        runner.callbacks[CallbackType.PRE_CONTROL_STEP] = scenario.pre_control_step
        runner.callbacks[CallbackType.POST_CONTROL_STEP] = scenario.post_control_step

        if options.runner.record is not None:
            runner.callbacks[CallbackType.VIDEO_FRAME_CAPTURED] = scenario.process_video_frame

        r.reject = False
        brain_controller = runner.controller.actor_controller
        brain: abrain.ANN = brain_controller.brain
        r.reject |= (brain.empty())

        if genome.with_vision():
            brain_controller.vision = \
                OpenGLVision(runner.model, genome.vision, runner.headless)

        if (p := options.ann_save_path) is not None:
            p: Path = options.runner.save_folder.joinpath(p)
            pr = plotly_render(brain, runner.controller.actor_controller.labels)
            pr.write_html(p)
            logging.info(f"Generated {p}")

            if False:
                p_ = p.with_suffix(".mp4")
                frames = 100

                xd = dict(title=None, visible=False, showticklabels=False)
                pr.update_layout(title=dict(automargin=False),
                                 scene=dict(xaxis=xd, yaxis=xd, zaxis=xd),
                                 margin=dict(autoexpand=False, b=0, l=0, r=0, t=0),
                                 width=500, height=500)
                for i in range(frames):
                    cr = 2
                    ct = 2 * math.pi * i / frames
                    eye = dict(x=cr * math.cos(ct), y=cr * math.sin(ct), z=0)
                    print(f"{100*i/frames:03.2f}% {eye}")
                    pr.update_layout(scene_camera=dict(eye=eye))
                    pr.write_image(f"test_image_{i:03d}.png")
                logging.info(f"Generated images")

        if rerun:
            viewer = runner.viewer

        if not r.reject:
            runner.run()

        scenario.finalize()

        r.fitnesses = cls._clip(scenario.fitness(), scenario.fitness_bounds(),
                                "fitness")
        r.descriptors = cls._clip(scenario.descriptors(), scenario.descriptor_bounds(),
                                  "features")

        r.stats = brain.stats().dict()
        r.stats.update({
            'brain': float(not brain.empty()),
            'perceptron': float(brain.perceptron()),
            'neurons': len(brain.neurons()),
        })

        if brain_controller.vision is not None:
            r.stats.update({
                'vision': brain_controller.vision.img.tolist()
            })

        r.stats['items'] = scenario.collected

        if rerun:
            return r, viewer
        else:
            return r

    @classmethod
    def evaluate_evo(cls, genome: RVGenome) -> Result:
        return cls._evaluate(genome, cls.options, False)

    @classmethod
    def evaluate_rerun(cls, genome: RVGenome, options: EvalOptions):
        return cls._evaluate(genome, options, True)
