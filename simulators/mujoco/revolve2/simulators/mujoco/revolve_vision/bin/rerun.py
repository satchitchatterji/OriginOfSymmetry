#!/usr/bin/env python3

import argparse
import builtins
import json
import logging
import math
import numbers
import os
import pprint
import time
from datetime import timedelta
from pathlib import Path
from random import Random
from typing import Dict, Optional

import cv2
import humanize
import numpy as np
import pandas as pd
from PIL.Image import Image
from abrain import CPPN, Point
from colorama import Fore, Style
from matplotlib import pyplot as plt

from abrain.core.genome import GIDManager
from src.default_experiment.evaluator import Evaluator, EvalOptions
from src.evolution.common import Individual
from src.misc.config import Config
from src.misc.genome import RVGenome
from src.simulation.runner import RunnerOptions, ANNDataLogging


class Options:
    def __init__(self):
        self.robot: Optional[Path] = None
        self.env: Optional[Path] = None
        self.config: Optional[Path] = None
        self.verbosity: int = 0

        self.perf_check: bool = True

        self.save_ann: bool = False
        self.save_cppn: bool = False
        self.draw_cppn: bool = False
        self.save_neurons: str = str(ANNDataLogging.NONE)
        self.save_path: bool = False

        self.flip_items: bool = False

        self.view: bool = False
        self.speed: float = 1.0
        self.auto_start: bool = True
        self.auto_quit: bool = True
        self.cam_id: Optional[int] = None
        self.settings_save: bool = True
        self.settings_restore: bool = True

        self.record: bool = False
        self.width: int = 500
        self.height: int = 500

    @staticmethod
    def populate(parser: argparse.ArgumentParser):
        parser.add_argument('-r', '--robot', dest="robot",
                            metavar='FILE', type=Path, required=True,
                            help="Location of the robot (ann) genome")
        parser.add_argument('-e', '--environment', dest="env",
                            metavar='FILE', type=str,
                            help="Location of the environmental parameters")
        parser.add_argument('-c', '--config', dest="config",
                            metavar='FILE', type=str,
                            help="Location of the configuration file")
        parser.add_argument('-v', '--verbose', dest="verbosity", default=0,
                            action='count', help="Print more information")

        group = parser.add_argument_group("Data collection",
                                          "Generate/Extract derived data from provided robot")
        group.add_argument('--no-performance-check', dest="perf_check",
                           action='store_false',
                           help="Do not check for identical behavior")
        group.add_argument('--write-cppn', dest="save_cppn", action='store_true',
                           help="Requests that the CPPN be written to file")
        group.add_argument('--draw-cppn', dest="draw_cppn", action='store_true',
                           help="Requests that the CPPN be used to draw some"
                                " of its patterns")
        group.add_argument('--write-ann', dest="save_ann", action='store_true',
                           help="Requests that the ANN be written to file")
        group.add_argument('--log-neurons', dest="save_neurons",
                           help=f"Requests that neuron states be written to file. "
                                f"Valid values are {[v.name for v in ANNDataLogging]}")
        group.add_argument('--log-trajectory', dest="save_path", action='store_true',
                           help=f"Log robot path (and instant fitness)")
        group.add_argument('--flip-items', dest="flip_items", action='store_true',
                           help=f"Flip items in the environment to test for robust strategy")

        group = parser.add_argument_group("Rendering",
                                          "Additional rendering flags")
        group.add_argument('--view', dest="view", action='store_true',
                           help="Requests a visual output")
        group.add_argument('--speed', dest="speed", type=float,
                           help="Initial execution speed-up")
        group.add_argument('--no-auto-start', dest="auto_start",
                           action='store_false',
                           help="Prohibits automatic launch of simulation")
        group.add_argument('--no-auto-quit', dest="auto_quit",
                           action='store_false',
                           help="Prohibits automatic closing of viewer")
        group.add_argument('--camera', dest="cam_id", type=int,
                           help="Select a camera by id")
        group.add_argument('--no-settings-save', dest="settings_save",
                           action='store_false',
                           help="Do not update viewer settings")
        group.add_argument('--no-settings-restore', dest="settings_restore",
                           action='store_false',
                           help="Do not restore viewer settings")

        group.add_argument('--movie', dest="record", action='store_true',
                           help="Requests a movie to be made")
        group.add_argument('--width', dest="width", type=int,
                           help="Specify the width of the movie")
        group.add_argument('--height', dest="height", type=int,
                           help="Specify the height of the movie")

        Config.argparse_setup(parser)


def generate_defaults(args):
    rng = Random(0)
    id_manager = GIDManager()
    genome = RVGenome.random(rng, id_manager)
    ind = Individual(genome)

    def_folder = Path("tmp/defaults/")
    def_folder.mkdir(parents=True, exist_ok=True)

    ind_file = args.robot = def_folder.joinpath("genome.json")
    with open(ind_file, 'w') as f:
        json.dump(ind.to_json(), f)

    cnf_file = args.config = def_folder.joinpath("config.json")
    Config.write_json(cnf_file)

    if args.verbosity > 0:
        print("Generated default files", [ind_file, cnf_file])


def try_locate(base: Path, name: str, levels: int = 0, strict: bool = True):
    if not base.exists():
        raise FileNotFoundError(f"Genome not found at {base}")

    path = base
    attempts = 0
    while attempts <= levels:
        path = path.parent
        candidate = path.joinpath(name)
        if candidate.exists():
            return candidate
        attempts += 1
    if strict:
        raise FileNotFoundError(f"Could not find file for '{name}' "
                                f"at most {levels} level(s) from '{base}'")


def main() -> int:
    start = time.perf_counter()
    # ==========================================================================
    # Parse command-line arguments

    args = Options()
    parser = argparse.ArgumentParser(description="Rerun evolved champions")
    Options.populate(parser)
    parser.parse_args(namespace=args)

    if args.verbosity <= 0:
        logging.basicConfig(level=logging.WARNING)
    elif args.verbosity == 1:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.DEBUG)

    if args.verbosity >= 2:
        print("Command line-arguments:")
        pprint.PrettyPrinter(indent=2, width=1).pprint(args.__dict__)

    defaults = (args.robot.name.upper() == 'NONE')
    if defaults:
        generate_defaults(args)

    if args.config is None:
        args.config = try_locate(args.robot, "config.json", 2)
    Config.read_json(args.config)
    Config.argparse_process(args)

    options = EvalOptions()
    save_folder = False

    if args.record:
        save_folder = True
        options.runner.record = RunnerOptions.Record(
            video_file_path=Path(args.robot.stem + ".movie.mp4"),
            width=args.width, height=args.height)

    elif args.view:
        options.runner.view = RunnerOptions.View(
            start_paused=(not args.record and not args.auto_start),
            speed=args.speed,
            auto_quit=args.auto_quit,
            cam_id=args.cam_id,
            settings_save=args.settings_save,
            settings_restore=args.settings_restore,
        )

    if args.save_ann:
        save_folder = True
        options.ann_save_path = args.robot.stem + ".ann.html"

    if args.save_neurons != str(ANNDataLogging.NONE):
        save_folder = True
        options.runner.ann_data_logging = ANNDataLogging[args.save_neurons]
        options.runner.ann_data_file = args.robot.stem + ".neurons.dat"

    if args.save_path:
        save_folder = True
        options.runner.log_path = args.robot.stem + ".trajectory.dat"

    options.runner.flipped_items = args.flip_items

    if save_folder:
        options.runner.save_folder = args.robot.parent

    if args.verbosity > 1:
        print("Deduced options:", end='\n\t')
        pprint.pprint(options)

        print("Loaded configuration from", args.config.absolute().resolve())
        Config.print()

    # ==========================================================================
    # Prepare and launch

    ind = Individual.from_file(args.robot)
    genome = ind.genome

    if args.save_cppn:
        path = str(args.robot.parent.joinpath(args.robot.stem + ".cppn"))
        genome.brain.to_dot(path + ".png")
        genome.brain.to_dot(path + ".pdf")
        genome.brain.to_dot(path + ".dot")
        return 0

    if args.draw_cppn:
        cppn = CPPN(genome.brain)
        s = 100
        # dp = Point(0, 0, 0)
        o_types = {CPPN.Output.Weight, CPPN.Output.LEO}
        outputs = cppn.outputs()
        wl_data = np.zeros((2, 2*s+1, 2*s+1, 3), dtype=np.uint8)
        b_data = np.zeros((s, s, 3), dtype=np.uint8)

        def to_uint(v):
            if v > 0:
                return np.full(3, np.uint8(v*255))
            else:
                return np.array([0, 0, np.uint8(-v*255)])

        for di, dj in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            dp = Point(-.5 + di, 1, -.5 + dj)
            i_, j_ = (s+1)*di, (s+1)*dj
            for i, x in enumerate(np.linspace(-1, 1, s)):
                for j, z in enumerate(np.linspace(-1, 1, s)):
                    cppn(Point(x, -.9, z), dp, outputs, o_types)
                    wl_data[0, i_+i, j_+j, :] = to_uint(outputs[0])
                    wl_data[1, i_+i, j_+j, :] = to_uint(outputs[1])
        dp = Point(0, 0, 0)
        for i, x in enumerate(np.linspace(-1, 1, s)):
            for j, z in enumerate(np.linspace(-1, 1, s)):
                b_data[i, j, :] = \
                    to_uint(cppn(Point(x, 1, z), dp, CPPN.Output.Bias))
        for name, a in [("weight", wl_data[0]),
                        ("leo", wl_data[1]),
                        ("bias", b_data)]:
            path = str(args.robot.parent.joinpath(
                f"{args.robot.stem}.cppn.{name}.png"))
            if cv2.imwrite(path, a):
                print("Generated", path)
            else:
                print("Error generating", path)
        return 0

    result, viewer = Evaluator.evaluate_rerun(genome, options)

    # ==========================================================================
    # Process results

    err = 0

    if args.perf_check and not defaults:
        err = performance_compare(ind.evaluation_result(), result, args.verbosity)

    if options.runner.record is not None:
        output = options.runner.save_folder.joinpath(options.runner.record.video_file_path)
        if not os.path.exists(output):
            print(f"Failed to generate {output}")
            err |= 1
        else:
            print(f"Successfully generated {output}")

    if args.verbosity > 0:
        duration = humanize.precisedelta(timedelta(seconds=time.perf_counter() - start))
        print(f"Evaluated {args.robot.absolute().resolve()} in {duration} / {Config.simulation_time}s")

    if viewer is not None and options.runner.view is not None \
            and not options.runner.view.auto_quit:
        while viewer.is_alive:
            viewer.render()
            time.sleep(1/60)

    return err


def performance_compare(lhs: Evaluator.Result, rhs: Evaluator.Result, verbosity):
    width = 20
    key_width = max(len(k) for keys in [lhs.fitnesses, lhs.stats, rhs.fitnesses, rhs.stats] for k in keys) + 1

    def s_format(s=''): return f"{s:{width}}"

    def f_format(f):
        if type(f) == builtins.list:
            # return f"{f}"[:width-3] + "..."
            return "\n" + pprint.pformat(f, width=width)
        else:
            return s_format(f"{f:g}")

    def map_compare(lhs_d: Dict[str, float], rhs_d: Dict[str, float]):
        output, code = "", 0
        lhs_keys, rhs_keys = set(lhs_d.keys()), set(rhs_d.keys())
        all_keys = sorted(lhs_keys.union(rhs_keys))
        for k in all_keys:
            output += f"{k:>{key_width}}: "
            lhs_v, rhs_v = lhs_d.get(k), rhs_d.get(k)
            if lhs_v is None:
                output += f"{Fore.YELLOW}{s_format()} > {f_format(rhs_v)}"
            elif rhs_v is None:
                output += f"{Fore.YELLOW}{f_format(lhs_v)} <"
            else:
                if lhs_v != rhs_v:
                    lhs_str, rhs_str = f_format(lhs_v), f_format(rhs_v)
                    if isinstance(lhs_v, numbers.Number):
                        diff = rhs_v - lhs_v
                        ratio = math.inf if lhs_v == 0 else diff/math.fabs(lhs_v)
                        output += f"{Fore.RED}{lhs_str} | {rhs_str}" \
                                  f"\t({diff}, {100*ratio:.2f}%)"
                    else:
                        output += "\n"
                        for lhs_item, rhs_item in zip(lhs_str.split('\n'), rhs_str.split('\n')):
                            if lhs_item != rhs_item:
                                output += Fore.RED
                            output += f"{lhs_item:{width}s} | {rhs_item:{width}s}"
                            if lhs_item != rhs_item:
                                output += Style.RESET_ALL
                            output += "\n"
                    code = 1
                else:
                    output += f"{Fore.GREEN}{f_format(lhs_v)}"

            output += f"{Style.RESET_ALL}\n"
        return output, code

    f_str, f_code = map_compare(lhs.fitnesses, rhs.fitnesses)
    d_str, d_code = map_compare(lhs.descriptors, rhs.descriptors)
    s_str, s_code = map_compare(lhs.stats, rhs.stats)
    max_width = max(len(line) for text in [f_str, s_str] for line in text.split('\n'))
    if verbosity > 0:
        def header(): print("-"*max_width)
        print("Performance summary:")
        header()
        print(f_str, end='')
        header()
        print(d_str, end='')
        header()
        print(s_str, end='')
        header()
        print()

    return max([f_code, d_code, s_code])


if __name__ == "__main__":
    exit(main())
