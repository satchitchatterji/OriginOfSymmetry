#!/usr/bin/env python3

import argparse
import glob
import logging
import pickle
import pprint
import sys
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Tuple, Optional, List, Union

from qdpy.plots import plot_evals

from src.evolution.map_elite import Logger, summary_plots
from src.evolution.map_elite import Grid


@dataclass
class BehavioralSpaceBoundingBox:
    class Type(Enum):
        Index = auto()
        Value = auto()
        Ratio = auto()

    type: Type = None
    data: Tuple[float] = None

    def scaled(self, shape, bounds) -> Tuple[int]:
        if self.type == self.Type.Index:
            assert 0 <= self.data[0] <= self.data[2] <= shape[0]
            assert 0 <= self.data[1] <= self.data[3] <= shape[1]
            data = self.data

        elif self.type == self.Type.Value:
            assert bounds[0][0] <= self.data[0] <= self.data[2] <= bounds[0][1]
            assert bounds[1][0] <= self.data[1] <= self.data[3] <= bounds[1][1]
            data = []
            for di, bi in enumerate([0, 1, 0, 1]):
                data.append(round((self.data[di] - bounds[bi][0]) / (bounds[bi][1] - bounds[bi][0]) * shape[bi]))

        elif self.type == self.Type.Ratio:
            assert 0 <= self.data[0] <= self.data[2] <= 100
            assert 0 <= self.data[1] <= self.data[3] <= 100
            data = []
            for di, bi in enumerate([0, 1, 0, 1]):
                data.append(round(self.data[di] * shape[bi] / 100))

        else:
            raise NotImplementedError(f"Unknown type {self.type}")

        print(self.data, data)
        return tuple(data)


class Options:
    def __init__(self):
        self.src_folder: Path = Path()
        self.dst_folder: Optional[Path] = None

        self.iteration: str = Logger.final_filename

        self.champions: List[BehavioralSpaceBoundingBox] = []
        self.do_plot: bool = False
        self.format: str = "png"
        self.fig_size: Tuple[float, float] = (4., 4.)

    @staticmethod
    def tuple_parser(string: str):
        return tuple(map(int, string.replace("(", "").replace(")", "").split(",")))

    @staticmethod
    def coord_parser(string: str):
        string = string.replace("(", "").replace(")", "")

        dim = 4
        if '%' in string:
            c_type = BehavioralSpaceBoundingBox.Type.Ratio
            string = string.replace('%', '')
            data = tuple(map(float, string.split(",")))
        elif '.' in string:
            c_type = BehavioralSpaceBoundingBox.Type.Value
            data = tuple(map(float, string.split(",")))
        else:
            c_type = BehavioralSpaceBoundingBox.Type.Index
            data = tuple(map(int, string.split(",")))

        # print(string, data, bsbb)
        if len(data) != dim:
            raise argparse.ArgumentTypeError("Coordinate tuples should contain"
                                             " 4 values")
        return BehavioralSpaceBoundingBox(c_type, data)

    @staticmethod
    def populate(p: argparse.ArgumentParser):
        group = p.add_argument_group("Storage",
                                     "Defines data/outputs location")
        group.add_argument('src_folder', metavar='DIR', type=Path,
                           help="Path to data folder")
        group.add_argument('--out', dest="dst_folder", metavar='DIR',
                           type=Path, help="Where to output the summaries"
                                           " (defaults to the source folder)")
        group.add_argument('--iteration', dest="iteration", metavar='FILE',
                           type=str, help="Name (not path) of the file to load "
                                          "or 'all' to process *.p")

        group = p.add_argument_group("Extraction",
                                     "Retrieve specific data from the pickled"
                                     " file")
        group.add_argument('--champion', dest="champions",
                           metavar='X0,Y0,X1,Y1',
                           type=Options.coord_parser, action='append',
                           help="Bounding box in the behavioral space from"
                                " which the best performing individual, if any,"
                                " will be returned")
        group.add_argument('--all-champions', dest='champions',
                           action='store_const', const='all',
                           help="Extract all champions from the grid")

        group = p.add_argument_group("Plots",
                                     "Defines what and how to plot")
        group.add_argument('--plot', dest="do_plot", action='store_true',
                           help="Whether to generate summary plots")
        group.add_argument('--format', dest="format", metavar='F',
                           choices=["png", "pdf"],
                           type=str, help="File format for the plots")
        group.add_argument('--fig-size', dest="fig_size", metavar='W,H',
                           type=Options.tuple_parser,
                           help="Size of the generated plots")


def do_plots(args, data, name):
    summary_plots(evals=data["evals"], iterations=data["iterations"],
                  grid=data["container"],
                  output_dir=args.dst_folder, name=name,
                  labels=data["labels"],
                  fig_size=args.fig_size, ext=args.format)


def extract_champions(args, data, name):
    grid: Grid = data['container']
    assert isinstance(grid, Grid)

    bounds = grid.features_domain
    shape = grid.shape

    print("Looking for champions in", args.champions)
    print(shape, bounds)

    folder = args.dst_folder.joinpath(name)
    folder.mkdir(exist_ok=True)

    def _write(coord, champ):
        filename = folder.joinpath(
            f"best_{champ.features[0]:.2f}_{champ.features[1]:.2f}.json")
        print(filename)
        champ.to_file(filename)
        if filename.exists():
            logging.info(f"Saved best individual for {coord} in {filename}")
        else:
            logging.warning(f"Failed to save best individual for {coord} to"
                            f" {filename}")

    if args.champions == "all":
        for i, c in grid.solutions.items():
            if len(c) > 0:
                print(i, c)
                _write(i, c[0])
            # print(c)

    else:
        for c in args.champions:
            i0, j0, i1, j1 = c.scaled(shape, bounds)
            print(i0, j0, i1, j1)

            best = None
            for i in range(i0, i1):
                for j in range(j0, j1):
                    cell = grid.solutions[(i, j)]
                    if len(cell) > 0:
                        ind = cell[0]
                        if best is None or ind.fitness[0] > best.fitness[0]:
                            best = ind

            print("Best:", best)
            if best is not None:
                _write(c, best)
            else:
                logging.warning(f"No best individual for {c}")


def main():
    args = Options()
    parser = argparse.ArgumentParser(
        description="Summary/plotter for QDPy map-elite")
    Options.populate(parser)
    parser.parse_args(namespace=args)

    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format="%(message)s")

    files = [args.src_folder.joinpath(args.iteration)]
    if args.iteration == "all":
        files = sorted(args.src_folder.glob("*.p"))
        if len(files) == 0:
            logging.critical(f"No file to process for {args.src_folder}/*.p")

    if args.dst_folder is None:
        args.dst_folder = args.src_folder

    for f in files:
        f: Path = f
        logging.info(f"Loading from {f}")
        with open(f, "rb") as df:
            data = pickle.load(df)

            if len(args.champions) > 0:
                extract_champions(args, data, f.stem)

            if args.do_plot:
                do_plots(args, data, f.stem)


if __name__ == '__main__':
    main()
