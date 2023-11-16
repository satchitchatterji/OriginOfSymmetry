#!/usr/bin/env python3

import argparse
import pprint
import signal
import sys
import time
from collections import namedtuple
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
import sched
from typing import Callable

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection


class Options:
    def __init__(self):
        self.folder: Path = "./tmp/last"
        self.sleep: float = 2

        self.with_max = True
        self.with_med = False
        self.with_avg = False
        self.with_fill = False

    @staticmethod
    def populate(p: argparse.ArgumentParser):
        p.add_argument("folder", metavar="DIR", type=Path, nargs='?', default="./tmp/last",
                       help="The folder to monitor")

        group = p.add_argument_group("Updating", "Controls for updating frequency ...")
        group.add_argument("--sleep", dest="sleep", metavar='X', type=float,
                           help="Sets sleep period")

        group = p.add_argument_group("Data", "Controls what to plot")
        group.add_argument("--max", dest="with_max", nargs='?', const=True, metavar='ON', type=bool,
                           help="Plot the maximal value")
        group.add_argument("--med", dest="with_med", nargs='?', const=True, metavar='ON', type=bool,
                           help="Plot the median value")
        group.add_argument("--avg", dest="with_avg", nargs='?', const=True, metavar='ON', type=bool,
                           help="Plot the average value")
        group.add_argument("--fill", dest="with_fill", nargs='?', const=True, metavar='ON', type=bool,
                           help="Plot the filled area between min-max")


def main():
    class RunGuard:
        def __init__(self): self.running = True
        def __call__(self, s, f): self.running = False
    runner = RunGuard()

    signal.signal(signal.SIGINT, runner)

    args = Options()
    parser = argparse.ArgumentParser(description="Monitor for evolutionary algorithms")
    Options.populate(parser)
    parser.parse_args(namespace=args)

    gen_stats = args.folder.joinpath("stats_gen.dat")
    while runner.running and not gen_stats.exists():
        print(f"Waiting for {gen_stats} to be created", end='\r')
        sys.stdout.flush()
        time.sleep(args.sleep)

    def get_data():
        l_df = pd.read_csv(gen_stats, sep=' ', index_col='Gen')
        l_fitnesses = {name.split('_')[0] for name in l_df.columns}
        return l_fitnesses, l_df

    fitnesses, df = get_data()
    fig, axs = plt.subplots(len(fitnesses), squeeze=False, sharex=True)
    axs = axs.flatten()

    plotter_t = Callable[[], None]

    @dataclass
    class Arts:
        ax: plt.Axes = None
        name: str = None
        max: plotter_t = None
        med: plotter_t = None
        avg: plotter_t = None
        fill: plotter_t = None
    arts = []

    for ax, fitness in zip(axs, fitnesses):
        arts.append(Arts(
            ax=ax, name=fitness,
            max=(lambda a, f: a.plot(df.index, df[f"{f}_max"])[0]) if args.with_max else None,
            med=(lambda a, f: a.plot(df.index, df[f"{f}_med"], linestyle='dashed')[0]) if args.with_med else None,
            avg=(lambda a, f: a.plot(df.index, df[f"{f}_avg"], linestyle='dotted')[0]) if args.with_avg else None,
            fill=(lambda a, f: a.fill_between(df.index, df[f"{f}_max"], df[f"{f}_min"],
                                              alpha=.1)) if args.with_fill else None,
        ))
    fig.show()
    plt.tight_layout()

    s = sched.scheduler(time.time, time.sleep)

    def should_quit():
        return args.sleep <= 0 or not runner.running or not plt.fignum_exists(fig.number)

    def replot():
        if gen_stats.exists():
            nonlocal fitnesses, df
            old_fitnesses = fitnesses

            fitnesses, df = get_data()
            if old_fitnesses != fitnesses:
                raise ValueError("Fitness set changed. Please restart")

            for i, art in enumerate(arts):
                art.ax.clear()
                if i == len(arts) - 1:
                    art.ax.set_xlabel('Generation')
                art.ax.set_ylabel(art.name)
                for plotter in filter(lambda x: x is not None, [art.fill, art.med, art.avg, art.max]):
                    plotter(art.ax, art.name)
                art.ax.relim()
                art.ax.autoscale_view(True, False, True)
            plt.tight_layout()

            name = gen_stats.parent.resolve().relative_to(Path.cwd())
            fig.suptitle(f"{name} @ {time.asctime()}")
            plt.get_current_fig_manager().set_window_title(f"Monitoring {name}")
            fig.align_ylabels()
            fig.canvas.draw()
        else:
            print(f'Waiting for {gen_stats} to be created', end='\r')
            sys.stdout.flush()

        if not should_quit():
            s.enter(args.sleep, 1, replot)

    def redraw():
        fig.canvas.flush_events()

        if not should_quit():
            s.enter(1/60, 1, redraw)

    s.enter(args.sleep, 1, replot)
    s.enter(1/60, 1, redraw)
    s.run()


if __name__ == '__main__':
    main()
