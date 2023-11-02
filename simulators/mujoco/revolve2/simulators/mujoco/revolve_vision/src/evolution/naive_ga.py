import concurrent
import dataclasses
import json
import logging
import math
import os.path
import pprint
import shutil
import time
from abc import ABC, abstractmethod
from collections import namedtuple
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from random import Random
from typing import Dict, NamedTuple, Callable, List, Type

import numpy as np

from ..evolution.common import normalize_run_parameters
from ..misc.config import Config
from ..misc.genome import RVGenome


class SelectionAlgorithm(dict, ABC):
    Population = List['Evolver.Ind']

    def __init__(self, **kwargs):
        dict.__init__(self, name=self.__class__.__name__, **kwargs)

    @abstractmethod
    def process(self, population: Population, callbacks, rng: Random, data_folder: Path) -> Population:
        """
        Generate the next population based current fitness and derived algorithm

        :param population: The collection of individuals to work on
        :param callbacks: Set of functions for producing mutated/crossed individuals
        :param rng: The source of randomness
        :param data_folder: Base folder under which to store intermediate results (such as champions)
        :return: the resulting population
        """
        pass


class NaiveTournamentSelection(SelectionAlgorithm):
    def __init__(self, size=5, elites=1):
        SelectionAlgorithm.__init__(self, size=size, elites=elites)
        if elites != 1:
            raise NotImplementedError

    def process(self, population: Dict[int, 'Evolver.Ind'], callbacks, rng: Random,
                data_folder: Path, generation: int, max_generation: int):
        new_population = {}
        fitnesses = [k for k in population[list(population.keys())[0]].fitnesses]
        # max_fitnesses = {k: (v, ind) for ind in population for ind.fitnesses.}

        elites = [max(((f, ind.fitnesses[f], ind) for ind in population.values()),
                      key=lambda v: v[1]) for f in fitnesses]
        new_population.update({elite.id: elite for _, _, elite in elites})

        g_width = math.ceil(math.log10(max_generation))
        g_folder = data_folder.joinpath(f"gen{generation:0{g_width}d}")
        Path.mkdir(g_folder, exist_ok=False)
        for f, _, elite in elites:
            path = g_folder.joinpath(f"{f}.json")
            with open(path, 'w') as file:
                json.dump(elite.to_json(), file)

        link = data_folder.joinpath("last")
        if link.exists():
            os.unlink(link)
        os.symlink(g_folder.absolute(), link)

        while len(new_population) < len(population):
            tournament = rng.sample(sorted(population.values(),
                                           key=lambda lhs: lhs.id),
                                    self['size'])

            champion = rng.choice(self.pareto_front(tournament))
            offspring = callbacks.mutated(champion)
            new_population[offspring.id] = offspring

        return new_population

    @staticmethod
    def dominated(lhs: 'Evolver.Ind', rhs: 'Evolver.Ind'):
        return all(lhs_f <= rhs_f for lhs_f, rhs_f in zip(lhs.fitnesses.values(), rhs.fitnesses.values()))

    @classmethod
    def pareto_front(cls, tournament):
        pareto = []
        for i, ind in enumerate(tournament):
            dominated = (len(pareto) > 0 and all(not cls.dominated(ind, opp) for opp in pareto)) \
                        and all(not cls.dominated(ind, opp) for opp in tournament[i + 1:])
            if not dominated:
                pareto.append(ind)
        return pareto


class Evolver:
    genome_class: Type
    algorithm_class: Type

    CallbackRandomG = Callable[[Random], RVGenome]
    CallbackMutateG = Callable[[Random, RVGenome], RVGenome]
    CallbackCrossoverG = Callable[[Random, RVGenome, RVGenome], RVGenome]
    CallbackEvaluate = Callable[['Ind'], 'Result']

    class Reproduction(Enum):
        CLONE = auto()
        MUTATE = auto()
        CROSS = auto()

    @dataclass
    class Stats:
        generation: int = 0
        cum_time: float = 0

    class IDGenerator:
        next_value = 0

        @classmethod
        def next(cls) -> int:
            value = cls.next_value
            cls.next_value += 1
            return value

    CallbackRandomI = Callable[[], Ind]
    CallbackMutateI = Callable[[Ind], Ind]
    CallbackCrossoverI = Callable[[Ind, Ind], Ind]
    Callbacks = namedtuple("Callbacks", ['random', 'mutated', 'crossed', 'evaluate', 'misc'])

    class CallbackType(Enum):
        BEFORE_GEN = auto()
        BEFORE_NEW_POP = auto()
        AFTER_GEN = auto()

    Genealogy = Dict[int, List[int]]

    def __init__(self, options: NamedTuple, algorithm: SelectionAlgorithm):
        # Manage run id, seed, data folder...
        normalize_run_parameters(options)

        # Normalize reproduction rates
        self._normalized_rates(options.reproduction)

        self.options = options

        self.rng = Random(self.options.seed)

        self.callbacks = Evolver.Callbacks(None, None, None, None, None)
        self.algorithm = algorithm

        self.population = None
        self.idGenerator = Evolver.IDGenerator()
        self.stats = Evolver.Stats()

        logging.info("Created evolver with options:")
        options.algorithm = self.algorithm
        Config.evolution = options.__dict__
        logging.info(pprint.PrettyPrinter(indent=2).pformat(options.__dict__))

        try:
            # Prepare storage locations
            self.run_folder = Path(self.options.run_folder)
            self.run_folder.parent.mkdir(parents=True, exist_ok=True)
            if self.options.overwrite and self.run_folder.exists():
                logging.warning("Overwriting run folder (as requested)")
                shutil.rmtree(self.run_folder)
            self.run_folder.mkdir(exist_ok=False)
            if not self.run_folder.exists():
                raise OSError(f"Failed to create run folder "
                              f"{self.run_folder.absolute()}")
            logging.info(f"Created run folder {self.run_folder.absolute()}")

            tee.set_log_path(self.run_folder.joinpath("log"))

        except Exception as e:  # And if it fails make sure some output is produced
            print(e)
            tee.teardown()
            raise e

        link = self.run_folder.parent.joinpath("last")
        if link.exists():
            os.unlink(link)
        os.symlink(self.run_folder.absolute(), link)
        logging.info(f"Created link {link.absolute()} -> {self.run_folder.absolute()}")

        # Store configuration
        config_path = self.run_folder.joinpath("config.json")
        Config.write_json(config_path)
        logging.info(f"Stored configuration in {config_path.absolute()}")

    @staticmethod
    def _ind_evaluate(evaluate, ind: Ind) -> 'Result':
        start_time = time.perf_counter()
        r = evaluate(ind.genome)
        r.eval_time = time.perf_counter() - start_time
        return r

    def set_callbacks(self, evaluate: CallbackEvaluate,
                      random: CallbackRandomG,
                      mutate: CallbackMutateG,
                      crossover: CallbackCrossoverG = None):

        self.callbacks = Evolver.Callbacks(
            random=lambda: Evolver.Ind(id=self.idGenerator.next(), genome=random(self.rng), parents=[]),
            mutated=lambda ind: Evolver.Ind(id=self.idGenerator.next(), genome=mutate(ind.genome, self.rng),
                                            parents=[ind.id]),
            crossed=lambda lhs, rhs: Evolver.Ind(id=self.idGenerator.next(),
                                                 genome=crossover(lhs.genome, rhs.genome, self.rng),
                                                 parents=[lhs.id, rhs.id]
                                                 if crossover is not None else None),
            evaluate=evaluate,
            misc={}
        )

    def add_misc_callback(self, t: CallbackType, c: Callable):
        self.callbacks.misc[t] = c

    def run(self, generations: int = None):
        if self.callbacks.random is None:
            raise ValueError("Missing callback for random generation")

        if self.callbacks.mutated is None:
            raise ValueError("Missing callback for asexual reproduction")

        if self.callbacks.crossed is None and \
                self.options.reproduction[self.Reproduction.CROSS.name] > 0:
            raise ValueError("Missing callback for sexual reproduction")

        if self.callbacks.evaluate is None:
            raise ValueError("Missing evaluation callback")

        if generations is None:
            generations = self.options.generations

        if self.population is None:
            logging.info(f"Generating {self.options.pop_size} individuals")
            self.population = {
                ind.id: ind
                for _ in range(self.options.pop_size) if (ind := self.callbacks.random())
            }
            self._log_genealogy()

        assert len(self.population) == self.options.pop_size

        for g in range(generations):
            self._step()

            self.stats.generation += 1

    def _step(self):
        start_time = time.perf_counter()

        if (f := self.callbacks.misc.get(self.CallbackType.BEFORE_GEN)) is not None:
            f()

        with concurrent.futures.ProcessPoolExecutor(
                max_workers=self.options.threads
        ) as executor:
            futures = [
                (ind.id,
                 executor.submit(self._ind_evaluate, self.callbacks.evaluate, ind))
                for ind in filter(lambda i: i.is_new, self.population.values())
            ]
            for gid, future in futures:
                ind = self.population[gid]
                result = future.result()
                ind.fitnesses = result.fitnesses
                ind.stats = result.stats
                ind.eval_time = result.eval_time

        self.stats.wall_time = time.perf_counter() - start_time
        self.stats.speedup = sum(ind.eval_time for ind in self.population.values()) \
            / self.stats.wall_time
        self._process_stats()

        new_population = self.algorithm.process(self.population, self.callbacks, self.rng,
                                                self.run_folder, self.stats.generation, self.options.generations)

        if (f := self.callbacks.misc.get(self.CallbackType.BEFORE_NEW_POP)) is not None:
            f()

        self.population = new_population

        self._log_genealogy()

        if (f := self.callbacks.misc.get(self.CallbackType.AFTER_GEN)) is not None:
            f()

    def _process_stats(self):
        def get_stats(getter):
            data = {k: np.array([getter(ind)[k]
                                 for ind in self.population.values()])
                    for k in getter(self.population[list(self.population.keys())[0]])}
            return data, {k: (*np.quantile(data[k], [0, .5, 1]), np.mean(data[k])) for k in data}

        f_data, f_stats = get_stats(lambda ind: ind.fitnesses)
        s_data, s_stats = get_stats(lambda ind: ind.stats)

        zip_data = {}
        for i, gid in enumerate(self.population.keys()):
            zip_data[gid] = {k: data[k][i] for data in [f_data, s_data] for k in data}

        self._log_stats(zip_data, {s: stats[s] for stats in [f_stats, s_stats] for s in stats})
        self._print_stats(f_stats, s_stats)

    def _log_stats(self, data, stats):
        with open(self.run_folder.joinpath("stats_gen.dat"), 'a') as f:
            if self.stats.generation == 0:
                f.write('Gen')
                for stat in stats:
                    for suffix in ["min", "med", "max", "avg"]:
                        f.write(f" {stat}_{suffix}")
                f.write('\n')
            f.write(str(self.stats.generation))
            for stat in stats.values():
                for val in stat:
                    f.write(f" {val:g}")
            f.write('\n')

        with open(self.run_folder.joinpath("stats_ind.dat"), 'a') as f:
            if self.stats.generation == 0:
                f.write('ID')
                for stat in stats:
                    f.write(f" {stat}")
                f.write('\n')
            for iid, items in data.items():
                f.write(str(iid))
                for d in items.values():
                    f.write(f" {d:g}")
                f.write('\n')

    def _print_stats(self, f_stats, s_stats):
        B = namedtuple('Borders', ['h', 'v', 'ul', 'ur', 'ml', 'mr', 'bl', 'br', 'um', 'cm', 'bm'])
        b = B(*[c for c in '\u2500\u2502\u250C\u2510\u251C\u2524\u2514\u2518\u252C\u253C\u2534'])

        tw = 79
        lw = 3 + max(10, max(len(k) for ls in [f_stats, s_stats] for k in ls))

        def line(lc, rc, ticks=None):
            string = list(f"{lc}{b.h * tw}{rc}")
            if ticks is not None:
                for i, c in ticks:
                    string[i] = c
            print("".join(string))

        print(f"{b.ul}{b.h * (tw // 2)}{b.ur}")
        print(f"{b.v} {time.asctime():^{tw // 2 - 2}s} {b.v}")
        line(b.ml, b.ur, [(tw // 2 + 1, b.cm)])

        h_gen = f"Generation {self.stats.generation:3d}"
        h_time = f" {self.stats.wall_time:.2f}s (x{self.stats.speedup:.2f})"
        print(f"{b.v} {h_gen:^{tw // 2 - 2}s} {b.v} {h_time:^{tw // 2 - 2}s} {b.v}")
        line(b.ml, b.mr, [(lw, b.um), (tw // 2 + 1, b.bm)])

        spacing = (tw - lw - 4 * (6 + 1 + 4)) / 5
        spacer = " " * int(spacing)
        pad = " " * int(5 * spacing - 5 * int(spacing))

        def show_stats(stats):
            for k, s in stats.items():
                print(f"{b.v} {k:>{lw - 3}s} {b.v}"
                      f"{spacer}min: {s[0]:6.2f}{spacer}med: {s[1]:6.2f}"
                      f"{spacer}avg: {s[3]:6.2f}{spacer}max: {s[2]:6.2f}{spacer}{pad}{b.v}")

        show_stats(f_stats)
        if len(s_stats) > 0:
            line(b.ml, b.mr, [(lw, b.cm)])
            show_stats(s_stats)
        line(b.bl, b.br, [(lw, b.bm)])

    def _log_genealogy(self):
        with open(self.run_folder.joinpath("genealogy.dat"), 'a') as f:
            for ind in self.population.values():
                if ind.is_new:
                    f.write(f"{ind.id} {' '.join(str(p_id) for p_id in ind.parents)}\n")

    @classmethod
    def _normalized_rates(cls, rates):
        r_sum = 0
        weights = dict(rates)
        for r in cls.Reproduction:
            if r.name in weights:
                r_sum += weights[r.name]
        rates.clear()
        for r in cls.Reproduction:
            rates[r.name] = weights.get(r.name, 0) / r_sum


if __name__ == '__main__':
    raise UserWarning("The algorithm should not be invoked directly")
