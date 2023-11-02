import dataclasses
import json
import logging
import os
import pprint
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, NamedTuple, Callable, Optional

from colorama import Fore, Style

from ..misc.genome import RVGenome


class Tee:
    """Ensure that everything that's printed is also saved
    """

    class PassthroughStream:
        """Forwards received messages to log/file"""
        def __init__(self, parent: 'Tee'):
            self.tee = parent

        def write(self, msg):
            self.tee.write(msg)

        def flush(self):
            self.tee.flush()

        def isatty(self): return self.tee.out.isatty()

        def close(self): pass

    class FormattedStream(PassthroughStream):
        """Forwards received messages to log/file """
        def __init__(self, parent: 'Tee', formatter: str):
            super().__init__(parent)
            self.formatter = formatter

        def write(self, msg):
            super().write(self.formatter.format(msg))

    def __init__(self, filter_out: Optional[Callable[[str], bool]] = lambda _: False):
        self.out = sys.stdout
        self.log = None
        self.msg_queue = []    # Collect until log file is available
        self.registered = False
        self.filter = filter_out

    def register(self):
        if not self.registered:
            sys.stdout = self.PassthroughStream(self)
            sys.stderr = self.FormattedStream(self, Fore.RED + "{}" + Style.RESET_ALL)
            self.registered = True

    def teardown(self):
        if self.registered:
            self.flush()
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            self.registered = False

    def set_log_path(self, path: Path):
        self.register()
        self.log = open(path, 'wt', buffering=1)
        for msg in self.msg_queue:
            self._write(msg)

    def _write(self, msg: str):
        if not self.filter(msg):
            self.log.write(msg)

    def write(self, msg: str):
        if self.log is None:
            self.msg_queue.append(msg)
        else:
            self._write(msg)
        self.out.write(msg)

    def flush(self):
        self.out.flush()
        if self.log is not None:
            self.log.flush()


def normalize_run_parameters(options: NamedTuple):
    if options.id is None:
        options.id = int(time.strftime('%Y%m%d%H%M%S'))
        logging.info(f"Generated run id: {options.id}")

    if options.seed is None:
        try:
            options.seed = int(options.id)
        except ValueError:
            options.seed = round(1000 * time.time())
        logging.info(f"Deduced seed: {options.seed}")

    # Define the run folder
    folder_name = options.id
    if not isinstance(folder_name, str):
        folder_name = f"run{options.id}"
    options.run_folder = os.path.normpath(f"{options.base_folder}/{folder_name}/")
    logging.info(f"Run folder: {options.run_folder}")

    # Check the thread parameter
    options.threads = max(1, min(options.threads, len(os.sched_getaffinity(0))))
    logging.info(f"Parallel: {options.threads}")

    if options.verbosity >= 0:
        raw_dict = {k: v for k, v in options.__dict__.items() if not k.startswith('_')}
        logging.info(f"Post-processed command line arguments:\n{pprint.pformat(raw_dict)}")


@dataclass
class EvaluationResult:
    DataCollection = Dict[str, float]
    fitnesses: DataCollection = field(default_factory=dict)
    descriptors: DataCollection = field(default_factory=dict)
    stats: DataCollection = field(default_factory=dict)


@dataclass
class Individual:
    DataCollection = EvaluationResult.DataCollection

    genome: RVGenome
    fitnesses: DataCollection = field(default_factory=dict)
    descriptors: DataCollection = field(default_factory=dict)
    stats: DataCollection = field(default_factory=dict)
    eval_time: float = 0

    def id(self):
        return self.genome.id()

    def __eq__(self, other):
        assert self.id() is not None
        assert other.id() is not None
        return self.id() == other.id()

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        return f"{{id={self.id()}, fitness={self.fitnesses}, features={self.descriptors}}}"

    def update(self, r: EvaluationResult):
        self.fitnesses = r.fitnesses
        self.descriptors = r.descriptors
        self.stats = r.stats

    def evaluation_result(self) -> EvaluationResult:
        return EvaluationResult(
            fitnesses=self.fitnesses,
            descriptors=self.descriptors,
            stats=self.stats,
        )

    def to_json(self):
        dct = dataclasses.asdict(self)
        dct["genome"] = self.genome.to_json()
        return dct

    def to_file(self, path):
        with open(path, 'w') as f:
            json.dump(self.to_json(), f)

    @classmethod
    def from_json(cls, data):
        data.pop('id', None)
        data.pop('parents', None)
        ind = cls(**data)
        ind.genome = RVGenome.from_json(data['genome'])
        return ind

    @classmethod
    def from_file(cls, path):
        with open(path, 'r') as f:
            return cls.from_json(json.load(f))
