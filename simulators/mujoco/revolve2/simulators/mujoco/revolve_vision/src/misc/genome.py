import copy
from dataclasses import dataclass, astuple
from random import Random
from typing import Optional

from abrain import Genome as ANNGenome
from abrain.core.genome import GIDManager
from .config import Config


@dataclass
class VisionData:
    w: int = 3
    h: int = 2

    @staticmethod
    def random(rng: Random):
        return VisionData()

    def mutate(self, rng: Random):
        f, field, bounds = rng.choice([("w", self.w, Config.vision_mutation_range[0]),
                                       ("h", self.h, Config.vision_mutation_range[1])])
        if field == bounds[0]:
            field += 1
        elif field == bounds[1]:
            field -= 1
        else:
            field += rng.choice([-1, 1])
        setattr(self, f, field)

    def __iter__(self):
        return iter(astuple(self))


class RVGenome:
    __private_key = object()

    def __init__(self, brain: ANNGenome, vision: Optional[VisionData] = None):
        self.brain = brain
        self.vision = vision

    def with_vision(self):
        return self.vision is not None

    def id(self):
        return self.brain.id()

    def parents(self):
        return self.brain.parents()

    def __repr__(self):
        if self.with_vision():
            return f"{{{str(self.vision)}, {str(self.brain)}}}"
        else:
            return str(self.brain)

    def mutate(self, rng: Random) -> None:
        if self.with_vision() and rng.random() < Config.vision_mutation_rate:
            self.vision.mutate(rng)
        else:
            self.brain.mutate(rng)

    def mutated(self, rng: Random, id_manager: GIDManager):
        clone = self.copy()
        clone.mutate(rng)
        clone.brain.update_lineage(id_manager, [self.brain])
        return clone

    @staticmethod
    def random(rng: Random, id_manager: GIDManager) -> 'RVGenome':
        return RVGenome(
            ANNGenome.random(rng, id_manager),
            VisionData.random(rng) if Config.with_vision else None
        )

    def copy(self) -> 'RVGenome':
        return RVGenome(
            self.brain.copy(),
            copy.deepcopy(self.vision)
        )

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, _):
        return self.copy()

    def __getstate__(self):
        return dict(brain=self.brain, vision=self.vision)

    def __setstate__(self, state):
        self.__dict__ = state
        assert isinstance(self.brain, ANNGenome)
        if self.vision is not None:
            assert isinstance(self.vision, VisionData)

    def to_json(self):
        if self.vision is None:
            return self.brain.to_json()
        else:
            return dict(brain=self.brain.to_json(), vision=self.vision.__dict__)

    @staticmethod
    def from_json(data) -> 'RVGenome':
        """Recreate a RVGenome from string json representation
        """

        ## TODO Remove retro-compatibility
        if "brain" not in data:
            return RVGenome(ANNGenome.from_json(data), VisionData())

        return RVGenome(ANNGenome.from_json(data["brain"]),
                        VisionData(**data["vision"]))

    @staticmethod
    def from_dot(path: str, rng: Random):
        """Does not make sense for embedded abrain genome"""
        raise RuntimeError
