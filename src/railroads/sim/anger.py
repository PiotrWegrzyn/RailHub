"""How angry people are with the company (W) for moving their buildings out of the way of its
track, stations and facilities, or bulldozing them.

W starts at nothing and rises by ANGER_PER_BUILDING for every building moved aside or bulldozed.
Each point of it makes moving or bulldozing one dearer by ANGER_SURCHARGE of its price, added up and not compounded: at
W = 2 a building costs twice its price. On 1 January of every year divisible by COOLING_YEARS
it cools to W - ceil(log2(1 + W)), and the player is told.
"""

import math
from dataclasses import dataclass

ANGER_PER_BUILDING = 1
ANGER_SURCHARGE = 0.5
COOLING_YEARS = 5


def cooled(anger: int) -> int:
    """What W cools to: W - ceil(log2(1 + W))."""
    return anger - math.ceil(math.log2(1 + anger))


def surcharge(anger: int) -> float:
    """What moving a building costs at this W, against its price at none."""
    return 1.0 + ANGER_SURCHARGE * anger


def cooling_year(year: int) -> int:
    """The first year from this one on that W cools on 1 January of."""
    return -(-year // COOLING_YEARS) * COOLING_YEARS


@dataclass
class Anger:
    value: int = 0

    @property
    def surcharge(self) -> float:
        return surcharge(self.value)

    def moved(self, buildings: int) -> None:
        """Rises for so many buildings moved aside."""
        self.value += ANGER_PER_BUILDING * buildings

    def new_year(self, year: int) -> int:
        """Cools if the year is one it cools in; returns by how much."""
        if year % COOLING_YEARS:
            return 0
        before, self.value = self.value, cooled(self.value)
        return before - self.value
