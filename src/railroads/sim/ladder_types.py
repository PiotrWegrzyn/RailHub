"""What a ladder is: the kinds of building it raises one step at a time, where they grow, how
likely, and how they join, merge and rise (Ladder, Join, Merge, Rise)."""

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from railroads.sim.constructions import BuildingKind
from railroads.sim.grids import FloatArray
from railroads.sim.points import Points
from railroads.sim.surroundings import Land


@dataclass(frozen=True, slots=True)
class Merge:
    """How many of a kind a neighbourhood needs before they merge into the next kind, and how
    many of them the merge pulls down."""

    needed: int
    pulled_down: int


@dataclass(frozen=True, slots=True)
class Join:
    """So many of a ladder's first kind in a neighbourhood that become one building of another
    kind, once the neighbourhood holds at least so many population points for each of its land
    cells off the mountains, and none of that kind stands within so many cells."""

    into: BuildingKind
    needed: int
    people_per_cell: float
    apart: int


@dataclass(frozen=True, slots=True)
class Rise:
    """The population a neighbourhood must hold for a building to become the next kind, and
    whether the map only ever gets one of that kind, for the first to get there."""

    people: int
    only_one: bool = False


@dataclass(frozen=True, slots=True)
class Ladder:
    """A line of buildings that grows: a turn puts up the first kind, and merges climb it."""

    kinds: tuple[BuildingKind, ...]
    land: Land
    # The chance a turn puts one up in a cell, before what is around it, in CHANCE_UNITs.
    base: float = 0.0
    # What each served point of its own kind in a cell adds there per doubling of what draws it,
    # in CHANCE_UNITs; railroads.sim.breeding.BREEDING_TIMES times the base when not given.
    per_doubling: float | None = None
    # Points the neighbourhood must hold before the ladder has any chance there.
    needs: Mapping[Points, int] = field(default_factory=dict)
    merges: Mapping[BuildingKind, Merge] = field(default_factory=dict)
    # The most of the first kind one cell, and one neighbourhood, may hold, beyond what fits.
    most_per_cell: int | None = None
    most_nearby: int | None = None
    skyscrapers: bool = False
    # The first year it grows, and if it is rare, one on the map for this many land cells at most.
    from_year: int = 0
    land_cells_each: int | None = None
    # New ones keep at least this many cells past any building on the ladder.
    apart: int | None = None
    # Kinds that climb the ladder by the people around them rather than by merging.
    rises: Mapping[BuildingKind, Rise] = field(default_factory=dict)
    # What its first kind joins into where enough of them gather among people.
    joins: Join | None = None
    # Grows only where the neighbourhood holds fewer population points than this.
    crowded_at: int | None = None
    # Points of a kind nearby draw more only once there are at least this many of them.
    draws_from: int = 1
    # How many years its buildings last starving (railroads.sim.dying.dies_after), when not the
    # usual DIE_AFTER_MONTHS there.
    starving_years: int | None = None

    def pull(self, nearby: npt.NDArray[np.int64]) -> FloatArray:
        """What the points nearby add to the chance, in doublings."""
        return np.where(nearby >= self.draws_from, np.log2(1 + nearby), 0.0)

    @property
    def name(self) -> str:
        return self.kinds[0].value
