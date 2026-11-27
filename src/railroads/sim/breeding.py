"""How likely each ladder's buildings are to breed: BREEDING_TIMES its base pull, what draws them
(DRAWN_BY, DRAWN_BY_KIND) and what keeps them away (SHUNNED_BY), and the lively venues that make
homes grow faster."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from railroads.sim.constructions import (
    BuildingKind,
)
from railroads.sim.ladder_types import Ladder
from railroads.sim.ladders import (
    CHANCE_UNIT,
    GrowthRule,
)
from railroads.sim.points import Points

# How many times its base chance each served point of a ladder's own kind in a cell adds there,
# per doubling of what draws it, unless the ladder gives its own (railroads.sim.ladder_types.Ladder).
BREEDING_TIMES = 3.0


# Ladders drawn by industry are drawn instead by their own kind around them OWN_KIND_TIMES as
# strongly as industry drew them, and by industry CATEGORY_SHARE of that; in a cell, each served
# point of their own kind counts OWN_KIND_TIMES and each of the rest of their kind of points
# CATEGORY_SHARE. So a served sawmill breeds sawmills, not brickworks.
OWN_KIND_TIMES = 10.0


CATEGORY_SHARE = 0.1


# What a ladder's chances can follow: points of a kind, or the points the buildings of one
# ladder give, named by its first kind.
Channel = Points | BuildingKind


# The points around a cell that draw a ladder's buildings, and how much each counts, by the kind
# of points they give.
INDUSTRIAL = {Points.POPULATION: 1.0, Points.INDUSTRY: 1.0}


SOCIAL = {Points.POPULATION: 2.0}


DRAWN_BY: Mapping[Points, Mapping[Points, float]] = MappingProxyType(
    {
        Points.POPULATION: {Points.POPULATION: 7.0, Points.INDUSTRY: 2.0},
        Points.FOOD: {Points.FOOD: 1.0, Points.POPULATION: 1.0},
        Points.ORE: INDUSTRIAL,
        Points.TIMBER: INDUSTRIAL,
        Points.TRADE: INDUSTRIAL,
        Points.INDUSTRY: INDUSTRIAL,
        Points.CORPORATE: SOCIAL,
        Points.WORSHIP: SOCIAL,
        Points.AIR: SOCIAL,
        Points.MILITARY: {Points.MILITARY: 1.0},
    }
)


# The farms that raise animals, which draw butchers as strongly as the people around do, each
# point of them ANIMALS_DRAW times.
ANIMAL_FARMS = (
    BuildingKind.CATTLE_RANCH,
    BuildingKind.PIG_FARM,
    BuildingKind.CHICKEN_FARM,
    BuildingKind.SHEEP_FARM,
)
ANIMALS_DRAW = 2.0


# Kinds drawn by other points than the rest of their kind of points (refineries by the oil wells
# and mines and the ports around them, butchers by people and the animal farms), and kinds that
# keep away
# from some: their whole chance is divided by one plus the points they shun around them, each
# times its weight, so heavy works stay out of towns.
DRAWN_BY_KIND: Mapping[BuildingKind, Mapping[Channel, float]] = MappingProxyType(
    {
        BuildingKind.REFINERY: {Points.ORE: 1.0, Points.TRADE: 2.0},
        BuildingKind.FACTORY: {Points.POPULATION: 1.0, Points.INDUSTRY: 1.5},
        BuildingKind.BAKERY: {Points.POPULATION: 1.0},
        BuildingKind.MEAT_FACTORY: {
            Points.POPULATION: 1.0,
            **dict.fromkeys(ANIMAL_FARMS, ANIMALS_DRAW),
        },
        BuildingKind.SAUSAGE_MAKER: {Points.POPULATION: 1.0},
        BuildingKind.CHEMICAL_WORKS: {Points.INDUSTRY: 1.5},
        BuildingKind.CEMENT_WORKS: {Points.INDUSTRY: 1.5},
        BuildingKind.GASWORKS: {Points.INDUSTRY: 1.0},
        BuildingKind.DAIRY: {Points.POPULATION: 2.0, Points.INDUSTRY: 1.0},
        BuildingKind.FARM: {Points.FOOD: 1.5, Points.POPULATION: 1.0},
        BuildingKind.FISHERY: {Points.POPULATION: 2.0, Points.FOOD: 0.5},
    }
)


SHUNNED_BY: Mapping[BuildingKind, Mapping[Points, float]] = MappingProxyType(
    {
        kind: {Points.POPULATION: 1.0}
        for kind in (
            BuildingKind.STEEL_MILL,
            BuildingKind.CHEMICAL_WORKS,
            BuildingKind.CEMENT_WORKS,
            BuildingKind.GASWORKS,
        )
    }
)


@dataclass(frozen=True, slots=True)
class Breeding:
    """How a ladder's buildings come up: its base chance, the kind of points its buildings give,
    which points around draw them and how much, and what each served point adds per doubling of
    those."""

    spawn: float
    own: Points
    drawn_by: Mapping[Channel, float]
    per_doubling: float
    # The points nearby that its whole chance is divided by, one plus each times its weight.
    shunned_by: Mapping[Channel, float] = field(default_factory=dict)
    # The served points in a cell that make it breed, each times its weight; its own kind of
    # points when not given.
    served_by: Mapping[Channel, float] = field(default_factory=dict)


_BREEDINGS: dict[tuple[int, int], tuple[Ladder, Mapping, Breeding]] = {}


def breeding_of(ladder: Ladder, points: Mapping[BuildingKind, tuple[Points, int]]) -> Breeding:
    """How a ladder's buildings come up, worked out once for each ladder and points."""
    kept = _BREEDINGS.get((id(ladder), id(points)))
    if kept is not None and kept[0] is ladder and kept[1] is points:
        return kept[2]
    own = points[ladder.kinds[0]][0]
    kind = ladder.kinds[0]
    spawn = CHANCE_UNIT * ladder.base
    drawn_by: dict[Channel, float] = dict(DRAWN_BY_KIND.get(kind) or DRAWN_BY.get(own, {own: 1.0}))
    served_by: dict[Channel, float] = {own: 1.0}
    industry = drawn_by.get(Points.INDUSTRY)
    if industry and own != Points.POPULATION:
        drawn_by[kind] = OWN_KIND_TIMES * industry
        drawn_by[Points.INDUSTRY] = CATEGORY_SHARE * industry
        served_by = {kind: OWN_KIND_TIMES, own: CATEGORY_SHARE}
    shunned_by = SHUNNED_BY.get(kind, {})
    pull = (
        BREEDING_TIMES * spawn if ladder.per_doubling is None else CHANCE_UNIT * ladder.per_doubling
    )
    breeding = Breeding(spawn, own, drawn_by, pull, shunned_by, served_by)
    _BREEDINGS[(id(ladder), id(points))] = (ladder, points, breeding)
    return breeding


def kind_channels(rule: GrowthRule) -> dict[BuildingKind, BuildingKind]:
    """Every kind on a ladder whose points some ladder's chances follow apart from the rest of
    their kind of points, its own or another's (DRAWN_BY_KIND), to the first kind of that
    ladder; a kind followed whose ladder does not grow here, to itself."""
    channels: dict[BuildingKind, BuildingKind] = {}
    breedings = [breeding_of(ladder, rule.points) for ladder in rule.ladders]
    followed = {
        channel
        for breeding in breedings
        for channel in (*breeding.drawn_by, *breeding.served_by)
        if isinstance(channel, BuildingKind)
    }
    for ladder in rule.ladders:
        first = ladder.kinds[0]
        if first in followed:
            for kind in ladder.kinds:
                channels[kind] = first
    for kind in followed:
        channels.setdefault(kind, kind)
    return channels


# The homes around a tavern, club, sushi hut or filling station that gets all it needs grow this
# much faster, and those around one that gets some of it that share as much faster.
LIVELY_GROWTH = 0.5


VENUES = (
    BuildingKind.TAVERN,
    BuildingKind.CLUB,
    BuildingKind.SUSHI_HUT,
    BuildingKind.FILLING_STATION,
)
