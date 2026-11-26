"""What buildings make, need and are boosted by in a given year: the tables of railroads.sim.goods
as the supply and demand boards (railroads.sim.tiles) and history (railroads.sim.events) change
them."""

from collections.abc import Mapping
from types import MappingProxyType

from railroads.sim.constructions import PORT_KINDS, BuildingKind
from railroads.sim.events import added_needs, make_factor, want_factor
from railroads.sim.goods import (
    BONUS_NEEDS,
    BONUS_PART,
    HOME_DEMANDS,
    HOMES,
    HOUSE_DEMANDS,
    KIND_BONUSES,
    KIND_BOOST_MOST,
    KIND_GOOD_YEARS,
    LAND_BOOST_MOST,
    MAKES,
    NEEDS,
    OWNED_BOOSTS,
    RECIPES,
    VENUE_BONUSES,
    Good,
    Need,
    good_exists,
    need_exists,
    need_year,
)
from railroads.sim.tiles import opens
from railroads.sim.year_caches import year_cached


@year_cached
def makes_of(kind: BuildingKind, year: int) -> Mapping[Good, float]:
    """The most a kind of building makes of each good in a month in a given year: nothing of a
    good that does not exist yet, and less of one history makes scarce (railroads.sim.events)."""

    return MappingProxyType(
        {
            good: amount * make_factor(good, year)
            for good, amount in MAKES.get(kind, {}).items()
            if KIND_GOOD_YEARS.get((kind, good), 0) <= year and good_exists(good, year)
        }
    )


def need_open(kind: BuildingKind, need: Need, year: int) -> bool:
    """Whether a kind of building needs something in a year: once its year has come and, for a
    need a demand tile opens, once that tile is at its level (railroads.sim.tiles.UNLOCKS)."""

    return need_year(kind, need) <= year and need_exists(need, year) and opens(kind, need, year)


def owned_boosts(kind: BuildingKind, year: int) -> frozenset[Need]:
    """The boosts only the player's own buildings of a kind take that are open in a year."""
    return frozenset(
        need for taker, need in OWNED_BOOSTS if taker == kind and need_open(kind, need, year)
    )


@year_cached
def needs_of(kind: BuildingKind, year: int, east: bool = False) -> Mapping[Need, float]:
    """What a kind of building needs in a month in a given year, on a map in the old Eastern Bloc
    or not: what it needs once that has begun, and what it wants only while an event lasts, times
    what history makes of each (railroads.sim.events); nothing of what it no longer wants."""

    needs = {
        need: amount for need, amount in NEEDS.get(kind, {}).items() if need_open(kind, need, year)
    }
    needs.update(
        (need, amount) for need, amount in added_needs(kind, year).items() if need not in needs
    )
    scaled = (
        (need, amount * want_factor(kind, need, year, east)) for need, amount in needs.items()
    )
    return MappingProxyType({need: amount for need, amount in scaled if amount > 0})


@year_cached
def bonus_part(kind: BuildingKind, need: Need, year: int, east: bool = False) -> float | None:
    """How much of a need a kind of building takes only as a bonus in a year (BONUS_PART), grown as
    what it needs of it grows (railroads.sim.events); None if all of it must arrive."""

    part = BONUS_PART.get((kind, need))
    return None if part is None else part * want_factor(kind, need, year, east)


@year_cached
def boosts_of(kind: BuildingKind, year: int) -> Mapping[Need, float]:
    """What a kind of building that makes goods can do without but makes up to BOOST_MOST more
    with, and how much of each it needs in a month for all of that: a processor's recipe boosts,
    and the bonus goods of one that makes goods from the land, such as wheat for its animals."""
    recipe = RECIPES.get(kind)
    if recipe is not None:
        needs = needs_of(kind, year)
        return MappingProxyType({need: needs[need] for need in recipe.boosts if need in needs})
    if kind not in MAKES and kind not in PORT_KINDS:
        return MappingProxyType({})

    return MappingProxyType(
        {
            need: amount
            for need, amount in needs_of(kind, year).items()
            if need in BONUS_NEEDS
            or need in LAND_BOOST_MOST
            or (kind, need) in KIND_BOOST_MOST
            or (kind, need) in OWNED_BOOSTS
            or need in added_needs(kind, year)
        }
    )


@year_cached
def bonus_of(kind: BuildingKind, year: int, east: bool = False) -> Mapping[Need, float]:
    """The bonus goods among what a kind of building needs in a month in a given year: for homes
    all but food, common goods and heating; those of buildings that make goods boost what they make instead
    (boosts_of)."""
    needs = needs_of(kind, year, east)
    if kind in VENUE_BONUSES:
        return MappingProxyType(
            {need: amount for need, amount in needs.items() if need in VENUE_BONUSES[kind]}
        )
    if kind in HOMES:
        demands = HOUSE_DEMANDS if kind == BuildingKind.HOUSE else HOME_DEMANDS
        return MappingProxyType(
            {need: amount for need, amount in needs.items() if need not in demands}
        )
    extra = KIND_BONUSES.get(kind, frozenset())
    bonus = {need: amount for need, amount in needs.items() if need in BONUS_NEEDS or need in extra}
    if kind in MAKES or kind in PORT_KINDS or len(bonus) == len(needs):
        return MappingProxyType({})
    return MappingProxyType(bonus)


@year_cached
def demands_of(kind: BuildingKind, year: int, east: bool = False) -> Mapping[Need, float]:
    """What a kind of building must get in a month in a given year to be satisfied: its needs
    that are neither bonus goods nor boosts."""
    left_out = {**bonus_of(kind, year, east), **boosts_of(kind, year)}
    return MappingProxyType(
        {
            need: amount
            for need, amount in needs_of(kind, year, east).items()
            if need not in left_out
        }
    )
