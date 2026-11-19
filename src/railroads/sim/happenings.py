"""What history (railroads.sim.events) does to one game's map, and the news the player gets of it:
a headline for every event as its month comes, for every decade people want more in, and for
every good that is first made (GOOD_YEARS).

The year before a war NEW_GARRISONS patches of military ground of GARRISON_SIZE cells go up, at
least GARRISONS_APART steps from any other, each with a base and two outposts; as it begins
GUN_FACTORIES gun factories open by the biggest towns, and GUN_FACTORIES more on a map with no
port; as it ends WAR_LOSSES of the military buildings are pulled down. Falling towers pull down
skyscrapers, those with the most people around first. An event bringing pilgrims picks the town
of St Paul's as its news comes, and the railway pays them there (Railway.pilgrims).
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from railroads.sim.clock import MONTH_NAMES
from railroads.sim.constructions import Building, BuildingKind
from railroads.sim.events import (
    EVENTS,
    MAKE_RISE,
    PILGRIM_MONTHS,
    RISE,
    RISE_EVERY,
    RISE_FROM,
    SLOW_MAKE_RISE,
    SLOW_RISE,
    WAR_GROWTH,
    Event,
    price_fall,
    rise_of,
)
from railroads.sim.goods import GOOD_YEARS, Need
from railroads.sim.grids import BoolGrid, Cell, Grid
from railroads.sim.ladders import MILITARY, Points
from railroads.sim.tilemap import Deposit, Relief, Terrain

if TYPE_CHECKING:
    from railroads.sim.growth import Growth

NEW_GARRISONS = 2
GARRISON_SIZE = (10, 18)
GARRISONS_APART = 15
GUN_FACTORIES = 2
WAR_LOSSES = 0.5


@dataclass(frozen=True, slots=True)
class Headline:
    date: str
    title: str
    story: str
    # What it changes in the game, a line each.
    changes: tuple[str, ...] = ()
    # Whether it stops the game until the player has read it.
    big: bool = False


def _percent(factor: float) -> str:
    return f"{(factor - 1) * 100:+.0f}%"


def changes_of(event: Event) -> tuple[str, ...]:
    """What an event changes in the game, a line each."""
    who = "Homes want" if event.homes_only else "Buildings want"
    lines = [
        f"{who} no {need.value}" if factor == 0 else f"{who} {_percent(factor)} {need.value}"
        for need, factor in event.wants.items()
        if factor != 1
    ]
    if event.rest != 1:
        lines.append(f"{who} {_percent(event.rest)} of everything else")
    lines += [
        f"Each {kind.value} wants {_percent(factor)} {need.value}"
        for (kind, need), factor in event.kind_wants.items()
    ]
    lines += [
        f"Each {kind.value} takes {need.value}, and makes more with them"
        for kind, need in event.adds
    ]
    lines += [f"{good.value.capitalize()} made {_percent(f)}" for good, f in event.makes.items()]
    lines += [f"{good.value.capitalize()} earns {_percent(f)}" for good, f in event.prices.items()]
    lines += [f"{good.value.capitalize()} {_percent(f)}" for good, f in event.travels.items()]
    if event.electricity_use != 1:
        lines.append(f"Electricity wanted {_percent(event.electricity_use)}")
    if event.electricity_price != 1:
        lines.append(f"Electricity costs {_percent(event.electricity_price)}")
    if event.pilgrims != 1:
        lines.append(
            f"Passengers getting off in the town of St Paul's earn {_percent(event.pilgrims)} "
            f"for {PILGRIM_MONTHS} months"
        )
    if event.war:
        lines += [
            f"The military grows {WAR_GROWTH} times as fast",
            f"{GUN_FACTORIES} gun factories open, {GUN_FACTORIES} more where there is no port",
            f"When it ends, {WAR_LOSSES:.0%} of the military buildings are pulled down",
        ]
    if event.towers_fall:
        lines.append(f"{event.towers_fall} skyscrapers are pulled down")
    if event.years:
        lines.append(f"For {event.years} years" if event.years > 1 else "For a year")
    return tuple(lines)


@dataclass(slots=True)
class Happenings:
    """What history did to this game, and the headlines not yet taken by the game."""

    gun_factories: list[Cell] = field(default_factory=list)
    # The town pilgrims earn more in, how many times as much, and for how many months more.
    pilgrim_town: int = 0
    pilgrim_pay: float = 1.0
    pilgrim_months: int = 0
    news: list[Headline] = field(default_factory=list)

    def take(self) -> list[Headline]:
        news, self.news = self.news, []
        return news

    def months(self, growth: Growth, first: int, last: int, rng: np.random.Generator) -> set[Cell]:
        """Lets history happen in the months from `first` to `last` since play began, as many
        as went by at once; returns the cells that changed."""
        changed: set[Cell] = set()
        for months in range(first, last + 1):
            changed |= self._month(growth, growth.start_year + months // 12, months % 12, rng)
        return changed

    def _month(self, growth: Growth, year: int, month: int, rng: np.random.Generator) -> set[Cell]:
        changed: set[Cell] = set()
        if self.pilgrim_months:
            self.pilgrim_months -= 1
            if not self.pilgrim_months:
                self.pilgrim_town, self.pilgrim_pay = 0, 1.0
        date = f"{MONTH_NAMES[month]} {year}"
        if month == 0:
            changed |= self._new_year(growth, year, date, rng)
        east = growth.economy.east
        for event in EVENTS:
            if event.year == year and event.month == month and (east or not event.east_only):
                changed |= self._begin(growth, event, date)
        return changed

    def _new_year(
        self, growth: Growth, year: int, date: str, rng: np.random.Generator
    ) -> set[Cell]:
        changed: set[Cell] = set()
        for event in EVENTS:
            if event.war and event.year == year + 1:
                changed |= new_garrisons(growth, rng)
            if event.war and event.year + event.years == year:
                gone = disband(growth, rng)
                changed |= gone
                self.news.append(
                    Headline(
                        date,
                        "Peace",
                        f"{event.title} is over, and the soldiers go home.",
                        (f"{len(gone)} military buildings pulled down",),
                    )
                )
        if year > RISE_FROM and (year - RISE_FROM) % RISE_EVERY == 0:
            goods = rise_of(BuildingKind.HOUSE, Need.COMMON, year)
            food = rise_of(BuildingKind.HOUSE, Need.FOOD, year)
            self.news.append(
                Headline(
                    date,
                    "People want more",
                    f"A new decade: homes, shops and everything else people use want {RISE:.0%} "
                    f"more of every good than ten years ago, and {SLOW_RISE:.0%} more food and "
                    f"heating. Farms, mines and the rest make {MAKE_RISE:.0%} more, "
                    f"{SLOW_MAKE_RISE:.0%} more food and heating, and every unit carried earns "
                    "as much less as people want more of it.",
                    (
                        f"Goods wanted: {goods:.2f} times as much as in {RISE_FROM}",
                        f"Food and heating wanted: {food:.2f} times as much as in {RISE_FROM}",
                        f"A unit of goods earns {price_fall(year):.0%} of its {RISE_FROM} price",
                        f"Food and heating: {price_fall(year, slow=True):.0%} of it",
                    ),
                )
            )
        for good, first in GOOD_YEARS.items():
            if first == year:
                self.news.append(
                    Headline(
                        date,
                        f"New: {good.value}",
                        f"{good.value.capitalize()} is made for the first time this year, by "
                        "buildings that go up from now on.",
                    )
                )
        return changed

    def _begin(self, growth: Growth, event: Event, date: str) -> set[Cell]:
        changed: set[Cell] = set()
        if event.war:
            changed |= self._open_gun_factories(growth)
        if event.towers_fall:
            changed |= fell_towers(growth, event.towers_fall)
        if event.pilgrims != 1:
            self.pilgrim_town = st_pauls_town(growth)
            self.pilgrim_pay = event.pilgrims
            self.pilgrim_months = PILGRIM_MONTHS
        self.news.append(Headline(date, event.title, event.story, changes_of(event), event.big))
        return changed

    def _open_gun_factories(self, growth: Growth) -> set[Cell]:
        from railroads.sim.novelties import by_biggest_town

        ports = bool(growth.count(BuildingKind.PORT).any())
        changed: set[Cell] = set()
        for _ in range(GUN_FACTORIES if ports else 2 * GUN_FACTORIES):
            built = by_biggest_town(growth, BuildingKind.GUN_FACTORY, self.gun_factories)
            if built is None:
                break
            self.gun_factories.append(built[0])
            changed.add(built[1])
        return changed


def _spread(grid: Grid, mask: BoolGrid, steps: int) -> BoolGrid:
    rows, cols = grid.neighbour_index_grids()
    for _ in range(steps):
        mask = mask | mask[rows, cols].any(axis=0)
    return mask


def _patch(
    grid: Grid, ground: BoolGrid, start: Cell, size: int, rng: np.random.Generator
) -> list[Cell]:
    """Up to `size` cells of the ground grown out from `start`."""
    patch, edge, seen = [start], [start], {start}
    while edge and len(patch) < size:
        here = edge.pop(int(rng.integers(len(edge))))
        for near in grid.neighbours(*here):
            if near not in seen and ground[near] and len(patch) < size:
                seen.add(near)
                patch.append(near)
                edge.append(near)
    return patch


def new_garrisons(growth: Growth, rng: np.random.Generator) -> set[Cell]:
    """Lays NEW_GARRISONS patches of military ground on open land far from the others, and
    musters a base and two outposts on each; returns the cells built on."""
    tile_map = growth.constructions.tile_map
    deposits = tile_map.deposits
    if deposits is None:
        return set()
    grid = tile_map.grid
    ground = (
        (tile_map.terrain == Terrain.LAND)
        & (tile_map.relief != Relief.MOUNTAINS)
        & (deposits == Deposit.NONE)
    )
    if tile_map.locked is not None:
        ground &= ~tile_map.locked
    changed: set[Cell] = set()
    for _ in range(NEW_GARRISONS):
        near = _spread(grid, deposits == Deposit.GARRISON, GARRISONS_APART)
        starts = np.argwhere(ground & ~near)
        if not len(starts):
            break
        row, col = starts[int(rng.integers(len(starts)))].tolist()
        low, high = GARRISON_SIZE
        patch = _patch(grid, ground, (row, col), int(rng.integers(low, high + 1)), rng)
        for cell in patch:
            deposits[cell] = Deposit.GARRISON
            ground[cell] = False
        flat = np.ravel_multi_index(tuple(np.array(patch).T), grid.shape)
        growth.reweigh_deposits(np.sort(np.asarray(flat, dtype=np.intp)))
        changed |= _muster(growth, patch, rng)
    return changed


def _muster(growth: Growth, patch: list[Cell], rng: np.random.Generator) -> set[Cell]:
    changed: set[Cell] = set()
    for kind in (BuildingKind.BASE, BuildingKind.OUTPOST, BuildingKind.OUTPOST):
        for index in rng.permutation(len(patch)).tolist():
            building = Building(patch[index], kind)
            if growth.constructions.why_not(building) is None:
                growth.build(building)
                changed.add(building.cell)
                break
    return changed


def disband(growth: Growth, rng: np.random.Generator) -> set[Cell]:
    """Pulls down WAR_LOSSES of the military buildings, drawn at random; returns their cells."""
    military = [
        built
        for built in growth.constructions
        if isinstance(built, Building) and built.kind in MILITARY.kinds
    ]
    gone = rng.choice(len(military), size=int(len(military) * WAR_LOSSES), replace=False)
    for index in sorted(gone.tolist()):
        growth.take_down(military[index])
    return {military[index].cell for index in gone.tolist()}


def fell_towers(growth: Growth, count: int) -> set[Cell]:
    """Pulls down this many skyscrapers, those with the most people around first."""
    near = growth.nearby(Points.POPULATION)
    towers = sorted(
        (
            built
            for built in growth.constructions
            if isinstance(built, Building) and built.kind == BuildingKind.SKYSCRAPER
        ),
        key=lambda built: -int(near[built.cell]),
    )[:count]
    for tower in towers:
        growth.take_down(tower)
    return {tower.cell for tower in towers}


def st_pauls_town(growth: Growth) -> int:
    """The town St Paul's stands in, 0 if it stands nowhere."""
    cathedral = growth.count(BuildingKind.ST_PAULS) > 0
    if not cathedral.any():
        return 0
    return growth.station_trade.town_near(divmod(int(np.argmax(cathedral)), growth.grid.cols))
