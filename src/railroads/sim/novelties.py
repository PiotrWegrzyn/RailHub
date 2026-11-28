"""Novelties: one-off buildings that fill a whole hex, want and send only goods the game already
has, and come up in some games and not in others.

When a game starts its seed picks NOVELTIES_PER_GAME of them, more on a place with more land
(REFERENCE_LAND), counting the whole place with the lots for sale, not only the land a game starts
on. Nothing is worked out cell by cell: once a year each picked novelty whose era
has come rolls YEARLY_CHANCE, and only on a hit does it look for a hex by its own rules, building
nothing that year if none fits. Olympic stadiums are apart from the pick: in every real Summer
Games year a map hosts one with OLYMPIC_CHANCE, scaled by its land, in its biggest town without a
stadium yet. Railway works are apart from the pick too: from RAILWAY_WORKS_YEAR a map gets a
few of them, RAILWAY_WORKS_PER_GAME scaled by its land, one a year at most with YEARLY_CHANCE,
each by the biggest town no other stands near.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from railroads.sim.constructions import Building, BuildingKind
from railroads.sim.grids import BoolGrid, Cell
from railroads.sim.growth import Growth
from railroads.sim.ladders import DEPARTMENT_STORES, HOTELS
from railroads.sim.move import Move
from railroads.sim.points import Points
from railroads.sim.tilemap import REFERENCE_LAND, Deposit, Relief, Terrain, TileMap

NOVELTIES_PER_GAME = (2, 3)
RAILWAY_WORKS_PER_GAME = (2, 3)
RAILWAY_WORKS_YEAR = 1855
YEARLY_CHANCE = 0.15
OLYMPIC_CHANCE = 0.05
MOST_OLYMPIC_CHANCE = 0.5
# The real Summer Games, without the cancelled ones of 1916, 1940 and 1944 and with Tokyo's a year
# late; from FUTURE_OLYMPICS on, every four years for ever (olympic_year).
PAST_OLYMPIC_YEARS = frozenset(
    [1896, 1900, 1904, 1908, 1912, *range(1920, 1937, 4), *range(1948, 2017, 4), 2021]
)
FUTURE_OLYMPICS = 2024


def olympic_year(year: int) -> bool:
    """Whether Summer Games are held in a year."""
    if year >= FUTURE_OLYMPICS:
        return (year - FUTURE_OLYMPICS) % 4 == 0
    return year in PAST_OLYMPIC_YEARS


# How far from its town a novelty looks for a free hex, and how far apart host towns must be.
TOWN_STEPS = 8
HOSTS_APART = 10
# How much likelier the space center is near military ground, and how near that is.
NEAR_MILITARY = 20.0
MILITARY_STEPS = 10
SPACE_CENTER_TRIES = 200
# A world's fair, an amusement park or a gigafactory wants a town big enough for a hotel, a film
# studio one big enough for a department store.
FAIR_TOWN = HOTELS.needs[Points.POPULATION]
FILM_TOWN = DEPARTMENT_STORES.needs[Points.POPULATION]

Placer = Callable[[Growth, BuildingKind, np.random.Generator], Cell | None]


@dataclass(frozen=True, slots=True)
class Novelty:
    kind: BuildingKind
    from_year: int
    place: Placer
    # Where it goes up, as the player reads it when no place fits.
    where: str
    until_year: int = 9999


@dataclass(frozen=True, slots=True)
class NoveltyOdds:
    """A one-off building this game may get: its chance in a year it may come up, when that is,
    and why none can come up now if so."""

    kind: BuildingKind
    chance: float
    when: str
    why_none: str | None = None


def _free(growth: Growth, kind: BuildingKind, cell: Cell) -> bool:
    return growth.constructions.why_not(Building(cell, kind)) is None


def _town(growth: Growth, least: int = 1) -> Cell | None:
    """The cell with the most people around it, if that many."""
    near = growth.nearby(Points.POPULATION)
    cell = _highest(near)
    return cell if near[cell] >= least else None


def _highest(values: npt.NDArray[Any]) -> Cell:
    """The cell of the highest value, the first of them if several."""
    best = np.unravel_index(int(np.argmax(values)), values.shape)
    return int(best[0]), int(best[1])


def by_biggest_town(
    growth: Growth, kind: BuildingKind, away_from: list[Cell]
) -> tuple[Cell, Cell] | None:
    """Builds the kind as near as it can to the biggest town more than HOSTS_APART steps from
    each of `away_from`; the town and the building's cell, or None if there was none."""
    grid = growth.constructions.tile_map.grid
    near = growth.nearby(Points.POPULATION).astype(np.float64)
    for spot in away_from:
        for cell in grid.within(spot, HOSTS_APART):
            near[cell] = -1
    town = _highest(near)
    if near[town] <= 0:
        return None
    cell = growth.constructions.nearest_free(kind, town, TOWN_STEPS)
    if cell is None:
        return None
    growth.build(Building(cell, kind))
    return town, cell


def _pick(
    growth: Growth, kind: BuildingKind, weights: npt.NDArray[np.float64], rng: np.random.Generator
) -> Cell | None:
    """A free cell drawn by weight, trying the likeliest draws first."""
    flat = np.flatnonzero(weights > 0)
    if not flat.size:
        return None
    keys = rng.random(flat.size) ** (1.0 / weights[flat])
    cols = growth.constructions.tile_map.grid.cols
    for index in flat[np.argsort(-keys)][:SPACE_CENTER_TRIES].tolist():
        cell = divmod(index, cols)
        if _free(growth, kind, cell):
            return cell
    return None


def _spread(tile_map: TileMap, mask: BoolGrid, steps: int) -> BoolGrid:
    """The cells at most `steps` steps from any cell of the mask."""
    rows, cols = tile_map.grid.neighbour_index_grids()
    for _ in range(steps):
        mask = mask | mask[rows, cols].any(axis=0)
    return mask


def _land(tile_map: TileMap) -> BoolGrid:
    return tile_map.terrain == Terrain.LAND


def place_space_center(growth: Growth, kind: BuildingKind, rng: np.random.Generator) -> Cell | None:
    """Anywhere on flat land, much likelier near military ground."""
    tile_map = growth.constructions.tile_map
    flat_land = _land(tile_map) & (tile_map.relief == Relief.FLAT)
    deposits = tile_map.deposits
    military = np.zeros(tile_map.grid.shape, dtype=bool)
    if deposits is not None:
        military = deposits == Deposit.GARRISON
    near = _spread(tile_map, military, MILITARY_STEPS)
    weights = np.where(flat_land, np.where(near, NEAR_MILITARY, 1.0), 0.0)
    return _pick(growth, kind, weights.ravel(), rng)


def place_pear_hq(growth: Growth, kind: BuildingKind, rng: np.random.Generator) -> Cell | None:
    """Only once a skyscraper stands: as near as it can to the one with most people around."""
    towers = growth.count(BuildingKind.SKYSCRAPER) > 0
    if not towers.any():
        return None
    near = np.where(towers, growth.nearby(Points.POPULATION), -1)
    return growth.constructions.nearest_free(kind, _highest(near), TOWN_STEPS)


def place_royal_palace(growth: Growth, kind: BuildingKind, rng: np.random.Generator) -> Cell | None:
    """By the biggest town, best on a hill or a river."""
    town = _town(growth)
    if town is None:
        return None
    tile_map = growth.constructions.tile_map
    grid = tile_map.grid
    weights = np.zeros(grid.rows * grid.cols)
    by_river = _spread(tile_map, tile_map.terrain == Terrain.RIVER, 1)
    for cell, steps in grid.within(town, TOWN_STEPS).items():
        hill = 3.0 if tile_map.relief[cell] == Relief.HILLS else 1.0
        river = 3.0 if by_river[cell] else 1.0
        weights[np.ravel_multi_index(cell, grid.shape)] = hill * river / (1 + steps)
    return _pick(growth, kind, weights, rng)


def place_salt_cathedral(
    growth: Growth, kind: BuildingKind, rng: np.random.Generator
) -> Cell | None:
    """On a salt deposit."""
    deposits = growth.constructions.tile_map.deposits
    if deposits is None:
        return None
    return _pick(growth, kind, (deposits == Deposit.SALT).ravel().astype(float), rng)


def place_spa(growth: Growth, kind: BuildingKind, rng: np.random.Generator) -> Cell | None:
    """In hills beside a river or a lake."""
    tile_map = growth.constructions.tile_map
    water = np.isin(tile_map.terrain, [Terrain.RIVER, Terrain.LAKE])
    hills = _land(tile_map) & (tile_map.relief == Relief.HILLS)
    return _pick(growth, kind, (hills & _spread(tile_map, water, 1)).ravel().astype(float), rng)


@dataclass(frozen=True)
class ByTown:
    """As near as it can to the biggest town, once it has `least` people around."""

    least: int

    def __call__(self, growth: Growth, kind: BuildingKind, rng: np.random.Generator) -> Cell | None:
        town = _town(growth, self.least)
        return None if town is None else growth.constructions.nearest_free(kind, town, TOWN_STEPS)


def _by_town(least: int) -> str:
    return f"by a town of at least {least} population points around"


NOVELTIES = (
    Novelty(BuildingKind.ROYAL_PALACE, 1850, place_royal_palace, "by the biggest town"),
    Novelty(BuildingKind.SALT_CATHEDRAL, 1850, place_salt_cathedral, "on a salt deposit"),
    Novelty(BuildingKind.SPA_RESORT, 1850, place_spa, "in hills beside a river or a lake"),
    Novelty(
        BuildingKind.WORLDS_FAIR, 1880, ByTown(FAIR_TOWN), _by_town(FAIR_TOWN), until_year=1940
    ),
    Novelty(BuildingKind.FILM_STUDIO, 1920, ByTown(FILM_TOWN), _by_town(FILM_TOWN)),
    Novelty(BuildingKind.AMUSEMENT_PARK, 1950, ByTown(FAIR_TOWN), _by_town(FAIR_TOWN)),
    Novelty(BuildingKind.SPACE_CENTER, 1960, place_space_center, "on flat land"),
    Novelty(BuildingKind.PEAR_HQ, 1980, place_pear_hq, "by a skyscraper"),
    Novelty(BuildingKind.GIGAFACTORY, 2030, ByTown(FAIR_TOWN), _by_town(FAIR_TOWN)),
)


@dataclass(slots=True)
class Novelties:
    """The novelties one game may get, and the Olympic stadiums it hosts."""

    picked: list[Novelty]
    olympic_chance: float
    built: set[BuildingKind] = field(default_factory=set)
    hosts: list[Cell] = field(default_factory=list)
    # How many railway works the map gets, and where those put up so far stand. Saves from
    # before them get none.
    railway_works: int = 0
    works: list[Cell] = field(default_factory=list)

    @classmethod
    def pick(cls, place: TileMap, rng: np.random.Generator) -> Novelties:
        """Picks this game's novelties, more of them on a place with more land: the map of the
        whole place, lots for sale included."""
        land = int(_land(place).sum()) / REFERENCE_LAND
        low, high = NOVELTIES_PER_GAME
        count = min(len(NOVELTIES), max(1, round(int(rng.integers(low, high + 1)) * land)))
        chosen = rng.choice(len(NOVELTIES), size=count, replace=False)
        olympic = min(MOST_OLYMPIC_CHANCE, OLYMPIC_CHANCE * land)
        low, high = RAILWAY_WORKS_PER_GAME
        works = max(1, round(int(rng.integers(low, high + 1)) * land))
        return cls([NOVELTIES[i] for i in sorted(chosen.tolist())], olympic, railway_works=works)

    def year(self, growth: Growth, rng: np.random.Generator) -> set[Cell]:
        """Rolls once for every novelty whose era has come and, on a hit, puts it up where its
        rules say; returns the cells that changed."""
        changed: set[Cell] = set()
        year = growth.year
        for novelty in self.picked:
            if novelty.kind in self.built or not novelty.from_year <= year <= novelty.until_year:
                continue
            if rng.random() >= YEARLY_CHANCE:
                continue
            cell = novelty.place(growth, novelty.kind, rng)
            if cell is not None:
                growth.build(Building(cell, novelty.kind))
                self.built.add(novelty.kind)
                changed.add(cell)
        if olympic_year(year) and rng.random() < self.olympic_chance:
            changed |= self._host(growth)
        works = getattr(self, "works", [])
        wanted = getattr(self, "railway_works", 0)
        if year >= RAILWAY_WORKS_YEAR and len(works) < wanted and rng.random() < YEARLY_CHANCE:
            changed |= self._railway_works(growth, works)
        return changed

    def odds(self, growth: Growth) -> list[NoveltyOdds]:
        """Each one-off building this game may get, with its chance and why none comes up now."""
        year = growth.year
        found = []
        for novelty in self.picked:
            when = f"a year from {novelty.from_year}"
            if novelty.until_year < 9999:
                when += f" to {novelty.until_year}"
            why = None
            if novelty.kind in self.built:
                why = "Built: one a game."
            elif year < novelty.from_year:
                why = f"Not before {novelty.from_year}."
            elif year > novelty.until_year:
                why = f"Only until {novelty.until_year}."
            elif novelty.place(growth, novelty.kind, np.random.default_rng(0)) is None:
                why = f"No place fits now: it goes up {novelty.where}"
                if isinstance(novelty.place, ByTown):
                    biggest = int(growth.nearby(Points.POPULATION).max())
                    why += f", and the biggest town has {biggest}"
                why += "."
            found.append(NoveltyOdds(novelty.kind, YEARLY_CHANCE, when, why))
        games = next(y for y in range(year, year + 5) if olympic_year(y))
        found.append(
            NoveltyOdds(
                BuildingKind.OLYMPIC_STADIUM,
                self.olympic_chance,
                f"in each Summer Games year, next in {games}, in the biggest town without one",
            )
        )
        wanted = getattr(self, "railway_works", 0)
        works = len(getattr(self, "works", []))
        why = None
        if works >= wanted:
            why = f"All {wanted} this map gets stand."
        elif year < RAILWAY_WORKS_YEAR:
            why = f"Not before {RAILWAY_WORKS_YEAR}."
        found.append(
            NoveltyOdds(
                BuildingKind.RAILWAY_WORKS,
                YEARLY_CHANCE,
                f"a year, {works} of {wanted} built, by the biggest town with none near",
                why,
            )
        )
        return found

    def _railway_works(self, growth: Growth, works: list[Cell]) -> set[Cell]:
        """Puts railway works by the biggest town with none near it yet."""
        built = by_biggest_town(growth, BuildingKind.RAILWAY_WORKS, works)
        if built is None:
            return set()
        works.append(built[1])
        self.works = works
        return {built[1]}

    def _host(self, growth: Growth) -> set[Cell]:
        """Puts a stadium in the biggest town far enough from every earlier host."""
        built = by_biggest_town(growth, BuildingKind.OLYMPIC_STADIUM, self.hosts)
        if built is None:
            return set()
        self.hosts.append(built[0])
        return {built[1]}

    def move(self, move: Move) -> None:
        """Moves the hosts and railway works onto the moved map."""
        self.hosts = [move.cell(cell) for cell in self.hosts]
        self.works = [move.cell(cell) for cell in self.works]
