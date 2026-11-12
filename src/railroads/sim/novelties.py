"""Novelties: one-off buildings that fill a whole hex, want and send only goods the game already
has, and come up in some games and not in others.

When a game starts its seed picks NOVELTIES_PER_GAME of them, more on a bigger map
(REFERENCE_LAND). Nothing is worked out cell by cell: once a year each picked novelty whose era
has come rolls YEARLY_CHANCE, and only on a hit does it look for a hex by its own rules, building
nothing that year if none fits. Olympic stadiums are apart from the pick: in every real Summer
Games year a map hosts one with OLYMPIC_CHANCE, scaled by its land, in its biggest town without a
stadium yet.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from railroads.sim.constructions import Building, BuildingKind
from railroads.sim.grids import Cell
from railroads.sim.growth import Growth
from railroads.sim.ladders import Points
from railroads.sim.tilemap import Deposit, Relief, Terrain, TileMap

REFERENCE_LAND = 121_000
NOVELTIES_PER_GAME = (2, 3)
YEARLY_CHANCE = 0.15
OLYMPIC_CHANCE = 0.05
MOST_OLYMPIC_CHANCE = 0.5
# The real Summer Games, without the cancelled ones of 1916, 1940 and 1944 and with Tokyo's a year
# late, then every four years to the end of play.
OLYMPIC_YEARS = frozenset(
    [1896, 1900, 1904, 1908, 1912, *range(1920, 1937, 4), *range(1948, 2017, 4), 2021]
    + list(range(2024, 2050, 4))
)
# How far from its town a novelty looks for a free hex, and how far apart host towns must be.
TOWN_STEPS = 8
HOSTS_APART = 10
# How much likelier the space center is near military ground, and how near that is.
NEAR_MILITARY = 20.0
MILITARY_STEPS = 10
SPACE_CENTER_TRIES = 200
FILM_TOWN = 150
FAIR_TOWN = 100

BoolGrid = npt.NDArray[np.bool_]
Placer = Callable[[Growth, BuildingKind, np.random.Generator], Cell | None]


@dataclass(frozen=True, slots=True)
class Novelty:
    kind: BuildingKind
    from_year: int
    place: Placer
    until_year: int = 9999


def _free(growth: Growth, kind: BuildingKind, cell: Cell) -> bool:
    return growth.constructions.why_not(Building(cell, kind)) is None


def _town(growth: Growth, least: int = 1) -> Cell | None:
    """The cell with the most people around it, if that many."""
    near = growth.nearby(Points.POPULATION)
    best = np.unravel_index(int(np.argmax(near)), near.shape)
    cell = (int(best[0]), int(best[1]))
    return cell if near[cell] >= least else None


def _nearest_free(growth: Growth, kind: BuildingKind, centre: Cell) -> Cell | None:
    reached = growth.constructions.tile_map.grid.within(centre, TOWN_STEPS)
    for cell, _ in sorted(reached.items(), key=lambda item: item[1]):
        if _free(growth, kind, cell):
            return cell
    return None


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
    best = np.unravel_index(int(np.argmax(near)), near.shape)
    return _nearest_free(growth, kind, (int(best[0]), int(best[1])))


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
        return None if town is None else _nearest_free(growth, kind, town)


NOVELTIES = (
    Novelty(BuildingKind.ROYAL_PALACE, 1850, place_royal_palace),
    Novelty(BuildingKind.SALT_CATHEDRAL, 1850, place_salt_cathedral),
    Novelty(BuildingKind.SPA_RESORT, 1850, place_spa),
    Novelty(BuildingKind.WORLDS_FAIR, 1880, ByTown(FAIR_TOWN), until_year=1940),
    Novelty(BuildingKind.FILM_STUDIO, 1920, ByTown(FILM_TOWN)),
    Novelty(BuildingKind.AMUSEMENT_PARK, 1950, ByTown(FAIR_TOWN)),
    Novelty(BuildingKind.SPACE_CENTER, 1960, place_space_center),
    Novelty(BuildingKind.PEAR_HQ, 1980, place_pear_hq),
)


@dataclass(slots=True)
class Novelties:
    """The novelties one game may get, and the Olympic stadiums it hosts."""

    picked: list[Novelty]
    olympic_chance: float
    built: set[BuildingKind] = field(default_factory=set)
    hosts: list[Cell] = field(default_factory=list)

    @classmethod
    def pick(cls, tile_map: TileMap, rng: np.random.Generator) -> Novelties:
        """Picks this game's novelties, more of them on a map with more land."""
        land = int(_land(tile_map).sum()) / REFERENCE_LAND
        low, high = NOVELTIES_PER_GAME
        count = min(len(NOVELTIES), max(1, round(int(rng.integers(low, high + 1)) * land)))
        chosen = rng.choice(len(NOVELTIES), size=count, replace=False)
        olympic = min(MOST_OLYMPIC_CHANCE, OLYMPIC_CHANCE * land)
        return cls([NOVELTIES[i] for i in sorted(chosen.tolist())], olympic)

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
        if year in OLYMPIC_YEARS and rng.random() < self.olympic_chance:
            changed |= self._host(growth)
        return changed

    def _host(self, growth: Growth) -> set[Cell]:
        """Puts a stadium in the biggest town far enough from every earlier host."""
        grid = growth.constructions.tile_map.grid
        near = growth.nearby(Points.POPULATION).astype(np.float64)
        for host in self.hosts:
            for cell in grid.within(host, HOSTS_APART):
                near[cell] = -1
        best = np.unravel_index(int(np.argmax(near)), near.shape)
        town = (int(best[0]), int(best[1]))
        if near[town] <= 0:
            return set()
        cell = _nearest_free(growth, BuildingKind.OLYMPIC_STADIUM, town)
        if cell is None:
            return set()
        growth.build(Building(cell, BuildingKind.OLYMPIC_STADIUM))
        self.hosts.append(town)
        return {cell}
