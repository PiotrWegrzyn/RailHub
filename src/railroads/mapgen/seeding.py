"""Seeds a new map with its towns and industry by letting it grow (railroads.sim.growth) for the
years before play: houses, farms, orchards, mines, lumber mills, ports and factories all come up
where chance, the land and their neighbours take them.

Until players can build them, stations are put down by the seeding too: STATIONS_AFTER_MONTHS
into the growth, a few go up at random beside the bigger groups of buildings, spaced apart, and
the growth carries on around them. Track is left for the player.
"""

import math
from dataclasses import replace

import numpy as np

from railroads.sim.constructions import Constructions, Station
from railroads.sim.growth import GROWTH, MONTHS_BEFORE_PLAY, Growth, GrowthRule, Points
from railroads.sim.tilemap import Terrain, TileMap

STATIONS_AFTER_MONTHS = 100 * 12
# Before play, goods are shared out and forests change once a year instead of every month, to
# seed quickly.
TRADE_EVERY_MONTHS = 12
FORESTS_EVERY_MONTHS = 12
# One station for this many land cells, beside at least STATION_PEOPLE population points.
LAND_CELLS_PER_STATION = 3000
STATION_PEOPLE = 3
STATION_SPACING_CELLS = 20.0


def seed_buildings(
    tile_map: TileMap,
    rng: np.random.Generator,
    growth: GrowthRule = GROWTH,
    months: int = MONTHS_BEFORE_PLAY,
    stations_after: int | None = STATIONS_AFTER_MONTHS,
) -> Constructions:
    """Grows the map's buildings for `months` on a copy of the map, whose forests change as
    they grow; the copy is the constructions' tile_map."""
    tile_map = replace(tile_map, vegetation=tile_map.vegetation.copy())
    constructions = Constructions(tile_map)
    rule = replace(
        growth,
        trade_every_months=TRADE_EVERY_MONTHS,
        forests_every_months=FORESTS_EVERY_MONTHS,
    )
    grower = Growth(constructions, rule)
    if stations_after is None or months <= stations_after:
        grower.run(months, rng)
        return constructions
    grower.run(stations_after, rng)
    place_stations(grower, rng)
    grower.run(months - stations_after, rng)
    return constructions


def place_stations(growth: Growth, rng: np.random.Generator) -> list[tuple[int, int]]:
    """Puts stations on empty cells beside the bigger groups of buildings, drawn at random with
    the most people around likeliest, and spaced apart; returns where they went."""
    constructions = growth.constructions
    tile_map = constructions.tile_map
    grid = tile_map.grid
    wanted = round(int((tile_map.terrain == Terrain.LAND).sum()) / LAND_CELLS_PER_STATION)
    people = growth.nearby(Points.POPULATION).ravel()
    candidates = np.flatnonzero(people >= STATION_PEOPLE)
    keys = np.log(rng.random(candidates.size)) / people[candidates] ** 2
    step = math.dist(grid.center(*grid.neighbours(0, 0)[0]), grid.center(0, 0))
    placed: list[tuple[int, int]] = []
    for index in candidates[np.argsort(-keys)].tolist():
        if len(placed) >= wanted:
            break
        cell = divmod(index, grid.cols)
        sx, sy = grid.center(*cell)
        if any(
            math.dist((sx, sy), grid.center(*other)) / step < STATION_SPACING_CELLS
            for other in placed
        ):
            continue
        if constructions.why_not(Station(cell)) is not None:
            continue
        constructions.build(Station(cell))
        placed.append(cell)
    return placed
