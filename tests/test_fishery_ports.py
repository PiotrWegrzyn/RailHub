import numpy as np
import pytest
from test_growth import build, make_map

from railroads.sim.constructions import BuildingKind, Constructions
from railroads.sim.growth import Growth
from railroads.sim.ladders import (
    FISHERIES_PER_PORT,
    PORT_PEOPLE_PER_CELL,
    RESOURCES,
)
from railroads.sim.tilemap import Terrain, TileMap

FISHERIES = next(ladder for ladder in RESOURCES if ladder.kinds[0] == BuildingKind.FISHERY)
COAST = (5, 1)


def land_around(tile_map: TileMap, cell: tuple[int, int]) -> list[tuple[int, int]]:
    around = [cell, *tile_map.grid.neighbours(*cell)]
    return [near for near in around if tile_map.terrain[near] == Terrain.LAND]


def fishing_village(
    houses_per_cell: int,
    fisheries: int = FISHERIES_PER_PORT,
    port: tuple[int, int] | None = None,
) -> Growth:
    tile_map = make_map()
    constructions = Constructions(tile_map)
    if port is not None:
        build(constructions, port, BuildingKind.PORT, 1)
    land = land_around(tile_map, COAST)
    for cell in land:
        build(constructions, cell, BuildingKind.HOUSE, houses_per_cell)
    for cell in land[:fisheries]:
        build(constructions, cell, BuildingKind.FISHERY, 1)
    return Growth(constructions)


def test_three_fisheries_among_enough_people_join_into_a_port() -> None:
    growth = fishing_village(PORT_PEOPLE_PER_CELL)

    growth._turn(COAST, FISHERIES, np.random.default_rng(0))

    assert growth.count(BuildingKind.PORT)[COAST] == 1
    assert growth.count(BuildingKind.FISHERY).sum() == 0


@pytest.mark.parametrize(
    ("houses_per_cell", "fisheries"),
    [
        (PORT_PEOPLE_PER_CELL - 1, FISHERIES_PER_PORT),
        (PORT_PEOPLE_PER_CELL, FISHERIES_PER_PORT - 1),
    ],
)
def test_fisheries_stay_without_enough_people_or_enough_of_them(
    houses_per_cell: int, fisheries: int
) -> None:
    growth = fishing_village(houses_per_cell, fisheries)

    growth._turn(COAST, FISHERIES, np.random.default_rng(0))

    assert growth.count(BuildingKind.PORT).sum() == 0
    assert growth.count(BuildingKind.FISHERY).sum() == fisheries


def test_fisheries_do_not_join_into_a_port_beside_another_port() -> None:
    growth = fishing_village(PORT_PEOPLE_PER_CELL, port=(COAST[0] + 2, COAST[1]))

    growth._turn(COAST, FISHERIES, np.random.default_rng(0))

    assert growth.count(BuildingKind.PORT).sum() == 1
    assert growth.count(BuildingKind.FISHERY).sum() == FISHERIES_PER_PORT
