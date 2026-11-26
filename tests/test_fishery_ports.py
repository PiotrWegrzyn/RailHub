import numpy as np
from growing import build, make_map

from builders import flat_map, layer
from railroads.sim.constructions import BuildingKind, Constructions
from railroads.sim.growth import Growth
from railroads.sim.ladders import FISHERIES, FISHERIES_PER_PORT
from railroads.sim.tilemap import Terrain, TileMap

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


def test_three_fisheries_join_into_a_port() -> None:
    growth = fishing_village(1)

    growth._turn(COAST, FISHERIES, np.random.default_rng(0))

    assert growth.count(BuildingKind.PORT)[COAST] == 1
    assert growth.count(BuildingKind.FISHERY).sum() == 0


def test_three_fisheries_on_an_empty_shore_join_into_a_port() -> None:
    growth = fishing_village(0)

    growth._turn(COAST, FISHERIES, np.random.default_rng(0))

    assert growth.count(BuildingKind.PORT)[COAST] == 1


def test_two_fisheries_stay_fisheries() -> None:
    growth = fishing_village(1, FISHERIES_PER_PORT - 1)

    growth._turn(COAST, FISHERIES, np.random.default_rng(0))

    assert growth.count(BuildingKind.PORT).sum() == 0
    assert growth.count(BuildingKind.FISHERY).sum() >= FISHERIES_PER_PORT - 1


def test_fisheries_join_into_a_port_beside_another_port() -> None:
    growth = fishing_village(1, port=(COAST[0] + 2, COAST[1]))

    growth._turn(COAST, FISHERIES, np.random.default_rng(0))

    assert growth.count(BuildingKind.PORT).sum() == 2


def test_fisheries_do_not_join_into_a_cell_holding_a_port() -> None:
    growth = fishing_village(1, port=COAST)

    growth._turn(COAST, FISHERIES, np.random.default_rng(0))

    assert growth.count(BuildingKind.PORT).sum() == 1
    assert growth.count(BuildingKind.FISHERY).sum() >= FISHERIES_PER_PORT


def test_fisheries_grow_mostly_by_the_sea_less_by_lakes_and_rivers_and_never_inland() -> None:
    terrain = layer(12, 12, Terrain.LAND)
    terrain[:, 0] = Terrain.SEA
    terrain[0, 4:8] = Terrain.LAKE
    terrain[11, 4:8] = Terrain.RIVER
    tile_map = flat_map(12, 12, terrain=terrain)

    weights = FISHERIES.land.weights(tile_map)

    by_sea, by_lake, by_river, inland = weights[6, 1], weights[1, 5], weights[10, 5], weights[6, 6]
    assert by_sea > by_lake > 0
    assert by_sea > by_river > 0
    assert inland == 0


def test_fisheries_on_an_empty_shore_join_into_a_port_within_a_year_without_growing() -> None:
    growth = fishing_village(0)
    rng = np.random.default_rng(0)

    for _ in range(12):
        growth.month(rng)

    assert growth.count(BuildingKind.PORT).sum() >= 1
