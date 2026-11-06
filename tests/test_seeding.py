import math
from collections import Counter
from dataclasses import replace

import numpy as np
import pytest

from railroads.mapgen.seeding import (
    ANY_TEMPERATURE,
    SEED_RULES,
    SeedRule,
    TownSizes,
    seed_buildings,
    suitability,
    town_cells,
)
from railroads.sim.constructions import CELL_SPACE, Building, BuildingKind, Constructions
from railroads.sim.geo import Bounds
from railroads.sim.grids import HexGrid
from railroads.sim.tilemap import Relief, Temperature, Terrain, TileMap, Vegetation

AS_MANY_AS_FIT = {kind: replace(rule, per_10k_km2=None) for kind, rule in SEED_RULES.items()}
FLAT_GRASS = SeedRule(
    relief={Relief.FLAT: 1.0, Relief.HILLS: 0.5},
    vegetation={Vegetation.GRASS: 1.0, Vegetation.FOREST: 0.5},
    temperature=ANY_TEMPERATURE,
    per_10k_km2=1.0,
    spacing_km=10.0,
)


def make_map(
    terrain: np.ndarray,
    relief: np.ndarray | None = None,
    vegetation: np.ndarray | None = None,
) -> TileMap:
    grid = HexGrid(rows=terrain.shape[0], cols=terrain.shape[1], km_per_cell=2.0)
    return TileMap(
        name="Test",
        grid=grid,
        bounds=Bounds(west=14.0, east=15.0, south=50.0, north=51.0),
        terrain=terrain.astype(np.uint8),
        vegetation=np.full(grid.shape, Vegetation.GRASS, dtype=np.uint8)
        if vegetation is None
        else vegetation.astype(np.uint8),
        temperature=np.full(grid.shape, Temperature.NORMAL, dtype=np.uint8),
        height_m=np.zeros(grid.shape, dtype=np.int16),
        relief=np.full(grid.shape, Relief.FLAT, dtype=np.uint8)
        if relief is None
        else relief.astype(np.uint8),
    )


def make_country(size: int = 80) -> TileMap:
    terrain = np.full((size, size), Terrain.LAND)
    terrain[:, :5] = Terrain.SEA
    terrain[:, size // 2] = Terrain.RIVER
    relief = np.full((size, size), Relief.FLAT)
    relief[size // 2 :, -25:] = Relief.HILLS
    relief[-15:, -25:] = Relief.MOUNTAINS
    vegetation = np.full((size, size), Vegetation.GRASS)
    vegetation[: size // 3, -30:] = Vegetation.FOREST
    return make_map(terrain, relief, vegetation)


def buildings_of(constructions: Constructions, kind: BuildingKind) -> list[Building]:
    return [b for b in constructions if isinstance(b, Building) and b.kind == kind]


def buildings_of_all(constructions: Constructions) -> list[Building]:
    return [built for built in constructions if isinstance(built, Building)]


def test_suitability_multiplies_the_weights_of_a_cell() -> None:
    tile_map = make_map(
        np.full((1, 4), Terrain.LAND),
        relief=np.array([[Relief.FLAT, Relief.HILLS, Relief.HILLS, Relief.MOUNTAINS]]),
        vegetation=np.array(
            [[Vegetation.GRASS, Vegetation.GRASS, Vegetation.FOREST, Vegetation.GRASS]]
        ),
    )

    scores = suitability(tile_map, FLAT_GRASS)

    assert scores.tolist() == [[1.0, 0.5, 0.25, 0.0]]


def test_suitability_counts_the_best_water_next_to_a_cell() -> None:
    tile_map = make_map(
        np.array([[Terrain.SEA, Terrain.LAND, Terrain.RIVER, Terrain.LAND, Terrain.LAND]])
    )
    rule = replace(FLAT_GRASS, water={Terrain.LAND: 0.2, Terrain.RIVER: 0.5, Terrain.SEA: 1.0})

    scores = suitability(tile_map, rule)

    assert scores.tolist() == [[0.0, 1.0, 0.0, 0.5, 0.2]]


def test_suitability_weighs_cells_in_and_next_to_towns() -> None:
    tile_map = make_map(np.full((1, 5), Terrain.LAND))
    constructions = Constructions(tile_map)
    constructions.build(Building((0, 0), BuildingKind.HOUSE))
    rule = replace(FLAT_GRASS, town=(0.1, 1.0))

    scores = suitability(tile_map, rule, town_cells(constructions))

    assert scores[0].tolist() == pytest.approx([1.0, 1.0, 0.1, 0.1, 0.1])


def test_seeded_buildings_stand_on_cells_their_rules_suit() -> None:
    tile_map = make_country()

    constructions = seed_buildings(tile_map, np.random.default_rng(1), AS_MANY_AS_FIT)

    in_town = town_cells(constructions)
    unsuitable = [
        built
        for built in constructions
        if isinstance(built, Building)
        and suitability(tile_map, SEED_RULES[built.kind], in_town)[built.cell] <= 0
    ]
    assert {b.kind for b in buildings_of_all(constructions)} == set(BuildingKind)
    assert unsuitable == []


@pytest.mark.parametrize("kind", [kind for kind in BuildingKind if kind != BuildingKind.HOUSE])
def test_buildings_of_one_kind_keep_their_spacing(kind: BuildingKind) -> None:
    tile_map = make_country()

    constructions = seed_buildings(tile_map, np.random.default_rng(2), AS_MANY_AS_FIT)

    grid = tile_map.grid
    spots = [grid.center(*b.cell) for b in buildings_of(constructions, kind)]
    closest_km = min(
        (math.dist(a, b) * grid.km_per_cell for i, a in enumerate(spots) for b in spots[i + 1 :]),
        default=math.inf,
    )
    assert len(spots) >= 2
    assert closest_km >= SEED_RULES[kind].spacing_km


def test_seeding_with_the_same_seed_gives_the_same_buildings() -> None:
    tile_map = make_country()

    first = list(seed_buildings(tile_map, np.random.default_rng(3)))
    second = list(seed_buildings(tile_map, np.random.default_rng(3)))

    assert first == second


def test_more_farmland_gets_more_farms() -> None:
    land = np.full((80, 80), Terrain.LAND)
    half_forest = np.full((80, 80), Vegetation.GRASS)
    half_forest[:, 40:] = Vegetation.FOREST

    all_grass = seed_buildings(make_map(land), np.random.default_rng(4))
    half_grass = seed_buildings(make_map(land, vegetation=half_forest), np.random.default_rng(4))

    assert len(buildings_of(all_grass, BuildingKind.FARM)) > len(
        buildings_of(half_grass, BuildingKind.FARM)
    )


def test_flat_grassland_away_from_water_gets_no_mines_lumber_mills_or_ports() -> None:
    tile_map = make_map(np.full((60, 60), Terrain.LAND))

    constructions = seed_buildings(tile_map, np.random.default_rng(5), AS_MANY_AS_FIT)

    kinds = Counter(b.kind for b in buildings_of_all(constructions))
    assert kinds[BuildingKind.MINE] == kinds[BuildingKind.LUMBER_MILL] == 0
    assert kinds[BuildingKind.PORT] == 0
    assert kinds[BuildingKind.FARM] > 0


def test_a_town_is_fullest_in_the_middle() -> None:
    tile_map = make_map(np.full((40, 40), Terrain.LAND))
    one_town = {
        BuildingKind.HOUSE: replace(
            SEED_RULES[BuildingKind.HOUSE], per_10k_km2=None, spacing_km=1000.0
        )
    }

    constructions = seed_buildings(
        tile_map, np.random.default_rng(6), one_town, TownSizes(largest=40)
    )

    per_cell = Counter(b.cell for b in buildings_of(constructions, BuildingKind.HOUSE))
    assert sum(per_cell.values()) == 40
    assert max(per_cell.values()) == CELL_SPACE
    assert min(per_cell.values()) <= 2


def test_seeding_leaves_track_and_stations_to_the_player() -> None:
    tile_map = make_country()

    constructions = seed_buildings(tile_map, np.random.default_rng(7))

    assert all(isinstance(built, Building) for built in constructions)
