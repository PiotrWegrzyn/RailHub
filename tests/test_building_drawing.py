import math
from collections.abc import Iterator

import numpy as np
import pytest
from imgui_bundle import imgui

from railroads.sim.constructions import Building, BuildingKind, Constructions, Track
from railroads.sim.geo import Bounds
from railroads.sim.grids import Grid, HexGrid, SquareGrid
from railroads.sim.tilemap import Relief, Temperature, Terrain, TileMap, Vegetation
from railroads.ui.buildings import BuildingSpots, draw_building_dots
from railroads.ui.iso_camera import IsoCamera


def make_constructions(grid_type: type[Grid] = HexGrid) -> Constructions:
    grid = grid_type(rows=5, cols=5, km_per_cell=2.0)
    terrain = np.full(grid.shape, Terrain.LAND, dtype=np.uint8)
    terrain[:, 0] = Terrain.SEA
    tile_map = TileMap(
        name="Test",
        grid=grid,
        bounds=Bounds(west=14.0, east=15.0, south=50.0, north=51.0),
        terrain=terrain,
        vegetation=np.full(grid.shape, Vegetation.GRASS, dtype=np.uint8),
        temperature=np.full(grid.shape, Temperature.NORMAL, dtype=np.uint8),
        height_m=np.zeros(grid.shape, dtype=np.int16),
        relief=np.full(grid.shape, Relief.FLAT, dtype=np.uint8),
    )
    constructions = Constructions(tile_map)
    for _ in range(6):
        constructions.build(Building((2, 2), BuildingKind.HOUSE))
    constructions.build(Track((2, 3), (0, len(grid.track_points()) // 2)))
    for kind in (BuildingKind.FARM, BuildingKind.LUMBER_MILL):
        constructions.build(Building((2, 3), kind))
    for kind in (BuildingKind.FACTORY, BuildingKind.MINE):
        constructions.build(Building((3, 2), kind))
    constructions.build(Building((2, 1), BuildingKind.PORT))
    constructions.build(Building((1, 3), BuildingKind.FLATS))
    constructions.build(Building((1, 3), BuildingKind.TENEMENT))
    constructions.build(Building((1, 2), BuildingKind.BIG_FACTORY))
    constructions.build(Building((1, 2), BuildingKind.SKYSCRAPER))
    constructions.build(Building((3, 4), BuildingKind.MEGA_FACTORY))
    for kind in (BuildingKind.ORCHARD, BuildingKind.FOOD_PLANT, BuildingKind.STEEL_MILL):
        constructions.build(Building((4, 3), kind))
    constructions.build(Building((2, 1), BuildingKind.SHIPYARD))
    return constructions


@pytest.fixture
def draw_list() -> Iterator[imgui.ImDrawList]:
    context = imgui.create_context()
    io = imgui.get_io()
    io.display_size = imgui.ImVec2(1280, 800)
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures.value
    imgui.new_frame()
    yield imgui.get_background_draw_list()
    imgui.render()
    imgui.destroy_context(context)


@pytest.mark.parametrize("grid_type", [HexGrid, SquareGrid], ids=["hex", "square"])
def test_buildings_stand_inside_their_cells(grid_type: type[Grid]) -> None:
    constructions = make_constructions(grid_type)

    spots = BuildingSpots.of(constructions)

    grid = constructions.tile_map.grid
    rows, cols = grid.cell_at(spots.x, spots.y)
    standing = list(zip(rows.tolist(), cols.tolist(), strict=True))
    built = [b.cell for b in constructions if isinstance(b, Building)]
    assert standing == built


def test_houses_sharing_a_cell_stand_apart() -> None:
    constructions = make_constructions()

    spots = BuildingSpots.of(constructions)

    houses = [
        (x, y)
        for x, y, kind in zip(spots.x, spots.y, spots.kinds, strict=True)
        if kind == BuildingKind.HOUSE
    ]
    gaps = [math.dist(a, b) for i, a in enumerate(houses) for b in houses[i + 1 :]]
    assert len(houses) == 6
    assert min(gaps) > 0.2


@pytest.mark.parametrize("zoom_level", [0, 1, 2])
def test_every_kind_of_building_shows_as_a_dot_when_zoomed_out(
    draw_list: imgui.ImDrawList, zoom_level: int
) -> None:
    spots = BuildingSpots.of(make_constructions())
    camera = IsoCamera(center_x=2.5, center_y=2.0, angle_deg=0.0, cell_units=1.0)
    camera.zoom_level = zoom_level
    before = draw_list.vtx_buffer.size()

    draw_building_dots(draw_list, camera, (640.0, 400.0), ((0.0, 0.0), (1280.0, 800.0)), spots)

    assert set(spots.kinds) == set(BuildingKind)
    assert draw_list.vtx_buffer.size() > before


def test_buildings_off_screen_are_not_drawn(draw_list: imgui.ImDrawList) -> None:
    spots = BuildingSpots.of(make_constructions())
    camera = IsoCamera(center_x=500.0, center_y=500.0, angle_deg=0.0, cell_units=1.0)
    before = draw_list.vtx_buffer.size()

    draw_building_dots(draw_list, camera, (640.0, 400.0), ((0.0, 0.0), (1280.0, 800.0)), spots)

    assert draw_list.vtx_buffer.size() == before
