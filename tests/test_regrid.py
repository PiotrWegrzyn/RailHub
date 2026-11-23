from dataclasses import replace

import numpy as np
import pytest

from builders import flat_map, played_game
from railroads.saves import SavedGame, snapshot, unpickled
from railroads.sim.constructions import LOCKED, Building, BuildingKind, Station
from railroads.sim.game import Game
from railroads.sim.grids import HEX_ROW_STEP, HexGrid
from railroads.sim.regrid import Move, regrid
from railroads.sim.tilemap import TileMap

DOWN, RIGHT = 2, 3


def bigger_map(game: Game) -> TileMap:
    """The game's map with DOWN rows above it, RIGHT columns to its left and as many below and to
    its right, all locked but the old map."""
    rows, cols = game.tile_map.grid.shape
    tile_map = flat_map(rows + 2 * DOWN, cols + 2 * RIGHT)
    locked = np.ones(tile_map.grid.shape, dtype=bool)
    locked[DOWN : DOWN + rows, RIGHT : RIGHT + cols] = False
    return replace(tile_map, locked=locked)


def moved_game() -> tuple[Game, Game]:
    """A played game moved onto a bigger map, and the same game left where it was."""
    return moved_save().game, played_game().game


def moved_save() -> SavedGame:
    saved = played_game()
    regrid(saved.game, bigger_map(saved.game), DOWN, RIGHT)
    return saved


def shifted(cell: tuple[int, int]) -> tuple[int, int]:
    return cell[0] + DOWN, cell[1] + RIGHT


def test_hex_rows_move_only_in_twos() -> None:
    grid = HexGrid(rows=4, cols=4, km_per_cell=2.0)

    with pytest.raises(ValueError, match="twos"):
        Move(grid, HexGrid(rows=8, cols=8, km_per_cell=2.0), 1, 0)


def test_an_array_over_the_flat_cells_moves_with_its_cells() -> None:
    move = Move(
        HexGrid(rows=2, cols=3, km_per_cell=2.0), HexGrid(rows=4, cols=5, km_per_cell=2.0), 2, 1
    )

    moved = move.flat_cells(np.arange(6), fill=-1)

    assert moved.reshape(4, 5)[2:, 1:4].tolist() == [[0, 1, 2], [3, 4, 5]]
    assert (moved.reshape(4, 5)[:2] == -1).all()
    assert move.flat(4) == 3 * 5 + 2


def test_stations_track_and_buildings_move_with_the_map() -> None:
    moved, kept = moved_game()

    stations = moved.railway.stations()
    houses = moved.growth.count(BuildingKind.HOUSE)

    assert stations == [shifted(station) for station in kept.railway.stations()]
    assert sorted(cell for cell, _ in moved.constructions.cells()) == sorted(
        shifted(cell) for cell, _ in kept.constructions.cells()
    )
    assert houses.shape == moved.tile_map.grid.shape
    assert np.array_equal(houses[DOWN:-DOWN, RIGHT:-RIGHT], kept.growth.count(BuildingKind.HOUSE))
    assert moved.railway.name_of(shifted((8, 6))) == kept.railway.name_of((8, 6))


def test_what_the_economy_remembers_moves_with_its_cells() -> None:
    moved, kept = moved_game()
    rows, cols = kept.tile_map.grid.shape

    levels = moved.growth.economy.levels.reshape(moved.tile_map.grid.shape)

    assert np.array_equal(
        levels[DOWN : DOWN + rows, RIGHT : RIGHT + cols],
        kept.growth.economy.levels.reshape(rows, cols),
    )
    assert moved.growth.year == kept.growth.year


def test_trains_keep_their_stops_cargo_and_way_on_the_moved_map() -> None:
    moved, kept = moved_game()

    moved.time_passed(0.3)
    kept.time_passed(0.3)

    for there, here in zip(moved.railway.trains, kept.railway.trains, strict=True):
        assert [stop.cell for stop in there.stops] == [shifted(stop.cell) for stop in here.stops]
        assert there.along == pytest.approx(here.along)
        assert there.load == here.load
        assert there.path == [
            pytest.approx((x + RIGHT, y + DOWN * HEX_ROW_STEP)) for x, y in here.path
        ]


def test_the_moved_game_plays_on_and_saves() -> None:
    saved = moved_save()
    moved = saved.game
    rng = np.random.default_rng(3)

    for _ in range(3):
        moved.month(rng)
        moved.time_passed(0.4)
    loaded = unpickled(snapshot(saved).data).game  # type: ignore[attr-defined]

    assert loaded.railway.stations() == moved.railway.stations()
    assert loaded.growth.population() == moved.growth.population()


def test_nothing_is_built_or_grows_on_the_locked_land_around_the_moved_map() -> None:
    moved, _ = moved_game()
    rng = np.random.default_rng(5)

    for _ in range(12):
        moved.month(rng)
    problem = moved.constructions.why_not(Station((0, 0)))

    locked = moved.tile_map.locked
    assert locked is not None
    assert problem == LOCKED
    assert not any(
        locked[building.cell] for building in moved.constructions if isinstance(building, Building)
    )
