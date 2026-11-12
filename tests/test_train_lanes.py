import itertools
import math

import pytest

from builders import flat_map
from railroads.sim.constructions import Constructions, Track, TrackWidth
from railroads.sim.growth import Growth
from railroads.sim.ladders import GrowthRule
from railroads.sim.railway import Railway
from railroads.sim.routes import build_station, lay_route, plan_route
from railroads.sim.trains import Stop, Train
from railroads.ui.iso_camera import PIXEL_CELL_WIDTH
from railroads.ui.pixel.projection import PixelProjection
from railroads.ui.pixel.track_art import DOUBLE_SPACING, TrackArt
from railroads.ui.pixel.train_layer import Lanes

WEST, EAST = (6, 3), (6, 20)
FULL = DOUBLE_SPACING / 2 * PIXEL_CELL_WIDTH


def line(width: TrackWidth) -> tuple[Railway, TrackArt]:
    constructions = Constructions(flat_map())
    growth = Growth(constructions, GrowthRule(trades=False))
    build_station(growth, WEST)
    build_station(growth, EAST)
    route = plan_route(constructions, WEST, EAST, width)
    assert route is not None
    lay_route(growth, route)
    railway = Railway(constructions)
    projection = PixelProjection.of(constructions.tile_map.grid, 0.0, 1.0)
    return railway, TrackArt(constructions.tile_map, constructions, projection)


def path_of(railway: Railway, start, end) -> list:
    train = Train("Test", stops=[Stop(start), Stop(end)])
    way = railway.route(start, end)
    assert way is not None
    train.path = way[0]
    return train.path


def halfway(path: list) -> float:
    return sum(math.dist(a, b) for a, b in itertools.pairwise(path)) / 2


def test_a_train_on_double_track_rides_one_track_half_the_spacing_off_the_middle() -> None:
    railway, track = line(TrackWidth.DOUBLE)
    path = path_of(railway, WEST, EAST)

    shift = Lanes(track).sideways(path, halfway(path))

    assert math.hypot(*shift) == pytest.approx(FULL)


def test_trains_going_opposite_ways_on_double_track_ride_different_tracks() -> None:
    railway, track = line(TrackWidth.DOUBLE)
    east, west = path_of(railway, WEST, EAST), path_of(railway, EAST, WEST)

    eastward = Lanes(track).sideways(east, halfway(east))
    westward = Lanes(track).sideways(west, halfway(west))

    assert eastward[0] == pytest.approx(-westward[0], abs=0.5)
    assert eastward[1] == pytest.approx(-westward[1], abs=0.5)


def test_a_train_stays_on_the_middle_of_single_track() -> None:
    railway, track = line(TrackWidth.SINGLE)
    path = path_of(railway, WEST, EAST)

    shifts = [Lanes(track).sideways(path, along / 10) for along in range(int(halfway(path) * 20))]

    assert all(shift == (0.0, 0.0) for shift in shifts)


def test_the_track_a_train_rides_on_closes_in_where_double_track_runs_into_single() -> None:
    _railway, track = line(TrackWidth.DOUBLE)
    piece = track.constructions.track_at((6, 10))[0]
    track.constructions.remove(track.constructions.track_at((6, 11))[0])
    track.constructions.build(Track((6, 11), piece.ends))

    east_first = track.tile_map.grid.track_exit((6, 10), piece.ends[0])[0] == (6, 11)

    at_single_end = track.riding_side(piece, 0.0 if east_first else 1.0, True)
    on_the_way = track.riding_side(piece, 0.25 if east_first else 0.75, True)
    at_double_end = track.riding_side(piece, 1.0 if east_first else 0.0, True)

    assert at_single_end == pytest.approx(0.0)
    assert on_the_way == pytest.approx(DOUBLE_SPACING / 4)
    assert at_double_end == pytest.approx(DOUBLE_SPACING / 2)


def test_either_way_along_double_track_a_train_keeps_to_the_same_side_of_its_way() -> None:
    _railway, track = line(TrackWidth.DOUBLE)
    piece = track.constructions.track_at((6, 10))[0]

    forward = track.riding_side(piece, 0.5, True)
    backward = track.riding_side(piece, 0.5, False)

    assert forward == pytest.approx(DOUBLE_SPACING / 2)
    assert backward == pytest.approx(DOUBLE_SPACING / 2)
