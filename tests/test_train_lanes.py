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
from railroads.ui.pixel.track_art import DOUBLE_SPACING, ISLAND_TRACKS, TrackArt
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


@pytest.mark.parametrize("main", [TrackWidth.DOUBLE, TrackWidth.QUADRUPLE])
def test_single_track_branching_off_wider_track_leaves_from_its_outer_track_on_that_side(
    main: TrackWidth,
) -> None:
    railway, track = line(main)
    constructions = railway.constructions
    growth = Growth(constructions, GrowthRule(trades=False))
    build_station(growth, (11, 15))
    route = plan_route(constructions, (6, 11), (11, 15), TrackWidth.SINGLE)
    assert route is not None
    lay_route(growth, route)
    double = next(
        found
        for found in (constructions.track_at((6, col)) for col in range(4, 20))
        if {piece.width for piece in found} == {TrackWidth.SINGLE, main}
    )
    branch = next(piece for piece in double if piece.width == TrackWidth.SINGLE)
    shared = next(
        point for point in branch.ends if any(p.width == main and point in p.ends for p in double)
    )
    share = 0.0 if branch.ends[0] == shared else 1.0
    outer = DOUBLE_SPACING / 2 * (int(main) - 1)

    at_the_junction = track.riding_side(branch, share, True)
    far_end = track.riding_side(branch, 1.0 - share, True)

    assert abs(at_the_junction) == pytest.approx(outer)
    assert far_end == pytest.approx(0.0)


@pytest.mark.parametrize(("lane", "off"), [(0, FULL), (1, 3 * FULL)])
def test_a_train_on_quadruple_track_rides_the_inner_track_fast_and_the_outer_one_slow(
    lane: int, off: float
) -> None:
    railway, track = line(TrackWidth.QUADRUPLE)
    path = path_of(railway, WEST, EAST)
    middle = railway.network.cell_at(path[len(path) // 2])

    shift = Lanes(track).sideways(path, halfway(path), {middle: lane})

    assert math.hypot(*shift) == pytest.approx(off)


def test_quadruple_track_closes_its_outer_tracks_onto_the_inner_ones_where_it_meets_double() -> (
    None
):
    constructions = Constructions(flat_map())
    for col, width in (
        (5, TrackWidth.DOUBLE),
        (6, TrackWidth.QUADRUPLE),
        (7, TrackWidth.QUADRUPLE),
    ):
        constructions.build(Track((6, col), (0, 6), width))
    projection = PixelProjection.of(constructions.tile_map.grid, 0.0, 1.0)
    track = TrackArt(constructions.tile_map, constructions, projection)
    piece = Track((6, 6), (0, 6), TrackWidth.QUADRUPLE)

    sides = track._sides_at(piece, [0.0, 0.5, 1.0])

    meeting_double, middle, onward = ([lane[k] for lane in sides] for k in range(3))
    assert {round(abs(side) / DOUBLE_SPACING, 2) for side in middle} == {0.5, 1.5}
    half = DOUBLE_SPACING / 2
    assert sorted(meeting_double + onward) == pytest.approx(
        sorted([-half, -half, half, half] + [-3 * half, -half, half, 3 * half])
    )


def test_a_station_on_quadruple_track_opens_its_outer_tracks_round_two_island_platforms() -> None:
    constructions = Constructions(flat_map())
    growth = Growth(constructions, GrowthRule(trades=False))
    build_station(growth, (6, 6))
    for col in (5, 7):
        constructions.build(Track((6, col), (0, 6), TrackWidth.QUADRUPLE))
    projection = PixelProjection.of(constructions.tile_map.grid, 0.0, 1.0)
    track = TrackArt(constructions.tile_map, constructions, projection)
    piece = track.station_piece((6, 6))

    sides = sorted(lane[0] for lane in track._sides_at(piece, [0.5]))

    half = DOUBLE_SPACING / 2
    assert piece.width == TrackWidth.QUADRUPLE
    assert sides == pytest.approx([-(half + ISLAND_TRACKS), -half, half, half + ISLAND_TRACKS])
