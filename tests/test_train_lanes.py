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
from railroads.ui.pixel.track_art import DOUBLE_SPACING, ISLAND_TRACKS, QUADRUPLE_OUTER, TrackArt
from railroads.ui.pixel.train_layer import Lanes, lanes_of

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


def single_branch(main: TrackWidth) -> tuple[TrackArt, Track, float]:
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
    return track, branch, 0.0 if branch.ends[0] == shared else 1.0


@pytest.mark.parametrize(
    ("main", "leaves_from"),
    [(TrackWidth.DOUBLE, DOUBLE_SPACING / 2), (TrackWidth.QUADRUPLE, DOUBLE_SPACING / 2)],
)
def test_single_track_branching_off_wider_track_leaves_from_the_outermost_track_it_can_turn_out_of(
    main: TrackWidth, leaves_from: float
) -> None:
    track, branch, share = single_branch(main)

    at_the_junction = track.riding_side(branch, share, True)
    far_end = track.riding_side(branch, 1.0 - share, True)

    assert abs(at_the_junction) == pytest.approx(leaves_from)
    assert far_end == pytest.approx(0.0, abs=1e-3)


def test_single_track_branching_off_quadruple_track_joins_all_four_of_its_tracks() -> None:
    track, branch, share = single_branch(TrackWidth.QUADRUPLE)

    drawn = track._sides_at(branch, [share], forks=True)

    half = DOUBLE_SPACING / 2
    assert sorted(abs(lane[0]) for lane in drawn) == pytest.approx(
        [half, half, QUADRUPLE_OUTER, QUADRUPLE_OUTER]
    )


@pytest.mark.parametrize(("lane", "off"), [(0, FULL), (1, QUADRUPLE_OUTER * PIXEL_CELL_WIDTH)])
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
    assert {round(abs(side), 2) for side in middle} == {DOUBLE_SPACING / 2, QUADRUPLE_OUTER}
    half = DOUBLE_SPACING / 2
    assert sorted(meeting_double + onward) == pytest.approx(
        sorted([-half, -half, half, half] + [-QUADRUPLE_OUTER, -half, half, QUADRUPLE_OUTER])
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


def test_track_branching_off_double_track_where_it_narrows_to_single_leaves_from_the_middle() -> (
    None
):
    constructions = Constructions(flat_map())
    grid = constructions.tile_map.grid
    cell = (6, 6)
    narrow, into_narrow = grid.track_exit(cell, 8)
    wide, into_wide = grid.track_exit(cell, 2)
    constructions.build(Track(narrow, (into_narrow, (into_narrow + 6) % 12)))
    constructions.build(Track(wide, (into_wide, (into_wide + 6) % 12), TrackWidth.DOUBLE))
    constructions.build(Track(cell, (2, 8), TrackWidth.DOUBLE))
    branch = Track(cell, (0, 8))
    constructions.build(branch)
    projection = PixelProjection.of(grid, 0.0, 1.0)
    track = TrackArt(constructions.tile_map, constructions, projection)

    drawn = track._sides_at(branch, [1.0], forks=True)

    assert [lane[0] for lane in drawn] == pytest.approx([0.0])


def test_a_train_is_drawn_on_the_track_of_quadruple_track_it_holds_through_the_junctions_between() -> (
    None
):
    four_lanes = {(5, 5), (5, 8)}
    point = (0.0, 0.0)
    train = Train("Test")
    train.claims = [
        ((5, 4), point, point, 0.0, 1.0, False, 0),
        ((5, 5), point, point, 1.0, 2.0, False, 1),
        ((5, 6), point, point, 2.0, 3.0, False, 0),
        ((5, 7), point, point, 3.0, 4.0, False, 0),
        ((5, 8), point, point, 4.0, 5.0, False, 0),
        ((5, 9), point, point, 5.0, 6.0, False, 0),
    ]

    lanes = lanes_of(train, four_lanes)

    assert [lanes[(5, col)] for col in range(4, 10)] == [1, 1, 1, 1, 0, 0]
    assert lanes[None] == 0


def test_a_junction_keeps_the_track_it_was_drawn_on_once_the_quadruple_track_behind_is_let_go() -> (
    None
):
    railway, track = line(TrackWidth.QUADRUPLE)
    four_lanes = {(5, 5), (5, 8)}
    point = (0.0, 0.0)
    train = Train("Test")
    train.path = path_of(railway, WEST, EAST)
    train.along = 4.5
    train.claims = [
        ((5, 5), point, point, 1.0, 2.0, False, 0),
        ((5, 6), point, point, 2.0, 3.0, False, 0),
        ((5, 7), point, point, 3.0, 4.0, False, 0),
        ((5, 8), point, point, 4.0, 5.0, False, 1),
    ]
    lanes = Lanes(track)
    lanes.of(train, four_lanes)
    train.claims = train.claims[1:]

    held = lanes.of(train, four_lanes)

    assert [held[(5, col)] for col in range(5, 9)] == [0, 0, 0, 1]


def test_a_train_on_the_outer_track_crosses_over_on_quadruple_track_into_double_at_a_junction() -> (
    None
):
    railway, track = line(TrackWidth.QUADRUPLE)
    constructions = track.constructions
    piece = constructions.track_at((6, 11))[0]
    constructions.remove(piece)
    constructions.build(Track((6, 11), piece.ends, TrackWidth.DOUBLE))
    constructions.build(Track((6, 11), (piece.ends[0], 4), TrackWidth.QUADRUPLE))
    path = path_of(railway, WEST, EAST)
    lanes = Lanes(track)

    shifts = [
        lanes.sideways(path, step / 50, {None: 1}) for step in range(int(halfway(path) * 100))
    ]

    assert max(math.dist(a, b) for a, b in itertools.pairwise(shifts)) < 1.0


def test_double_track_branching_off_quadruple_track_can_be_ridden_on_from_its_outer_track() -> None:
    _railway, track = line(TrackWidth.QUADRUPLE)
    constructions = track.constructions
    main = constructions.track_at((6, 11))[0]
    branch = Track((6, 11), (main.ends[0], 4), TrackWidth.DOUBLE)
    constructions.build(branch)
    end = 0.0 if branch.ends[0] == main.ends[0] else 1.0
    outer = track.riding_side(main, 0.0, True, 1)

    found = track.riding_choices(branch, end, end == 0.0, 1)

    assert found is not None
    assert found[0] == main.ends[0]
    assert any(at_end == pytest.approx(outer) for _, at_end in found[1])


def test_a_piece_ridden_before_the_track_it_branches_off_is_laid_is_ridden_as_a_branch_after() -> (
    None
):
    _railway, track = line(TrackWidth.DOUBLE)
    constructions = track.constructions
    main = constructions.track_at((6, 11))[0]
    constructions.remove(main)
    branch = Track((6, 11), (main.ends[0], 4), TrackWidth.SINGLE)
    constructions.build(branch)
    before = track.riding_choices(branch, 0.0, True)

    constructions.build(main)
    after = track.riding_choices(branch, 0.0, True)

    assert before is None
    assert after is not None
    assert after[0] == main.ends[0]
