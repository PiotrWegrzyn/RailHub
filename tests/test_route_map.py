from collections.abc import Iterator

import numpy as np
import pytest
from imgui_bundle import hello_imgui, imgui
from screens import FakeTexture, gui_context
from test_railway import joined_stations

from railroads.sim.clock import QUARTER
from railroads.sim.constructions import Building, BuildingKind
from railroads.sim.demand import DEMAND_PER_NEED
from railroads.sim.goods import Good, Need
from railroads.sim.steps import finish
from railroads.sim.trains import Stop
from railroads.ui.minimap import Panel
from railroads.ui.route_map import (
    GENERAL_SHADES,
    LEAST_MADE,
    MADE_SHADES,
    MOST_FRAMING_ZOOM,
    MOST_ZOOM,
    SHADE_FROM,
    SOFTEN_ROWS,
    SPECIAL_SHADES,
    SUMMARY_MOST,
    MapFrame,
    Marks,
    RouteMap,
    add_stop,
    can_stop_at,
    edge_spot,
    framing,
    made_at,
    marks_at,
    needed,
    needs_named,
    shade_of,
    soften,
    soften_steps,
    station_at,
    summary_of,
    track_lines,
)
from railroads.ui.station_offers import Offer

PANEL = Panel(100.0, 50.0, 400.0, 300.0)
MAP_SIZE = (40.0, 20.0)


def test_a_click_picks_the_station_drawn_nearest_it() -> None:
    spots = {(1, 1): (10.0, 10.0), (1, 2): (16.0, 10.0)}

    picked = station_at(spots, 14.0, 11.0, 9.0)

    assert picked == (1, 2)


def test_a_click_away_from_every_station_picks_none() -> None:
    spots = {(1, 1): (10.0, 10.0)}

    picked = station_at(spots, 30.0, 10.0, 9.0)

    assert picked is None


def test_a_station_off_the_map_is_pointed_at_from_the_edge_on_its_side() -> None:
    middle_y = PANEL.top + PANEL.height / 2

    spot = edge_spot(PANEL, PANEL.right + 200.0, middle_y, 10.0)

    assert spot == pytest.approx((PANEL.right - 10.0, middle_y))


def test_a_station_off_a_corner_is_pointed_at_from_inside_the_map() -> None:
    spot = edge_spot(PANEL, PANEL.left - 500.0, PANEL.top - 500.0, 10.0)

    assert PANEL.contains(*spot)
    assert spot[0] == pytest.approx(PANEL.left + 10.0) or spot[1] == pytest.approx(PANEL.top + 10.0)


def test_grain_wanted_by_homes_and_a_bakery_is_both_a_general_and_a_special_need() -> None:
    _, railway = joined_stations()
    demand = railway.station_demand((6, 3))
    demand.capacity = {Need.FOOD: 100.0, Need.WHEAT: 300.0, Need.COAL: 80.0}

    general, special = needed(demand, Good.WHEAT)

    assert general == pytest.approx(100.0 / DEMAND_PER_NEED)
    assert special == pytest.approx(300.0 / DEMAND_PER_NEED)


def test_heating_is_a_general_need_for_coal_and_a_boiler_a_special_one() -> None:
    _, railway = joined_stations()
    demand = railway.station_demand((6, 3))
    demand.capacity = {Need.HEATING: 20.0, Need.BOILER_FUEL: 60.0}

    general, special = needed(demand, Good.COAL)

    assert general == pytest.approx(20.0 / DEMAND_PER_NEED)
    assert special == pytest.approx(60.0 / DEMAND_PER_NEED)


def test_mail_wanted_and_sent_at_a_station_is_a_general_need_and_made_there() -> None:
    _, railway = joined_stations()
    railway.station_demand((6, 3)).capacity = {Need.MAIL: 30.0}
    made = {(6, 3): Offer(Good.MAIL, 0.0, LEAST_MADE + 5)}

    marks = marks_at(railway, Good.MAIL, made)

    assert marks == {(6, 3): Marks(general=30.0 / DEMAND_PER_NEED, made=LEAST_MADE + 5)}


def test_a_supplier_wanting_a_little_of_its_good_takes_the_lightest_shade_of_both() -> None:
    _, railway = joined_stations()
    railway.station_demand((6, 3)).capacity = {Need.FOOD: 30.0}
    made = {(6, 3): Offer(Good.APPLES, 115.0, LEAST_MADE + 5)}

    shades = marks_at(railway, Good.APPLES, made)[(6, 3)].shades()

    assert shades == [GENERAL_SHADES[0], MADE_SHADES[0]]


def test_a_station_is_marked_as_making_a_good_only_above_the_least_made() -> None:
    _, railway = joined_stations()
    made = {
        (6, 3): Offer(Good.COAL, 0.0, LEAST_MADE + 1),
        (6, 20): Offer(Good.COAL, 200.0, LEAST_MADE - 1),
    }

    marks = marks_at(railway, Good.COAL, made)

    assert marks == {(6, 3): Marks(made=LEAST_MADE + 1)}


def test_a_good_wanted_and_made_nowhere_marks_no_station() -> None:
    _, railway = joined_stations()
    railway.station_demand((6, 3)).capacity = {Need.FOOD: 30.0}

    marks = marks_at(railway, Good.DIAMONDS, {})

    assert marks == {}


@pytest.mark.parametrize(
    ("amount", "shade"),
    [(1.0, 0), (SHADE_FROM[0] - 0.1, 0), (SHADE_FROM[0], 1), (SHADE_FROM[1] - 0.1, 1), (500.0, 2)],
)
def test_a_marks_shade_darkens_at_each_threshold(amount: float, shade: int) -> None:
    result = shade_of(amount)

    assert result == shade


def test_a_station_gets_a_shade_for_each_kind_of_mark_it_has() -> None:
    mark = Marks(general=200.0, special=10.0, made=QUARTER * SHADE_FROM[0])

    shades = mark.shades()

    assert shades == [GENERAL_SHADES[2], SPECIAL_SHADES[0], MADE_SHADES[1]]


def test_a_good_is_made_where_buildings_around_make_it_or_where_it_waits() -> None:
    _, railway = joined_stations()
    railway.station_goods((6, 20)).stockpile[Good.COAL] = 12.0
    made = {(6, 3): {Good.COAL: 30.0}, (6, 20): {}}

    found = made_at(railway, Good.COAL, made.__getitem__)

    assert {station: (o.gathered, o.waiting) for station, o in found.items()} == {
        (6, 3): (30.0, 0.0),
        (6, 20): (0.0, 12.0),
    }


def test_a_good_made_and_waiting_nowhere_is_made_at_no_station() -> None:
    _, railway = joined_stations()

    found = made_at(railway, Good.IRON, lambda _: {Good.COAL: 30.0})

    assert found == {}


def test_a_station_summary_lists_what_it_wants_and_makes_the_most_first() -> None:
    _, railway = joined_stations()
    railway.station_demand((6, 3)).capacity = {Need.FOOD: 30.0, Need.COAL: 80.0}
    railway.station_goods((6, 3)).stockpile[Good.WHEAT] = 7.0

    summary = summary_of(railway, (6, 3), {Good.COAL: 20.0, Good.IRON: 50.0})

    assert summary.wants == [(Need.COAL, 80.0), (Need.FOOD, 30.0)]
    assert summary.makes == [(Good.IRON, 50.0), (Good.COAL, 20.0), (Good.WHEAT, 7.0)]
    assert summary.more_wants == summary.more_makes == 0


def test_a_station_summary_keeps_only_the_most_and_counts_the_rest() -> None:
    _, railway = joined_stations()
    needs = list(Need)[: SUMMARY_MOST + 2]
    railway.station_demand((6, 3)).capacity = {need: 10.0 + i for i, need in enumerate(needs)}

    summary = summary_of(railway, (6, 3), {})

    assert len(summary.wants) == SUMMARY_MOST
    assert summary.wants[0] == (needs[-1], 10.0 + len(needs) - 1)
    assert summary.more_wants == 2
    assert summary.makes == []


def test_softening_blurs_a_hard_edge_and_keeps_flat_ground_as_it_was() -> None:
    rgba = np.full((32, 32, 4), 255, dtype=np.uint8)
    rgba[:, :16, :3] = 0

    soft = soften(rgba)

    row = soft[8, :, 0].astype(int)
    assert soft.shape == (16, 16, 4)
    assert row[0] == 0 and row[-1] == 255
    assert 0 < row[7] < 255 or 0 < row[8] < 255
    assert list(row) == sorted(row)


@pytest.mark.parametrize("shape", [(2 * SOFTEN_ROWS * 3 + 7, 31), (5, 3), (1, 1)])
def test_softening_in_steps_gives_the_picture_softened_at_once(shape: tuple[int, int]) -> None:
    rgba = np.random.default_rng(5).integers(0, 256, size=(*shape, 4), dtype=np.uint8)
    steps = soften_steps(rgba)

    taken = 0
    while True:
        try:
            next(steps)
        except StopIteration as done:
            soft = done.value
            break
        taken += 1

    assert np.array_equal(soft, soften(rgba))
    assert taken == 2 * -(-soft.shape[0] // SOFTEN_ROWS)


def striped(rows: int, cols: int) -> np.ndarray:
    """A picture whose half is random with white and black rows every few rows, so that some
    stripe lies as far from every band's edge as a blur reaches and shows a band cut too short."""
    half = np.random.default_rng(7).integers(0, 256, size=(rows, cols, 4), dtype=np.uint8)
    half[::7] = 255
    half[3::11] = 0
    return np.repeat(np.repeat(half, 2, axis=0), 2, axis=1)


@pytest.mark.parametrize("radius", [0.3, 2.99, 3.0, 6.0, 10.5])
def test_softening_in_steps_gives_the_picture_softened_at_once_however_wide_the_blur(
    radius: float,
) -> None:
    rgba = striped(2 * SOFTEN_ROWS + 3, 5)

    soft = finish(soften_steps(rgba, radius))

    assert np.array_equal(soft, soften(rgba, radius))


def test_a_station_can_be_added_unless_the_train_already_ends_there() -> None:
    _, railway = joined_stations()
    train = railway.buy()
    assert train is not None
    train.stops.append(Stop((6, 3)))

    add_stop(train, (6, 20))

    assert [stop.cell for stop in train.stops] == [(6, 3), (6, 20)]
    assert not can_stop_at(train, (6, 20))
    assert can_stop_at(train, (6, 3))


def test_the_whole_map_fits_the_panel_and_turns_points_into_spots_and_back() -> None:
    frame = MapFrame.of(PANEL, MAP_SIZE, 1.0)

    spot = frame.to_screen(30.0, 5.0)
    back = frame.to_map(*spot)

    assert PANEL.contains(*frame.to_screen(0.0, 10.0))
    assert PANEL.contains(*frame.to_screen(39.9, 10.0))
    assert back == pytest.approx((30.0, 5.0))


def test_zoomed_in_the_map_stays_on_the_map_near_its_edge() -> None:
    frame = MapFrame.of(PANEL, MAP_SIZE, 4.0, (0.0, 0.0))

    corner = frame.to_map(PANEL.left, PANEL.top)

    assert corner == pytest.approx((0.0, 0.0))
    assert frame.scale == pytest.approx(4 * PANEL.width / MAP_SIZE[0])


def test_zooming_keeps_the_map_point_under_the_mouse_in_place() -> None:
    route_map = RouteMap()
    frame = MapFrame.of(PANEL, MAP_SIZE, route_map.zoom)
    mouse = (PANEL.left + 240.0, PANEL.top + 160.0)
    under = frame.to_map(*mouse)

    route_map.zoom_at(frame, MAP_SIZE, *mouse, 3)

    zoomed = MapFrame.of(PANEL, MAP_SIZE, route_map.zoom, route_map.focus)
    assert route_map.zoom > 1.0
    assert zoomed.to_map(*mouse) == pytest.approx(under)


def test_zoom_stays_between_the_whole_map_and_the_closest_view() -> None:
    route_map = RouteMap()
    frame = MapFrame.of(PANEL, MAP_SIZE, route_map.zoom)

    route_map.zoom_at(frame, MAP_SIZE, PANEL.left, PANEL.top, -5)
    widest = route_map.zoom
    route_map.zoom_at(frame, MAP_SIZE, PANEL.left, PANEL.top, 100)

    assert widest == 1.0
    assert route_map.zoom == MOST_ZOOM


@pytest.fixture
def context(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    with gui_context((800, 600)):
        monkeypatch.setattr(
            hello_imgui, "create_texture_gpu_from_rgba_data", lambda _: FakeTexture()
        )
        imgui.new_frame()
        imgui.begin("map", None)
        yield
        imgui.end()
        imgui.render()


def test_the_map_centres_on_the_trains_stops(context: None) -> None:
    _, railway = joined_stations()
    railway.station_demand((6, 3)).capacity = {Need.COAL: 40.0}
    train = railway.buy()
    assert train is not None
    train.stops = [Stop((6, 3)), Stop((6, 20))]
    route_map = RouteMap()
    route_map.good = Good.COAL
    ground = (imgui.ImTextureRef(1), railway.constructions.tile_map.grid.size())

    route_map.draw(railway, train, lambda: ground, lambda _: {Good.COAL: 10.0})

    grid = railway.constructions.tile_map.grid
    (x0, y0), (x1, y1) = grid.center(6, 3), grid.center(6, 20)
    assert route_map.focus == pytest.approx(((x0 + x1) / 2, (y0 + y1) / 2))


def test_a_station_whose_buildings_lack_nothing_wants_nothing() -> None:
    growth, railway = joined_stations()
    growth.build(Building((6, 21), BuildingKind.CATTLE_RANCH))
    railway.station_goods((6, 20)).brought[Good.WHEAT] = 500.0
    growth.trade_goods()

    summary = summary_of(railway, (6, 20), {})

    assert railway.station_demand((6, 20)).capacity[Need.WHEAT] > 0
    assert summary.wants == []


def test_track_pieces_end_to_end_are_drawn_as_one_line_in_order() -> None:
    pieces = np.array([[[1.0, 0.0], [1.5, 0.0], [2.0, 0.0]], [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]]])

    lines = track_lines(pieces, [False, False])

    assert len(lines) == 1
    points, double = lines[0]
    assert points[:, 0].tolist() == [0.0, 0.5, 1.0, 1.5, 2.0]
    assert not double


def test_track_splits_into_lines_where_three_pieces_meet_or_its_width_changes() -> None:
    meeting = np.array(
        [
            [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]],
            [[1.0, 0.0], [1.5, 0.0], [2.0, 0.0]],
            [[1.0, 0.0], [1.0, 0.5], [1.0, 1.0]],
            [[2.0, 0.0], [2.5, 0.0], [3.0, 0.0]],
        ]
    )

    lines = track_lines(meeting, [False, False, False, True])

    assert len(lines) == 4
    assert [double for _, double in lines].count(True) == 1


def test_general_needs_are_named_in_the_order_needs_are_listed_the_last_after_or() -> None:
    needs = frozenset({Need.MAIL, Need.FOOD, Need.HEATING})

    named = needs_named(needs)

    assert named == "food, heating or mail"


def test_the_zoom_buttons_zoom_on_the_middle_of_the_trains_stops() -> None:
    _, railway = joined_stations()
    grid = railway.constructions.tile_map.grid
    map_size = grid.size()
    train = railway.buy()
    assert train is not None
    train.stops = [Stop((6, 3)), Stop((6, 20))]
    route_map = RouteMap()
    frame = MapFrame.of(PANEL, map_size, route_map.zoom)

    route_map.zoom_on_stops(railway, train, frame, map_size, 6)

    zoomed = MapFrame.of(PANEL, map_size, route_map.zoom, route_map.focus)
    middle = (
        (grid.center(6, 3)[0] + grid.center(6, 20)[0]) / 2,
        (grid.center(6, 3)[1] + grid.center(6, 20)[1]) / 2,
    )
    left, top = zoomed.to_screen(*middle)
    assert route_map.zoom > 1.0
    assert PANEL.left < left < PANEL.right
    assert PANEL.top < top < PANEL.bottom


def test_the_map_frames_a_new_train_again_once_it_gets_its_first_stops() -> None:
    _, railway = joined_stations()
    map_size = railway.constructions.tile_map.grid.size()
    train = railway.buy()
    assert train is not None
    route_map = RouteMap()
    route_map.frame(railway, train, map_size, PANEL)
    unframed = route_map.focus
    train.stops = [Stop((6, 3)), Stop((6, 8))]

    stale = route_map._framed != framing(train)
    route_map.frame(railway, train, map_size, PANEL)

    assert stale
    assert unframed is None
    assert route_map.focus is not None


def test_framing_a_train_with_one_stop_comes_only_a_little_closer_and_shows_much_of_the_map() -> (
    None
):
    _, railway = joined_stations()
    map_size = railway.constructions.tile_map.grid.size()
    train = railway.buy()
    assert train is not None
    train.stops = [Stop((6, 3))]
    route_map = RouteMap()

    route_map.frame(railway, train, map_size, PANEL)

    assert 1.0 <= route_map.zoom <= MOST_FRAMING_ZOOM < MOST_ZOOM
