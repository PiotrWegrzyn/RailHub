import math

import numpy as np
import pytest

from builders import flat_map
from railroads.sim.lots import Lots
from railroads.ui.land_offers import (
    LandOffers,
    PlacePicture,
    lot_corners,
    lot_under,
    reaches_box,
    seen_part,
    shown_lots,
    world_uv,
)


def offering_lots() -> Lots:
    lots = Lots.around_middle(flat_map(80, 70))
    lots.offers = [(-2, 0), (2, 0)]
    return lots


def test_a_lot_is_a_hexagon_round_its_centre_on_the_map() -> None:
    lots = offering_lots()

    corners = lot_corners(lots, (0, 0))

    middle = (sum(x for x, _ in corners) / 6, sum(y for _, y in corners) / 6)
    assert len(corners) == 6
    assert all(math.dist(middle, corner) == pytest.approx(lots.radius) for corner in corners)


def test_a_point_past_the_left_of_the_map_lies_in_the_lot_there() -> None:
    lots = offering_lots()
    x, y = lots.map_point(lots.centre((-2, 0)))

    lot = lot_under(lots, x, y)

    assert x < 0
    assert lot == (-2, 0)


def test_lots_on_offer_and_those_in_the_maps_corners_are_shown() -> None:
    lots = offering_lots()

    shown = dict(shown_lots(lots))

    assert shown[(-2, 0)] is True
    assert shown[(2, 0)] is True
    assert any(not offered for offered in shown.values())
    assert not set(shown) & lots.owned


def test_the_world_map_is_read_where_the_map_lies_on_the_globe() -> None:
    tile_map = flat_map(8, 6)
    west, north = tile_map.bounds.west, tile_map.bounds.north

    u, v = world_uv(tile_map, (0.0, 0.0))

    assert u == pytest.approx((west + 180) / 360)
    assert v == pytest.approx((90 - north) / 180)


def test_a_click_on_a_lot_on_offer_asks_to_buy_it() -> None:
    lots = offering_lots()
    land = LandOffers()
    land.hover(lots, lots.map_point(lots.centre((-2, 0))))

    took = land.click(lots)

    assert took
    assert land.asked == (-2, 0)


def test_a_point_over_no_lot_shown_hovers_none() -> None:
    lots = offering_lots()
    land = LandOffers()

    land.hover(lots, lots.map_point(lots.centre((0, 0))))

    assert land.hovered is None
    assert not land.click(lots)


def test_a_lot_is_shown_with_the_ground_of_the_places_map_where_it_lies() -> None:
    place = flat_map(80, 70)
    lots = Lots.around_middle(place)

    picture = PlacePicture.of(place, lots)
    u, v = picture.uv(lots.centre((0, 0)))

    assert picture.top % 2 == 0
    assert u == pytest.approx((lots.middle[1] + 0.5 - picture.left) / picture.pixels.shape[1])
    assert v == pytest.approx((lots.middle[0] + 0.5 - picture.top) / picture.pixels.shape[0])


def test_past_the_places_map_a_lot_shows_sea() -> None:
    place = flat_map(80, 70)
    lots = Lots.around_middle(place)

    picture = PlacePicture.of(place, lots)

    assert not np.array_equal(
        picture.pixels[0, 0], picture.pixels[-picture.top + 5, -picture.left + 10]
    )


def test_a_side_crossing_the_screen_is_seen_only_where_it_is_on_it() -> None:
    box = (0.0, 0.0, 100.0, 100.0)

    part = seen_part((-100.0, 50.0), (300.0, 50.0), box)

    assert part == pytest.approx((0.25, 0.5))


def test_a_side_off_the_screen_is_not_seen() -> None:
    box = (0.0, 0.0, 100.0, 100.0)

    part = seen_part((-50.0, 150.0), (200.0, 300.0), box)

    assert part is None


def test_a_side_on_the_screen_is_seen_whole() -> None:
    box = (0.0, 0.0, 100.0, 100.0)

    part = seen_part((10.0, 10.0), (90.0, 60.0), box)

    assert part == (0.0, 1.0)


def test_a_lot_far_off_the_screen_is_not_drawn() -> None:
    box = (0.0, 0.0, 100.0, 100.0)

    near = reaches_box([(400.0, 400.0), (500.0, 450.0), (450.0, 500.0)], box, 50.0)

    assert not near


def test_a_lot_just_off_the_screen_is_drawn_for_its_tag_to_show() -> None:
    box = (0.0, 0.0, 100.0, 100.0)

    near = reaches_box([(120.0, 40.0), (160.0, 60.0), (140.0, 80.0)], box, 50.0)

    assert near
