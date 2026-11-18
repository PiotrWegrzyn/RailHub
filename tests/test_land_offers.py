import numpy as np
import pytest

from builders import flat_map
from railroads.sim.grids import HEX_ROW_STEP
from railroads.sim.lots import Lots
from railroads.ui.land_offers import (
    LandOffers,
    PlacePicture,
    lot_corners,
    lot_under,
    shown_lots,
    world_uv,
)


def offering_lots() -> Lots:
    lots = Lots.of_square(top=0, left=0, rows=8, cols=6)
    lots.offers = [(0, -1), (-1, 0)]
    return lots


def test_a_lot_past_the_left_edge_lies_a_lot_wide_left_of_the_map() -> None:
    grid = flat_map(8, 6).grid
    lots = offering_lots()

    corners = lot_corners(grid, lots, (0, -1))

    assert corners == [
        (-3.0, 0.0),
        (0.0, 0.0),
        (0.0, pytest.approx(4 * HEX_ROW_STEP)),
        (-3.0, pytest.approx(4 * HEX_ROW_STEP)),
    ]


def test_a_point_past_the_map_lies_in_the_lot_there() -> None:
    grid = flat_map(8, 6).grid
    lots = offering_lots()

    lot = lot_under(grid, lots, -1.5, -0.5)

    assert lot == (-1, -1)


def test_lots_on_offer_and_those_in_the_maps_gaps_are_shown() -> None:
    lots = offering_lots()
    lots.buy((0, -1))
    lots.offers = [(-1, 0)]

    shown = shown_lots(lots)

    assert shown == [((-1, 0), True), ((1, -1), False)]


def test_the_world_map_is_read_where_the_map_lies_on_the_globe() -> None:
    tile_map = flat_map(8, 6)
    west, north = tile_map.bounds.west, tile_map.bounds.north

    u, v = world_uv(tile_map, (0.0, 0.0))

    assert u == pytest.approx((west + 180) / 360)
    assert v == pytest.approx((90 - north) / 180)


def test_a_click_on_a_lot_on_offer_asks_to_buy_it() -> None:
    grid = flat_map(8, 6).grid
    lots = offering_lots()
    land = LandOffers()
    land.hover(lots, grid, (-1.5, 1.0))

    took = land.click(lots)

    assert took
    assert land.asked == (0, -1)


def test_a_point_over_no_lot_shown_hovers_none() -> None:
    grid = flat_map(8, 6).grid
    land = LandOffers()

    land.hover(offering_lots(), grid, (30.0, 30.0))

    assert land.hovered is None
    assert not land.click(offering_lots())


def test_a_lot_is_shown_with_the_ground_of_the_places_map_where_it_lies() -> None:
    place = flat_map(16, 12)
    lots = Lots.of_square(top=4, left=3, rows=8, cols=6)

    picture = PlacePicture.of(place, lots)
    uvs = picture.uvs(lots, (0, -1))

    assert picture.pixels.shape == (16 + 8, 12 + 6, 4)
    assert (picture.top, picture.left) == (-4, -3)
    assert uvs[0] == pytest.approx((3 / 18, 8 / 24))
    assert uvs[2] == pytest.approx((6 / 18, 12 / 24))


def test_past_the_places_map_a_lot_shows_sea() -> None:
    place = flat_map(16, 12)
    lots = Lots.of_square(top=4, left=3, rows=8, cols=6)

    picture = PlacePicture.of(place, lots)

    assert not np.array_equal(picture.pixels[0, 0], picture.pixels[10, 10])
