import numpy as np
import pytest

from builders import flat_map
from railroads.sim.constructions import BuildingKind, Constructions
from railroads.sim.events import make_factor
from railroads.sim.game import Game
from railroads.sim.goods import Good, Need, makes_of, needs_of
from railroads.sim.tiles import (
    CLOSES,
    FIRST_YEAR,
    TILE_GOODS,
    TILE_OF,
    Board,
    Tile,
    Tiles,
    activate,
    demand_factor,
    price,
    supply_factor,
)


def test_every_good_belongs_to_exactly_one_tile() -> None:
    counted = [good for goods in TILE_GOODS.values() for good in goods]

    assert sorted(counted, key=lambda g: g.name) == sorted(Good, key=lambda g: g.name)
    assert len(TILE_OF) == len(Good)


def test_the_world_reaches_each_era_in_its_year_when_the_lowest_tiles_go_up() -> None:
    tiles = Tiles.until(2050)

    years = tiles.era_years

    assert years == {2: 1843, 3: 1873, 4: 1901, 5: 1933, 6: 1957, 7: 1974}


def test_every_supply_level_adds_its_periods_share_of_the_base_not_compounded() -> None:
    made = supply_factor(Tile.COAL, 1, 9)

    assert made == pytest.approx(1.0 + 7 * 0.10 + 0.40)


def test_a_closed_tile_makes_no_more_past_its_closing_level() -> None:
    made = supply_factor(Tile.COAL, 1, 40)

    assert made == pytest.approx(supply_factor(Tile.COAL, 1, CLOSES[Tile.COAL]))


def test_every_demand_level_adds_the_same_share_of_the_base() -> None:
    wanted = demand_factor(Tile.FARMING, 1, 11)

    assert wanted == pytest.approx(1.5)


def test_a_unit_earns_its_base_times_demand_over_supply_to_the_power_of_four_fifths() -> None:
    earned = price(4.0, 2.0)

    assert earned == pytest.approx(0.5**0.8)


def test_every_good_earns_its_base_price_before_the_first_year_of_the_boards() -> None:
    tiles = Tiles.until(1900)

    earned = tiles.of_year(FIRST_YEAR - 1).price(Good.COAL)

    assert earned == 1.0


def test_coal_earns_about_a_fifth_of_its_price_by_2050_and_luxuries_about_half() -> None:
    boards = Tiles.until(2050).of_year(2050)

    coal, luxuries = boards.price(Good.COAL), boards.price(Good.GOLD)

    assert coal == pytest.approx(0.21, abs=0.01)
    assert luxuries == pytest.approx(0.55, abs=0.02)


def test_the_picked_tiles_go_up_on_1_january_and_the_lowest_for_picks_not_made() -> None:
    tiles = Tiles()
    tiles.pick(Board.SUPPLY, Tile.WOOD)

    tiles.turn(FIRST_YEAR)

    raised = {tile for tile, level in tiles.supply.items() if level == 2}
    assert raised == {Tile.WOOD, Tile.COAL}
    assert tiles.picks[Board.SUPPLY] == []


def test_no_tile_can_go_more_than_two_levels_above_the_lowest() -> None:
    tiles = Tiles()
    tiles.pick(Board.SUPPLY, Tile.WOOD)
    tiles.pick(Board.SUPPLY, Tile.WOOD)
    tiles.turn(FIRST_YEAR)

    can = tiles.can_pick(Board.SUPPLY, Tile.WOOD)

    assert tiles.supply[Tile.WOOD] == 3
    assert not can


def test_only_two_supply_tiles_and_one_demand_tile_can_be_picked_a_year() -> None:
    tiles = Tiles()
    tiles.pick(Board.SUPPLY, Tile.WOOD)
    tiles.pick(Board.SUPPLY, Tile.COAL)
    tiles.pick(Board.DEMAND, Tile.WOOD)

    more = tiles.pick(Board.SUPPLY, Tile.ORE), tiles.pick(Board.DEMAND, Tile.ORE)

    assert more == (False, False)


def test_taking_a_pick_back_frees_it() -> None:
    tiles = Tiles()
    tiles.pick(Board.SUPPLY, Tile.WOOD)
    tiles.pick(Board.SUPPLY, Tile.WOOD)

    tiles.unpick(Board.SUPPLY, Tile.WOOD)

    assert tiles.picks[Board.SUPPLY] == [Tile.WOOD]


def test_a_closed_tile_cannot_be_picked_nor_holds_the_others_back() -> None:
    tiles = Tiles.until(2010)

    can = tiles.can_pick(Board.SUPPLY, Tile.COAL)

    assert tiles.supply[Tile.COAL] == CLOSES[Tile.COAL]
    assert not can
    assert min(tiles.supply[tile] for tile in tiles.open_tiles(Board.SUPPLY)) > CLOSES[Tile.COAL]


def test_an_eras_new_tiles_come_at_the_lowest_level_on_each_board() -> None:
    tiles = Tiles.until(1843)

    supply, demand = tiles.supply[Tile.STEEL], tiles.demand[Tile.STEEL]

    assert supply == min(tiles.supply[tile] for tile in TILE_GOODS if tile in tiles.supply)
    assert demand == min(tiles.demand.values())
    assert tiles.of_year(1843).make_factor(Good.STEEL) == 1.0


def test_the_era_brings_its_generation_of_wagons_and_news() -> None:
    tiles = Tiles.until(1842)

    news = tiles.turn(1843)

    assert tiles.of_year(1843).wagons == 2
    assert news == [
        (
            "A new era: wagons of generation 2 and new tiles: oil and fuel, steel and metals. "
            "From now on buildings make oil, steel, fuel, gas, castings."
        )
    ]


def test_picking_supply_lowers_what_a_tiles_goods_will_earn_and_demand_raises_it() -> None:
    tiles = Tiles.until(1850)
    now = tiles.of_year(1850).price(Good.WHEAT)

    supply = tiles.price_after(Board.SUPPLY, Tile.FARMING)
    demand = tiles.price_after(Board.DEMAND, Tile.FARMING)

    assert supply < now < demand


def test_buildings_make_and_homes_want_as_much_as_the_boards_of_the_game_in_play() -> None:
    boards = Tiles.until(1900)
    activate(boards)

    wheat = makes_of(BuildingKind.FARM, 1900)[Good.WHEAT]
    food = needs_of(BuildingKind.HOUSE, 1900)[Need.FOOD]

    assert wheat == pytest.approx(10 * boards.of_year(1900).make_factor(Good.WHEAT))
    assert food > needs_of(BuildingKind.HOUSE, 1700)[Need.FOOD]


def test_a_game_begins_on_the_boards_of_its_year_and_turns_them_every_1_january() -> None:
    game = Game(Constructions(flat_map(12, 12)), start_year=1850)
    before = make_factor(Good.COAL, 1850)
    game.railway.tiles.pick(Board.SUPPLY, Tile.COAL)

    for _ in range(12):
        game.month(np.random.default_rng(0))

    assert game.growth.year == 1851
    assert game.railway.tiles.supply[Tile.COAL] == Tiles.until(1850).supply[Tile.COAL] + 1
    assert make_factor(Good.COAL, 1851) > before
