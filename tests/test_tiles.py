import numpy as np
import pytest

from builders import flat_map
from railroads.sim.constructions import BuildingKind, Constructions
from railroads.sim.events import make_factor
from railroads.sim.game import Game
from railroads.sim.goods import Good, Need
from railroads.sim.tiles import (
    BASE_LEVELS,
    BENCHMARK_YEAR,
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
from railroads.sim.yearly import makes_of, needs_of


def test_every_good_belongs_to_exactly_one_tile() -> None:
    counted = [good for goods in TILE_GOODS.values() for good in goods]

    assert sorted(counted, key=lambda g: g.name) == sorted(Good, key=lambda g: g.name)
    assert len(TILE_OF) == len(Good)


def test_the_world_reaches_each_era_in_its_year_when_the_lowest_tiles_go_up() -> None:
    tiles = Tiles.until(2050)

    years = tiles.era_years

    assert years == {2: 1843, 3: 1873, 4: 1901, 5: 1933, 6: 1957, 7: 1974}


def test_every_supply_level_above_the_base_adds_its_periods_share_not_compounded() -> None:
    made = supply_factor(Tile.COAL, 1, 16)

    assert made == pytest.approx(1.0 + 3 * 0.15 + 2 * 0.18)


def test_every_supply_level_below_the_base_takes_away_its_share() -> None:
    made = supply_factor(Tile.WOOD, 1, 1)

    assert made == pytest.approx(1.0 - 9 * 0.04)


def test_every_demand_level_above_the_base_adds_the_same_share() -> None:
    wanted = demand_factor(Tile.FARMING, 1, 10)

    assert wanted == pytest.approx(1.0 + 4 * 0.31)


def test_every_demand_level_below_the_base_takes_away_its_share() -> None:
    wanted = demand_factor(Tile.MAIL, 1, 1)

    assert wanted == pytest.approx(1.0 - 4 * 0.09)


def test_a_tile_that_comes_after_the_benchmark_year_makes_its_base_at_the_level_it_comes_at() -> (
    None
):
    made, wanted = supply_factor(Tile.COPPER, 18, 18), demand_factor(Tile.COPPER, 9, 9)

    assert (made, wanted) == (1.0, 1.0)


def test_the_base_levels_are_those_of_the_benchmark_year_as_the_world_grows_by_itself() -> None:
    boards = Tiles.until(BENCHMARK_YEAR).of_year(BENCHMARK_YEAR)

    levels = {tile: (boards.supply[tile], boards.demand[tile]) for tile in boards.supply}

    assert levels == dict(BASE_LEVELS)


def test_every_good_earns_its_base_price_in_the_benchmark_year() -> None:
    boards = Tiles.until(BENCHMARK_YEAR).of_year(BENCHMARK_YEAR)

    earned = {tile: boards.price(TILE_GOODS[tile][0]) for tile in boards.supply}

    assert earned == {tile: pytest.approx(1.0) for tile in boards.supply}


def test_a_unit_earns_its_base_times_demand_over_supply_to_the_power_of_four_fifths() -> None:
    earned = price(4.0, 2.0)

    assert earned == pytest.approx(0.5**0.8)


def test_every_good_earns_its_base_price_before_the_first_year_of_the_boards() -> None:
    tiles = Tiles.until(1900)

    earned = tiles.of_year(FIRST_YEAR - 1).price(Good.COAL)

    assert earned == 1.0


def test_every_tile_makes_about_ten_times_and_is_wanted_about_five_times_its_base_by_2050() -> None:
    boards = Tiles.until(2050).of_year(2050)

    made, wanted = boards.made, boards.wanted

    assert made == {tile: pytest.approx(10.0, abs=0.1) for tile in Tile}
    assert wanted == {tile: pytest.approx(5.0, abs=0.1) for tile in Tile}


def test_every_good_earns_a_little_under_three_fifths_of_its_base_price_by_2050() -> None:
    boards = Tiles.until(2050).of_year(2050)

    earned = {tile: boards.price(TILE_GOODS[tile][0]) for tile in Tile}

    assert earned == {tile: pytest.approx(0.58, abs=0.015) for tile in Tile}


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


def test_coal_keeps_going_up_after_2000() -> None:
    before, after = Tiles.until(2000).supply[Tile.COAL], Tiles.until(2050).supply[Tile.COAL]

    grown = after - before

    assert grown >= 5


def test_an_eras_new_tiles_come_at_the_lowest_level_on_each_board() -> None:
    tiles = Tiles.until(1843)

    supply, demand = tiles.supply[Tile.STEEL], tiles.demand[Tile.STEEL]

    assert supply == min(tiles.supply[tile] for tile in TILE_GOODS if tile in tiles.supply)
    assert demand == min(tiles.demand.values())


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
