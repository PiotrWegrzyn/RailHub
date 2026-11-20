import numpy as np
import pytest

from builders import flat_map
from railroads.sim import economy as economy_module
from railroads.sim.clock import QUARTER, START_YEAR
from railroads.sim.constructions import BuildingKind
from railroads.sim.economy import (
    COLLECTION_RADIUS,
    SCARCE_WORTH,
    Economy,
    OwnedShare,
    Satisfaction,
    StationOrder,
    levels_of,
)
from railroads.sim.events import make_factor
from railroads.sim.goods import BONUS_PART, PIPELINE, Good, Need, makes_of, needs_of
from railroads.sim.grids import HexGrid
from railroads.sim.ladders import LOGGING_CAMPS, MINES
from railroads.sim.steps import finish
from railroads.sim.tilemap import Relief, Terrain, TileMap, Vegetation
from railroads.sim.tiles import Tiles, activate

STATION = (8, 10)
FARM_MONTH = makes_of(BuildingKind.FARM, START_YEAR)[Good.WHEAT]
HOUSE_FOOD = needs_of(BuildingKind.HOUSE, START_YEAR)[Need.FOOD]
HOUSE_MUST = HOUSE_FOOD - BONUS_PART[(BuildingKind.HOUSE, Need.FOOD)]


def make_map(size: int = 12) -> TileMap:
    grid = HexGrid(rows=size, cols=size, km_per_cell=2.0)
    terrain = np.full(grid.shape, Terrain.LAND, dtype=np.uint8)
    terrain[:, 0] = Terrain.SEA
    relief = np.full(grid.shape, Relief.FLAT, dtype=np.uint8)
    relief[:, -3:] = Relief.HILLS
    relief[:, -1] = Relief.MOUNTAINS
    vegetation = np.full(grid.shape, Vegetation.GRASS, dtype=np.uint8)
    vegetation[:6, 4:8] = Vegetation.FOREST
    return flat_map(grid=grid, terrain=terrain, vegetation=vegetation, relief=relief)


def counts(tile_map: TileMap, *placed: tuple[tuple[int, int], BuildingKind, int]) -> dict:
    grids = {kind: np.zeros(tile_map.grid.shape, dtype=np.int64) for kind in BuildingKind}
    for cell, kind, n in placed:
        grids[kind][cell] += n
    return grids


def level_at(economy: Economy, trade: object, cell: tuple[int, int], kind: BuildingKind) -> int:
    return int(trade.levels((kind,))[position(trade, economy, cell)])  # type: ignore[attr-defined]


def flat(economy: Economy, cell: tuple[int, int]) -> int:
    return int(np.ravel_multi_index(cell, economy.tile_map.grid.shape))


def position(trade: object, economy: Economy, cell: tuple[int, int]) -> int:
    return int(np.searchsorted(trade.cells, flat(economy, cell)))  # type: ignore[attr-defined]


def holding(room: float = 0.0, **goods: float) -> dict:
    brought = {Good[name]: amount for name, amount in goods.items()}
    return {STATION: StationOrder(brought, room, delivery=COLLECTION_RADIUS)}


def eaten_at(economy: Economy, trade: object, cell: tuple[int, int]) -> float:
    eaten, _ = trade.goods_got[(BuildingKind.HOUSE, Need.FOOD)]  # type: ignore[attr-defined]
    return float(eaten[position(trade, economy, cell)])


@pytest.mark.parametrize(
    ("house", "share"), [((8, 4), 1.0), ((8, 5), 13 / 21), ((8, 6), 8 / 21), ((8, 8), 3 / 21)]
)
def test_a_house_takes_less_of_a_farms_month_the_further_off_it_is_by_fibonacci_shares(
    house: tuple[int, int], share: float
) -> None:
    tile_map = make_map()
    economy = Economy(tile_map)
    placed = counts(tile_map, (house, BuildingKind.HOUSE, 3), ((8, 4), BuildingKind.FARM, 1))

    trade = economy.trade(placed)

    assert eaten_at(economy, trade, house) == pytest.approx(share * FARM_MONTH)


@pytest.mark.parametrize(("house", "share"), [((8, 10), 1 / 21), ((8, 11), 0.0)])
def test_a_house_six_cells_from_a_farm_takes_a_21st_of_its_month_and_one_further_none(
    house: tuple[int, int], share: float
) -> None:
    tile_map = make_map(size=24)
    economy = Economy(tile_map)
    placed = counts(tile_map, (house, BuildingKind.HOUSE, 3), ((8, 4), BuildingKind.FARM, 1))

    trade = economy.trade(placed)

    assert eaten_at(economy, trade, house) == pytest.approx(share * FARM_MONTH)


def test_farms_feed_the_nearer_house_first_and_the_farther_one_only_its_share() -> None:
    tile_map = make_map(size=24)
    economy = Economy(tile_map)
    placed = counts(
        tile_map,
        ((8, 4), BuildingKind.FARM, 4),
        ((8, 5), BuildingKind.HOUSE, 1),
        ((8, 9), BuildingKind.HOUSE, 1),
    )

    trade = economy.trade(placed)

    assert eaten_at(economy, trade, (8, 5)) == pytest.approx(HOUSE_FOOD)
    assert eaten_at(economy, trade, (8, 9)) == pytest.approx(2 / 21 * 4 * FARM_MONTH)


def test_nothing_a_farm_hands_out_is_lost_on_the_way() -> None:
    tile_map = make_map()
    economy = Economy(tile_map)
    placed = counts(tile_map, ((8, 4), BuildingKind.FARM, 1), ((8, 8), BuildingKind.HOUSE, 3))

    trade = economy.trade(placed)

    gave, _ = trade.goods_gave[(BuildingKind.FARM, Good.WHEAT)]
    assert gave[position(trade, economy, (8, 4))] == pytest.approx(eaten_at(economy, trade, (8, 8)))
    assert 0 < eaten_at(economy, trade, (8, 8)) < FARM_MONTH


def test_buildings_hand_out_their_own_goods_before_a_station_hands_out_what_trains_brought() -> (
    None
):
    tile_map = make_map()
    economy = Economy(tile_map)
    placed = counts(tile_map, ((8, 9), BuildingKind.HOUSE, 1), ((8, 9), BuildingKind.FARM, 2))

    trade = economy.trade(placed, holding(BREAD=20.0))

    assert eaten_at(economy, trade, (8, 9)) == pytest.approx(HOUSE_FOOD)
    assert trade.handed_for[STATION] == {}
    assert trade.local[Need.FOOD][position(trade, economy, (8, 9))] == pytest.approx(HOUSE_FOOD)


def test_a_station_hands_out_only_what_the_buildings_around_left_short() -> None:
    tile_map = make_map()
    economy = Economy(tile_map)
    placed = counts(tile_map, ((8, 9), BuildingKind.HOUSE, 2), ((8, 9), BuildingKind.FARM, 1))

    trade = economy.trade(placed, holding(BREAD=50.0))

    assert trade.handed_for[STATION] == {
        (Need.FOOD, Good.BREAD): pytest.approx(2 * HOUSE_FOOD - FARM_MONTH)
    }


def test_a_station_feeds_the_nearest_house_first_however_satisfied_it_was() -> None:
    tile_map = make_map()
    economy = Economy(tile_map)
    near, far = (8, 11), (8, 7)
    placed = counts(tile_map, (near, BuildingKind.HOUSE, 1), (far, BuildingKind.HOUSE, 1))
    economy.levels[flat(economy, near)] = Satisfaction.THRIVING
    economy.levels[flat(economy, far)] = Satisfaction.UNSATISFIED

    trade = economy.trade(placed, holding(BREAD=HOUSE_MUST))

    assert eaten_at(economy, trade, near) == pytest.approx(HOUSE_MUST)
    assert eaten_at(economy, trade, far) == 0.0


def test_a_station_shares_out_among_houses_as_near_by_how_short_they_are() -> None:
    tile_map = make_map()
    economy = Economy(tile_map)
    placed = counts(tile_map, ((8, 11), BuildingKind.HOUSE, 1), ((8, 9), BuildingKind.HOUSE, 3))

    trade = economy.trade(placed, holding(BREAD=2 * HOUSE_FOOD))

    assert eaten_at(economy, trade, (8, 11)) == pytest.approx(HOUSE_FOOD / 2)
    assert eaten_at(economy, trade, (8, 9)) == pytest.approx(3 * HOUSE_FOOD / 2)


def test_a_station_hands_out_the_freight_it_gathered_to_the_buildings_around_it() -> None:
    tile_map = make_map()
    economy = Economy(tile_map)
    placed = counts(tile_map, ((8, 9), BuildingKind.HOUSE, 1))
    order = StationOrder({}, 100.0, delivery=COLLECTION_RADIUS, stockpile={Good.WHEAT: 50.0})

    trade = economy.trade(placed, {STATION: order})

    assert eaten_at(economy, trade, (8, 9)) == pytest.approx(HOUSE_FOOD)
    assert trade.handed[STATION] == {Good.WHEAT: pytest.approx(HOUSE_FOOD)}


def test_a_farm_is_as_satisfied_as_the_share_of_its_wheat_that_was_taken() -> None:
    tile_map = make_map()
    economy = Economy(tile_map)
    placed = counts(tile_map, ((8, 4), BuildingKind.FARM, 1), ((8, 6), BuildingKind.HOUSE, 5))

    trade = economy.trade(placed)

    share = trade.share((BuildingKind.FARM,))[position(trade, economy, (8, 4))]
    assert share == pytest.approx(8 / 21)


def test_a_station_gathers_only_what_the_buildings_around_left() -> None:
    tile_map = make_map()
    economy = Economy(tile_map)
    placed = counts(tile_map, ((8, 9), BuildingKind.FARM, 4), ((8, 8), BuildingKind.HOUSE, 1))

    trade = economy.trade(placed, {STATION: StationOrder({}, 100.0)})

    assert eaten_at(economy, trade, (8, 8)) == pytest.approx(HOUSE_FOOD)
    assert trade.gathered[STATION] == {Good.WHEAT: pytest.approx(4 * FARM_MONTH - HOUSE_FOOD)}
    assert trade.handed_for[STATION] == {}


@pytest.mark.parametrize(
    ("share", "gathered"), [(0.0, 0.0), (0.5, 0.5 * FARM_MONTH), (1.0, FARM_MONTH)]
)
def test_a_station_hands_out_the_players_share_of_a_farm_they_own_as_if_a_train_brought_it(
    share: float, gathered: float
) -> None:
    tile_map = make_map()
    economy = Economy(tile_map)
    placed = counts(tile_map, ((8, 9), BuildingKind.FARM, 1), ((8, 9), BuildingKind.HOUSE, 5))
    owned = counts(tile_map, ((8, 9), BuildingKind.FARM, 1))[BuildingKind.FARM].ravel()
    owners = {BuildingKind.FARM: OwnedShare(owned.astype(np.float64), share)}

    trade = economy.trade(placed, {STATION: StationOrder({}, 100.0)}, owned=owners)

    assert trade.owned_brought[STATION].get(Good.WHEAT, 0.0) == pytest.approx(gathered)
    assert trade.handed[STATION].get(Good.WHEAT, 0.0) == pytest.approx(gathered)
    assert eaten_at(economy, trade, (8, 9)) == pytest.approx(FARM_MONTH)


@pytest.mark.parametrize(
    ("room", "level"), [(3.0, Satisfaction.CONTENT), (100.0, Satisfaction.THRIVING)]
)
def test_a_farm_is_as_satisfied_as_the_share_of_its_wheat_a_station_takes(
    room: float, level: Satisfaction
) -> None:
    tile_map = make_map()
    economy = Economy(tile_map)
    placed = counts(tile_map, ((8, 9), BuildingKind.FARM, 1))

    trade = economy.trade(placed, {STATION: StationOrder({}, room)})

    assert level_at(economy, trade, (8, 9), BuildingKind.FARM) == level


@pytest.mark.parametrize(
    ("planks", "steel", "goods"), [(0.0, 9.0, 0.0), (2.0, 9.0, 6.0), (9.0, 1.0, 3.0)]
)
def test_a_factory_makes_only_as_much_as_its_planks_and_steel_allow(
    planks: float, steel: float, goods: float
) -> None:
    tile_map = make_map()
    economy = Economy(tile_map)
    economy.year = 1855
    placed = counts(tile_map, ((8, 8), BuildingKind.FACTORY, 1))

    trade = economy.trade(placed, holding(100.0, PLANKS=planks, STEEL=steel))

    assert trade.gave[BuildingKind.FACTORY][position(trade, economy, (8, 8))] == pytest.approx(
        goods
    )


@pytest.mark.parametrize(
    ("share", "level"),
    [
        (0.0, Satisfaction.UNSATISFIED),
        (0.2, Satisfaction.CONTENT),
        (0.99, Satisfaction.CONTENT),
        (1.0, Satisfaction.THRIVING),
    ],
)
def test_satisfaction_goes_by_the_share_met(share: float, level: Satisfaction) -> None:
    levels = levels_of(np.array([share]))

    assert levels[0] == level


def test_logging_camps_only_grow_at_the_edge_of_a_forest() -> None:
    weights = LOGGING_CAMPS.land.weights(make_map())

    assert weights[2, 4] > 0
    assert weights[2, 5] == 0


def test_mines_like_hills_beside_mountains_best() -> None:
    weights = MINES.land.weights(make_map())

    assert weights[8, 10] > 2 * weights[8, 9]


def test_a_steel_mill_makes_nothing_with_nowhere_for_its_steel_and_keeps_its_inputs() -> None:
    tile_map = make_map()
    economy = Economy(tile_map)
    placed = counts(tile_map, ((8, 8), BuildingKind.STEEL_MILL, 1))

    trade = economy.trade(placed, holding(COAL=10.0, IRON=12.0))

    made, _ = trade.goods_gave[(BuildingKind.STEEL_MILL, Good.STEEL)]
    assert made[position(trade, economy, (8, 8))] == 0.0
    assert economy.stock[(BuildingKind.STEEL_MILL, Need.COAL)][flat(economy, (8, 8))] == 10.0
    assert economy.stock[(BuildingKind.STEEL_MILL, Need.IRON)][flat(economy, (8, 8))] == 12.0


def test_ten_coal_and_five_iron_ore_make_one_steel_once_a_station_takes_it() -> None:
    tile_map = make_map()
    economy = Economy(tile_map)
    placed = counts(tile_map, ((8, 8), BuildingKind.STEEL_MILL, 1))

    trade = economy.trade(placed, holding(100.0, COAL=10.0, IRON=5.0))

    assert trade.gathered[STATION][Good.STEEL] == pytest.approx(1.0)
    assert economy.stock[(BuildingKind.STEEL_MILL, Need.COAL)][flat(economy, (8, 8))] == 0.0


def test_inputs_wait_in_a_stockpile_of_three_hundred_until_the_other_input_comes() -> None:
    tile_map = make_map()
    economy = Economy(tile_map)
    placed = counts(tile_map, ((8, 8), BuildingKind.STEEL_MILL, 1))

    for _ in range(5):
        economy.trade(placed, holding(COAL=100.0))

    assert economy.stock[(BuildingKind.STEEL_MILL, Need.COAL)][flat(economy, (8, 8))] == 300.0


def test_nine_fruit_and_three_steel_make_thirty_cans() -> None:
    tile_map = make_map()
    economy = Economy(tile_map)
    economy.stock[(BuildingKind.FOOD_PLANT, Need.FRUIT)][flat(economy, (8, 8))] = 9.0
    economy.stock[(BuildingKind.FOOD_PLANT, Need.STEEL)][flat(economy, (8, 8))] = 3.0
    placed = counts(tile_map, ((8, 8), BuildingKind.FOOD_PLANT, 1))

    trade = economy.trade(placed, {STATION: StationOrder({}, 100.0)})

    assert trade.gathered[STATION][Good.CANS] == pytest.approx(30.0)
    assert economy.stock[(BuildingKind.FOOD_PLANT, Need.FRUIT)][flat(economy, (8, 8))] == 0.0


def test_houses_eat_raw_fruit_as_food() -> None:
    tile_map = make_map()
    economy = Economy(tile_map)
    placed = counts(tile_map, ((8, 9), BuildingKind.HOUSE, 1))

    trade = economy.trade(placed, holding(APPLES=20.0))

    eaten, needed = trade.goods_got[(BuildingKind.HOUSE, Need.FOOD)]
    at = position(trade, economy, (8, 9))
    assert eaten[at] == needed[at] == HOUSE_FOOD


def test_canneries_take_their_fruit_before_houses_eat_it() -> None:
    tile_map = make_map()
    economy = Economy(tile_map)
    economy.year = 1855
    placed = counts(tile_map, ((8, 9), BuildingKind.HOUSE, 1), ((8, 8), BuildingKind.FOOD_PLANT, 1))

    trade = economy.trade(placed, holding(APPLES=3.0))

    eaten, _ = trade.goods_got[(BuildingKind.HOUSE, Need.FOOD)]
    assert eaten[position(trade, economy, (8, 9))] == 0.0
    assert economy.stock[(BuildingKind.FOOD_PLANT, Need.FRUIT)][flat(economy, (8, 8))] == 3.0


@pytest.mark.parametrize(("year", "goods"), [(1842, 10.0), (1843, 0.0)])
def test_a_factory_works_on_planks_alone_until_steel_comes_with_its_era(
    year: int, goods: float
) -> None:
    tile_map = make_map()
    economy = Economy(tile_map)
    economy.year = year
    placed = counts(tile_map, ((8, 8), BuildingKind.FACTORY, 1))

    trade = economy.trade(placed, holding(100.0, PLANKS=10.0 / 3))

    made = trade.gave[BuildingKind.FACTORY][position(trade, economy, (8, 8))]
    assert made == pytest.approx(goods)


def test_a_station_counts_what_it_hands_out_of_what_trains_brought_by_need() -> None:
    tile_map = make_map()
    economy = Economy(tile_map)
    placed = counts(tile_map, ((8, 9), BuildingKind.HOUSE, 4))

    trade = economy.trade(placed, holding(WHEAT=5.0))

    assert trade.handed_for[STATION] == {(Need.FOOD, Good.WHEAT): 5.0}
    assert trade.handed[STATION] == {Good.WHEAT: 5.0}


def test_a_bakery_bakes_each_unit_of_wheat_into_two_and_a_half_of_bread() -> None:
    tile_map = make_map()
    economy = Economy(tile_map)
    placed = counts(tile_map, ((8, 8), BuildingKind.BAKERY, 1))

    trade = economy.trade(placed, holding(100.0, WHEAT=5.0))

    assert trade.gathered[STATION][Good.BREAD] == pytest.approx(12.5)
    assert economy.stock[(BuildingKind.BAKERY, Need.WHEAT)][flat(economy, (8, 9))] == 0.0


def test_bakeries_take_their_wheat_before_houses_eat_it_and_bread_feeds_houses() -> None:
    tile_map = make_map()
    economy = Economy(tile_map)
    placed = counts(tile_map, ((8, 9), BuildingKind.HOUSE, 1), ((8, 9), BuildingKind.BAKERY, 1))

    trade = economy.trade(placed, holding(WHEAT=3.0, BREAD=20.0))

    baked = 3.0 / 0.4
    assert economy.stock[(BuildingKind.BAKERY, Need.WHEAT)][flat(economy, (8, 8))] == pytest.approx(
        0.0
    )
    assert eaten_at(economy, trade, (8, 9)) == pytest.approx(HOUSE_FOOD)
    assert trade.handed_for[STATION][(Need.FOOD, Good.BREAD)] == pytest.approx(HOUSE_FOOD - baked)


@pytest.mark.parametrize(("year", "goods"), [(1900, 30.0), (1901, 6.0)])
def test_a_big_factory_works_on_steel_alone_until_copper_comes_with_its_era(
    year: int, goods: float
) -> None:
    tile_map = make_map()
    economy = Economy(tile_map)
    economy.year = year
    placed = counts(tile_map, ((8, 8), BuildingKind.BIG_FACTORY, 1))

    trade = economy.trade(placed, holding(100.0, STEEL=10.0, COPPER=2.0))

    assert trade.gave[BuildingKind.BIG_FACTORY][position(trade, economy, (8, 8))] == pytest.approx(
        goods
    )


def test_farms_take_limestone_machinery_fertiliser_and_diesel_as_their_demand_tiles_open_them() -> (
    None
):
    activate(Tiles.until(1925))

    taken = [set(needs_of(BuildingKind.FARM, year)) for year in (1805, 1820, 1910, 1925)]

    assert taken == [
        set(),
        {Need.LIMESTONE},
        {Need.LIMESTONE, Need.MACHINERY, Need.FERTILISER},
        {Need.LIMESTONE, Need.MACHINERY, Need.FERTILISER, Need.FUEL},
    ]


@pytest.mark.parametrize(
    ("kind", "inputs", "made", "amount"),
    [
        (BuildingKind.BRICKWORKS, {"CLAY": 10.0}, Good.BRICKS, 20.0),
        (BuildingKind.BRICKWORKS, {"CLAY": 10.0, "COAL": 5.0}, Good.BRICKS, 40.0),
        (BuildingKind.TANNERY, {"CATTLE": 10.0}, Good.LEATHER, 20.0),
        (BuildingKind.SHOE_FACTORY, {"LEATHER": 10.0}, Good.SHOES, 5.0),
        (BuildingKind.GLASSWORKS, {"SAND": 10.0}, Good.GLASS, 20.0),
        (BuildingKind.GLASSWORKS, {"SAND": 10.0, "COAL": 5.0}, Good.GLASS, 30.0),
        (BuildingKind.GLASSWORKS, {"SAND": 10.0, "COAL": 5.0, "LIMESTONE": 5.0}, Good.GLASS, 40.0),
        (BuildingKind.TANNERY, {"CATTLE": 10.0, "LIMESTONE": 5.0}, Good.LEATHER, 40.0),
        (BuildingKind.PRINTING_WORKS, {"PAPER": 4.0}, Good.NEWSPAPERS, 80.0),
    ],
)
def test_new_works_turn_their_inputs_into_their_goods(
    kind: BuildingKind, inputs: dict[str, float], made: Good, amount: float
) -> None:
    tile_map = make_map()
    economy = Economy(tile_map)
    placed = counts(tile_map, ((8, 8), kind, 1))

    trade = economy.trade(placed, holding(100.0, **inputs))

    assert trade.gathered[STATION][made] == pytest.approx(amount)


def test_a_brewery_takes_its_hops_and_its_beer_quenches_houses() -> None:
    tile_map = make_map()
    economy = Economy(tile_map)
    placed = counts(tile_map, ((8, 9), BuildingKind.HOUSE, 1), ((8, 8), BuildingKind.BREWERY, 1))

    trade = economy.trade(placed, holding(HOPS=3.0, BEER=1.0))

    drunk, _ = trade.goods_got[(BuildingKind.HOUSE, Need.BEER)]
    assert drunk[position(trade, economy, (8, 9))] == 1.0


@pytest.mark.parametrize(("barrels", "beer"), [(0.0, 10.0), (2.0, 15.0), (4.0, 20.0), (9.0, 20.0)])
def test_barrels_let_a_brewery_make_up_to_twice_the_beer_from_its_hops(
    barrels: float, beer: float
) -> None:
    tile_map = make_map()
    economy = Economy(tile_map)
    placed = counts(tile_map, ((8, 8), BuildingKind.BREWERY, 1))

    trade = economy.trade(placed, holding(100.0, HOPS=10.0, BARRELS=barrels))

    made = trade.gathered[STATION].get(Good.BEER, 0.0)
    assert made == pytest.approx(beer)


def test_a_brewery_makes_no_beer_from_barrels_without_hops() -> None:
    tile_map = make_map()
    economy = Economy(tile_map)
    placed = counts(tile_map, ((8, 8), BuildingKind.BREWERY, 1))

    trade = economy.trade(placed, holding(100.0, BARRELS=10.0))

    assert trade.gathered[STATION].get(Good.BEER, 0.0) == 0.0


@pytest.mark.parametrize(("wheat", "cattle"), [(0.0, 10.0), (60.0, 15.0), (120.0, 20.0)])
def test_wheat_lets_a_cattle_ranch_raise_up_to_twice_the_cattle(
    wheat: float, cattle: float
) -> None:
    tile_map = make_map()
    economy = Economy(tile_map)
    placed = counts(tile_map, ((8, 8), BuildingKind.CATTLE_RANCH, 1))

    trade = economy.trade(placed, holding(100.0, WHEAT=wheat))

    assert trade.gathered[STATION].get(Good.CATTLE, 0.0) == pytest.approx(cattle)


def test_flax_spins_into_clothes_like_wool_and_cotton() -> None:
    tile_map = make_map()
    economy = Economy(tile_map)
    placed = counts(tile_map, ((8, 8), BuildingKind.TEXTILE_MILL, 1))

    trade = economy.trade(placed, holding(100.0, FLAX=10.0))

    assert trade.gathered[STATION][Good.CLOTHES] == pytest.approx(4.0)


@pytest.mark.parametrize("animal", ["CATTLE", "PIGS", "CHICKENS", "SHEEP"])
def test_a_butcher_turns_each_animal_into_two_and_a_half_of_meat(animal: str) -> None:
    tile_map = make_map()
    economy = Economy(tile_map)
    placed = counts(tile_map, ((8, 8), BuildingKind.MEAT_FACTORY, 1))

    trade = economy.trade(placed, holding(100.0, **{animal: 10.0}))

    assert trade.gathered[STATION][Good.MEAT] == pytest.approx(25.0)


@pytest.mark.parametrize(
    ("kind", "held", "made", "most", "need", "left"),
    [
        (BuildingKind.BAKERY, {"WHEAT": 80.0}, Good.BREAD, 120.0, Need.WHEAT, 32.0),
        (BuildingKind.MEAT_FACTORY, {"CATTLE": 80.0}, Good.MEAT, 120.0, Need.ANIMALS, 32.0),
        (BuildingKind.DAIRY, {"MILK": 80.0}, Good.DAIRY, 120.0, Need.MILK, 32.0),
        (
            BuildingKind.FOOD_PLANT,
            {"APPLES": 130.0, "STEEL": 80.0},
            Good.CANS,
            300.0,
            Need.FRUIT,
            40.0,
        ),
        (
            BuildingKind.SAUSAGE_MAKER,
            {"MEAT": 130.0, "SALT": 80.0},
            Good.SAUSAGES,
            240.0,
            Need.MEAT,
            34.0,
        ),
    ],
)
def test_a_food_processor_makes_at_most_its_month_of_food_from_its_main_input(
    kind: BuildingKind, held: dict[str, float], made: Good, most: float, need: Need, left: float
) -> None:
    tile_map = make_map()
    economy = Economy(tile_map)
    placed = counts(tile_map, ((8, 8), kind, 1))

    trade = economy.trade(placed, holding(1000.0, **held))

    assert trade.gathered[STATION][made] == pytest.approx(most)
    assert economy.stock[(kind, need)][flat(economy, (8, 8))] == pytest.approx(left)


@pytest.mark.parametrize(
    ("kind", "wheat"), [(BuildingKind.CATTLE_RANCH, 120), (BuildingKind.SHEEP_FARM, 90)]
)
def test_herds_eat_a_lot_of_wheat(kind: BuildingKind, wheat: int) -> None:
    tile_map = make_map()
    economy = Economy(tile_map)
    placed = counts(tile_map, ((8, 9), kind, 1))

    trade = economy.trade(placed, holding(WHEAT=200.0))

    eaten, needed = trade.goods_got[(kind, Need.WHEAT)]
    at = position(trade, economy, (8, 9))
    assert eaten[at] == needed[at] == wheat


def test_houses_eat_meat_as_food() -> None:
    tile_map = make_map()
    economy = Economy(tile_map)
    placed = counts(tile_map, ((8, 9), BuildingKind.HOUSE, 1))

    trade = economy.trade(placed, holding(MEAT=20.0))

    eaten, _ = trade.goods_got[(BuildingKind.HOUSE, Need.FOOD)]
    assert eaten[position(trade, economy, (8, 9))] == HOUSE_FOOD


def test_stations_hand_out_the_most_valuable_food_first() -> None:
    tile_map = make_map()
    economy = Economy(tile_map)
    placed = counts(tile_map, ((8, 9), BuildingKind.HOUSE, 1))
    order = StationOrder({Good.WHEAT: 20.0, Good.MEAT: 20.0}, 0.0)

    trade = economy.trade(placed, {STATION: order})

    assert trade.handed_for[STATION] == {(Need.FOOD, Good.MEAT): HOUSE_FOOD}


@pytest.mark.parametrize(("year", "planes"), [(1956, 2.0), (1957, 1.0)])
def test_aircraft_factories_want_electronics_once_their_era_comes(year: int, planes: float) -> None:
    tile_map = make_map()
    economy = Economy(tile_map)
    economy.year = year
    placed = counts(tile_map, ((8, 8), BuildingKind.PLANE_FACTORY, 1))

    trade = economy.trade(placed, holding(100.0, STEEL=6.0, COPPER=2.0, FUEL=2.0, ELECTRONICS=1.0))

    assert trade.gathered[STATION][Good.PLANES] == pytest.approx(planes)


@pytest.mark.parametrize("fuel", ["LOGS", "COAL"])
def test_houses_heat_with_logs_or_coal(fuel: str) -> None:
    tile_map = make_map()
    economy = Economy(tile_map)
    placed = counts(tile_map, ((8, 9), BuildingKind.HOUSE, 1))

    trade = economy.trade(placed, holding(**{fuel: 5.0}))

    burned, _ = trade.goods_got[(BuildingKind.HOUSE, Need.HEATING)]
    assert burned[position(trade, economy, (8, 9))] == 3.0


@pytest.mark.parametrize(
    ("year", "made"),
    [(1949, {Good.GOODS, Good.OIL}), (1950, {Good.GOODS, Good.OIL, Good.GAS})],
)
def test_ports_bring_in_gas_only_once_there_is_any_in_1950(year: int, made: set) -> None:
    makes = makes_of(BuildingKind.PORT, year)

    assert set(makes) == made


@pytest.mark.parametrize(
    ("kind", "need", "wanted"),
    [
        (BuildingKind.POWER_PLANT, Need.COAL, 200.0),
        (BuildingKind.DIESEL_POWER_PLANT, Need.FUEL, 100.0),
        (BuildingKind.GAS_POWER_PLANT, Need.GAS, 150.0),
    ],
)
def test_each_kind_of_power_plant_burns_only_its_own_fuel(
    kind: BuildingKind, need: Need, wanted: float
) -> None:
    tile_map = make_map()
    economy = Economy(tile_map)
    economy.year = 1950
    placed = counts(tile_map, ((8, 9), kind, 1))

    trade = economy.trade(placed, holding(COAL=50.0, FUEL=50.0, GAS=50.0, OIL=50.0))

    burned, needed = trade.goods_got[(kind, need)]
    at = position(trade, economy, (8, 9))
    assert burned[at] == 50.0
    assert needed[at] == wanted


def test_bonus_goods_raise_a_fed_warm_and_furnished_house_past_full_satisfaction() -> None:
    tile_map = make_map()
    economy = Economy(tile_map)
    placed = counts(tile_map, ((8, 9), BuildingKind.HOUSE, 1))

    trade = economy.trade(
        placed, holding(FISH=HOUSE_FOOD, COAL=3.0, POTS=5.0, BEER=1.0, NEWSPAPERS=0.5)
    )

    share = trade.share((BuildingKind.HOUSE,))[position(trade, economy, (8, 9))]
    assert share == pytest.approx(1 + 0.3 * (1 + 1 + 1 + 0.5))
    assert level_at(economy, trade, (8, 9), BuildingKind.HOUSE) == Satisfaction.THRIVING


@pytest.mark.parametrize("given", [{"BEER": 1.0}, {"LOGS": 3.0}, {"COAL": 3.0, "BEER": 1.0}])
def test_a_house_that_gets_no_food_is_unsatisfied_whatever_else_it_gets(given: dict) -> None:
    tile_map = make_map()
    economy = Economy(tile_map)
    placed = counts(tile_map, ((8, 9), BuildingKind.HOUSE, 1))

    trade = economy.trade(placed, holding(**given))

    share = trade.share((BuildingKind.HOUSE,))[position(trade, economy, (8, 9))]
    assert share == 0.0
    assert level_at(economy, trade, (8, 9), BuildingKind.HOUSE) == Satisfaction.UNSATISFIED


def test_a_house_with_a_little_food_gets_its_beer_bonus_only_on_that_little() -> None:
    tile_map = make_map()
    economy = Economy(tile_map)
    placed = counts(tile_map, ((8, 9), BuildingKind.HOUSE, 1))

    trade = economy.trade(placed, holding(FISH=1.0, COAL=3.0, BEER=1.0))

    share = trade.share((BuildingKind.HOUSE,))[position(trade, economy, (8, 9))]
    assert share == pytest.approx(4 / 13 * 1.3)


def test_each_bonus_good_counts_what_the_economy_is_told_it_is_worth() -> None:
    tile_map = make_map()
    economy = Economy(tile_map)
    economy.bonus_per_good = 0.4
    placed = counts(tile_map, ((8, 9), BuildingKind.HOUSE, 1))

    trade = economy.trade(placed, holding(FISH=10.0, COAL=3.0, POTS=5.0, BEER=1.0))

    assert trade.share((BuildingKind.HOUSE,))[position(trade, economy, (8, 9))] == pytest.approx(
        1 + 0.4 * 2
    )


def test_a_station_full_of_one_good_still_gathers_another() -> None:
    tile_map = make_map()
    economy = Economy(tile_map)
    placed = counts(tile_map, ((8, 9), BuildingKind.FARM, 1))

    trade = economy.trade(placed, holding(100.0, COAL=100.0))

    assert trade.gathered[STATION][Good.WHEAT] > 0
    assert level_at(economy, trade, (8, 9), BuildingKind.FARM) == Satisfaction.THRIVING


def test_a_station_full_of_a_good_gathers_no_more_of_it() -> None:
    tile_map = make_map()
    economy = Economy(tile_map)
    placed = counts(tile_map, ((8, 9), BuildingKind.FARM, 1))

    trade = economy.trade(placed, holding(100.0, WHEAT=100.0))

    assert trade.gathered[STATION].get(Good.WHEAT, 0.0) == 0.0


@pytest.mark.parametrize(("planks", "coal"), [(0.0, 100.0), (10.0, 150.0), (20.0, 200.0)])
def test_planks_for_pit_props_lets_a_mine_dig_up_to_twice_the_coal(
    planks: float, coal: float
) -> None:
    tile_map = make_map()
    economy = Economy(tile_map)
    placed = counts(tile_map, ((8, 9), BuildingKind.MINE, 1))

    trade = economy.trade(placed, holding(1000.0, PLANKS=planks))

    assert trade.gathered[STATION].get(Good.COAL, 0.0) == pytest.approx(coal)


def test_gasworks_make_town_gas_out_of_coal() -> None:
    tile_map = make_map()
    economy = Economy(tile_map)
    placed = counts(tile_map, ((8, 8), BuildingKind.GASWORKS, 1))

    trade = economy.trade(placed, holding(1000.0, COAL=50.0))

    assert trade.gathered[STATION][Good.GAS] == pytest.approx(100.0)


def test_food_taken_at_a_station_after_its_need_was_served_counts_for_the_next_month(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(economy_module, "STEP_SECONDS", -1.0)
    tile_map = make_map()
    economy = Economy(tile_map)
    placed = counts(tile_map, ((8, 9), BuildingKind.HOUSE, 1))
    steps = economy.trade_steps(placed)
    needs = [step for step in PIPELINE if isinstance(step, Need)]
    for _ in range(2 + needs.index(Need.FOOD) + 1):
        next(steps)
    late = {(BuildingKind.HOUSE, Need.FOOD): {flat(economy, (8, 9)): 4.0}}
    economy.taken = late

    finish(steps)

    assert economy.taken == late


@pytest.mark.parametrize("good", ["SHOES", "CLOTHES", "FURNITURE", "POTS"])
def test_any_common_good_meets_a_homes_need_for_common_goods(good: str) -> None:
    tile_map = make_map()
    economy = Economy(tile_map)
    placed = counts(tile_map, ((8, 9), BuildingKind.HOUSE, 1))

    trade = economy.trade(placed, holding(**{good: 10.0}))

    got, wanted = trade.goods_got[(BuildingKind.HOUSE, Need.COMMON)]
    at = position(trade, economy, (8, 9))
    assert got[at] == wanted[at] == 5.0


@pytest.mark.parametrize(
    ("held", "juice"),
    [
        ({"APPLES": 1.0, "GLASS": 1.0}, 2.0),
        ({"APPLES": 1.0, "PLASTICS": 1.0}, 4.0),
        ({"APPLES": 1.0, "GLASS": 1.0, "REFINED_SUGAR": 5.0}, 3.0),
        ({"APPLES": 1.0, "PLASTICS": 1.0, "REFINED_SUGAR": 5.0}, 6.0),
    ],
)
def test_a_juice_works_bottles_fruit_in_glass_twice_as_far_in_plastic_and_more_with_sugar(
    held: dict[str, float], juice: float
) -> None:
    tile_map = make_map()
    economy = Economy(tile_map)
    economy.year = 1960
    placed = counts(tile_map, ((8, 8), BuildingKind.JUICE_WORKS, 1))

    trade = economy.trade(placed, holding(100.0, **held))

    assert trade.gathered[STATION][Good.JUICE] == pytest.approx(juice)


def test_plastic_bottles_only_once_there_is_plastic_in_its_era() -> None:
    tile_map = make_map()
    economy = Economy(tile_map)
    economy.year = 1956
    placed = counts(tile_map, ((8, 8), BuildingKind.JUICE_WORKS, 1))

    trade = economy.trade(placed, holding(100.0, APPLES=1.0, PLASTICS=1.0))

    assert trade.gathered[STATION].get(Good.JUICE, 0.0) == 0.0


@pytest.mark.parametrize(("sugar", "bread"), [(0.0, 12.5), (1.0, 18.75)])
def test_refined_sugar_lets_a_bakery_bake_half_as_much_bread_again(
    sugar: float, bread: float
) -> None:
    tile_map = make_map()
    economy = Economy(tile_map)
    placed = counts(tile_map, ((8, 8), BuildingKind.BAKERY, 1))

    trade = economy.trade(placed, holding(100.0, WHEAT=5.0, REFINED_SUGAR=sugar))

    assert trade.gathered[STATION][Good.BREAD] == pytest.approx(bread)


def test_a_house_fed_only_this_month_counts_as_a_third_satisfied_over_the_last_quarter() -> None:
    tile_map = make_map()
    economy = Economy(tile_map)
    alone = counts(tile_map, ((5, 5), BuildingKind.HOUSE, 1))
    fed = counts(tile_map, ((5, 5), BuildingKind.HOUSE, 1), ((5, 6), BuildingKind.FARM, 1))
    for _ in range(QUARTER):
        economy.trade(alone)

    trade = economy.trade(fed)

    house = trade.share((BuildingKind.HOUSE,))[position(trade, economy, (5, 5))]
    assert house > 0
    assert economy.last_quarter()[flat(economy, (5, 5))] == pytest.approx(house / QUARTER)


def test_a_buyer_starving_over_the_last_quarter_makes_what_it_is_handed_worth_the_most() -> None:
    tile_map = make_map()
    economy = Economy(tile_map)
    house = counts(tile_map, ((5, 5), BuildingKind.HOUSE, 1))
    fed = counts(tile_map, ((5, 5), BuildingKind.HOUSE, 1), ((5, 6), BuildingKind.FARM, 1))
    for _ in range(QUARTER):
        economy.trade(house)

    trade = economy.trade(fed)

    assert trade.worth[Good.WHEAT][position(trade, economy, (5, 6))] == pytest.approx(SCARCE_WORTH)


@pytest.mark.parametrize(
    ("takes", "owned", "wheat"),
    [
        (frozenset(), 1, FARM_MONTH),
        (frozenset({Need.LIMESTONE}), 0, FARM_MONTH),
        (frozenset({Need.LIMESTONE}), 1, 1.5 * FARM_MONTH),
        (frozenset({Need.MACHINERY}), 1, 3.0 * FARM_MONTH),
        (frozenset({Need.LIMESTONE, Need.MACHINERY}), 1, 3.5 * FARM_MONTH),
    ],
)
def test_only_a_farm_the_player_owns_takes_limestone_and_machinery_once_their_tech_opens_them(
    takes: frozenset[Need], owned: int, wheat: float
) -> None:
    tile_map = make_map()
    economy = Economy(tile_map)
    economy.year = 1900
    placed = counts(tile_map, (STATION, BuildingKind.FARM, 1))
    mine = counts(tile_map, (STATION, BuildingKind.FARM, owned))[BuildingKind.FARM].ravel()
    owners = {BuildingKind.FARM: OwnedShare(mine.astype(np.float64), 0.3, takes)}

    trade = economy.trade(placed, holding(1000.0, LIMESTONE=50.0, MACHINERY=50.0), owned=owners)

    gathered = trade.gathered[STATION].get(Good.WHEAT, 0.0)
    brought = trade.owned_brought[STATION].get(Good.WHEAT, 0.0)
    assert gathered + brought == pytest.approx(wheat * make_factor(Good.WHEAT, 1900))


@pytest.mark.parametrize("drink", ["BEER", "WINE", "VODKA"])
def test_a_tavern_takes_any_alcohol_and_food_only_as_a_bonus(drink: str) -> None:
    tile_map = make_map()
    economy = Economy(tile_map)
    placed = counts(tile_map, ((8, 9), BuildingKind.TAVERN, 1))

    trade = economy.trade(placed, holding(**{drink: 90.0}))

    share = trade.share((BuildingKind.TAVERN,))[position(trade, economy, (8, 9))]
    assert share == pytest.approx(1.0)


@pytest.mark.parametrize(("owned_at", "fed_at"), [(None, (8, 9)), ((8, 7), (8, 7))])
def test_a_station_hands_out_to_the_players_own_buildings_before_nearer_ones(
    owned_at: tuple[int, int] | None, fed_at: tuple[int, int]
) -> None:
    tile_map = make_map()
    economy = Economy(tile_map)
    placed = counts(tile_map, ((8, 9), BuildingKind.BAKERY, 1), ((8, 7), BuildingKind.BAKERY, 1))
    mine = np.zeros(tile_map.grid.shape)
    if owned_at is not None:
        mine[owned_at] = 1.0
    owners = {BuildingKind.BAKERY: OwnedShare(mine.ravel(), 1.0)}

    trade = economy.trade(placed, holding(WHEAT=50.0), owned=owners)

    got, _ = trade.goods_got[(BuildingKind.BAKERY, Need.WHEAT)]
    assert got[position(trade, economy, fed_at)] == pytest.approx(50.0)


@pytest.mark.parametrize(("satisfaction", "worth"), [(0.0, SCARCE_WORTH), (0.5, 2.0), (1.0, 1.0)])
def test_what_a_station_hands_out_is_worth_more_the_less_satisfied_its_buyer_was(
    satisfaction: float, worth: float
) -> None:
    tile_map = make_map()
    economy = Economy(tile_map)
    economy.months = [np.full(economy.cells, satisfaction) for _ in range(QUARTER)]
    placed = counts(tile_map, ((8, 9), BuildingKind.HOUSE, 1))

    trade = economy.trade(placed, holding(FISH=5.0))

    handed = trade.handed_for[STATION][(Need.FOOD, Good.FISH)]
    assert trade.handed_worth[STATION][(Need.FOOD, Good.FISH)] == pytest.approx(worth * handed)


WOOL_MONTH = makes_of(BuildingKind.SHEEP_FARM, 1900)[Good.WOOL]


@pytest.mark.parametrize(
    ("wheat", "takes", "wool"),
    [
        (0.0, frozenset(), WOOL_MONTH),
        (90.0, frozenset(), 2.0 * WOOL_MONTH),
        (0.0, frozenset({Need.MACHINERY}), 2.0 * WOOL_MONTH),
        (90.0, frozenset({Need.MACHINERY}), 3.0 * WOOL_MONTH),
    ],
)
def test_wheat_doubles_a_sheep_farms_wool_and_machinery_adds_as_much_again_to_one_the_player_owns(
    wheat: float, takes: frozenset[Need], wool: float
) -> None:
    tile_map = make_map()
    economy = Economy(tile_map)
    economy.year = 1900
    placed = counts(tile_map, (STATION, BuildingKind.SHEEP_FARM, 1))
    mine = placed[BuildingKind.SHEEP_FARM].ravel().astype(np.float64)
    owners = {BuildingKind.SHEEP_FARM: OwnedShare(mine, 0.3, takes)}

    trade = economy.trade(placed, holding(1000.0, WHEAT=wheat, MACHINERY=50.0), owned=owners)

    gathered = trade.gathered[STATION].get(Good.WOOL, 0.0)
    brought = trade.owned_brought[STATION].get(Good.WOOL, 0.0)
    assert gathered + brought == pytest.approx(wool)


def test_a_farm_the_player_controls_takes_machinery_whole_and_they_get_their_part_of_it() -> None:
    tile_map = make_map()
    economy = Economy(tile_map)
    economy.year = 1900
    placed = counts(tile_map, (STATION, BuildingKind.FARM, 1))
    one = placed[BuildingKind.FARM].ravel().astype(np.float64)
    owners = {BuildingKind.FARM: OwnedShare(0.6 * one, 1.0, frozenset({Need.MACHINERY}), one)}

    trade = economy.trade(placed, holding(1000.0, MACHINERY=50.0), owned=owners)

    gathered = trade.gathered[STATION].get(Good.WHEAT, 0.0)
    brought = trade.owned_brought[STATION].get(Good.WHEAT, 0.0)
    rise = make_factor(Good.WHEAT, 1900)
    assert gathered + brought == pytest.approx(3.0 * FARM_MONTH * rise)
    assert brought == pytest.approx(0.6 * 3.0 * FARM_MONTH * rise)


@pytest.mark.parametrize(
    ("logs", "planks", "chips"), [(0.0, 0.0, 0.0), (20.0, 20.0, 4.0), (100.0, 40.0, 8.0)]
)
def test_a_sawmill_saws_a_plank_from_every_log_and_a_load_of_chips_from_every_five(
    logs: float, planks: float, chips: float
) -> None:
    tile_map = make_map()
    economy = Economy(tile_map)
    placed = counts(tile_map, ((8, 8), BuildingKind.SAWMILL, 1))

    trade = economy.trade(placed, holding(1000.0, LOGS=logs))

    assert trade.gathered[STATION].get(Good.PLANKS, 0.0) == pytest.approx(planks)
    assert trade.gathered[STATION].get(Good.WOOD_CHIPS, 0.0) == pytest.approx(chips)


@pytest.mark.parametrize(
    ("machinery", "fuel", "planks"),
    [(0.0, 0.0, 40.0), (50.0, 0.0, 80.0), (0.0, 50.0, 60.0), (50.0, 50.0, 100.0)],
)
def test_machinery_and_diesel_let_a_sawmill_saw_more_logs(
    machinery: float, fuel: float, planks: float
) -> None:
    tile_map = make_map()
    economy = Economy(tile_map)
    economy.year = 1925
    placed = counts(tile_map, ((8, 8), BuildingKind.SAWMILL, 1))
    economy.trade(placed, holding(1000.0, LOGS=200.0, MACHINERY=machinery, FUEL=fuel))

    trade = economy.trade(placed, holding(1000.0, LOGS=200.0))

    assert trade.gathered[STATION].get(Good.PLANKS, 0.0) == pytest.approx(planks)
    assert trade.gathered[STATION].get(Good.WOOD_CHIPS, 0.0) == pytest.approx(planks / 5)


def test_a_sawmill_keeps_only_three_months_of_machinery() -> None:
    tile_map = make_map()
    economy = Economy(tile_map)
    economy.year = 1925
    placed = counts(tile_map, ((8, 8), BuildingKind.SAWMILL, 1))

    trade = economy.trade(placed, holding(1000.0, LOGS=40.0, MACHINERY=50.0))

    took, _ = trade.goods_got[(BuildingKind.SAWMILL, Need.MACHINERY)]
    assert took[position(trade, economy, (8, 8))] == pytest.approx(15.0)


@pytest.mark.parametrize(
    ("year", "machinery", "logs"), [(1872, 5.0, 20.0), (1873, 0.0, 20.0), (1873, 5.0, 40.0)]
)
def test_machinery_lets_any_logging_camp_fell_twice_the_logs_once_its_era_comes(
    year: int, machinery: float, logs: float
) -> None:
    tile_map = make_map()
    economy = Economy(tile_map)
    economy.year = year
    placed = counts(tile_map, ((8, 9), BuildingKind.LOGGING_CAMP, 1))

    trade = economy.trade(placed, holding(1000.0, MACHINERY=machinery))

    assert trade.gathered[STATION].get(Good.LOGS, 0.0) == pytest.approx(
        logs * make_factor(Good.LOGS, year)
    )


@pytest.mark.parametrize(("chips", "paper"), [(0.0, 0.0), (5.0, 50.0), (20.0, 100.0)])
def test_a_paper_mill_makes_paper_out_of_wood_chips(chips: float, paper: float) -> None:
    tile_map = make_map()
    economy = Economy(tile_map)
    placed = counts(tile_map, ((8, 8), BuildingKind.PAPER_MILL, 1))

    trade = economy.trade(placed, holding(1000.0, WOOD_CHIPS=chips, PLANKS=50.0, LOGS=50.0))

    assert trade.gathered[STATION].get(Good.PAPER, 0.0) == pytest.approx(paper)


def test_houses_burn_logs_but_not_planks() -> None:
    tile_map = make_map()
    economy = Economy(tile_map)
    placed = counts(tile_map, ((8, 9), BuildingKind.HOUSE, 1))

    trade = economy.trade(placed, holding(PLANKS=5.0))

    burned, _ = trade.goods_got[(BuildingKind.HOUSE, Need.HEATING)]
    assert burned[position(trade, economy, (8, 9))] == 0.0
