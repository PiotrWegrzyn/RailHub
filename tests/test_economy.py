import numpy as np
import pytest

from builders import flat_map
from railroads.sim.constructions import BuildingKind
from railroads.sim.economy import Economy, Satisfaction, StationOrder, levels_of, served_multipliers
from railroads.sim.goods import Good, Need, makes_of, needs_of
from railroads.sim.grids import HexGrid
from railroads.sim.ladders import LUMBER_MILLS, MINES
from railroads.sim.tilemap import Relief, Terrain, TileMap, Vegetation

STATION = (8, 10)


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
    stockpile = {Good[name]: amount for name, amount in goods.items()}
    return {STATION: StationOrder({}, room, stockpile=stockpile)}


def eaten_at(economy: Economy, trade: object, cell: tuple[int, int]) -> float:
    eaten, _ = trade.goods_got[(BuildingKind.HOUSE, Need.FOOD)]  # type: ignore[attr-defined]
    return float(eaten[position(trade, economy, cell)])


@pytest.mark.parametrize(("house", "eaten"), [((8, 4), 10.0), ((8, 5), 9.0), ((8, 8), 6.0)])
def test_without_a_station_a_farm_sells_to_a_house_losing_a_tenth_for_every_cell_between(
    house: tuple[int, int], eaten: float
) -> None:
    tile_map = make_map()
    economy = Economy(tile_map)
    placed = counts(tile_map, (house, BuildingKind.HOUSE, 3), ((8, 4), BuildingKind.FARM, 1))

    trade = economy.trade(placed)

    assert eaten_at(economy, trade, house) == pytest.approx(eaten)


def test_nothing_a_farm_sells_reaches_ten_cells_away() -> None:
    tile_map = make_map(size=24)
    economy = Economy(tile_map)
    placed = counts(tile_map, ((8, 14), BuildingKind.HOUSE, 1), ((8, 4), BuildingKind.FARM, 1))

    trade = economy.trade(placed)

    assert eaten_at(economy, trade, (8, 14)) == 0.0
    assert level_at(economy, trade, (8, 4), BuildingKind.FARM) == Satisfaction.UNSATISFIED


def test_a_farm_feeds_the_nearer_house_first_and_the_farther_one_what_is_left() -> None:
    tile_map = make_map()
    economy = Economy(tile_map)
    placed = counts(
        tile_map,
        ((8, 4), BuildingKind.FARM, 2),
        ((8, 5), BuildingKind.HOUSE, 1),
        ((8, 7), BuildingKind.HOUSE, 1),
    )

    trade = economy.trade(placed)

    assert eaten_at(economy, trade, (8, 5)) == pytest.approx(10.0)
    assert eaten_at(economy, trade, (8, 7)) == pytest.approx((20 - 10 / 0.9) * 0.7)


def test_a_farm_is_as_satisfied_as_the_share_of_its_wheat_that_arrives() -> None:
    tile_map = make_map()
    economy = Economy(tile_map)
    placed = counts(tile_map, ((8, 4), BuildingKind.FARM, 1), ((8, 6), BuildingKind.HOUSE, 5))

    trade = economy.trade(placed)

    share = trade.share((BuildingKind.FARM,))[position(trade, economy, (8, 4))]
    assert share == pytest.approx(0.8)


def test_a_station_gathers_before_buildings_nearby_buy() -> None:
    tile_map = make_map()
    economy = Economy(tile_map)
    placed = counts(tile_map, ((8, 9), BuildingKind.FARM, 1), ((6, 4), BuildingKind.HOUSE, 1))

    trade = economy.trade(
        placed, {STATION: StationOrder({}, 100.0, delivery=0, kept=frozenset(Good))}
    )

    assert trade.gathered[STATION] == {Good.WHEAT: 10.0}
    assert eaten_at(economy, trade, (6, 4)) == 0.0


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
    ("lumber", "steel", "goods"), [(0.0, 9.0, 0.0), (2.0, 9.0, 6.0), (9.0, 1.0, 3.0)]
)
def test_a_factory_makes_only_as_much_as_its_lumber_and_steel_allow(
    lumber: float, steel: float, goods: float
) -> None:
    tile_map = make_map()
    economy = Economy(tile_map)
    economy.year = 1855
    placed = counts(tile_map, ((8, 8), BuildingKind.FACTORY, 1))

    trade = economy.trade(placed, holding(100.0, LUMBER=lumber, STEEL=steel))

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


def test_lumber_mills_only_grow_at_the_edge_of_a_forest() -> None:
    weights = LUMBER_MILLS.land.weights(make_map())

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


def test_ten_coal_and_ten_iron_ore_make_twenty_steel_once_a_station_takes_it() -> None:
    tile_map = make_map()
    economy = Economy(tile_map)
    placed = counts(tile_map, ((8, 8), BuildingKind.STEEL_MILL, 1))

    trade = economy.trade(placed, holding(100.0, COAL=10.0, IRON=10.0))

    assert trade.gathered[STATION][Good.STEEL] == pytest.approx(20.0)
    assert economy.stock[(BuildingKind.STEEL_MILL, Need.COAL)][flat(economy, (8, 8))] == 0.0


def test_inputs_wait_in_a_stockpile_of_three_hundred_until_the_other_input_comes() -> None:
    tile_map = make_map()
    economy = Economy(tile_map)
    placed = counts(tile_map, ((8, 8), BuildingKind.STEEL_MILL, 1))

    for _ in range(5):
        economy.trade(placed, holding(COAL=100.0))

    assert economy.stock[(BuildingKind.STEEL_MILL, Need.COAL)][flat(economy, (8, 8))] == 300.0


def test_ten_fruit_and_ten_steel_make_twenty_five_cans() -> None:
    tile_map = make_map()
    economy = Economy(tile_map)
    economy.stock[(BuildingKind.FOOD_PLANT, Need.FRUIT)][flat(economy, (8, 8))] = 10.0
    economy.stock[(BuildingKind.FOOD_PLANT, Need.STEEL)][flat(economy, (8, 8))] = 10.0
    placed = counts(tile_map, ((8, 8), BuildingKind.FOOD_PLANT, 1))

    trade = economy.trade(placed, {STATION: StationOrder({}, 100.0)})

    assert trade.gathered[STATION][Good.CANS] == pytest.approx(25.0)
    assert economy.stock[(BuildingKind.FOOD_PLANT, Need.FRUIT)][flat(economy, (8, 8))] == 0.0


def test_houses_eat_raw_fruit_as_food() -> None:
    tile_map = make_map()
    economy = Economy(tile_map)
    placed = counts(tile_map, ((8, 9), BuildingKind.HOUSE, 1))

    trade = economy.trade(placed, holding(APPLES=20.0))

    eaten, needed = trade.goods_got[(BuildingKind.HOUSE, Need.FOOD)]
    at = position(trade, economy, (8, 9))
    assert eaten[at] == needed[at] == 10.0


def test_canneries_take_their_fruit_before_houses_eat_it() -> None:
    tile_map = make_map()
    economy = Economy(tile_map)
    economy.year = 1855
    placed = counts(tile_map, ((8, 9), BuildingKind.HOUSE, 1), ((8, 8), BuildingKind.FOOD_PLANT, 1))

    trade = economy.trade(placed, holding(APPLES=3.0))

    eaten, _ = trade.goods_got[(BuildingKind.HOUSE, Need.FOOD)]
    assert eaten[position(trade, economy, (8, 9))] == 0.0
    assert economy.stock[(BuildingKind.FOOD_PLANT, Need.FRUIT)][flat(economy, (8, 8))] == 3.0


@pytest.mark.parametrize(("year", "goods"), [(1854, 10.0), (1855, 0.0)])
def test_a_factory_works_on_lumber_alone_until_steel_comes_in_1855(year: int, goods: float) -> None:
    tile_map = make_map()
    economy = Economy(tile_map)
    economy.year = year
    placed = counts(tile_map, ((8, 8), BuildingKind.FACTORY, 1))

    trade = economy.trade(placed, holding(100.0, LUMBER=10.0 / 3))

    made = trade.gave[BuildingKind.FACTORY][position(trade, economy, (8, 8))]
    assert made == pytest.approx(goods)


def test_a_station_counts_what_it_passes_along_from_its_stockpile_by_need() -> None:
    tile_map = make_map()
    economy = Economy(tile_map)
    placed = counts(tile_map, ((8, 9), BuildingKind.HOUSE, 4))

    trade = economy.trade(placed, holding(WHEAT=5.0))

    assert trade.supplied_for[STATION] == {(Need.FOOD, Good.WHEAT): 5.0}
    assert trade.supplied[STATION] == {Good.WHEAT: 5.0}
    assert trade.handed_for[STATION] == {}


def test_a_bakery_bakes_each_unit_of_wheat_into_two_of_bread() -> None:
    tile_map = make_map()
    economy = Economy(tile_map)
    placed = counts(tile_map, ((8, 8), BuildingKind.BAKERY, 1))

    trade = economy.trade(placed, holding(100.0, WHEAT=5.0))

    assert trade.gathered[STATION][Good.BREAD] == pytest.approx(10.0)
    assert economy.stock[(BuildingKind.BAKERY, Need.WHEAT)][flat(economy, (8, 8))] == 0.0


def test_bakeries_take_their_wheat_before_houses_eat_it_and_bread_feeds_houses() -> None:
    tile_map = make_map()
    economy = Economy(tile_map)
    placed = counts(tile_map, ((8, 9), BuildingKind.HOUSE, 1), ((8, 8), BuildingKind.BAKERY, 1))

    trade = economy.trade(placed, holding(WHEAT=3.0, BREAD=5.0))

    baked_for_the_house = 5.0 / 0.9
    assert economy.stock[(BuildingKind.BAKERY, Need.WHEAT)][flat(economy, (8, 8))] == pytest.approx(
        3.0 - baked_for_the_house / 2
    )
    assert eaten_at(economy, trade, (8, 9)) == pytest.approx(10.0)
    assert trade.supplied_for[STATION][(Need.FOOD, Good.BREAD)] == 5.0


@pytest.mark.parametrize(("year", "goods"), [(1879, 30.0), (1880, 6.0)])
def test_a_big_factory_works_on_steel_alone_until_copper_comes_in_1880(
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


def test_farms_want_machinery_only_from_1920() -> None:
    before = needs_of(BuildingKind.FARM, 1919)
    after = needs_of(BuildingKind.FARM, 1920)

    assert before == {}
    assert after == {Need.MACHINERY: 1}


@pytest.mark.parametrize(
    ("kind", "inputs", "made", "amount"),
    [
        (BuildingKind.BRICKWORKS, {"CLAY": 10.0}, Good.BRICKS, 20.0),
        (BuildingKind.BRICKWORKS, {"CLAY": 10.0, "COAL": 5.0}, Good.BRICKS, 40.0),
        (BuildingKind.TANNERY, {"CATTLE": 10.0}, Good.LEATHER, 20.0),
        (BuildingKind.SHOE_FACTORY, {"LEATHER": 10.0}, Good.SHOES, 20.0),
        (BuildingKind.GLASSWORKS, {"SAND": 10.0}, Good.GLASS, 20.0),
        (BuildingKind.GLASSWORKS, {"SAND": 10.0, "COAL": 5.0}, Good.GLASS, 40.0),
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


@pytest.mark.parametrize(("barrels", "beer"), [(0.0, 40.0), (2.0, 60.0), (4.0, 80.0), (9.0, 80.0)])
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

    assert trade.gathered[STATION][Good.CLOTHES] == pytest.approx(10.0)


@pytest.mark.parametrize("animal", ["CATTLE", "PIGS", "CHICKENS", "SHEEP"])
def test_a_butcher_turns_each_animal_into_two_of_meat(animal: str) -> None:
    tile_map = make_map()
    economy = Economy(tile_map)
    placed = counts(tile_map, ((8, 8), BuildingKind.MEAT_FACTORY, 1))

    trade = economy.trade(placed, holding(100.0, **{animal: 10.0}))

    assert trade.gathered[STATION][Good.MEAT] == pytest.approx(20.0)


@pytest.mark.parametrize(
    ("kind", "held", "made", "most", "need", "left"),
    [
        (BuildingKind.BAKERY, {"WHEAT": 80.0}, Good.BREAD, 100.0, Need.WHEAT, 30.0),
        (BuildingKind.MEAT_FACTORY, {"CATTLE": 80.0}, Good.MEAT, 100.0, Need.ANIMALS, 30.0),
        (BuildingKind.DAIRY, {"MILK": 80.0}, Good.DAIRY, 100.0, Need.MILK, 30.0),
        (
            BuildingKind.FOOD_PLANT,
            {"APPLES": 130.0, "STEEL": 80.0},
            Good.CANS,
            250.0,
            Need.FRUIT,
            30.0,
        ),
        (
            BuildingKind.SAUSAGE_MAKER,
            {"MEAT": 130.0, "SALT": 80.0},
            Good.SAUSAGES,
            200.0,
            Need.MEAT,
            30.0,
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
    assert eaten[position(trade, economy, (8, 9))] == 10.0


def test_stations_hand_out_the_most_valuable_food_first_even_from_their_stockpile() -> None:
    tile_map = make_map()
    economy = Economy(tile_map)
    placed = counts(tile_map, ((8, 9), BuildingKind.HOUSE, 1))
    order = StationOrder({Good.WHEAT: 20.0}, 0.0, stockpile={Good.MEAT: 20.0})

    trade = economy.trade(placed, {STATION: order})

    assert trade.supplied_for[STATION] == {(Need.FOOD, Good.MEAT): 10.0}
    assert trade.handed_for[STATION].get((Need.FOOD, Good.WHEAT), 0.0) == 0.0


@pytest.mark.parametrize(("year", "planes"), [(1959, 2.0), (1960, 1.0)])
def test_aircraft_factories_want_electronics_once_there_are_any_in_1960(
    year: int, planes: float
) -> None:
    tile_map = make_map()
    economy = Economy(tile_map)
    economy.year = year
    placed = counts(tile_map, ((8, 8), BuildingKind.PLANE_FACTORY, 1))

    trade = economy.trade(placed, holding(100.0, STEEL=6.0, COPPER=2.0, FUEL=2.0, ELECTRONICS=1.0))

    assert trade.gathered[STATION][Good.PLANES] == pytest.approx(planes)


@pytest.mark.parametrize("fuel", ["LUMBER", "COAL"])
def test_houses_heat_with_lumber_or_coal(fuel: str) -> None:
    tile_map = make_map()
    economy = Economy(tile_map)
    placed = counts(tile_map, ((8, 9), BuildingKind.HOUSE, 1))

    trade = economy.trade(placed, holding(**{fuel: 5.0}))

    burned, _ = trade.goods_got[(BuildingKind.HOUSE, Need.HEATING)]
    assert burned[position(trade, economy, (8, 9))] == 1.0


@pytest.mark.parametrize(
    ("year", "made"),
    [(1949, {Good.GOODS, Good.OIL}), (1950, {Good.GOODS, Good.OIL, Good.GAS})],
)
def test_ports_bring_in_gas_only_once_there_is_any_in_1950(year: int, made: set) -> None:
    makes = makes_of(BuildingKind.PORT, year)

    assert set(makes) == made


def test_a_power_plant_burns_coal_oil_or_gas() -> None:
    tile_map = make_map()
    economy = Economy(tile_map)
    economy.year = 1950
    placed = counts(tile_map, ((8, 9), BuildingKind.POWER_PLANT, 1))

    trade = economy.trade(placed, holding(COAL=50.0, OIL=50.0, GAS=50.0))

    burned, needed = trade.goods_got[(BuildingKind.POWER_PLANT, Need.BOILER_FUEL)]
    at = position(trade, economy, (8, 9))
    assert burned[at] == 150.0
    assert needed[at] == 200.0


def test_bonus_goods_raise_a_fed_and_warm_house_past_full_satisfaction() -> None:
    tile_map = make_map()
    economy = Economy(tile_map)
    placed = counts(tile_map, ((8, 9), BuildingKind.HOUSE, 1))

    trade = economy.trade(placed, holding(FISH=10.0, COAL=1.0, BEER=1.0, NEWSPAPERS=0.5))

    share = trade.share((BuildingKind.HOUSE,))[position(trade, economy, (8, 9))]
    assert share == pytest.approx(1.45)
    assert level_at(economy, trade, (8, 9), BuildingKind.HOUSE) == Satisfaction.THRIVING


def test_a_house_that_only_gets_its_beer_is_content_at_thirty_percent() -> None:
    tile_map = make_map()
    economy = Economy(tile_map)
    placed = counts(tile_map, ((8, 9), BuildingKind.HOUSE, 1))

    trade = economy.trade(placed, holding(BEER=1.0))

    share = trade.share((BuildingKind.HOUSE,))[position(trade, economy, (8, 9))]
    assert share == pytest.approx(0.3)
    assert level_at(economy, trade, (8, 9), BuildingKind.HOUSE) == Satisfaction.CONTENT


def test_each_bonus_good_counts_what_the_economy_is_told_it_is_worth() -> None:
    tile_map = make_map()
    economy = Economy(tile_map)
    economy.bonus_per_good = 0.4
    placed = counts(tile_map, ((8, 9), BuildingKind.HOUSE, 1))

    trade = economy.trade(placed, holding(BEER=1.0))

    assert trade.share((BuildingKind.HOUSE,))[position(trade, economy, (8, 9))] == pytest.approx(
        0.4
    )


def test_growth_counts_satisfaction_past_full_without_a_ceiling() -> None:
    share = np.array([0.5, 1.0, 1.45])

    multipliers = served_multipliers(share)

    assert multipliers.tolist() == pytest.approx([5.0, 10.0, 14.5])


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


@pytest.mark.parametrize(("lumber", "coal"), [(0.0, 100.0), (10.0, 150.0), (20.0, 200.0)])
def test_lumber_for_pit_props_lets_a_mine_dig_up_to_twice_the_coal(
    lumber: float, coal: float
) -> None:
    tile_map = make_map()
    economy = Economy(tile_map)
    placed = counts(tile_map, ((8, 9), BuildingKind.MINE, 1))

    trade = economy.trade(placed, holding(1000.0, LUMBER=lumber))

    assert trade.gathered[STATION].get(Good.COAL, 0.0) == pytest.approx(coal)


def test_gasworks_make_town_gas_out_of_coal() -> None:
    tile_map = make_map()
    economy = Economy(tile_map)
    placed = counts(tile_map, ((8, 8), BuildingKind.GASWORKS, 1))

    trade = economy.trade(placed, holding(1000.0, COAL=50.0))

    assert trade.gathered[STATION][Good.GAS] == pytest.approx(100.0)
