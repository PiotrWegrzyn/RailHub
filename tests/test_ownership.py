import numpy as np
import pytest

from railroads.sim.constructions import Building, BuildingKind, Constructions
from railroads.sim.geo import Bounds
from railroads.sim.goods import Good, price_of
from railroads.sim.grids import HexGrid
from railroads.sim.growth import OWNER_SHARE, PAYBACK_YEARS, Growth
from railroads.sim.ladders import FACTORIES, FARMS_CROWDED_OUT, GrowthRule
from railroads.sim.railway import Railway
from railroads.sim.routes import build_station, lay_route, plan_route
from railroads.sim.tilemap import Relief, Temperature, Terrain, TileMap, Vegetation

QUIET = GrowthRule(trades=False, ladders=(), forests=None)


def make_map() -> TileMap:
    grid = HexGrid(rows=12, cols=30, km_per_cell=2.0)
    return TileMap(
        name="Test",
        grid=grid,
        bounds=Bounds(west=14.0, east=15.0, south=50.0, north=51.0),
        terrain=np.full(grid.shape, Terrain.LAND, dtype=np.uint8),
        vegetation=np.full(grid.shape, Vegetation.GRASS, dtype=np.uint8),
        temperature=np.full(grid.shape, Temperature.NORMAL, dtype=np.uint8),
        height_m=np.zeros(grid.shape, dtype=np.int16),
        relief=np.full(grid.shape, Relief.FLAT, dtype=np.uint8),
    )


def grown(*placed: tuple[tuple[int, int], BuildingKind]) -> Growth:
    constructions = Constructions(make_map())
    for cell, kind in placed:
        constructions.build(Building(cell, kind))
    growth = Growth(constructions, QUIET)
    growth.railway = Railway(constructions)
    return growth


def test_a_building_costs_five_years_of_the_players_share_of_what_it_can_make() -> None:
    growth = grown(((5, 5), BuildingKind.FARM))
    cash = growth.railway.finances.cash

    problem = growth.buy((5, 5), BuildingKind.FARM)

    price = 12 * PAYBACK_YEARS * OWNER_SHARE * 5 * price_of(Good.WHEAT)
    assert problem is None
    assert growth.owned(BuildingKind.FARM)[5, 5] == 1
    assert growth.railway.finances.cash == pytest.approx(cash - price)


@pytest.mark.parametrize(
    ("placed", "kind", "problem"),
    [
        ([((5, 5), BuildingKind.HOUSE)], BuildingKind.HOUSE, "A house cannot be bought"),
        ([((5, 5), BuildingKind.MINE)], BuildingKind.MINE, "You already own every mine here"),
    ],
)
def test_homes_cannot_be_bought_nor_more_than_stand_there(
    placed: list, kind: BuildingKind, problem: str
) -> None:
    growth = grown(*placed)
    if kind == BuildingKind.MINE:
        growth.buy((5, 5), kind)

    said = growth.buy((5, 5), kind)

    assert said == problem


def test_the_player_earns_half_of_what_their_farm_sells_to_a_joined_station() -> None:
    constructions = Constructions(make_map())
    growth = Growth(constructions, GrowthRule(trades=False))
    build_station(growth, (6, 3))
    build_station(growth, (6, 20))
    route = plan_route(constructions, (6, 3), (6, 20))
    assert route is not None
    lay_route(growth, route)
    growth.railway = Railway(constructions)
    growth.build(Building((6, 22), BuildingKind.FARM))
    growth.own((6, 22), BuildingKind.FARM)

    growth.trade_goods()

    earned = growth.railway.finances.month["owned buildings"]
    assert earned == pytest.approx(5 * price_of(Good.WHEAT) * OWNER_SHARE)
    assert growth.owner_income == {BuildingKind.FARM: pytest.approx(earned)}


def test_a_crowded_out_farm_of_the_players_goes_up_again_nearby_and_stays_theirs() -> None:
    growth = grown(((5, 5), BuildingKind.FARM))
    growth.own((5, 5), BuildingKind.FARM)
    neighbours = growth.constructions.tile_map.grid.neighbours(5, 5)
    for i in range(FARMS_CROWDED_OUT):
        growth.build(Building(neighbours[i % len(neighbours)], BuildingKind.HOUSE))

    growth._crowd_out_farms()

    owned = growth.owned(BuildingKind.FARM)
    assert growth.count(BuildingKind.FARM)[5, 5] == 0
    assert owned.sum() == 1
    assert growth.count(BuildingKind.FARM)[tuple(np.argwhere(owned)[0])] == 1


def test_factories_merging_into_a_big_one_hand_it_to_whoever_owned_one_of_them() -> None:
    growth = grown(
        ((5, 5), BuildingKind.FACTORY),
        ((5, 6), BuildingKind.FACTORY),
        ((5, 6), BuildingKind.FACTORY),
    )
    growth.own((5, 6), BuildingKind.FACTORY)

    growth._turn((5, 5), FACTORIES, np.random.default_rng(0))

    assert growth.count(BuildingKind.BIG_FACTORY)[5, 5] == 1
    assert growth.owned(BuildingKind.BIG_FACTORY)[5, 5] == 1
    assert growth.owned(BuildingKind.FACTORY).sum() == 0
