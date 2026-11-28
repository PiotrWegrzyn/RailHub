import pytest

from railroads.sim.constructions import BuildingKind
from railroads.sim.goods import Need
from railroads.sim.tiles import Tile, Tiles, activate, of_year
from railroads.sim.yearly import boosts_of, needs_of


def first_year_at(tile: Tile, level: int) -> int:
    return next(year for year in range(1850, 2100) if of_year(year).demand.get(tile, 0) >= level)


@pytest.mark.parametrize(
    ("kinds", "tile", "level"),
    [
        (
            (
                BuildingKind.FURNITURE_MAKER,
                BuildingKind.BIG_FURNITURE_MAKER,
                BuildingKind.MEGA_FURNITURE_MAKER,
            ),
            Tile.HOUSEHOLD,
            9,
        ),
        (
            (BuildingKind.CAR_FACTORY, BuildingKind.BIG_CAR_FACTORY, BuildingKind.MEGA_CAR_FACTORY),
            Tile.VEHICLES,
            12,
        ),
    ],
)
def test_works_take_leather_as_a_boost_only_once_their_demand_tile_reaches_its_level(
    kinds: tuple[BuildingKind, ...], tile: Tile, level: int
) -> None:
    activate(Tiles.until(2050))
    opened = first_year_at(tile, level)

    before = [Need.LEATHER in boosts_of(kind, opened - 1) for kind in kinds]
    after = [Need.LEATHER in boosts_of(kind, opened) for kind in kinds]

    assert before == [False, False, False]
    assert after == [True, True, True]


@pytest.mark.parametrize(
    "kind", [BuildingKind.OUTPOST, BuildingKind.BASE, BuildingKind.HEADQUARTERS]
)
def test_soldiers_wear_clothes_and_shoes_beside_their_arms(kind: BuildingKind) -> None:
    activate(Tiles.until(2050))

    needs = needs_of(kind, 1900)

    assert {Need.CLOTHES, Need.SHOES} <= set(needs)
