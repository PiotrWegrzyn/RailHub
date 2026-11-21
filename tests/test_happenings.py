import numpy as np

from builders import flat_map
from railroads.sim.constructions import Building, BuildingKind, Constructions
from railroads.sim.game import Game
from railroads.sim.goods import Good
from railroads.sim.growth import Growth
from railroads.sim.happenings import GUN_FACTORIES, NEW_GARRISONS, Happenings
from railroads.sim.ladders import MILITARY, GrowthRule
from railroads.sim.novelties import NOVELTIES
from railroads.sim.tilemap import Deposit, Terrain

QUIET = GrowthRule(ladders=(), forests=None)


def grown(
    year: int, buildings: dict[tuple[int, int], tuple[BuildingKind, int]], sea: bool = False
) -> Growth:
    deposits = np.zeros((40, 40), dtype=np.uint8)
    terrain = np.full((40, 40), Terrain.LAND, dtype=np.uint8)
    if sea:
        terrain[0] = Terrain.SEA
    constructions = Constructions(flat_map(40, 40, deposits=deposits, terrain=terrain))
    for cell, (kind, many) in buildings.items():
        for _ in range(many):
            constructions.build(Building(cell, kind))
    return Growth(constructions, QUIET, start_year=year)


def standing(growth: Growth, kinds: tuple[BuildingKind, ...]) -> list[Building]:
    return [
        built
        for built in growth.constructions
        if isinstance(built, Building) and built.kind in kinds
    ]


def happen(growth: Growth, year: int, month: int = 0) -> Happenings:
    happenings = Happenings()
    months = (year - growth.start_year) * 12 + month
    happenings.months(growth, months, months, np.random.default_rng(0))
    return happenings


def test_the_year_before_a_war_new_garrisons_go_up_with_a_base_and_two_outposts() -> None:
    growth = grown(1913, {})

    happen(growth, 1913)

    deposits = growth.constructions.tile_map.deposits
    assert deposits is not None and (deposits == Deposit.GARRISON).sum() >= 2 * NEW_GARRISONS
    assert len(standing(growth, (BuildingKind.BASE,))) == NEW_GARRISONS
    assert len(standing(growth, (BuildingKind.OUTPOST,))) == 2 * NEW_GARRISONS


def test_new_garrisons_let_the_military_grow_there() -> None:
    deposits = np.zeros((40, 40), dtype=np.uint8)
    growth = Growth(
        Constructions(flat_map(40, 40, deposits=deposits)),
        GrowthRule(ladders=(MILITARY,), forests=None),
        start_year=1913,
    )

    happen(growth, 1913)

    assert growth.chances(MILITARY).max() > 0


def test_a_war_opens_twice_as_many_gun_factories_on_a_map_without_a_port() -> None:
    towns = {
        (5, 5): (BuildingKind.HOUSE, 6),
        (5, 35): (BuildingKind.HOUSE, 6),
        (35, 5): (BuildingKind.HOUSE, 6),
        (35, 35): (BuildingKind.HOUSE, 6),
        (20, 20): (BuildingKind.HOUSE, 6),
    }
    inland = grown(1914, towns)
    coast = grown(1914, {**towns, (1, 20): (BuildingKind.PORT, 1)}, sea=True)

    happen(inland, 1914)
    happen(coast, 1914)

    assert len(standing(inland, (BuildingKind.GUN_FACTORY,))) == 2 * GUN_FACTORIES
    assert len(standing(coast, (BuildingKind.GUN_FACTORY,))) == GUN_FACTORIES


def test_as_a_war_ends_half_the_military_buildings_are_pulled_down() -> None:
    growth = grown(1919, {(row, 10): (BuildingKind.OUTPOST, 1) for row in range(2, 12)})

    happenings = happen(growth, 1919)

    assert len(standing(growth, MILITARY.kinds)) == 5
    assert any(headline.title == "Peace" for headline in happenings.news)


def test_the_towers_that_fall_are_those_with_the_most_people_around() -> None:
    growth = grown(
        2001,
        {
            (10, 10): (BuildingKind.SKYSCRAPER, 1),
            (10, 12): (BuildingKind.SKYSCRAPER, 1),
            (11, 11): (BuildingKind.FLATS, 4),
            (30, 30): (BuildingKind.SKYSCRAPER, 1),
        },
    )

    happen(growth, 2001, month=8)

    assert [tower.cell for tower in standing(growth, (BuildingKind.SKYSCRAPER,))] == [(30, 30)]


def test_pilgrims_crowd_the_town_of_st_peters_for_a_year_after_the_pope_dies() -> None:
    growth = grown(2005, {(20, 20): (BuildingKind.ST_PETERS, 1), (20, 21): (BuildingKind.FLATS, 4)})
    happenings = Happenings()
    rng = np.random.default_rng(0)

    happenings.months(growth, 3, 3, rng)
    crowded = (happenings.pilgrim_town, happenings.pilgrim_pay)
    happenings.months(growth, 4, 15, rng)

    assert crowded[0] > 0 and crowded[1] == 6.0
    assert (happenings.pilgrim_town, happenings.pilgrim_pay) == (0, 1.0)


def test_the_news_of_the_popes_death_gives_the_hour() -> None:
    growth = grown(2005, {})

    happenings = happen(growth, 2005, month=3)

    assert [(h.date, h.title) for h in happenings.news] == [("April 2005", "John Paul II dies")]
    assert "21:37" in happenings.news[0].story


def test_the_news_of_a_war_stops_the_game_until_the_player_reads_it() -> None:
    game = Game(Constructions(flat_map(20, 20)), start_year=1913)
    game.begin_history()
    game.growth.rule = QUIET
    rng = np.random.default_rng(0)

    months = [game.month(rng) for _ in range(12)]

    assert [h.title for h in months[-1].headlines] == ["The Great War"]
    assert [news.title for news in game.goal.news] == ["The Great War"]


def test_passengers_pay_the_pilgrims_share_only_in_the_town_of_st_peters() -> None:
    constructions = Constructions(flat_map(30, 30))
    constructions.build(Building((10, 10), BuildingKind.ST_PETERS))
    for _ in range(4):
        constructions.build(Building((10, 11), BuildingKind.FLATS))
        constructions.build(Building((25, 25), BuildingKind.FLATS))
    game = Game(constructions, start_year=2005)
    game.finances.cash = 1e9
    game.begin_history()
    game.build_station((10, 12))
    game.build_station((25, 23))
    railway = game.railway
    before = railway.facility_pay((10, 12), Good.PASSENGERS)

    happenings = game.growth.happenings
    assert happenings is not None
    happenings.months(game.growth, 3, 3, np.random.default_rng(0))

    assert railway.facility_pay((10, 12), Good.PASSENGERS) == before * 6.0
    assert railway.facility_pay((25, 23), Good.PASSENGERS) == before
    assert railway.facility_pay((10, 12), Good.MAIL) == railway.facility_pay((25, 23), Good.MAIL)


def test_a_battery_gigafactory_comes_up_by_the_biggest_town_from_2030() -> None:
    novelty = next(n for n in NOVELTIES if n.kind == BuildingKind.GIGAFACTORY)
    growth = grown(2030, {(20, 20): (BuildingKind.FLATS, 6)})

    cell = novelty.place(growth, novelty.kind, np.random.default_rng(0))

    assert novelty.from_year == 2030
    assert cell is not None and cell in growth.constructions.tile_map.grid.within((20, 20), 3)
