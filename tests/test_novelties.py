import numpy as np
import pytest

from builders import square_map
from railroads.newgame import make_game
from railroads.settings import MapChoice, Settings
from railroads.sim.clock import END_YEAR
from railroads.sim.constructions import Building, BuildingKind, Constructions
from railroads.sim.goods import MAKES, Good, Need
from railroads.sim.growth import Growth
from railroads.sim.ladders import GrowthRule
from railroads.sim.novelties import (
    NOVELTIES,
    ByTown,
    Novelties,
    Novelty,
    olympic_year,
    place_space_center,
)
from railroads.sim.tilemap import Deposit
from railroads.sim.yearly import needs_of

QUIET = GrowthRule(ladders=(), forests=None)
BY_KIND = {novelty.kind: novelty for novelty in NOVELTIES}


def grown(year: int, towns: dict[tuple[int, int], BuildingKind] | None = None, **kw) -> Growth:
    constructions = Constructions(square_map(**kw))
    for cell, kind in (towns or {}).items():
        for _ in range(6 if kind == BuildingKind.HOUSE else 1):
            constructions.build(Building(cell, kind))
    return Growth(constructions, QUIET, start_year=year)


def built(growth: Growth, kind: BuildingKind) -> list[tuple[int, int]]:
    return [tuple(cell) for cell in np.argwhere(growth.count(kind) > 0).tolist()]


@pytest.mark.parametrize(("size", "least", "most"), [(348, 2, 3), (174, 1, 1)])
def test_a_game_gets_two_or_three_novelties_on_a_map_like_poland_and_fewer_on_a_small_one(
    size: int, least: int, most: int
) -> None:
    tile_map = square_map(size)

    counts = {len(Novelties.pick(tile_map, np.random.default_rng(s)).picked) for s in range(20)}

    assert min(counts) >= least
    assert max(counts) <= most


@pytest.mark.parametrize(("size", "chance"), [(348, 0.05), (174, 0.0125)])
def test_the_chance_of_hosting_the_games_grows_with_the_land(size: int, chance: float) -> None:
    novelties = Novelties.pick(square_map(size), np.random.default_rng(0))

    assert novelties.olympic_chance == pytest.approx(chance, rel=0.01)


@pytest.mark.parametrize(
    ("year", "hosts"),
    [(1916, False), (1920, True), (2021, True), (2022, False), (2100, True), (2101, False)],
)
def test_stadiums_go_up_only_in_real_summer_games_years(year: int, hosts: bool) -> None:
    growth = grown(year, {(20, 20): BuildingKind.HOUSE})
    novelties = Novelties([], olympic_chance=1.0)

    novelties.year(growth, np.random.default_rng(0))

    assert bool(built(growth, BuildingKind.OLYMPIC_STADIUM)) == hosts
    assert olympic_year(year) == hosts


def test_each_games_goes_to_the_biggest_town_that_has_not_hosted_yet() -> None:
    towns = {(10, 10): BuildingKind.FLATS, (30, 30): BuildingKind.HOUSE}
    growth = grown(1920, towns)
    novelties = Novelties([], olympic_chance=1.0)

    novelties.year(growth, np.random.default_rng(0))
    novelties.year(growth, np.random.default_rng(0))

    grid = growth.constructions.tile_map.grid
    stadiums = built(growth, BuildingKind.OLYMPIC_STADIUM)
    assert len(stadiums) == 2
    assert any(cell in grid.within((10, 10), 2) for cell in stadiums)
    assert any(cell in grid.within((30, 30), 2) for cell in stadiums)


def test_the_pear_company_waits_for_a_skyscraper_and_settles_beside_it() -> None:
    growth = grown(1990, {(20, 20): BuildingKind.SKYSCRAPER})
    lonely = grown(1990, {(20, 20): BuildingKind.FLATS})
    place = BY_KIND[BuildingKind.PEAR_HQ].place

    cell = place(growth, BuildingKind.PEAR_HQ, np.random.default_rng(0))
    none = place(lonely, BuildingKind.PEAR_HQ, np.random.default_rng(0))

    assert cell is not None and cell in growth.constructions.tile_map.grid.within((20, 20), 1)
    assert none is None


def test_space_centers_mostly_go_up_near_military_ground() -> None:
    deposits = np.zeros((40, 40), dtype=np.uint8)
    deposits[2:5, 2:5] = Deposit.GARRISON
    growth = grown(1970, deposits=deposits)
    grid = growth.constructions.tile_map.grid
    near = set(grid.within((3, 3), 11))

    cells = [
        place_space_center(growth, BuildingKind.SPACE_CENTER, np.random.default_rng(s))
        for s in range(40)
    ]

    assert sum(cell in near for cell in cells) > 20


def test_a_picked_novelty_goes_up_once_and_not_before_its_year() -> None:
    royal = BY_KIND[BuildingKind.ROYAL_PALACE]
    palace = Novelty(BuildingKind.ROYAL_PALACE, 1900, royal.place, royal.where)
    early = grown(1899, {(20, 20): BuildingKind.HOUSE})
    later = grown(1900, {(20, 20): BuildingKind.HOUSE})
    rng = np.random.default_rng(0)

    for _ in range(100):
        Novelties([palace], 0.0).year(early, rng)
    novelties = Novelties([palace], 0.0)
    for _ in range(100):
        novelties.year(later, rng)

    assert built(early, BuildingKind.ROYAL_PALACE) == []
    assert len(built(later, BuildingKind.ROYAL_PALACE)) == 1


def test_the_worlds_fair_is_over_by_1941() -> None:
    growth = grown(1941, {(20, 20): BuildingKind.FLATS})
    novelties = Novelties([BY_KIND[BuildingKind.WORLDS_FAIR]], 0.0)

    for _ in range(100):
        novelties.year(growth, np.random.default_rng(0))

    assert built(growth, BuildingKind.WORLDS_FAIR) == []


def test_the_salt_mine_cathedral_stands_on_salt_and_digs_it() -> None:
    deposits = np.zeros((40, 40), dtype=np.uint8)
    deposits[30, 7] = Deposit.SALT
    growth = grown(1850, deposits=deposits)
    place = BY_KIND[BuildingKind.SALT_CATHEDRAL].place

    cell = place(growth, BuildingKind.SALT_CATHEDRAL, np.random.default_rng(0))

    assert cell == (30, 7)
    assert MAKES[BuildingKind.SALT_CATHEDRAL] == {Good.SALT: 120}


def test_novelties_want_only_goods_the_game_already_has_and_fill_a_whole_hex() -> None:
    constructions = Constructions(square_map())

    for novelty in (*NOVELTIES, Novelty(BuildingKind.OLYMPIC_STADIUM, 1896, lambda *_: None, "")):
        constructions.build(Building((5, 5), novelty.kind))
        assert constructions.space_left((5, 5)) == 0
        constructions.remove(Building((5, 5), novelty.kind))
        assert all(isinstance(need, Need) for need in needs_of(novelty.kind, 2050))


@pytest.mark.parametrize(("year", "most"), [(1854, 0), (1855, 2)])
def test_railway_works_go_up_from_1855_by_the_biggest_towns_apart_up_to_their_number(
    year: int, most: int
) -> None:
    towns = {
        (10, 10): BuildingKind.FLATS,
        (30, 30): BuildingKind.HOUSE,
        (10, 30): BuildingKind.HOUSE,
    }
    growth = grown(year, towns)
    novelties = Novelties([], olympic_chance=0.0, railway_works=2)

    for _ in range(200):
        novelties.year(growth, np.random.default_rng(len(novelties.works) + _))
    works = built(growth, BuildingKind.RAILWAY_WORKS)
    grid = growth.constructions.tile_map.grid
    by_the_biggest_town = any(cell in grid.within((10, 10), 2) for cell in works)

    assert len(works) == most
    assert by_the_biggest_town is (most > 0)


def test_a_long_game_on_poland_gets_one_off_buildings() -> None:
    game = make_game(Settings(seed=1, map=MapChoice.POLAND), report=lambda _: None).game
    growth = game.growth
    novelties = growth.novelties
    assert novelties is not None
    one_offs = {novelty.kind for novelty in NOVELTIES} | {BuildingKind.OLYMPIC_STADIUM}
    rng = np.random.default_rng(1)

    for _ in range(growth.year, END_YEAR):
        growth.start_year += 1
        novelties.year(growth, rng)
    standing = {kind for kind in one_offs if growth.count(kind).any()}

    assert len(novelties.picked) >= 2
    assert standing


def test_town_novelties_go_up_by_a_big_town_and_not_by_a_hamlet() -> None:
    city = grown(
        1950,
        {(20, 20): BuildingKind.FLATS, (20, 21): BuildingKind.FLATS, (21, 20): BuildingKind.FLATS},
    )
    hamlet = grown(1950, {(20, 20): BuildingKind.HOUSE})
    by_town = [novelty for novelty in NOVELTIES if isinstance(novelty.place, ByTown)]

    in_city = [n.place(city, n.kind, np.random.default_rng(0)) for n in by_town]
    in_hamlet = [n.place(hamlet, n.kind, np.random.default_rng(0)) for n in by_town]

    assert by_town and None not in in_city
    assert set(in_hamlet) == {None}


def test_the_odds_say_why_a_town_novelty_cannot_come_up_yet() -> None:
    hamlet = grown(1950, {(20, 20): BuildingKind.HOUSE})
    film = Novelties([BY_KIND[BuildingKind.FILM_STUDIO]], olympic_chance=0.05, railway_works=2)

    odds = {found.kind: found for found in film.odds(hamlet)}

    assert odds[BuildingKind.FILM_STUDIO].why_none is not None
    assert "biggest town has 6" in odds[BuildingKind.FILM_STUDIO].why_none
    assert odds[BuildingKind.OLYMPIC_STADIUM].chance == 0.05
    assert odds[BuildingKind.RAILWAY_WORKS].why_none is None
