import pytest

from builders import flat_map
from railroads.sim.constructions import Constructions, Track
from railroads.sim.finance import MAGLEV_TRACK_PRICE, TRACK_PRICE, track_upkeep
from railroads.sim.game import Game
from railroads.sim.growth import Growth
from railroads.sim.ladders import GrowthRule
from railroads.sim.railway import Railway
from railroads.sim.routes import build_station, lay_route, plan_route, route_price
from railroads.sim.tech import NODES, Effect, Research
from railroads.sim.trains import ENGINES, Stop, Train
from railroads.ui.build_menu import BuildMenu
from railroads.ui.tech_window import describe
from railroads.ui.trains_window import engine_summary

WEST, EAST = (6, 3), (6, 20)


def two_lines() -> Railway:
    constructions = Constructions(flat_map())
    growth = Growth(constructions, GrowthRule(trades=False))
    for cell in (WEST, EAST, (2, 3), (2, 20)):
        build_station(growth, cell)
    for start, end, maglev in ((WEST, EAST, False), ((2, 3), (2, 20), True)):
        route = plan_route(constructions, start, end, maglev=maglev)
        assert route is not None
        lay_route(growth, route)
    return Railway(constructions)


def maglev_game(year: int, *taken: str) -> Game:
    game = Game(Constructions(flat_map()))
    game.finances.cash = 10_000_000.0
    game.railway.year = year
    game.railway.research = Research(taken=set(taken))
    return game


def test_the_500_kmh_engine_is_a_maglev_opened_by_magnetic_levitation() -> None:
    engine = ENGINES["maglev"]

    _, node = NODES["maglev"]

    assert engine.maglev and not engine.electric
    assert engine.kmh == pytest.approx(500.0)
    assert "maglev" in engine.needs
    assert node.effect == Effect.UNLOCK
    assert "thyristors" in node.requires


def test_a_maglev_finds_a_way_only_along_maglev_track() -> None:
    railway = two_lines()

    on_rails = railway.route(WEST, EAST, maglev=True)
    on_guideway = railway.route((2, 3), (2, 20), maglev=True)

    assert on_rails is None
    assert on_guideway is not None


def test_nothing_but_a_maglev_runs_on_maglev_track() -> None:
    railway = two_lines()

    plain = railway.route((2, 3), (2, 20))
    wired = railway.route((2, 3), (2, 20), electric=True)

    assert plain is None and wired is None
    assert railway.route(WEST, EAST) is not None


def test_a_maglev_without_a_guideway_says_so() -> None:
    railway = two_lines()
    train = Train("Maglev 1", engine="maglev")
    train.stops = [Stop(WEST), Stop(EAST)]
    train.stops[0].wait_for_load = False

    railway.run(0.001)
    railway.trains.append(train)
    railway.run(0.001)

    assert train.status == "No maglev track to its next stop"


def test_maglev_track_costs_its_guideway_on_top_of_plain_track() -> None:
    constructions = Constructions(flat_map())
    plain = plan_route(constructions, WEST, (6, 9))
    guideway = plan_route(constructions, WEST, (6, 9), maglev=True)
    assert plain is not None and guideway is not None

    extra = route_price(constructions, guideway) - route_price(constructions, plain)

    assert extra == pytest.approx(MAGLEV_TRACK_PRICE * len(guideway.pieces))
    assert MAGLEV_TRACK_PRICE > 10 * TRACK_PRICE


def test_maglev_track_costs_more_upkeep_than_electric_track() -> None:
    piece = Track((6, 3), (0, 6))

    plain = track_upkeep([piece])
    wired = track_upkeep([Track((6, 3), (0, 6), electric=True)])
    guideway = track_upkeep([Track((6, 3), (0, 6), maglev=True)])

    assert plain < wired < guideway


def test_maglev_track_carries_no_wires_even_if_asked_for_electric() -> None:
    constructions = Constructions(flat_map())

    route = plan_route(constructions, WEST, (6, 9), electric=True, maglev=True)

    assert route is not None
    assert all(piece.maglev and not piece.electric for piece in route.pieces)


def test_a_guideway_laid_over_rails_takes_their_place() -> None:
    constructions = Constructions(flat_map())
    growth = Growth(constructions, GrowthRule(trades=False))
    rails = plan_route(constructions, WEST, (6, 12), electric=True)
    assert rails is not None
    lay_route(growth, rails)

    guideway = plan_route(constructions, (6, 12), WEST, maglev=True)
    assert guideway is not None
    lay_route(growth, guideway)

    pieces = constructions.track()
    assert len(pieces) == len(rails.pieces)
    assert all(piece.maglev and not piece.electric for piece in pieces)


def test_rails_laid_along_a_guideway_leave_it_a_guideway() -> None:
    constructions = Constructions(flat_map())
    growth = Growth(constructions, GrowthRule(trades=False))
    guideway = plan_route(constructions, WEST, (6, 12), maglev=True)
    assert guideway is not None
    lay_route(growth, guideway)

    rails = plan_route(constructions, (6, 12), WEST, electric=True)
    assert rails is not None
    lay_route(growth, rails)

    assert all(piece.maglev and not piece.electric for piece in constructions.track())


def test_maglev_track_opens_with_the_maglev_engine() -> None:
    early = maglev_game(2030, "express_trains", "maglev")
    unlearned = maglev_game(2040, "express_trains")
    ready = maglev_game(2040, "express_trains", "maglev")

    opened = [game.railway.maglev_open for game in (early, unlearned, ready)]

    assert opened == [False, False, True]
    assert "maglev" in ready.railway.engines()


def test_maglev_track_cannot_be_laid_before_it_opens() -> None:
    game = maglev_game(2030, "express_trains", "maglev")
    cash = game.finances.cash

    done = game.lay_track(WEST, (6, 9), maglev=True)

    assert not done.ok
    assert "Magnetic levitation" in done.message
    assert game.finances.cash == cash
    assert game.constructions.track() == []


def test_maglev_track_is_laid_once_it_opens() -> None:
    game = maglev_game(2040, "express_trains", "maglev")

    done = game.lay_track(WEST, (6, 9), maglev=True)

    assert done.ok
    assert all(piece.maglev for piece in game.constructions.track())


def test_a_maglev_does_not_end_the_sale_of_the_fastest_electric() -> None:
    railway = two_lines()
    railway.research = Research(taken=set(NODES))
    railway.year = 2040

    engines = railway.engines()

    assert {"maglev", "high_speed"} <= set(engines)


def test_choosing_maglev_track_leaves_out_the_wires() -> None:
    menu = BuildMenu(electric=True)

    menu.lay(menu.track_width, menu.electric, maglev=True)

    assert menu.maglev and not menu.electric


def test_a_maglev_tells_it_needs_maglev_track() -> None:
    summary = engine_summary(ENGINES["maglev"])

    assert "Runs on maglev track only" in summary
    assert "electrified" not in summary


def test_magnetic_levitation_says_what_it_opens() -> None:
    _, node = NODES["maglev"]

    text = describe(node)

    assert text == "opens maglev track and trains (from 2040)"
