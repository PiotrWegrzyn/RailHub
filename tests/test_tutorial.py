import pickle
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from test_app_frame import context, frame, shell_frame

from railroads.land import buy_lot_steps
from railroads.saves import SavedGame, Saves
from railroads.sim.constructions import Building, BuildingKind
from railroads.sim.finance import START_CASH
from railroads.sim.game import Game
from railroads.sim.goods import Good
from railroads.sim.steps import finish
from railroads.sim.tilemap import LAYERS
from railroads.sim.trains import Stop
from railroads.tutorial.game import make_tutorial
from railroads.tutorial.place import (
    ASHFORD,
    ASHFORD_STATION,
    BROOKBY_STATION,
    BUILDINGS,
    MAP_COLS,
    MAP_ROWS,
    SHOWN_FARM,
)
from railroads.tutorial.steps import STEPS, Seen, Step, Tutorial
from railroads.ui.app import App, Shell
from railroads.ui.main_menu import TutorialChoice
from railroads.ui.map_view import Lens

__all__ = ["context"]

NOTHING_SEEN = Seen()


def step(title: str) -> Step:
    return next(step for step in STEPS if step.title == title)


def built(saved: SavedGame) -> list[tuple[tuple[int, int], str]]:
    return sorted(
        (building.cell, building.kind.value)
        for building in saved.game.constructions
        if isinstance(building, Building)
    )


def with_stations() -> Game:
    game = make_tutorial().game
    game.build_station(ASHFORD_STATION)
    game.build_station(BROOKBY_STATION)
    return game


def with_line() -> Game:
    game = with_stations()
    game.lay_track(ASHFORD_STATION, BROOKBY_STATION)
    return game


def with_train(cars: list[Good], back: list[Good] | None = None, held: bool = False) -> Game:
    game = with_line()
    train = game.railway.buy()
    assert train is not None
    train.stops = [Stop(ASHFORD_STATION, list(cars)), Stop(BROOKBY_STATION, list(back or []))]
    train.held = held
    return game


def test_the_tutorial_starts_with_the_same_land_and_buildings_every_time() -> None:
    first = make_tutorial()

    second = make_tutorial()

    assert built(first) == built(second)
    for layer in LAYERS:
        assert np.array_equal(
            getattr(first.game.tile_map, layer), getattr(second.game.tile_map, layer)
        )
    assert first.game.finances.cash == second.game.finances.cash
    assert first.game.lots is not None and second.game.lots is not None
    assert first.game.lots.offers == second.game.lots.offers


def test_the_tutorial_starts_with_the_buildings_drawn_for_it_and_nothing_grown() -> None:
    drawn = sorted((cell, kind.value) for cell, kinds in BUILDINGS.items() for kind in kinds)

    saved = make_tutorial()

    assert built(saved) == drawn
    assert saved.game.tile_map.grid.shape == (MAP_ROWS, MAP_COLS)
    assert saved.game.growth.year == 1850


def test_the_tutorial_plays_its_first_year_the_same_every_time() -> None:
    first, second = make_tutorial(), make_tutorial()

    for saved in (first, second):
        for _ in range(12):
            saved.game.month(saved.rng)

    assert built(first) == built(second)


def test_the_tutorial_names_its_stations_after_its_own_towns() -> None:
    game = with_stations()

    names = [game.railway.name_of(station) for station in game.railway.stations()]

    assert sorted(names) == ["Ashford", "Brookby"]


def test_the_tutorial_starts_on_its_first_step_looking_at_ashford() -> None:
    saved = make_tutorial()

    assert saved.tutorial == Tutorial()
    assert saved.camera is not None
    assert (saved.camera[0], saved.camera[1]) == pytest.approx(
        saved.game.tile_map.grid.center(*ASHFORD)
    )


@pytest.mark.parametrize(
    ("title", "game"),
    [
        ("1. A station", lambda: make_tutorial().game),
        ("1. A second station", lambda: make_tutorial().game),
        ("1. Track", with_stations),
        ("2. A train", with_line),
        ("2. A train", lambda: with_train([Good.PASSENGERS])),
        ("2. A train", lambda: with_train([Good.PASSENGERS, Good.WHEAT], held=True)),
        ("3. Cars and hubs", lambda: with_train([Good.PASSENGERS, Good.WHEAT])),
        ("4. Making and using goods", lambda: make_tutorial().game),
        ("5. Satisfaction", lambda: make_tutorial().game),
        ("6. Passengers and towns", lambda: make_tutorial().game),
        ("7. Shares", lambda: make_tutorial().game),
        ("8. Tech", lambda: make_tutorial().game),
        ("9. Fuel and engines", lambda: make_tutorial().game),
        ("10. More land", lambda: make_tutorial().game),
    ],
)
def test_a_step_is_not_done_before_its_task(title: str, game) -> None:
    played = game()

    done = step(title).done

    assert done is not None
    assert not done(played, NOTHING_SEEN)


def test_building_a_station_at_ashford_does_the_first_step() -> None:
    game = make_tutorial().game

    game.build_station(ASHFORD_STATION)

    assert step("1. A station").done(game, NOTHING_SEEN)
    assert not step("1. A second station").done(game, NOTHING_SEEN)


def test_building_a_station_at_brookby_does_the_second_station_step() -> None:
    game = make_tutorial().game

    game.build_station(BROOKBY_STATION)

    assert step("1. A second station").done(game, NOTHING_SEEN)


def test_laying_track_between_the_stations_does_the_track_step() -> None:
    game = with_stations()

    game.lay_track(ASHFORD_STATION, BROOKBY_STATION)

    assert step("1. Track").done(game, NOTHING_SEEN)


def test_a_train_let_go_with_passengers_and_food_does_the_train_step() -> None:
    game = with_train([Good.PASSENGERS, Good.WHEAT])

    done = step("2. A train").done(game, NOTHING_SEEN)

    assert done


def test_cars_on_the_way_back_do_the_cars_step() -> None:
    game = with_train([Good.PASSENGERS, Good.WHEAT], [Good.PASSENGERS])

    done = step("3. Cars and hubs").done(game, NOTHING_SEEN)

    assert done


@pytest.mark.parametrize(
    ("title", "seen"),
    [
        ("4. Making and using goods", Seen(selected=SHOWN_FARM)),
        ("5. Satisfaction", Seen(lens=Lens.SATISFACTION.value)),
        ("6. Passengers and towns", Seen(lens=Lens.TOWNS.value)),
        ("8. Tech", Seen(tech_open=True)),
        ("9. Fuel and engines", Seen(buying_train=True)),
    ],
)
def test_a_step_waiting_for_something_opened_is_done_once_it_is_open(
    title: str, seen: Seen
) -> None:
    game = make_tutorial().game

    done = step(title).done

    assert done is not None and done(game, seen)


def test_the_satisfaction_step_is_not_done_by_the_towns_lens() -> None:
    game = make_tutorial().game

    done = step("5. Satisfaction").done

    assert done is not None and not done(game, Seen(lens=Lens.TOWNS.value))


def test_buying_shares_of_the_farm_does_the_shares_step() -> None:
    game = make_tutorial().game

    said = game.growth.owners.buy(SHOWN_FARM, BuildingKind.FARM)

    assert said is None
    assert step("7. Shares").done(game, NOTHING_SEEN)


def test_the_land_step_gives_the_money_for_a_lot_and_buying_one_does_it() -> None:
    saved = make_tutorial()
    game = saved.game
    assert game.lots is not None
    cash = game.finances.cash
    land = step("10. More land")
    land.begin(game)
    lot = game.lots.offers[0]

    bought = finish(buy_lot_steps(saved, lot, report=lambda text: None))

    assert bought.game.finances.cash == pytest.approx(cash)
    assert bought.tutorial == saved.tutorial
    assert land.done(bought.game, NOTHING_SEEN)


def test_the_land_step_marks_the_middle_of_every_lot_for_sale() -> None:
    game = make_tutorial().game
    assert game.lots is not None

    cells = step("10. More land").cells(game)

    assert len(cells) == len(game.lots.offers)
    assert all(
        not game.tile_map.grid.contains(*cell) or game.tile_map.is_locked(cell) for cell in cells
    )


def test_a_step_that_only_explains_is_done_at_once() -> None:
    tutorial = Tutorial()

    tutorial.next()

    assert tutorial.step == 1


def test_next_waits_for_the_task_and_skip_does_not() -> None:
    tutorial = Tutorial(step=1)
    game = make_tutorial().game
    tutorial.update(game, NOTHING_SEEN)

    tutorial.next()
    waited = tutorial.step
    tutorial.skip()

    assert waited == 1
    assert tutorial.step == 2


def test_a_task_done_once_stays_done_when_undone() -> None:
    tutorial = Tutorial(step=STEPS.index(step("8. Tech")))
    game = make_tutorial().game
    tutorial.update(game, Seen(tech_open=True))

    tutorial.update(game, NOTHING_SEEN)
    tutorial.next()

    assert tutorial.step == STEPS.index(step("9. Fuel and engines"))


def test_a_step_begins_only_once() -> None:
    tutorial = Tutorial(step=STEPS.index(step("10. More land")))
    game = make_tutorial().game
    tutorial.update(game, NOTHING_SEEN)
    game.finances.spend(1_000.0, "track")

    tutorial.update(game, NOTHING_SEEN)

    assert game.lots is not None
    assert game.finances.cash == START_CASH + game.lots.price() - 1_000.0


def test_the_tutorial_goes_on_where_it_was_after_saving_and_loading(tmp_path: Path) -> None:
    saved = make_tutorial()
    saved = replace(saved, tutorial=Tutorial(step=5, finished={1, 2}, begun=5))
    saves = Saves(tmp_path)

    saves.save("tutorial", saved)
    loaded = saves.load("tutorial")

    assert loaded.tutorial == saved.tutorial


def test_an_app_keeps_the_tutorial_in_what_it_saves() -> None:
    saved = make_tutorial()

    app = App.from_saved(saved)

    assert app.saved().tutorial is saved.tutorial
    assert pickle.loads(pickle.dumps(app.saved().tutorial)) == saved.tutorial


@pytest.mark.parametrize("index", range(len(STEPS)))
def test_every_step_draws_with_what_it_points_at(context: None, index: int) -> None:
    saved = make_tutorial()
    saved = replace(saved, tutorial=Tutorial(step=index))
    app = App.from_saved(saved)
    app.view.build.trains.open = True

    for _ in range(3):
        frame(app)

    assert app.tutorial is not None and app.tutorial.begun == index


def test_the_tutorial_chosen_in_the_menu_opens_on_its_first_step(
    context: None, tmp_path: Path
) -> None:
    shell = Shell(Saves(tmp_path))

    shell._choose(TutorialChoice())
    assert shell._task is not None
    shell._task.result()
    shell_frame(shell)

    assert shell.app is not None
    assert shell.app.tutorial == Tutorial()
    assert shell.app.game.lots is not None
    assert shell.app.view.land.wanted is None


def test_the_tutorial_plays_through_to_the_end_on_the_app_s_frames(context: None) -> None:
    app = App.from_saved(make_tutorial())
    game = app.game
    view = app.view

    def do_next(action=lambda: None) -> None:
        action()
        frame(app)
        assert app.tutorial is not None and app.tutorial.done
        app.tutorial.next()
        frame(app)

    def buy_train() -> None:
        train = game.railway.buy()
        assert train is not None
        train.stops = [
            Stop(ASHFORD_STATION, [Good.PASSENGERS, Good.WHEAT]),
            Stop(BROOKBY_STATION),
        ]

    def lens(chosen: Lens) -> None:
        view.lens = chosen

    def tech() -> None:
        app.tech.open = True

    def buying() -> None:
        view.build.trains.open = True
        view.build.trains.buy.open = True

    frame(app)
    do_next()
    do_next(lambda: game.build_station(ASHFORD_STATION))
    do_next(lambda: game.build_station(BROOKBY_STATION))
    do_next(lambda: game.lay_track(ASHFORD_STATION, BROOKBY_STATION))
    do_next(buy_train)
    do_next(lambda: game.railway.trains[0].stops[1].add_car(Good.PASSENGERS))
    do_next(lambda: setattr(view, "selected", SHOWN_FARM))
    do_next(lambda: lens(Lens.SATISFACTION))
    do_next(lambda: lens(Lens.TOWNS))
    do_next(lambda: game.growth.owners.buy(SHOWN_FARM, BuildingKind.FARM))
    do_next(tech)
    do_next(buying)
    view.build.trains.buy.open = False
    assert game.lots is not None
    saved = finish(buy_lot_steps(app.saved(), game.lots.offers[0], report=lambda text: None))
    app = App.from_saved(saved)
    do_next()
    tutorial = app.tutorial
    assert tutorial is not None and tutorial.current is not None
    tutorial.closed = True
    frame(app)

    assert tutorial.current is None
    assert tutorial.finished == set(range(1, len(STEPS) - 1))
