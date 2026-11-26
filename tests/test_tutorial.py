import pickle
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from imgui_bundle import imgui
from test_app_frame import context, frame, shell_frame

from railroads.land import buy_lot_steps
from railroads.saves import SavedGame, Saves
from railroads.sim.anger import ANGER_SURCHARGE
from railroads.sim.constructions import Building, BuildingKind
from railroads.sim.demolition import DEMOLITION_PER_POINT
from railroads.sim.finance import START_CASH, TRACK_PRICE
from railroads.sim.game import Game
from railroads.sim.goods import FREIGHT_SHARE, Good
from railroads.sim.ownership import HOLD_MONTHS, SHARES
from railroads.sim.points import Points
from railroads.sim.railway import PLATFORM_LEVEL, PLATFORM_REACH, QUADRUPLE_ERA, REACH_LEVEL
from railroads.sim.reach import LOCAL_REACH
from railroads.sim.schedules import Schedule
from railroads.sim.steps import finish
from railroads.sim.tilemap import LAYERS
from railroads.sim.trade import BONUS_PER_GOOD, SCARCE_WORTH
from railroads.sim.trains import Stop, Train
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
from railroads.tutorial.steps import SHARE_LOTS, STEPS, Seen, Step, Tutorial
from railroads.ui.app import App, Shell
from railroads.ui.main_menu import TutorialChoice
from railroads.ui.map_view import Lens
from railroads.words import dollars, share_of

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


def with_train(cars: list[Good], back: list[Good] | None = None) -> Game:
    game = with_line()
    train = game.railway.buy()
    assert train is not None
    routed(
        game, train, [Stop(ASHFORD_STATION, list(cars)), Stop(BROOKBY_STATION, list(back or []))]
    )
    return game


def routed(game: Game, train: Train, stops: list[Stop]) -> Schedule:
    schedules = game.railway.schedules
    schedule = schedules.of(train)
    assert schedule is not None
    schedule.stops[:] = stops
    schedules.settle(schedule, replan=True)
    return schedule


def test_the_tutorial_starts_with_the_same_land_and_buildings_every_time() -> None:
    first = make_tutorial()

    second = make_tutorial()
    differing = [
        layer
        for layer in LAYERS
        if not np.array_equal(
            getattr(first.game.tile_map, layer), getattr(second.game.tile_map, layer)
        )
    ]

    assert built(first) == built(second)
    assert differing == []
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


def test_the_line_between_the_towns_crosses_only_flat_open_land() -> None:
    game = with_stations()
    cash = game.finances.cash
    laid = BROOKBY_STATION[1] - ASHFORD_STATION[1] - 1

    done = game.lay_track(ASHFORD_STATION, BROOKBY_STATION)

    assert done.ok
    assert cash - game.finances.cash == pytest.approx(laid * TRACK_PRICE)


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
        ("3. Cars and hubs", lambda: with_train([Good.PASSENGERS, Good.WHEAT])),
        ("4. Making and using goods", lambda: make_tutorial().game),
        ("5. Satisfaction", lambda: make_tutorial().game),
        ("6. Passengers and towns", lambda: make_tutorial().game),
        ("7. Shares", lambda: make_tutorial().game),
        ("8. Supply and demand", lambda: make_tutorial().game),
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


def test_cars_on_the_way_back_do_the_cars_step() -> None:
    game = with_train([Good.PASSENGERS, Good.WHEAT], [Good.PASSENGERS])

    done = step("3. Cars and hubs").done(game, NOTHING_SEEN)

    assert done


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
    tutorial = Tutorial(step=STEPS.index(step("8. Supply and demand")))
    game = make_tutorial().game
    tutorial.update(game, Seen(boards_open=True))

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


# How many steps a month of the trains' running is taken in by months_pass.
TRAIN_STEPS = 50


def months_pass(app: App, months: int) -> None:
    """Lets months pass on the app's frames as they would in play, the trains running through
    each, picking the lowest tiles a player could on each 1 January the game waits on."""
    for _ in range(months):
        for _ in range(TRAIN_STEPS):
            app.view.time_passed(1 / TRAIN_STEPS)
        app.clock.seconds = app.clock.seconds_per_month - 0.001
        frame(app)
        while app._turning is not None or app.tiles.choosing:
            if app.tiles.choosing and app._turning is None:
                tiles = app.game.railway.tiles
                board = app.tiles.board
                while tiles.picks_left(board):
                    tiles.pick(
                        board, next(t for t in tiles.levels(board) if tiles.can_pick(board, t))
                    )
                app.tiles.confirm()
            frame(app)


def test_the_tutorial_plays_through_to_the_end_with_the_year_going_by(context: None) -> None:
    app = App.from_saved(make_tutorial())
    game = app.game
    view = app.view
    shown: list[str] = []

    def do_next(action=lambda: None, months: int = 0) -> None:
        assert app.tutorial is not None and app.tutorial.current is not None
        shown.append(app.tutorial.current.title)
        action()
        frame(app)
        months_pass(app, months)
        assert app.tutorial.done, app.tutorial.current.title
        app.tutorial.next()
        frame(app)

    def buy_train() -> None:
        train = game.railway.buy()
        assert train is not None
        routed(
            game,
            train,
            [Stop(ASHFORD_STATION, [Good.PASSENGERS, Good.WHEAT]), Stop(BROOKBY_STATION)],
        )

    def buying() -> None:
        view.build.trains.open = True
        view.build.trains.buy.open = True

    frame(app)
    do_next()
    do_next(lambda: game.build_station(ASHFORD_STATION))
    do_next(lambda: game.build_station(BROOKBY_STATION))
    do_next(lambda: game.lay_track(ASHFORD_STATION, BROOKBY_STATION))
    do_next(buy_train)
    with_a_train = game.finances.cash
    do_next(lambda: game.railway.schedules.all[0].stops[1].add_car(Good.PASSENGERS), months=12)
    earned = game.finances.cash - with_a_train
    do_next(lambda: setattr(view, "selected", SHOWN_FARM))
    do_next(lambda: setattr(view, "lens", Lens.SATISFACTION))
    do_next(lambda: setattr(view, "lens", Lens.TOWNS))
    do_next(lambda: game.growth.owners.buy(SHOWN_FARM, BuildingKind.FARM))
    do_next(lambda: setattr(app.tiles, "open", True))
    app.tiles.open = False
    do_next(buying)
    view.build.trains.buy.open = False
    frame(app)
    assert game.lots is not None
    saved = finish(buy_lot_steps(app.saved(), game.lots.offers[0], report=lambda text: None))
    app = App.from_saved(saved)
    frame(app)
    do_next()
    tutorial = app.tutorial
    assert tutorial is not None
    assert tutorial.current is not None
    tutorial.closed = True
    frame(app)

    assert shown == [step.title for step in STEPS[:-1]]
    assert earned > 0
    assert app.clock.date == "January 1851"
    assert tutorial.current is None
    assert tutorial.finished == set(range(1, len(STEPS) - 1))


def test_the_tutorial_window_comes_back_onto_a_screen_that_shrank(context: None) -> None:
    app = App.from_saved(make_tutorial())
    frame(app)
    io = imgui.get_io()
    io.display_size = imgui.ImVec2(900, 600)

    frame(app)
    window = imgui.internal.find_window_by_name("###tutorial")

    assert window is not None
    assert window.pos.x + window.size.x <= 900
    assert window.pos.y + window.size.y <= 600


def test_the_tutorial_gives_the_numbers_the_game_has() -> None:
    numbers = [
        f"{FREIGHT_SHARE:.0%} of a good's market price, up to {SCARCE_WORTH:g} times",
        f"add {BONUS_PER_GOOD:.0%} each",
        f"{SHARES:,} shares, bought {SHARE_LOTS} at a time",
        f"sold {HOLD_MONTHS // 12} years after",
        f"up to {LOCAL_REACH} cells away",
        f"up to level {REACH_LEVEL}; from level {PLATFORM_LEVEL}",
        f"a platform up to {PLATFORM_REACH} cells away",
        f"from {dollars(DEMOLITION_PER_POINT[Points.POPULATION])} each, and {share_of(ANGER_SURCHARGE)}",
        f"from era {QUADRUPLE_ERA}",
    ]

    text = " ".join(" ".join(step.text) for step in STEPS)

    assert [number for number in numbers if number not in text] == []
