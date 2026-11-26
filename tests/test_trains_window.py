from collections.abc import Iterator

import pytest
from imgui_bundle import hello_imgui, imgui
from screens import FakeTexture, gui_context

from builders import joined_stations
from railroads.catalogue import engine_figures
from railroads.sim.demand import StationDemand
from railroads.sim.goods import General, Good, Need
from railroads.sim.railway import Railway, Ride
from railroads.sim.trade import SCARCE_WORTH
from railroads.sim.trains import ENGINES, EXPRESS_FARE, OPENED_BY, Car, Stop, Train
from railroads.ui import buy_train
from railroads.ui.station_offers import Offer
from railroads.ui.train_texts import (
    carried_in,
    ride_rows,
    train_warning,
)
from railroads.ui.trains_window import (
    SCHEDULES,
    TRAINS,
    TrainsWindow,
    general_wanted,
    sort_offers,
)
from railroads.words import tonnes_text


def train_calling_at(*stops: Stop) -> Train:
    return Train("Test", list(stops))


def test_a_stop_offers_the_cars_the_train_brings_in_from_the_stop_before() -> None:
    train = train_calling_at(
        Stop((1, 1), [Good.CATTLE, Good.CATTLE, Good.COAL]), Stop((5, 5)), Stop((9, 9))
    )

    carried = carried_in(train, 1)

    assert carried == [Good.CATTLE, Good.COAL]


def test_the_first_stop_offers_what_the_train_brings_back_from_the_last() -> None:
    train = train_calling_at(Stop((1, 1)), Stop((5, 5), [Good.WHEAT]))

    carried = carried_in(train, 0)

    assert carried == [Good.WHEAT]


def test_the_list_draws_a_twenty_five_car_train_led_by_a_dining_car(gui: None) -> None:
    _, railway = joined_stations()
    railway.finances.cash = 1e9
    railway.buy()
    train = railway.trains[0]
    train.cars = [Car(Good.PASSENGERS, dining=True)]
    train.cars += [Car(Good.COAL, load=5.0) for _ in range(24)]
    window = TrainsWindow(open=True)

    draw_frames(window, railway)

    assert window.chosen == train.name


@pytest.fixture
def gui(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    with gui_context():
        monkeypatch.setattr(
            hello_imgui, "create_texture_gpu_from_rgba_data", lambda _: FakeTexture()
        )
        yield


def test_the_buy_button_stays_at_the_top_while_a_long_list_of_trains_scrolls(
    gui: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, railway = joined_stations()
    railway.finances.cash = 1e9
    for _ in range(60):
        railway.buy()
    window = TrainsWindow(open=True)
    ground = (imgui.ImTextureRef(1), railway.constructions.tile_map.grid.size())
    scrolls: list[float] = []
    button = imgui.button

    def watched(label: str, size: tuple[float, float] = (0, 0)) -> bool:
        if "Buy a train" in label:
            scrolls.append(imgui.get_scroll_max_y())
        return button(label, size)

    monkeypatch.setattr(imgui, "button", watched)

    for _ in range(3):
        imgui.new_frame()
        window.draw(railway, lambda _: {}, lambda: ground, picture)
        imgui.end_frame()

    assert scrolls[-1] == 0.0


def picture(_: str, __: int) -> tuple[imgui.ImTextureRef, tuple[float, float]]:
    return imgui.ImTextureRef(1), (40.0, 20.0)


def draw_frames(window: TrainsWindow, railway: Railway, count: int = 2) -> None:
    ground = (imgui.ImTextureRef(1), railway.constructions.tile_map.grid.size())
    for _ in range(count):
        imgui.new_frame()
        window.draw(railway, lambda _: {}, lambda: ground, picture)
        imgui.end_frame()


def pressing(monkeypatch: pytest.MonkeyPatch, wanted: str) -> None:
    button = imgui.button

    def pressed(label: str, size: tuple[float, float] = (0, 0)) -> bool:
        return button(label, size) or wanted in label

    monkeypatch.setattr(imgui, "button", pressed)


def test_a_train_bought_for_a_schedule_runs_it_and_the_buy_screen_closes(
    gui: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, railway = joined_stations()
    first = railway.buy()
    assert first is not None
    routed(railway, first, [Stop((6, 3)), Stop((6, 20))])
    schedule = railway.schedules.of(first)
    window = TrainsWindow(open=True)
    window.buy.open_for(schedule)
    pressing(monkeypatch, "Buy for")

    draw_frames(window, railway, 1)
    train = railway.trains[-1]

    assert len(railway.trains) == 2
    assert train.schedule == first.schedule
    assert window.chosen == train.name
    assert not window.buy.open


def test_a_train_bought_with_a_new_schedule_shows_that_schedule_to_give_it_stops(
    gui: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, railway = joined_stations()
    window = TrainsWindow(open=True)
    window.buy.open_for(None)
    pressing(monkeypatch, "Buy for")

    draw_frames(window, railway, 3)

    assert len(railway.trains) == 1
    assert window.chosen_schedule == railway.trains[0].schedule
    assert window.tab == SCHEDULES


def test_clicking_a_train_in_a_schedules_table_opens_that_train(
    gui: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, railway = joined_stations()
    train = railway.buy()
    assert train is not None
    window = showing_schedule(train)
    draw_frames(window, railway, 2)
    monkeypatch.setattr(window.schedule_table, "draw", lambda *args, **kwargs: (train.name, False))

    draw_frames(window, railway, 3)

    assert window.chosen == train.name
    assert window.tab == TRAINS


def test_closing_the_trains_window_closes_the_buy_screen_too(gui: None) -> None:
    _, railway = joined_stations()
    window = TrainsWindow(open=False)
    window.buy.open_for(None)

    draw_frames(window, railway, 1)

    assert not window.buy.open


def test_an_express_shows_its_dining_car_and_dearer_fares_among_its_figures() -> None:
    figures = dict(engine_figures(ENGINES["bullet"]))

    pulls, fares = figures["Pulls up to"], figures["Fares"]

    assert pulls.endswith("coaches and a dining car")
    assert fares == f"Passengers only, paying {EXPRESS_FARE - 1:.0%} more"


def test_an_engines_figures_show_the_fuel_it_burns_and_how_far_it_runs_between_stops() -> None:
    engine = ENGINES["planet"]

    figures = dict(engine_figures(engine, 1.75))

    assert figures["Burns"] == "0.75 coal a cell half loaded, about $2 a cell now"
    assert figures["Range"] == "21 cells between stops"


def test_an_engines_figures_say_which_supply_level_put_it_on_sale_and_when_it_was_first_built() -> (
    None
):
    figures = dict(engine_figures(ENGINES["goods"], opened=OPENED_BY["goods"]))

    assert figures["On sale from"] == "Steel and metals supply 11"
    assert figures["First built"] == str(ENGINES["goods"].year)


def test_a_train_whose_way_between_two_stops_is_longer_than_its_fuel_is_warned_of() -> None:
    _, railway = joined_stations()
    train = Train("Test", [Stop((6, 3)), Stop((6, 20))], engine="penydarren")

    warning = train_warning(railway, train)

    assert warning.startswith(f"From {railway.name_of((6, 3))} to {railway.name_of((6, 20))} is ")
    assert warning.endswith("Add a stop between them to refuel at.")


def test_stop_cards_warn_of_the_fuel_of_the_schedules_weakest_train_not_the_engine_standing_in() -> (
    None
):
    window = TrainsWindow()
    window._riders = [Train("A", engine="mixed_diesel"), Train("B", engine="pacific")]

    engine = window._fuel_engine(Train("Schedule", engine="big_boy"))

    assert engine == "pacific"


def test_stop_cards_of_a_schedule_run_by_electric_trains_warn_of_no_fuel() -> None:
    window = TrainsWindow()
    window._riders = [Train("A", engine="boxcab")]

    engine = window._fuel_engine(Train("Schedule", engine="big_boy"))

    assert not ENGINES[engine].tank


def test_stop_cards_of_a_schedule_with_no_trains_warn_of_the_engine_standing_in() -> None:
    window = TrainsWindow()

    engine = window._fuel_engine(Train("Schedule", engine="big_boy"))

    assert engine == "big_boy"


def routed(railway: Railway, train: Train, stops: list[Stop]) -> list[Stop]:
    schedule = railway.schedules.of(train)
    assert schedule is not None
    schedule.stops = stops
    railway.schedules.settle(schedule)
    return schedule.stops


def showing_schedule(train: Train) -> TrainsWindow:
    window = TrainsWindow(open=True, chosen_schedule=train.schedule)
    window._pick_tab = SCHEDULES
    return window


def test_stop_cards_scrolled_out_of_sight_are_not_drawn(
    gui: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, railway = joined_stations()
    train = railway.buy()
    assert train is not None
    routed(railway, train, [Stop((6, 3) if index % 2 else (6, 20)) for index in range(12)])
    window = showing_schedule(train)
    draw_frames(window, railway, 4)
    drawn: list[int] = []
    card = TrainsWindow._draw_stop

    def counted(self: TrainsWindow, *args: object) -> bool:
        drawn.append(1)
        return card(self, *args)

    monkeypatch.setattr(TrainsWindow, "_draw_stop", counted)

    draw_frames(window, railway, 1)

    assert 0 < len(drawn) < len(train.stops)


def test_the_cargo_line_says_goods_earn_freight_for_what_buildings_take_up_to_the_scarce_worth() -> (
    None
):
    ride = Ride(
        cells=10.0, cargo=120.0, fuel=5.0, loaded_cars=3.0, empty_cars=0.0, loaded=1, empty=0
    )

    label, amount, tip = ride_rows(ride)[0]

    assert (label, amount) == ("Cargo", 120.0)
    assert f"up to {SCARCE_WORTH:g} times as much" in tip
    assert "what the buildings around would take at once" in tip


def test_a_stop_card_stays_open_when_its_first_car_is_added(gui: None) -> None:
    _, railway = joined_stations()
    train = railway.buy()
    assert train is not None
    stops = routed(railway, train, [Stop((6, 3)), Stop((6, 20))])
    window = showing_schedule(train)
    draw_frames(window, railway)
    stops[1].add_car(Good.COAL)

    draw_frames(window, railway)

    assert window._folded[(train.schedule, 1)]


def test_moving_a_stop_later_swaps_it_with_the_next_keeping_its_cars_and_folding(
    gui: None,
) -> None:
    _, railway = joined_stations()
    train = railway.buy()
    assert train is not None
    train.stops = [Stop((6, 3), [Good.COAL]), Stop((6, 20)), Stop((6, 3), [Good.WHEAT])]
    train.heading = 2
    window = TrainsWindow(open=True, chosen=train.name)
    window._folded = {(train.name, 0): False, (train.name, 1): True}

    window._move_stop(train, 0, 1)

    assert [stop.cars for stop in train.stops] == [[], [Good.COAL], [Good.WHEAT]]
    assert window._folded == {(train.name, 0): True, (train.name, 1): False}
    assert train.heading == 1


def test_what_follows_a_stops_cars_stays_put_when_its_first_car_is_added(
    gui: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, railway = joined_stations()
    train = railway.buy()
    assert train is not None
    stops = routed(railway, train, [Stop((6, 3)), Stop((6, 20))])
    window = showing_schedule(train)
    window._folded = {(train.schedule, 0): True, (train.schedule, 1): False}
    below: list[float] = []
    cars = TrainsWindow._draw_cars

    def measured(self: TrainsWindow, *args: object) -> None:
        cars(self, *args)
        below.append(imgui.get_cursor_pos_y())

    monkeypatch.setattr(TrainsWindow, "_draw_cars", measured)
    draw_frames(window, railway)
    empty = below[-1]
    stops[0].add_car(Good.COAL)

    draw_frames(window, railway)

    assert below[-1] == empty


def test_the_buy_button_stays_in_sight_under_the_engines_figures(
    gui: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(buy_train, "PICTURE_HEIGHT", 600.0)
    _, railway = joined_stations()
    first = railway.buy()
    assert first is not None
    routed(railway, first, [Stop((6, 3)), Stop((6, 20))])
    window = TrainsWindow(open=True)
    window.buy.open_for(railway.schedules.of(first))
    ground = (imgui.ImTextureRef(1), railway.constructions.tile_map.grid.size())
    places: list[tuple[float, float]] = []
    button = imgui.button

    def watched(label: str, size: tuple[float, float] = (0, 0)) -> bool:
        clicked = button(label, size)
        if "Buy for" in label:
            shown = imgui.internal.get_current_window().inner_clip_rect.max.y
            places.append((imgui.get_item_rect_max().y, shown))
        return clicked

    def tall_picture(_: str, __: int) -> tuple[imgui.ImTextureRef, tuple[float, float]]:
        return imgui.ImTextureRef(1), (40.0, 70.0)

    monkeypatch.setattr(imgui, "button", watched)

    for _ in range(3):
        imgui.new_frame()
        window.draw(railway, lambda _: {}, lambda: ground, tall_picture)
        imgui.end_frame()

    bottom, edge = places[-1]
    assert bottom <= edge


@pytest.mark.parametrize(
    ("tonnes", "text"), [(12.5, "12.5"), (24.48, "24.5"), (12.0, "12"), (1234.0, "1,234")]
)
def test_tonnes_show_to_a_tenth_so_two_half_tonne_cars_do_not_look_as_if_they_fit(
    tonnes: float, text: str
) -> None:
    shown = tonnes_text(tonnes)

    assert shown == text


@pytest.mark.parametrize("engine", ["planet", "single", "tank", "goods"])
def test_the_table_draws_a_train_of_every_category_with_its_framed_engine(
    gui: None, engine: str
) -> None:
    _, railway = joined_stations()
    railway.finances.cash = 1e9
    railway.buy()
    train = railway.trains[0]
    train.engine = engine
    train.stops = [Stop(cell, [Good.PASSENGERS]) for cell in railway.stations()]
    window = TrainsWindow(open=True)

    draw_frames(window, railway)

    assert window.chosen == train.name


def test_goods_to_load_come_by_what_the_next_stop_wants_most_or_by_what_waits_most() -> None:
    demand = StationDemand(capacity={Need.FOOD: 40.0, Need(Good.PASSENGERS.value): 300.0})
    offers = [
        Offer(Good.COAL, 500.0, 0.0),
        Offer(Good.WHEAT, 80.0, 0.0),
        Offer(Good.PASSENGERS, 20.0, 0.0),
    ]

    wanted = [o.good for o in sort_offers(offers, demand, by_wanted=True)]
    waiting = [o.good for o in sort_offers(offers, demand, by_wanted=False)]

    assert wanted == [Good.PASSENGERS, Good.WHEAT, Good.COAL]
    assert waiting == [Good.COAL, Good.WHEAT, Good.PASSENGERS]


def test_the_schedules_tab_shows_the_first_schedule_with_its_trains_and_stops(gui: None) -> None:
    _, railway = joined_stations()
    train = railway.buy()
    assert train is not None
    routed(railway, train, [Stop((6, 3), [Good.COAL]), Stop((6, 20))])
    window = TrainsWindow(open=True)
    window._pick_tab = SCHEDULES

    draw_frames(window, railway)

    assert window.tab == SCHEDULES
    assert window.chosen_schedule == train.schedule


def test_a_schedules_stops_hold_cars_for_the_biggest_engine_on_sale_not_only_its_trains() -> None:
    _, railway = joined_stations()
    railway.year = 1850
    train = railway.buy("tank")
    assert train is not None

    schedule = railway.schedules.of(train)
    assert schedule is not None

    engine = railway.schedules.biggest_engine(schedule)

    assert ENGINES[engine].cars > ENGINES["tank"].cars


def test_a_schedules_stop_says_how_many_of_its_cars_its_trains_take(
    gui: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, railway = joined_stations()
    train = railway.buy()
    assert train is not None
    routed(railway, train, [Stop((6, 3), [Good.COAL] * 5), Stop((6, 20))])
    window = showing_schedule(train)
    window._folded = {(train.schedule, 0): True}
    notes: list[str] = []
    monkeypatch.setattr("railroads.ui.trains_window.colored", lambda _, text: notes.append(text))

    draw_frames(window, railway, 3)

    assert any(note.endswith("take the first 3") for note in notes)


def test_a_trains_view_shows_its_stops_without_cards_to_change_them(
    gui: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, railway = joined_stations()
    train = railway.buy()
    assert train is not None
    routed(railway, train, [Stop((6, 3), [Good.COAL]), Stop((6, 20))])
    window = TrainsWindow(open=True, chosen=train.name)
    cards: list[int] = []
    monkeypatch.setattr(TrainsWindow, "_draw_stop", lambda *_: cards.append(1))

    draw_frames(window, railway, 3)

    assert cards == []
    assert len(train.stops) == 2


@pytest.mark.parametrize(("on_sale", "said"), [(["tank"], True), (["tank", "planet"], False)])
def test_a_schedule_with_no_trains_says_when_no_engine_on_sale_could_run_it(
    gui: None, monkeypatch: pytest.MonkeyPatch, on_sale: list[str], said: bool
) -> None:
    _, railway = joined_stations()
    railway.year = 1850
    schedule = railway.schedules.new()
    schedule.stops = [Stop((6, 3)), Stop((6, 20))]
    monkeypatch.setattr(railway, "engines", lambda: on_sale)
    window = TrainsWindow(open=True, chosen_schedule=schedule.name)
    window._pick_tab = SCHEDULES
    notes: list[str] = []
    monkeypatch.setattr("railroads.ui.trains_window.colored", lambda _, text: notes.append(text))

    draw_frames(window, railway, 3)

    assert any("No engine on sale could run it" in note for note in notes) == said


def test_hovering_a_car_on_a_trains_stops_shows_on_its_map_where_that_good_is_wanted(
    gui: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, railway = joined_stations()
    train = railway.buy()
    assert train is not None
    routed(railway, train, [Stop((6, 3), [Good.COAL]), Stop((6, 20))])
    window = TrainsWindow(open=True, chosen=train.name)
    monkeypatch.setattr(imgui, "is_item_hovered", lambda *_: True)

    draw_frames(window, railway, 2)

    assert window.map.good == Good.COAL


@pytest.mark.parametrize(
    ("next_stop_wants", "wanted"),
    [({Need.COAL: 40.0}, False), ({Need.FISH: 40.0}, False), ({Need.FOOD: 40.0}, True)],
)
def test_a_general_food_car_is_wanted_only_where_the_next_stop_wants_every_food_waiting_here(
    next_stop_wants: dict[Need, float], wanted: bool
) -> None:
    demand = StationDemand(capacity=next_stop_wants)
    waiting = {Good.WHEAT: 30.0, Good.FISH: 10.0}

    shown = general_wanted(General.FOOD, lambda good: waiting.get(good, 0.0), demand)

    assert shown == wanted
