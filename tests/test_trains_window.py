from collections.abc import Iterator

import pytest
from imgui_bundle import hello_imgui, imgui
from test_app_frame import FakeTexture
from test_railway import joined_stations

from railroads.sim.goods import Good
from railroads.sim.railway import Railway
from railroads.sim.trains import ENGINES, EXPRESS_FARE, Car, Stop, Train
from railroads.ui.buy_train import engine_figures
from railroads.ui.trains_window import ROW_CAR_SIZES, TrainsWindow, carried_in, row_car_size


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


def test_a_short_train_shows_its_cars_largest_in_the_list() -> None:
    sizes = row_car_size(4, 160.0)

    assert sizes == ROW_CAR_SIZES[0]


def test_a_sixteen_car_train_fits_its_row_at_the_smallest_size() -> None:
    car_width, _, gap = row_car_size(16, 160.0)

    assert (car_width, gap) == ROW_CAR_SIZES[-1][::2]
    assert 16 * (car_width + gap) - gap <= 160.0


def test_cars_too_many_for_the_smallest_size_are_narrowed_to_fit_the_row() -> None:
    car_width, _, gap = row_car_size(25, 160.0)

    assert car_width < ROW_CAR_SIZES[-1][0]
    assert 25 * (car_width + gap) - gap == pytest.approx(160.0)


def test_the_list_draws_a_long_express_with_its_dining_car(gui: None) -> None:
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
    monkeypatch.setattr(hello_imgui, "create_texture_gpu_from_rgba_data", lambda _: FakeTexture())
    context = imgui.create_context()
    io = imgui.get_io()
    io.display_size = imgui.ImVec2(1280, 800)
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures.value
    yield
    imgui.destroy_context(context)


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


def test_a_train_bought_on_the_buy_screen_stands_until_the_screen_is_closed(
    gui: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, railway = joined_stations()
    window = TrainsWindow(open=True)
    window.buy.open_for(None)
    pressing(monkeypatch, "Buy for")

    draw_frames(window, railway, 1)

    assert len(railway.trains) == 1
    train = railway.trains[0]
    assert window.chosen == train.name
    assert window.buy.bought == train.name
    assert train.held


def test_closing_the_buy_screen_lets_the_trains_bought_on_it_set_off(gui: None) -> None:
    _, railway = joined_stations()
    train = railway.buy()
    assert train is not None
    train.held = True
    window = TrainsWindow(open=True)
    window.buy.open_for(None)
    window.buy.bought = train.name
    draw_frames(window, railway)
    held_while_open = train.held

    window.buy.open = False
    draw_frames(window, railway)

    assert held_while_open
    assert not train.held


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

    assert figures["Burns"] == "0.75 coal a cell, about $2 a cell now"
    assert figures["Range"] == "20 cells between stops"


def test_stop_cards_scrolled_out_of_sight_are_not_drawn(
    gui: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, railway = joined_stations()
    train = railway.buy()
    assert train is not None
    train.stops = [Stop((6, 3) if index % 2 else (6, 20)) for index in range(12)]
    window = TrainsWindow(open=True, chosen=train.name)
    draw_frames(window, railway, 3)
    drawn: list[int] = []
    card = TrainsWindow._draw_stop

    def counted(self: TrainsWindow, *args: object) -> bool:
        drawn.append(1)
        return card(self, *args)

    monkeypatch.setattr(TrainsWindow, "_draw_stop", counted)

    draw_frames(window, railway, 1)

    assert 0 < len(drawn) < len(train.stops)
