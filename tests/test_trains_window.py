from collections.abc import Iterator

import pytest
from imgui_bundle import hello_imgui, imgui
from test_app_frame import FakeTexture
from test_railway import joined_stations

from railroads.sim.goods import Good
from railroads.sim.trains import Stop, Train
from railroads.ui.trains_window import TrainsWindow, carried_in


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
        window.draw(railway, lambda _: {}, lambda: ground)
        imgui.end_frame()

    assert scrolls[-1] == 0.0
