from collections.abc import Iterator

import pytest
from imgui_bundle import imgui, implot

from railroads.sim.goods import Good
from railroads.sim.stats import History
from railroads.ui.reports_window import ReportsWindow, report


@pytest.fixture
def frame() -> Iterator[None]:
    context = imgui.create_context()
    plots = implot.create_context()
    io = imgui.get_io()
    io.display_size = imgui.ImVec2(1280, 800)
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures.value
    imgui.new_frame()
    yield
    imgui.render()
    implot.destroy_context(plots)
    imgui.destroy_context(context)


def made_history() -> History:
    history = History()
    history.deliver(1901, Good.COAL, 3)
    history.deliver(1903, Good.COAL)
    history.deliver(1903, Good.WOOL, 2)
    history.close_year(1901, 1, 100)
    history.close_year(1902, 2, 120)
    return history


def test_the_report_runs_from_the_first_delivery_to_this_year_with_today_last() -> None:
    history = made_history()

    shown = report(history, 1903, stations=4, population=150)

    assert shown.delivery_years == [1901, 1902, 1903]
    assert shown.goods == [g for g in Good if g in (Good.COAL, Good.WOOL)]
    coal = shown.goods.index(Good.COAL)
    assert shown.cars[coal].tolist() == [3, 0, 1]
    assert shown.years == [1901, 1902, 1903]
    assert shown.stations == [1, 2, 4]
    assert shown.population == [100, 120, 150]


def test_with_nothing_delivered_the_report_holds_only_this_year() -> None:
    shown = report(History(), 1850, stations=0, population=10)

    assert shown.goods == []
    assert shown.cars.shape == (0, 1)
    assert shown.years == [1850]


@pytest.mark.parametrize("history", [made_history(), History()])
def test_the_window_draws_its_charts(frame: None, history: History) -> None:
    window = ReportsWindow()
    window.open = True

    window.draw(history, 1903, 4, 150)

    assert window.open
    assert (implot.get_colormap_index("Goods") != -1) == bool(history.delivered)
