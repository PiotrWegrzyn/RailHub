import ast
from collections.abc import Iterator
from pathlib import Path

import pytest
from imgui_bundle import imgui

from railroads.ui.ornate import begin_ornate, end_ornate, keep_in_front

UI = Path(__file__).resolve().parents[1] / "src" / "railroads" / "ui"


@pytest.fixture
def context() -> Iterator[None]:
    gui = imgui.create_context()
    io = imgui.get_io()
    io.display_size = imgui.ImVec2(1280, 800)
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures.value
    yield
    imgui.destroy_context(gui)


def draw_long_window(scroll_to: float | None = None) -> None:
    imgui.new_frame()
    imgui.set_next_window_pos((0, 0))
    imgui.set_next_window_size((300, 200))
    begin_ornate("Long###long")
    for line in range(200):
        imgui.text(f"line {line}")
    if scroll_to is not None:
        imgui.set_scroll_y(scroll_to)
    end_ornate()
    imgui.end_frame()


def test_a_windows_title_and_close_cross_stay_at_its_top_while_what_it_holds_scrolls(
    context: None,
) -> None:
    for _ in range(3):
        draw_long_window()
    draw_long_window(scroll_to=1000)
    draw_long_window()

    window = imgui.internal.find_window_by_name("Long###long")
    body = window.dc.child_windows[0]
    assert window.scroll.y == 0
    assert body.scroll.y > 0


def titled_windows() -> list[str]:
    """Every window begun with imgui.begin and a title of its own, outside ornate.py."""
    titled = []
    for path in UI.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for call in ast.walk(tree):
            if not (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "begin"
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id == "imgui"
                and call.args
            ):
                continue
            name = call.args[0]
            hidden = isinstance(name, ast.Constant) and str(name.value).startswith("##")
            if not hidden and path.name != "ornate.py":
                titled.append(f"{path.name}:{call.lineno}")
    return titled


def test_every_window_with_a_title_is_begun_with_its_title_and_close_cross_fixed_at_its_top() -> (
    None
):
    sources = UI

    titled = titled_windows()

    assert sources.is_dir()
    assert titled == []


def draw_side_by_side(focus_big: bool = False) -> None:
    imgui.new_frame()
    imgui.set_next_window_pos((0, 0))
    imgui.set_next_window_size((400, 300))
    if focus_big:
        imgui.set_next_window_focus()
    begin_ornate("Big###big")
    imgui.text("a big window")
    end_ornate()
    imgui.set_next_window_pos((20, 20))
    imgui.set_next_window_size((200, 150))
    begin_ornate("Small###small")
    keep_in_front()
    imgui.text("a small window kept in front")
    end_ornate()
    imgui.end_frame()


def test_a_window_kept_in_front_stays_in_front_of_one_clicked_after_it(context: None) -> None:
    draw_side_by_side()
    draw_side_by_side(focus_big=True)

    draw_side_by_side()

    order = [window.name for window in imgui.get_current_context().windows]
    assert order.index("Small###small") > order.index("Big###big")


def test_a_self_sizing_window_taller_than_the_screen_ends_at_its_bottom_and_scrolls(
    context: None,
) -> None:
    bottom, scroll = 0.0, 0.0
    for _ in range(4):
        imgui.new_frame()
        imgui.set_next_window_pos((0, 100))
        imgui.set_next_window_size_constraints((300, 0), (300, 2000))
        begin_ornate("Tall###tall", imgui.WindowFlags_.always_auto_resize.value)
        for line in range(200):
            imgui.text(f"line {line}")
        scroll = imgui.get_scroll_max_y()
        end_ornate()
        window = imgui.internal.find_window_by_name("Tall###tall")
        bottom = window.pos.y + window.size.y
        imgui.end_frame()

    assert bottom <= 800
    assert scroll > 0
