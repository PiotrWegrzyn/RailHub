from railroads.ui.build_menu import BuildMenu, Tool


def test_opening_the_trains_window_puts_the_track_tool_away() -> None:
    menu = BuildMenu(tool=Tool.TRACK, track_from=(4, 4))

    menu.toggle_trains()

    assert menu.trains.open
    assert (menu.tool, menu.track_from) == (Tool.NONE, None)
