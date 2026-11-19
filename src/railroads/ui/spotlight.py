"""Pulsing brass frames round what the tutorial wants clicked (railroads.tutorial.steps): the
windows call spot() with a name after each button the tutorial may point at, and the ones it
wants this frame (want) are framed."""

import math
import time
from collections.abc import Iterable

from imgui_bundle import hello_imgui, imgui

from railroads.ui.theme import BRASS, rgba_u32

# How fast the frames pulse, in beats a second, and how far they reach past what they frame.
PULSE_HZ = 1.5
REACH = 3.0

_wanted: set[str] = set()


def want(names: Iterable[str]) -> None:
    """The names to frame from now on."""
    _wanted.clear()
    _wanted.update(names)


def wanted() -> frozenset[str]:
    return frozenset(_wanted)


def pulse() -> float:
    """Where the pulse is, from 0 to 1."""
    return 0.5 + 0.5 * math.sin(time.monotonic() * PULSE_HZ * 2 * math.pi)


def pulsing_brass() -> tuple[float, float, float, float]:
    return (*BRASS[:3], 0.45 + 0.55 * pulse())


def spot(name: str) -> None:
    """Frames the item just drawn if the tutorial wants it."""
    if name in _wanted:
        frame(imgui.get_item_rect_min(), imgui.get_item_rect_max())


def frame(low: imgui.ImVec2, high: imgui.ImVec2) -> None:
    """A pulsing frame round a rectangle on the screen, kept moving. It is drawn with the current
    window, so windows in front of it hide it as they hide what it frames, but past its edges."""
    reach = REACH + 2.0 * pulse()
    draw_list = imgui.get_window_draw_list()
    draw_list.push_clip_rect_full_screen()
    draw_list.add_rect(
        (low.x - reach, low.y - reach),
        (high.x + reach, high.y + reach),
        rgba_u32(pulsing_brass()),
        6.0,
        3.0,
    )
    draw_list.pop_clip_rect()
    hello_imgui.request_refresh()
