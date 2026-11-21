"""Frames kept coming for a moment after a click or a key. hello_imgui draws a frame for each
input event and then idles, but imgui takes a press a frame after the move before it, and shows a
new window only on its second frame; in the browser, where idling draws nothing at all, a click
would wait for the next mouse move to be taken and a window opened by it to appear."""

import time

from imgui_bundle import hello_imgui, imgui

# How long frames keep coming at full speed after the mouse moves or a button or key goes down
# or up.
LIVELY_SECONDS = 0.5
KEYS = [
    key
    for key in imgui.Key
    if imgui.Key.named_key_begin.value <= key.value < imgui.Key.named_key_end.value
]


class Lively:
    def __init__(self) -> None:
        self._until = 0.0

    def keep(self) -> None:
        """Asks for the next frame while input came lately or is held."""
        if _input_now():
            self._until = time.monotonic() + LIVELY_SECONDS
        if time.monotonic() < self._until:
            hello_imgui.request_refresh()


def _input_now() -> bool:
    io = imgui.get_io()
    if io.mouse_delta.x or io.mouse_delta.y or io.mouse_wheel or io.mouse_down.any():
        return True
    if any(imgui.is_mouse_released(button) for button in range(3)):
        return True
    return any(imgui.is_key_down(key) or imgui.is_key_released(key) for key in KEYS)
