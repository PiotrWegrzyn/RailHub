import ctypes

import numpy as np
import pytest

from railroads.ui.pixel import webgl


class FakeEmscripten:
    """Emscripten's exported GL functions, reading and writing memory at the addresses given."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def _glGenTextures(self, count: int, ids: int) -> None:
        ctypes.memmove(ids, np.array([7] * count, np.uint32).tobytes(), 4 * count)

    def _glGetIntegerv(self, name: int, values: int) -> None:
        ctypes.memmove(values, np.array([1, 2, 3, 4], np.int32).tobytes(), 16)

    def _glGetShaderInfoLog(self, shader: int, size: int, length: int, log: int) -> None:
        ctypes.memmove(log, b"oops", 4)
        ctypes.memmove(length, np.array([4], np.int32).tobytes(), 4)

    def _glBindTexture(self, target: int, texture: int) -> None:
        self.calls.append(("glBindTexture", target, texture))


@pytest.fixture
def emscripten(monkeypatch: pytest.MonkeyPatch) -> FakeEmscripten:
    fake = FakeEmscripten()
    monkeypatch.setattr(webgl, "module", lambda: fake)
    return fake


def test_a_new_texture_is_the_number_gl_wrote_at_the_address_handed_over(
    emscripten: FakeEmscripten,
) -> None:
    texture = webgl.glGenTextures(1)

    assert texture == 7


def test_the_viewport_reads_four_values_and_other_settings_one(emscripten: FakeEmscripten) -> None:
    viewport = webgl.glGetIntegerv(webgl.GL_VIEWPORT)
    program = webgl.glGetIntegerv(webgl.GL_CURRENT_PROGRAM)

    assert list(viewport) == [1, 2, 3, 4]
    assert program == 1


def test_a_shader_log_is_read_up_to_the_length_gl_gave(emscripten: FakeEmscripten) -> None:
    log = webgl.glGetShaderInfoLog(3)

    assert log == b"oops"


def test_calls_with_plain_numbers_go_straight_through(emscripten: FakeEmscripten) -> None:
    webgl.glBindTexture(webgl.GL_TEXTURE_2D, np.uint32(5))

    assert emscripten.calls == [("glBindTexture", webgl.GL_TEXTURE_2D, 5)]


def test_no_data_is_handed_over_as_address_zero() -> None:
    address = webgl.address(None)

    assert address == 0
