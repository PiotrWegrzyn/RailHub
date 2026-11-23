"""What tests that draw stand on: an imgui context like the one the game's window gives, and a
texture standing in for one on the GPU."""

from collections.abc import Iterator
from contextlib import contextmanager

from imgui_bundle import imgui, implot


class FakeTexture:
    def texture_id(self) -> int:
        return 1


@contextmanager
def gui_context(
    size: tuple[float, float] = (1280.0, 800.0), *, plots: bool = False, textures: bool = True
) -> Iterator[None]:
    """An imgui context with a screen this big, drawing textures if `textures`, and an implot
    one beside it if `plots`."""
    gui = imgui.create_context()
    plotting = implot.create_context() if plots else None
    io = imgui.get_io()
    io.display_size = imgui.ImVec2(*size)
    if textures:
        io.backend_flags |= imgui.BackendFlags_.renderer_has_textures.value
    try:
        yield
    finally:
        if plotting is not None:
            implot.destroy_context(plotting)
        imgui.destroy_context(gui)
