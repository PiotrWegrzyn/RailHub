"""The zoomed-out map's building dots drawn by the graphics card: every building's dot is two
triangles in one buffer, handed over once each time the buildings change, and the dots on screen
are drawn into a texture the size of the map area in a single call whenever the view moves. imgui
would take a call for each dot, and with thousands of buildings on screen those calls took most
of a zoomed-out frame.

Each dot covers the same device pixels as imgui's filled rectangle of the same corners
(railroads.ui.buildings.draw_building_dots), in the same order, so the map looks the same. Where
the window smooths the edges of what it draws (the browser's canvas does), a pixel a dot's edge
crosses takes as much of the dot's colour as the dot covers of it, as there.

It reaches OpenGL as the ground shader does (railroads.ui.pixel.ground_shader), and DotShader.make
returns None where that can't run, for the map to draw the dots one by one."""

import ctypes
import sys

import numpy as np
import numpy.typing as npt

from railroads.ui.pixel.ground_shader import (
    GL,
    HEADER,
    _program,
    _restore,
    _saved_state,
    _Target,
    uniform,
)
from railroads.ui.pixel.ground_shader import VERTEX as WHOLE_TARGET

VERTEX = """layout(location = 0) in vec2 place;
layout(location = 1) in vec2 corner;
layout(location = 2) in float radius;
layout(location = 3) in vec4 tint;
uniform mat2 to_screen;
uniform vec2 origin;
uniform float unit;
uniform float density;
uniform vec2 size;
uniform float fringe;
out vec4 shade;
flat out vec4 box;
void main() {
    float half_size = max(1.0, unit * radius);
    vec2 middle = to_screen * place + origin;
    box = vec4(middle - half_size, middle + half_size) * density;
    vec2 pixel = (middle + corner * half_size) * density + corner * fringe;
    gl_Position = vec4(pixel / size * 2.0 - 1.0, 0.0, 1.0);
    shade = tint;
}
"""

FRAGMENT = """in vec4 shade;
flat in vec4 box;
uniform float fringe;
out vec4 colour;
void main() {
    float cover = 1.0;
    if (fringe > 0.0) {
        vec2 low = floor(gl_FragCoord.xy);
        vec2 inside = clamp(min(box.zw, low + 1.0) - max(box.xy, low), 0.0, 1.0);
        cover = inside.x * inside.y;
    }
    colour = vec4(shade.rgb * cover, cover);
}
"""

# The dots are laid with their colours multiplied by how much of each pixel they cover, for
# their edges to blend; imgui multiplies a texture's colours by that itself, so they are divided
# back out for it.
UNMULTIPLY = """uniform highp sampler2D dots;
out vec4 colour;
void main() {
    vec4 laid = texelFetch(dots, ivec2(gl_FragCoord.xy), 0);
    colour = laid.a > 0.0 ? vec4(laid.rgb / laid.a, laid.a) : vec4(0.0);
}
"""

# Each vertex: where the dot's building stands on the map, which corner of the dot it is, the
# dot's size in cells and its colour.
VERTEX_KIND = np.dtype(
    [("place", np.float32, 2), ("corner", np.float32, 2), ("radius", np.float32),
     ("tint", np.uint8, 4)]
)  # fmt: skip
# The corners of a dot's two triangles.
CORNERS = np.array([(-1, -1), (1, -1), (1, 1), (-1, -1), (1, 1), (-1, 1)], dtype=np.float32)
GL_TRIANGLES = 0x0004
GL_TRUE = 1
GL_ONE = 1
GL_ONE_MINUS_SRC_ALPHA = 0x0303
GL_COLOR_BUFFER_BIT = 0x4000
GL_SAMPLES = 0x80A9
GL_DYNAMIC_DRAW = 0x88E8


def dot_vertices(
    x: npt.NDArray[np.floating],
    y: npt.NDArray[np.floating],
    radii: npt.NDArray[np.floating],
    colours: npt.NDArray[np.uint8],
) -> npt.NDArray[np.void]:
    """The six vertices of each dot, standing at (x, y) on the map, `radii` cells across and
    of RGBA `colours`, in the order they are drawn."""
    count = len(x)
    vertices = np.zeros((count, len(CORNERS)), dtype=VERTEX_KIND)
    vertices["place"][..., 0] = np.asarray(x, dtype=np.float32)[:, None]
    vertices["place"][..., 1] = np.asarray(y, dtype=np.float32)[:, None]
    vertices["corner"] = CORNERS
    vertices["radius"] = np.asarray(radii, dtype=np.float32)[:, None]
    vertices["tint"] = np.asarray(colours, dtype=np.uint8)[:, None, :]
    return vertices.reshape(-1)


class DotShader:
    """The dots' programs, their vertices on the graphics card, whether the window smooths
    edges, and the texture last drawn."""

    def __init__(self, dots: int, unmultiply: int, smooth: bool) -> None:
        self._dots = dots
        self._unmultiply = unmultiply
        self._smooth = smooth
        self._vao = GL.glGenVertexArrays(1)
        self._buffer = GL.glGenBuffers(1)
        self._count = 0
        self._laid = _Target()
        self._target = _Target()
        self.dots_version: object = None
        self._drawn: tuple | None = None
        GL.glBindVertexArray(self._vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._buffer)
        stride = VERTEX_KIND.itemsize
        for index, name, size, kind, normalized in (
            (0, "place", 2, GL.GL_FLOAT, GL.GL_FALSE),
            (1, "corner", 2, GL.GL_FLOAT, GL.GL_FALSE),
            (2, "radius", 1, GL.GL_FLOAT, GL.GL_FALSE),
            (3, "tint", 4, GL.GL_UNSIGNED_BYTE, GL_TRUE),
        ):
            GL.glEnableVertexAttribArray(index)
            offset = ctypes.c_void_p(VERTEX_KIND.fields[name][1])
            GL.glVertexAttribPointer(index, size, kind, normalized, stride, offset)
        self._whole = GL.glGenVertexArrays(1)
        self._corners = GL.glGenBuffers(1)
        GL.glBindVertexArray(self._whole)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._corners)
        corners = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype=np.float32)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, corners.nbytes, corners, GL.GL_STATIC_DRAW)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, GL.GL_FALSE, 0, ctypes.c_void_p(0))
        GL.glBindVertexArray(0)

    @classmethod
    def make(cls) -> DotShader | None:
        """The dots' shader, or None where OpenGL can't run it. Called while the window's own
        framebuffer is bound, to learn whether it smooths edges."""
        if GL is None:
            return None
        try:
            # The browser's canvas smooths edges, though GL says its framebuffer has no samples.
            smooth = sys.platform == "emscripten" or int(GL.glGetIntegerv(GL_SAMPLES)) > 1
            dots = _program(HEADER + VERTEX, HEADER + FRAGMENT)
            unmultiply = _program(HEADER + WHOLE_TARGET, HEADER + UNMULTIPLY)
            return cls(dots, unmultiply, smooth)
        except Exception:  # noqa: BLE001 - any GL failure means drawing the dots with imgui
            return None

    def set_dots(self, vertices: npt.NDArray[np.void], version: object) -> None:
        """Takes every dot's vertices (dot_vertices), for the buildings as they are at
        `version`."""
        data = np.ascontiguousarray(vertices)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._buffer)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, data.nbytes, data, GL_DYNAMIC_DRAW)
        self._count = len(data)
        self.dots_version = version
        self._drawn = None

    def draw(
        self,
        size: tuple[int, int],
        density: float,
        to_screen: tuple[float, float, float, float, float, float],
        unit: float,
    ) -> int:
        """The texture of a map area `size` device pixels big with every dot on it: a building at
        (x, y) on the map stands at (a x + b y + e, c x + d y + f) points from the area's top
        left corner for `to_screen` (a, b, c, d, e, f), and a dot `radius` cells across is
        max(1, `unit` radius) points either way of it. Drawn again only if any of these or the
        dots changed."""
        key = (size, density, to_screen, unit, self.dots_version)
        resized = self._laid.fit(*size) | self._target.fit(*size)
        if resized or self._drawn != key:
            saved = _saved_state()
            try:
                self._lay(size, density, to_screen, unit)
                GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._target.framebuffer)
                GL.glDisable(GL.GL_BLEND)
                GL.glUseProgram(self._unmultiply)
                GL.glUniform1i(uniform(self._unmultiply, "dots"), 0)
                GL.glActiveTexture(GL.GL_TEXTURE0)
                GL.glBindTexture(GL.GL_TEXTURE_2D, self._laid.texture)
                GL.glBindVertexArray(self._whole)
                GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)
            finally:
                _restore(saved)
            self._drawn = key
        return self._target.texture

    def _lay(
        self,
        size: tuple[int, int],
        density: float,
        to_screen: tuple[float, float, float, float, float, float],
        unit: float,
    ) -> None:
        """Draws the dots over a clear texture, their colours multiplied by their cover."""
        a, b, c, d, e, f = to_screen
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._laid.framebuffer)
        GL.glViewport(0, 0, *size)
        GL.glDisable(GL.GL_SCISSOR_TEST)
        GL.glClearColor(0.0, 0.0, 0.0, 0.0)
        GL.glClear(GL_COLOR_BUFFER_BIT)
        if not self._count:
            return
        program = self._dots
        GL.glUseProgram(program)
        matrix = np.array((a, c, b, d), np.float32)
        GL.glUniformMatrix2fv(uniform(program, "to_screen"), 1, GL.GL_FALSE, matrix)
        GL.glUniform2f(uniform(program, "origin"), e, f)
        GL.glUniform1f(uniform(program, "unit"), unit)
        GL.glUniform1f(uniform(program, "density"), density)
        GL.glUniform2f(uniform(program, "size"), *size)
        GL.glUniform1f(uniform(program, "fringe"), 1.0 if self._smooth else 0.0)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA)
        GL.glBindVertexArray(self._vao)
        GL.glDrawArrays(GL_TRIANGLES, 0, self._count)
