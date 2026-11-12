"""The ground of the pixel map drawn by the graphics card: a shader works out every screen pixel
from the art pixel under it, which hex cell that lies in, the cell's colour, a grain of light and
dark patches and the light on the slopes, so the whole view shows crisp ground at once, with no
chunk drawn yet.

It draws into a texture the size of the map on screen, which the map then shows like any picture,
under the chunks the CPU draws (railroads.ui.pixel.layer); where a chunk is ready it covers the
ground. A picture is drawn again only when the view or the cells' colours change.

On the desktop it reaches OpenGL through PyOpenGL; in the browser build, through the GL that
imgui draws with there (railroads.ui.pixel.webgl), as WebGL 2. Without either, or if the graphics
card refuses the shader, GroundShader.make returns None and the map keeps its smooth ground
picture under the chunks.
"""

import ctypes
import sys
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from railroads.sim.tilemap import TileMap
from railroads.ui.pixel.projection import PixelProjection

if sys.platform == "emscripten":
    from railroads.ui.pixel import webgl as GL

    HEADER = "#version 300 es\nprecision highp float;\nprecision highp int;\n"
else:
    try:
        from OpenGL import GL
    except ImportError:
        GL = None
    HEADER = "#version 330 core\n"

# How strongly the grain of light and dark patches, and the slopes, change the cells' colours.
GRAIN = 0.07
SLOPE_LIGHT = 0.0018

VERTEX = """layout(location = 0) in vec2 corner;
void main() {
    gl_Position = vec4(corner, 0.0, 1.0);
}
"""

FRAGMENT = """out vec4 colour;
uniform vec2 origin;
uniform float scale;
uniform float height;
uniform mat2 to_plane;
uniform vec2 projection_origin;
uniform ivec2 cells_size;
uniform sampler2D cells;
uniform sampler2D heights;
uniform vec2 light;
uniform float grain;
uniform float slope_light;

const float HEX_RADIUS = 0.5773502691896258;
const float SQRT3 = 1.7320508075688772;

ivec2 cell_at(vec2 p) {
    float px = p.x - 0.5;
    float py = p.y - HEX_RADIUS;
    float q = px - py / SQRT3;
    float r = py * 2.0 / SQRT3;
    float s = -q - r;
    float rq = roundEven(q), rr = roundEven(r), rs = roundEven(s);
    float eq = abs(rq - q), er = abs(rr - r), es = abs(rs - s);
    bool fix_q = eq > er && eq > es;
    bool fix_r = !fix_q && er > es;
    if (fix_q) rq = -rr - rs;
    if (fix_r) rr = -rq - rs;
    int row = int(rr);
    int col = int(rq) + (row - (row & 1)) / 2;
    return ivec2(col, row);
}

float hash(vec2 p) {
    p = fract(p * vec2(0.1031, 0.1030));
    p += dot(p, p.yx + 33.33);
    return fract((p.x + p.y) * p.x);
}

float patches(vec2 art, float size) {
    vec2 at = art / size;
    vec2 i = floor(at);
    vec2 f = at - i;
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i), hash(i + vec2(1.0, 0.0)), u.x),
               mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), u.x), u.y);
}

float height_at(vec2 p) {
    vec2 at = vec2(p.x, p.y / 0.8660254) - 0.5;
    ivec2 corner = ivec2(floor(at));
    vec2 f = at - vec2(corner);
    ivec2 last = cells_size - 1;
    float a = texelFetch(heights, clamp(corner, ivec2(0), last), 0).r;
    float b = texelFetch(heights, clamp(corner + ivec2(1, 0), ivec2(0), last), 0).r;
    float c = texelFetch(heights, clamp(corner + ivec2(0, 1), ivec2(0), last), 0).r;
    float d = texelFetch(heights, clamp(corner + ivec2(1, 1), ivec2(0), last), 0).r;
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

void main() {
    vec2 device = vec2(gl_FragCoord.x, height - gl_FragCoord.y);
    vec2 art = floor((device - origin) / scale);
    vec2 p = to_plane * (art + 0.5 + projection_origin);
    ivec2 cell = cell_at(p);
    if (cell.x < 0 || cell.y < 0 || cell.x >= cells_size.x || cell.y >= cells_size.y) {
        colour = vec4(0.0);
        return;
    }
    vec4 base = texelFetch(cells, cell, 0);
    float speckle = hash(art) - 0.5;
    float blotch = patches(art, 7.0) - 0.5 + 0.6 * (patches(art, 23.0) - 0.5);
    float slope = height_at(p + light * 0.35) - height_at(p - light * 0.35);
    float shade = 1.0 + grain * (blotch + 0.5 * speckle) - slope_light * slope;
    colour = vec4(clamp(base.rgb * shade, 0.0, 1.0), 1.0);
}
"""


@dataclass(frozen=True)
class View:
    """What a ground picture shows: its size in device pixels, where art pixel (0, 0) starts in
    it and how many device pixels each art pixel covers, and the way the light falls."""

    width: int
    height: int
    origin: tuple[float, float]
    scale: float
    light: tuple[float, float]


def view_of(
    origin: tuple[int, int],
    scale: int,
    top_left: tuple[float, float],
    size: tuple[float, float],
    density: int,
    light: tuple[float, float],
) -> View:
    """The view for a map drawn with art pixel (0, 0) at screen `origin` (as ChunkPlacement
    places it), `scale` screen pixels to an art pixel, in a map area at `top_left` of `size`
    screen pixels, `density` device pixels to a screen pixel."""
    return View(
        round(size[0] * density),
        round(size[1] * density),
        ((origin[0] - top_left[0]) * density, (origin[1] - top_left[1]) * density),
        scale * density,
        light,
    )


class GroundShader:
    """The compiled shader, the cells' colours and heights on the graphics card, and the texture
    the last picture was drawn into."""

    def __init__(self, program: int, projection: PixelProjection, tile_map: TileMap) -> None:
        self._program = program
        self._projection = projection
        self._shape = tile_map.grid.shape
        self._vao = GL.glGenVertexArrays(1)
        self._buffer = GL.glGenBuffers(1)
        GL.glBindVertexArray(self._vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._buffer)
        corners = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype=np.float32)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, corners.nbytes, corners, GL.GL_STATIC_DRAW)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, GL.GL_FALSE, 0, ctypes.c_void_p(0))
        GL.glBindVertexArray(0)
        self._cells = _texture(GL.GL_NEAREST)
        self._heights = _texture(GL.GL_NEAREST)
        heights = np.ascontiguousarray(tile_map.height_m.astype(np.float32))
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._heights)
        rows, cols = heights.shape
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D, 0, GL.GL_R32F, cols, rows, 0, GL.GL_RED, GL.GL_FLOAT, heights
        )
        self._picture = 0
        self._framebuffer = GL.glGenFramebuffers(1)
        self._picture_size = (0, 0)
        self._drawn: tuple[View, int] | None = None
        self.colors_version = -1

    @classmethod
    def make(cls, projection: PixelProjection, tile_map: TileMap) -> GroundShader | None:
        """The shader for this map, or None where OpenGL can't run it."""
        if GL is None:
            return None
        try:
            program = _program(HEADER + VERTEX, HEADER + FRAGMENT)
            return cls(program, projection, tile_map)
        except Exception:  # noqa: BLE001 - any GL failure means falling back to the CPU ground
            return None

    def set_colors(self, colors: npt.NDArray[np.uint8], version: int) -> None:
        """Takes every cell's RGBA colour, as (rows, cols, 4) bytes."""
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._cells)
        rows, cols = colors.shape[:2]
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_RGBA8,
            cols,
            rows,
            0,
            GL.GL_RGBA,
            GL.GL_UNSIGNED_BYTE,
            np.ascontiguousarray(colors),
        )
        self.colors_version = version

    def picture(self, view: View) -> int:
        """The texture holding the ground for this view, drawn again only if the view or the
        colours changed since."""
        if self._drawn == (view, self.colors_version):
            return self._picture
        self._ensure_picture(view.width, view.height)
        saved = _saved_state()
        try:
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._framebuffer)
            GL.glViewport(0, 0, view.width, view.height)
            GL.glDisable(GL.GL_BLEND)
            GL.glDisable(GL.GL_SCISSOR_TEST)
            GL.glUseProgram(self._program)
            self._set_uniforms(view)
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self._cells)
            GL.glActiveTexture(GL.GL_TEXTURE1)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self._heights)
            GL.glBindVertexArray(self._vao)
            GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)
        finally:
            _restore(saved)
        self._drawn = (view, self.colors_version)
        return self._picture

    def _set_uniforms(self, view: View) -> None:
        def at(name: str) -> int:
            return GL.glGetUniformLocation(self._program, name)

        (a, b), (c, d) = np.linalg.inv(self._projection.matrix)
        GL.glUniform2f(at("origin"), *view.origin)
        GL.glUniform1f(at("scale"), view.scale)
        GL.glUniform1f(at("height"), view.height)
        GL.glUniformMatrix2fv(at("to_plane"), 1, GL.GL_FALSE, np.array([a, c, b, d], np.float32))
        GL.glUniform2f(at("projection_origin"), *map(float, self._projection.origin))
        GL.glUniform2i(at("cells_size"), self._shape[1], self._shape[0])
        GL.glUniform1i(at("cells"), 0)
        GL.glUniform1i(at("heights"), 1)
        GL.glUniform2f(at("light"), *view.light)
        GL.glUniform1f(at("grain"), GRAIN)
        GL.glUniform1f(at("slope_light"), SLOPE_LIGHT)

    def _ensure_picture(self, width: int, height: int) -> None:
        if self._picture_size == (width, height):
            return
        if self._picture:
            GL.glDeleteTextures(1, [self._picture])
        self._picture = _texture(GL.GL_NEAREST)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_RGBA8,
            width,
            height,
            0,
            GL.GL_RGBA,
            GL.GL_UNSIGNED_BYTE,
            None,
        )
        saved = GL.glGetIntegerv(GL.GL_FRAMEBUFFER_BINDING)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._framebuffer)
        GL.glFramebufferTexture2D(
            GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, self._picture, 0
        )
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, int(saved))
        self._picture_size = (width, height)
        self._drawn = None


def _texture(filtering: int) -> int:
    texture = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, texture)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, filtering)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, filtering)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
    return texture


def _program(vertex: str, fragment: str) -> int:
    program = GL.glCreateProgram()
    for kind, source in ((GL.GL_VERTEX_SHADER, vertex), (GL.GL_FRAGMENT_SHADER, fragment)):
        shader = GL.glCreateShader(kind)
        GL.glShaderSource(shader, source)
        GL.glCompileShader(shader)
        if not GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS):
            raise RuntimeError(GL.glGetShaderInfoLog(shader))
        GL.glAttachShader(program, shader)
    GL.glLinkProgram(program)
    if not GL.glGetProgramiv(program, GL.GL_LINK_STATUS):
        raise RuntimeError(GL.glGetProgramInfoLog(program))
    return program


def _saved_state() -> tuple:
    return (
        int(GL.glGetIntegerv(GL.GL_FRAMEBUFFER_BINDING)),
        tuple(int(v) for v in GL.glGetIntegerv(GL.GL_VIEWPORT)),
        int(GL.glGetIntegerv(GL.GL_CURRENT_PROGRAM)),
        int(GL.glGetIntegerv(GL.GL_VERTEX_ARRAY_BINDING)),
        int(GL.glGetIntegerv(GL.GL_ACTIVE_TEXTURE)),
        bool(GL.glIsEnabled(GL.GL_BLEND)),
        bool(GL.glIsEnabled(GL.GL_SCISSOR_TEST)),
    )


def _restore(saved: tuple) -> None:
    framebuffer, viewport, program, vao, active, blend, scissor = saved
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, framebuffer)
    GL.glViewport(*viewport)
    GL.glUseProgram(program)
    GL.glBindVertexArray(vao)
    GL.glActiveTexture(GL.GL_TEXTURE1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
    GL.glActiveTexture(active)
    if blend:
        GL.glEnable(GL.GL_BLEND)
    if scissor:
        GL.glEnable(GL.GL_SCISSOR_TEST)
