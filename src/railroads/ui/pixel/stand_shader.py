"""What stands on the pixel map, drawn by the graphics card over the ground it draws
(railroads.ui.pixel.ground_shader): the buildings, stations, trees and rocks, and the track.

Each chunk's texture is made here, as big as the chunk shows on screen, and every pixel of it is
drawn at that size, so up close nothing is a smaller picture blown up into squares. The ground,
woods, fields and yards are laid first (GroundShader.dress_onto), or the ground the CPU worked out
finely where hills and mountains rise; then the sprites drawn under the track, the track, and the
rest of the sprites, back to front, as the CPU stands them (PixelWorld.render).

Sprites come from the sheet drawn at exactly as many pixels to an art pixel as the screen shows
(railroads.ui.pixel.sheet.DENSITIES), so a sprite pixel is a screen pixel at every zoom: the same
model drawn at the size it is seen, never a coarser picture blown up nor a finer one shrunk. Only
the sprites a chunk shows are unpacked and put on the graphics card, on pages of the sheet's own,
the first time a chunk stands them.

The track is laid out by the CPU, away from the frames, as triangles (TrackArt.shapes), which
the graphics card draws a few times finer than shown and averages down, as the CPU averages its
picture of the track, and hides behind raised ground in front of it as the CPU does
(PixelWorld._lay_lifted_track): where the ground showing lies further down the map than the
track is lifted, by more than HIDDEN_BEYOND."""

import ctypes
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from railroads.ui.iso_camera import TILT
from railroads.ui.pixel import relief
from railroads.ui.pixel.ground_shader import (
    GL,
    HEADER,
    VERTEX,
    ChunkInputs,
    ChunkTexture,
    GroundShader,
    RenderTarget,
    _Into,
    _number,
    _texture,
    free_textures,
    gl_program,
    restore_gl_state,
    saved_gl_state,
    use,
)
from railroads.ui.pixel.projection import CHUNK
from railroads.ui.pixel.scenery import Scenery
from railroads.ui.pixel.sheet import DENSITIES, ByteSheet
from railroads.ui.pixel.track_art import TrackShapes
from railroads.ui.pixel.world import HIDDEN_BEYOND

# How big a page of sprites on the graphics card is each way, and the picture the track is drawn
# into finely before it is averaged down: several of the biggest sprites fit on a page, and both
# stay well inside the biggest texture any graphics card takes.
PAGE = 2048
# How many sheets, and their pages on the graphics card, are kept: those of the closest zooms
# a screen shows, so zooming among them needn't load a sheet again.
SHEETS_KEPT = 3

GL_TRIANGLES = 0x0004
GL_ONE = 1
GL_SRC_ALPHA = 0x0302
GL_ONE_MINUS_SRC_ALPHA = 0x0303
GL_FUNC_ADD = 0x8006
GL_MAX = 0x8008
GL_COLOR_BUFFER_BIT = 0x4000
GL_DYNAMIC_DRAW = 0x88E8

SPRITE_VERTEX = """layout(location = 0) in vec2 corner;
layout(location = 1) in vec2 texel;
uniform vec2 size;
out vec2 at;
void main() {
    at = texel;
    gl_Position = vec4(corner / size * 2.0 - 1.0, 0.0, 1.0);
}
"""

# A sprite's pixel under each screen pixel, premultiplied by its alpha, to lay over the chunk.
SPRITE = """in vec2 at;
out vec4 colour;
uniform highp sampler2D page;
void main() {
    vec4 pixel = texelFetch(page, ivec2(floor(at)), 0);
    colour = vec4(pixel.rgb * pixel.a, pixel.a);
}
"""

SHAPE_VERTEX = """layout(location = 0) in vec2 corner;
layout(location = 1) in vec4 paint;
uniform vec2 origin;
uniform vec2 size;
out vec4 painted;
void main() {
    painted = paint;
    gl_Position = vec4((corner - origin) / size * 2.0 - 1.0, 0.0, 1.0);
}
"""

# The track's own colour, premultiplied by its alpha: the shadow is see-through.
SHAPE = """in vec4 painted;
out vec4 colour;
void main() {
    colour = vec4(painted.rgb * painted.a, painted.a);
}
"""

LIFT_VERTEX = """layout(location = 0) in vec2 corner;
layout(location = 1) in float lift;
uniform vec2 origin;
uniform vec2 size;
out float lifted;
void main() {
    lifted = lift;
    gl_Position = vec4((corner - origin) / size * 2.0 - 1.0, 0.0, 1.0);
}
"""

# How far the track is lifted, as a share of MOST_LIFT, kept the highest where several stretches
# cover a pixel (blended by GL_MAX), as TrackArt's picture of the heights keeps it.
LIFT = """in float lifted;
out vec4 colour;
void main() {
    colour = vec4(lifted / @MOST_LIFT@, 0.0, 0.0, 1.0);
}
"""

# The track drawn finely averaged down to one pixel of the chunk, shown only where the art pixel
# under it lies on the map and the raised ground showing there lies no further down the map than
# the track is lifted, by more than HIDDEN_BEYOND (PixelWorld._lay_lifted_track).
TRACK = """out vec4 colour;
uniform ivec2 tile;
uniform int quality;
uniform int scale;
uniform ivec2 chunk_origin;
uniform ivec2 relief_origin;
uniform ivec2 woods_origin;
uniform float metres_per_pixel;
uniform highp sampler2D fine;
uniform highp sampler2D heights;
uniform highp sampler2D relief;
uniform highp sampler2D wooded;

float lift_at(ivec2 art) {
    float rise = texelFetch(relief, art - relief_origin, 0).x;
    return clamp(rise / metres_per_pixel * @EXAGGERATION@ * @TILT@, 0.0, @MOST_LIFT@);
}

void main() {
    ivec2 pixel = ivec2(floor(gl_FragCoord.xy));
    ivec2 at = (pixel - tile) * quality;
    vec4 sum = vec4(0.0);
    float lift = 0.0;
    for (int dy = 0; dy < quality; dy++) {
        for (int dx = 0; dx < quality; dx++) {
            sum += texelFetch(fine, at + ivec2(dx, dy), 0);
            lift = max(lift, texelFetch(heights, at + ivec2(dx, dy), 0).r);
        }
    }
    if (sum.a == 0.0) discard;
    ivec2 art = chunk_origin + pixel / scale;
    int offset = 0;
    for (int d = 1; d <= @MOST_LIFT_ROWS@; d++) {
        if (floor(lift_at(art + ivec2(0, d))) >= float(d)) offset = d;
    }
    bool inside = texelFetch(wooded, art + ivec2(0, offset) - woods_origin, 0).b > 0.5;
    bool seen = float(offset) <= lift * @MOST_LIFT@ + @HIDDEN_BEYOND@;
    colour = inside && seen ? sum / float(quality * quality) : vec4(0.0);
}
"""

# The chunk made premultiplied by its alpha turned into the texture shown, which is not.
FINISH = """out vec4 colour;
uniform highp sampler2D made;
void main() {
    vec4 pixel = texelFetch(made, ivec2(floor(gl_FragCoord.xy)), 0);
    colour = pixel.a > 0.0 ? vec4(clamp(pixel.rgb / pixel.a, 0.0, 1.0), pixel.a) : vec4(0.0);
}
"""


def _filled(source: str) -> str:
    names = {
        "EXAGGERATION": _number(relief.EXAGGERATION),
        "TILT": _number(TILT),
        "MOST_LIFT": _number(relief.MOST_LIFT),
        "MOST_LIFT_ROWS": str(relief.MOST_LIFT),
        "HIDDEN_BEYOND": _number(HIDDEN_BEYOND),
    }
    for name, value in names.items():
        source = source.replace(f"@{name}@", value)
    assert "@" not in source
    return source


def track_source() -> str:
    """The shader averaging the track down and hiding it behind raised ground (TRACK)."""
    return _filled(TRACK)


def lift_source() -> str:
    return _filled(LIFT)


def sheet_for(scale: int) -> int:
    """The sheet a chunk shown `scale` pixels to an art pixel stands its sprites from: the one
    drawn at that scale, or the finest whose pixels blow up evenly to it."""
    return max(density for density in DENSITIES if scale % density == 0)


@dataclass(frozen=True)
class Standing:
    """The sprites standing in a chunk, back to front, with those drawn under the track first:
    each one's sprite, the corner of its box in the chunk's pixels, and its width and height
    there, and how many of them stand under the track."""

    sprite: npt.NDArray[np.int64]
    corner: npt.NDArray[np.int64]
    size: npt.NDArray[np.int64]
    under: int


def standing(
    scenery: Scenery,
    sizes: npt.NDArray[np.int64],
    anchors: npt.NDArray[np.int64],
    chunk: tuple[int, int],
    scale: int,
    times: int,
) -> Standing:
    """What stands in a chunk shown `scale` pixels to an art pixel, from sprites of `sizes` and
    `anchors` each blown up `times` times, as PixelWorld._stand_scenery finds it."""
    left, top = chunk[0] * CHUNK, chunk[1] * CHUNK
    sprite = scenery.sprite
    x0 = (scenery.x - left) * scale - anchors[sprite, 0] * times
    y0 = (scenery.y - scenery.lift - top) * scale - anchors[sprite, 1] * times
    w, h = sizes[sprite, 0] * times, sizes[sprite, 1] * times
    span = CHUNK * scale
    over = (x0 < span) & (x0 + w > 0) & (y0 < span) & (y0 + h > 0)
    under = np.flatnonzero(over & scenery.under_track)
    order = np.concatenate([under, np.flatnonzero(over & ~scenery.under_track)])
    corner = np.stack([x0[order], y0[order]], axis=-1).astype(np.int64)
    size = np.stack([w[order], h[order]], axis=-1).astype(np.int64)
    return Standing(sprite[order].astype(np.int64), corner, size, len(under))


def tiles(span: int, quality: int, box: tuple[float, float, float, float]) -> list[tuple]:
    """The parts of a chunk `span` pixels each way the track is drawn finely in, one at a time,
    each no bigger than a PAGE once `quality` times finer: as left, top, width and height in the
    chunk's pixels, those overlapping `box` (left, top, right, bottom) only."""
    step = PAGE // quality
    found = []
    for top in range(0, span, step):
        for left in range(0, span, step):
            width, height = min(step, span - left), min(step, span - top)
            if box[0] < left + width and box[2] > left and box[1] < top + height and box[3] > top:
                found.append((left, top, width, height))
    return found


class _Pages:
    """One sheet's sprites put on pages on the graphics card as they are first stood, in rows
    across each page."""

    def __init__(self, sheet: ByteSheet) -> None:
        self.sheet = sheet
        self.textures: list[int] = []
        self._places: dict[int, tuple[int, int, int]] = {}
        self._x = self._y = self._row = 0

    def place(self, sprite: int) -> tuple[int, int, int]:
        """The page a sprite is on and where it starts there, putting it on one first."""
        found = self._places.get(sprite)
        if found is not None:
            return found
        pixels = np.ascontiguousarray(self.sheet.pixels[sprite])
        h, w = pixels.shape[:2]
        if self._x + w > PAGE:
            self._x, self._y, self._row = 0, self._y + self._row, 0
        if not self.textures or self._y + h > PAGE:
            self.textures.append(_texture())
            GL.glTexImage2D(
                GL.GL_TEXTURE_2D,
                0,
                GL.GL_RGBA8,
                PAGE,
                PAGE,
                0,
                GL.GL_RGBA,
                GL.GL_UNSIGNED_BYTE,
                None,
            )
            self._x = self._y = self._row = 0
        page = len(self.textures) - 1
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.textures[page])
        GL.glTexSubImage2D(
            GL.GL_TEXTURE_2D, 0, self._x, self._y, w, h, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, pixels
        )
        found = self._places[sprite] = (page, self._x, self._y)
        self._x += w
        self._row = max(self._row, h)
        return found

    def free(self) -> None:
        if self.textures:
            GL.glDeleteTextures(len(self.textures), self.textures)
        self.textures = []


class _Geometry:
    """A buffer of vertices with how its program reads them, refilled for each chunk."""

    def __init__(self, layout: tuple[int, ...]) -> None:
        self.vao = GL.glGenVertexArrays(1)
        self.buffer = GL.glGenBuffers(1)
        self.count = 0
        GL.glBindVertexArray(self.vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.buffer)
        stride = sum(layout) * 4
        offset = 0
        for index, floats in enumerate(layout):
            GL.glEnableVertexAttribArray(index)
            GL.glVertexAttribPointer(
                index, floats, GL.GL_FLOAT, GL.GL_FALSE, stride, ctypes.c_void_p(offset)
            )
            offset += floats * 4
        GL.glBindVertexArray(0)

    def fill(self, vertices: npt.NDArray[np.float32]) -> None:
        data = np.ascontiguousarray(vertices, dtype=np.float32)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.buffer)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, data.nbytes, data, GL_DYNAMIC_DRAW)
        self.count = len(data)


@dataclass(frozen=True)
class _Programs:
    sprite: int
    shape: int
    lift: int
    track: int
    finish: int


class StandShader:
    """Makes chunks' textures with what stands on them (`stand`), over the ground `ground`
    draws, from the sheets `sheets` loads at each of DENSITIES."""

    def __init__(
        self, programs: _Programs, ground: GroundShader, sheets: Callable[[int], ByteSheet]
    ) -> None:
        self._programs = programs
        self._ground = ground
        self._sheets = sheets
        self._pages: OrderedDict[int, _Pages] = OrderedDict()
        self._made = RenderTarget()
        self._fine = RenderTarget()
        self._heights = RenderTarget()
        self._out = GL.glGenFramebuffers(1)
        self._sprites = _Geometry((2, 2))
        self._shapes = _Geometry((2, 4))
        self._lifts = _Geometry((2, 1))

    @classmethod
    def make(
        cls, ground: GroundShader | None, sheets: Callable[[int], ByteSheet]
    ) -> StandShader | None:
        """The shaders standing what stands on the map over the ground `ground` draws, or None
        where there is no ground shader or the graphics card refuses these."""
        if ground is None or GL is None:
            return None
        try:
            programs = _Programs(
                gl_program(HEADER + SPRITE_VERTEX, HEADER + SPRITE),
                gl_program(HEADER + SHAPE_VERTEX, HEADER + SHAPE),
                gl_program(HEADER + LIFT_VERTEX, HEADER + lift_source()),
                gl_program(HEADER + VERTEX, HEADER + track_source()),
                gl_program(HEADER + VERTEX, HEADER + FINISH),
            )
            shader = cls(programs, ground, sheets)
            shader._fine.fit(PAGE, PAGE)
            shader._heights.fit(PAGE, PAGE)
            return shader
        except Exception:  # noqa: BLE001 - any GL failure means the CPU draws what stands
            return None

    def stand(
        self,
        chunk: tuple[int, int],
        scale: int,
        k: int,
        scenery: Scenery,
        track: npt.NDArray[np.uint8],
        ground: npt.NDArray[np.uint8] | None = None,
    ) -> ChunkTexture:
        """A chunk's texture, `scale` pixels to an art pixel: what lies on its ground drawn `k`
        pixels to an art pixel and blown up to `scale`, or `ground` if the CPU drew it, with
        everything in `scenery` standing in the chunk and the track (`track`, as
        TrackShapes.packed) over it."""
        free_textures()
        pages = self.pages(sheet_for(scale))
        span = CHUNK * scale
        times = scale // k
        ground_shader = self._ground
        ground_shader.lay_surfaces(chunk, k)
        self._made.fit(span, span)
        if ground is None:
            ground_shader.dress_onto(self._made, chunk, k, times)
        else:
            ground_shader.blow_up_onto(self._made, ground, span // ground.shape[1])
        laid = standing_in(pages, scenery, chunk, scale)
        self._stand_sprites(laid, span, 0, laid.under)
        shapes = TrackShapes.unpacked(track)
        if len(shapes.triangles):
            self._lay_track(shapes, span, ground_shader.chunk_inputs(chunk))
        self._stand_sprites(laid, span, laid.under, len(laid.sprite))
        return self._finished(span)

    def pages(self, density: int) -> _Pages:
        """The pages of the sheet drawn at `density`, loading it the first time, and letting go
        of the sheets used longest ago beyond SHEETS_KEPT."""
        pages = self._pages.get(density)
        if pages is None:
            pages = self._pages[density] = _Pages(self._sheets(density))
            while len(self._pages) > SHEETS_KEPT:
                self._pages.popitem(last=False)[1].free()
        self._pages.move_to_end(density)
        return pages

    def _stand_sprites(self, laid: _Laid, span: int, first: int, last: int) -> None:
        if first == last:
            return
        vertices, runs = laid.vertices[first * 6 : last * 6], laid.runs(first, last)
        self._sprites.fill(vertices)
        uniforms = {"size": ("2f", (float(span), float(span))), "page": ("1i", (0,))}
        for start, count, texture in runs:
            self._draw(
                _Into(self._made.framebuffer, (span, span)),
                (0, 0, span, span),
                self._programs.sprite,
                uniforms,
                (texture,),
                self._sprites,
                (start, count),
                (GL_FUNC_ADD, GL_ONE, GL_ONE_MINUS_SRC_ALPHA),
            )

    def _lay_track(self, shapes: TrackShapes, span: int, inputs: ChunkInputs) -> None:
        q = shapes.quality
        corners = shapes.triangles[:, :2] / q
        box = (*corners.min(axis=0), *corners.max(axis=0))
        self._shapes.fill(shapes.triangles)
        self._lifts.fill(shapes.lifts)
        add = (GL_FUNC_ADD, GL_ONE, GL_ONE_MINUS_SRC_ALPHA)
        for left, top, width, height in tiles(span, q, box):
            fine = (0, 0, width * q, height * q)
            place = {
                "origin": ("2f", (float(left * q), float(top * q))),
                "size": ("2f", (float(width * q), float(height * q))),
            }
            self._clear(self._fine)
            self._draw(
                self._fine,
                fine,
                self._programs.shape,
                place,
                (),
                self._shapes,
                (0, self._shapes.count),
                None,
            )
            self._clear(self._heights)
            if self._lifts.count:
                self._draw(
                    self._heights,
                    fine,
                    self._programs.lift,
                    place,
                    (),
                    self._lifts,
                    (0, self._lifts.count),
                    (GL_MAX, GL_ONE, GL_ONE),
                )
            uniforms = {
                "tile": ("2i", (left, top)),
                "quality": ("1i", (q,)),
                "scale": ("1i", (shapes.scale,)),
                "chunk_origin": ("2i", inputs.chunk_origin),
                "relief_origin": ("2i", inputs.relief_origin),
                "woods_origin": ("2i", inputs.woods_origin),
                "metres_per_pixel": ("1f", (inputs.metres_per_pixel,)),
                "fine": ("1i", (0,)),
                "heights": ("1i", (1,)),
                "relief": ("1i", (2,)),
                "wooded": ("1i", (3,)),
            }
            textures = (self._fine.texture, self._heights.texture, inputs.relief, inputs.wooded)
            self._draw(
                _Into(self._made.framebuffer, (span, span)),
                (left, top, width, height),
                self._programs.track,
                uniforms,
                textures,
                None,
                (0, 4),
                add,
            )

    def _finished(self, span: int) -> ChunkTexture:
        texture = _texture()
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, span, span, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None
        )
        saved = GL.glGetIntegerv(GL.GL_FRAMEBUFFER_BINDING)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._out)
        GL.glFramebufferTexture2D(
            GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, texture, 0
        )
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, int(saved))
        uniforms = {"made": ("1i", (0,))}
        self._draw(
            _Into(self._out, (span, span)),
            (0, 0, span, span),
            self._programs.finish,
            uniforms,
            (self._made.texture,),
            None,
            (0, 4),
            None,
        )
        return ChunkTexture(texture)

    def _clear(self, target: RenderTarget) -> None:
        saved = saved_gl_state()
        try:
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, target.framebuffer)
            GL.glDisable(GL.GL_SCISSOR_TEST)
            GL.glClearColor(0.0, 0.0, 0.0, 0.0)
            GL.glClear(GL_COLOR_BUFFER_BIT)
        finally:
            restore_gl_state(saved)

    def _draw(
        self,
        target: RenderTarget | _Into,
        viewport: tuple[int, int, int, int],
        program: int,
        uniforms: dict[str, tuple[str, tuple]],
        inputs: tuple[int, ...],
        geometry: _Geometry | None,
        vertices: tuple[int, int],
        blend: tuple[int, int, int] | None,
    ) -> None:
        """Draws `vertices` (first and count) of `geometry` as triangles with `program` into
        `viewport` of `target`, or the whole viewport without geometry, blended by `blend`
        (equation, source and destination factors) or over what is there."""
        saved = saved_gl_state()
        try:
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, target.framebuffer)
            GL.glViewport(*viewport)
            GL.glDisable(GL.GL_SCISSOR_TEST)
            if blend is None:
                GL.glDisable(GL.GL_BLEND)
            else:
                GL.glEnable(GL.GL_BLEND)
                GL.glBlendEquation(blend[0])
                GL.glBlendFunc(blend[1], blend[2])
            use(program, uniforms, inputs)
            if geometry is None:
                GL.glBindVertexArray(self._ground._vao)
                GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, *vertices)
            else:
                GL.glBindVertexArray(geometry.vao)
                GL.glDrawArrays(GL_TRIANGLES, *vertices)
        finally:
            GL.glBlendEquation(GL_FUNC_ADD)
            GL.glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            restore_gl_state(saved)


@dataclass(frozen=True)
class _Laid:
    """The sprites standing in a chunk (Standing) as the quads drawing them, six vertices each,
    with the page each is on."""

    vertices: npt.NDArray[np.float32]
    textures: list[int]
    under: int
    sprite: npt.NDArray[np.int64]

    def runs(self, first: int, last: int) -> list[tuple[int, int, int]]:
        """The sprites from `first` to `last` as runs on one page each: the first vertex of the
        run counted from `first`'s, how many vertices and the page's texture."""
        found: list[tuple[int, int, int]] = []
        for i in range(first, last):
            texture = self.textures[i]
            if found and found[-1][2] == texture:
                start, count, _ = found[-1]
                found[-1] = (start, count + 6, texture)
            else:
                found.append(((i - first) * 6, 6, texture))
        return found


def standing_in(pages: _Pages, scenery: Scenery, chunk: tuple[int, int], scale: int) -> _Laid:
    """The sprites standing in a chunk shown `scale` pixels to an art pixel as quads from the
    pages of the sheet drawn for it, each sprite put on them first if it is not on them yet."""
    sheet = pages.sheet
    laid = standing(scenery, sheet.sizes, sheet.anchors, chunk, scale, scale // sheet_for(scale))
    places = [pages.place(int(sprite)) for sprite in laid.sprite]
    vertices = np.zeros((len(places) * 6, 4), dtype=np.float32)
    if places:
        x0, y0 = laid.corner[:, 0], laid.corner[:, 1]
        x1, y1 = x0 + laid.size[:, 0], y0 + laid.size[:, 1]
        sizes = sheet.sizes[laid.sprite]
        u0 = np.array([place[1] for place in places])
        v0 = np.array([place[2] for place in places])
        u1, v1 = u0 + sizes[:, 0], v0 + sizes[:, 1]
        corners = [(x0, y0, u0, v0), (x1, y0, u1, v0), (x1, y1, u1, v1)]
        corners += [(x0, y0, u0, v0), (x1, y1, u1, v1), (x0, y1, u0, v1)]
        for n, values in enumerate(corners):
            vertices[n::6] = np.stack(values, axis=-1)
    textures = [pages.textures[place[0]] for place in places]
    return _Laid(vertices, textures, laid.under, laid.sprite)
