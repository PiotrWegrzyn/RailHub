"""The ground of the pixel map drawn by the graphics card: a shader works out every screen pixel
from the art pixel under it the way railroads.ui.pixel.ground does on the CPU (Ground.render):
which hex cell it lies in, the cell's biome and its ramp of shades picked by noise, meadows,
savanna, tundra and desert with blades, flowers and dunes, tufts, water deep or shallow with
waves and glints and soft sandy shores, the light on the slopes, the glow where buildings are
served and the signs of deposits. Hills and mountains rise as railroads.ui.pixel.relief raises
them: the heights worked out from the same rasters, lit by their slopes and coloured with
foothills, bare rock and snow, then lifted, each art pixel showing the nearest ground below raised
up to it. The noise reads the same table of random numbers as the CPU's, so both draw the same
ground.

It draws one pixel per art pixel over the part of the map on screen, then blows that up into a
texture the size of the map on screen, every art pixel a crisp square, which the map shows like
any picture under the chunks (railroads.ui.pixel.layer). A picture is drawn again only when the
view, the cells or the bed graded for track change.

The woods, and the shade trees cast on the grass, are drawn here too, a chunk at a time as its
texture is made (GroundShader.lay_surfaces): where woods cover each art pixel round the chunk
(WOODS), how far crowns reach round them (EDGES), then the crowns, the floor between them, trunks
and shadows the way railroads.ui.pixel.ground paints them (Ground.canopy), finely where an art
pixel covers several of the chunk's pixels (CANOPY), with fields and yards, and last all of it
raised with the ground and with its texture into the chunk's texture (DRESS), over which
railroads.ui.pixel.stand_shader stands the track and everything standing on the ground. The crowns
differ from the CPU's only by the grain of that texture, as the ground does. Where the CPU works
out a chunk's raised ground finely, that is blown up into the chunk's texture instead
(GroundShader.blow_up_onto).

On the desktop it reaches OpenGL through PyOpenGL; in the browser build, through the GL that
imgui draws with there (railroads.ui.pixel.webgl), as WebGL 2. Without either, or if the graphics
card refuses the shaders or can't draw into textures of floats, GroundShader.make returns None and
the map keeps its smooth ground picture under whole chunks.
"""

import ctypes
import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from railroads.sim.grids import FloatArray
from railroads.ui.pixel import ground as cpu
from railroads.ui.pixel import relief
from railroads.ui.pixel.gl_util import (
    GL,
    HEADER,
    FrameBuffer,
    RenderTarget,
    gl_program,
    new_texture,
    restore_gl_state,
    saved_gl_state,
    use,
)
from railroads.ui.pixel.ground_glsl import (
    BLOW_UP,
    GLOW_STEPS,
    PLOT_REACH,
    TREE_SHADE_REACH,
    VERTEX,
    WOODS_MARGIN,
    canopy_source,
    dress_source,
    edges_source,
    fill_names,
    fragment_source,
    raise_source,
    relief_source,
    tint_source,
    woods_source,
)
from railroads.ui.pixel.noise import NOISE_TABLE
from railroads.ui.pixel.projection import CHUNK, PixelProjection
from railroads.ui.pixel.relief import Heights
from railroads.ui.pixel.sheet import FINE


def cell_info(
    biomes: npt.NDArray[np.integer],
    deposits: npt.NDArray[np.integer] | None,
    hills: npt.NDArray[np.bool_],
    glow: FloatArray,
    mountains: npt.NDArray[np.bool_] | None = None,
    streams: npt.NDArray[np.uint8] | None = None,
) -> npt.NDArray[np.uint8]:
    """What the shader needs of each cell, as (rows, cols, 4) bytes: its biome, its deposit, 1 on
    hills and 2 on mountains with its links along a minor river (streams.stream_links) four
    times over, and how strongly it glows in GLOW_STEPS."""
    info = np.zeros((*biomes.shape, 4), dtype=np.uint8)
    info[..., 0] = biomes
    if deposits is not None:
        info[..., 1] = deposits
    info[..., 2] = hills
    if mountains is not None:
        info[..., 2] |= mountains.astype(np.uint8) << 1
    if streams is not None:
        info[..., 2] |= streams << 2
    info[..., 3] = np.rint(np.asarray(glow) * GLOW_STEPS)
    return info


@dataclass(frozen=True)
class View:
    """The art pixels a ground picture shows: `width` by `height` of them from (`left`, `top`)."""

    left: int
    top: int
    width: int
    height: int


def view_of(
    origin: tuple[int, int],
    scale: int,
    top_left: tuple[float, float],
    size: tuple[float, float],
) -> View:
    """The art pixels showing in a map area at `top_left` of `size` screen pixels, with art pixel
    (0, 0) at screen `origin` (as ChunkPlacement places it) and `scale` screen pixels to one."""
    left = math.floor((top_left[0] - origin[0]) / scale)
    top = math.floor((top_left[1] - origin[1]) / scale)
    right = math.ceil((top_left[0] + size[0] - origin[0]) / scale)
    bottom = math.ceil((top_left[1] + size[1] - origin[1]) / scale)
    return View(left, top, right - left, bottom - top)


@dataclass(frozen=True)
class Screen:
    """Where a ground picture goes on screen: its size in device pixels, where art pixel (0, 0)
    starts in it and how many device pixels each art pixel covers."""

    width: int
    height: int
    origin: tuple[float, float]
    scale: float


def screen_of(
    origin: tuple[int, int],
    scale: int,
    top_left: tuple[float, float],
    size: tuple[float, float],
    density: int,
) -> Screen:
    """The screen for a map area at `top_left` of `size` screen pixels, `density` device pixels
    to a screen pixel, with art pixel (0, 0) at screen `origin` and `scale` screen pixels to one."""
    return Screen(
        round(size[0] * density),
        round(size[1] * density),
        ((origin[0] - top_left[0]) * density, (origin[1] - top_left[1]) * density),
        scale * density,
    )


@dataclass(frozen=True)
class ChunkInputs:
    """What the ground shader worked out for a chunk that what stands on it reads: where the
    chunk starts, its raised ground (RELIEF) and where its first art pixel lies, whether each art
    pixel lies on the map (WOODS, its third channel) and where its first lies, and the metres
    an art pixel of lift stands for."""

    chunk_origin: tuple[int, int]
    relief: int
    relief_origin: tuple[int, int]
    wooded: int
    woods_origin: tuple[int, int]
    metres_per_pixel: float


@dataclass(frozen=True)
class _Programs:
    ground: int
    relief: int
    raised: int
    blow_up: int
    tint: int
    woods: int
    edges: int
    canopy: int
    dress: int


class GroundShader:
    """The compiled shaders, what they read on the graphics card (the cells, the slopes' light
    with the water and mountains round the ground, the rasters hills and mountains rise from and
    the noise table), the art pixels last drawn and the picture last blown up from them.

    The ground in a view is drawn in three passes, each over the whole of it: the raised ground of
    every art pixel (RELIEF), the ground lit and coloured with it (FRAGMENT), both reaching
    MOST_LIFT rows and a few more below the view, where ground lifted into it lies, and last the
    ground raised (RAISE)."""

    def __init__(
        self,
        programs: _Programs,
        projection: PixelProjection,
        shape: tuple[int, int],
        light: FloatArray,
        heights: Heights,
    ) -> None:
        self._programs = programs
        self._projection = projection
        self._shape = shape
        self._light_shape = light.shape[:2]
        self._heights = heights
        self._vao = GL.glGenVertexArrays(1)
        self._buffer = GL.glGenBuffers(1)
        GL.glBindVertexArray(self._vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._buffer)
        corners = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype=np.float32)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, corners.nbytes, corners, GL.GL_STATIC_DRAW)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, GL.GL_FALSE, 0, ctypes.c_void_p(0))
        GL.glBindVertexArray(0)
        self._cells = new_texture()
        self._light = _floats(light)
        self._table = _floats(NOISE_TABLE)
        self._rasters = _floats(heights.rasters)
        self._beds = 0
        self._relief = RenderTarget(floats=True)
        self._lit = RenderTarget(floats=True)
        self._art = RenderTarget()
        self._screen = RenderTarget()
        self._tinted = RenderTarget()
        self._tints = new_texture()
        self.tints_version: object = None
        self._tinted_key: tuple | None = None
        self._drawn: tuple[View, int, int] | None = None
        self._shown: tuple[View, Screen, int, int, bool] | None = None
        self.cells_version = -1
        self.bed_version = -1
        self.set_bed(None, -1)
        self._woodland = new_texture()
        self._trees = np.zeros((0, 2), dtype=np.float32)
        self._clearings = np.zeros((0, 4), dtype=np.float32)
        self._plots = np.zeros((0, 10), dtype=np.float32)
        self._yards = np.zeros((0, 4), dtype=np.float32)
        self.surfaces_version: object = None
        self._chunk_relief = RenderTarget(floats=True)
        self._wooded = RenderTarget(floats=True)
        self._edges = RenderTarget(floats=True)
        self._canopy = RenderTarget(floats=True)

    @classmethod
    def make(
        cls,
        projection: PixelProjection,
        shape: tuple[int, int],
        light: FloatArray,
        heights: Heights,
    ) -> GroundShader | None:
        """The shaders for a map of `shape` cells, whose ground is lit and coloured by `light`
        (Ground.shading) and raised by `heights` (Elevation.heights), or None where OpenGL can't
        run them."""
        if GL is None:
            return None
        try:
            programs = _Programs(
                gl_program(HEADER + VERTEX, HEADER + fragment_source()),
                gl_program(HEADER + VERTEX, HEADER + relief_source()),
                gl_program(HEADER + VERTEX, HEADER + raise_source()),
                gl_program(HEADER + VERTEX, HEADER + fill_names(BLOW_UP, {})),
                gl_program(HEADER + VERTEX, HEADER + tint_source()),
                gl_program(HEADER + VERTEX, HEADER + woods_source()),
                gl_program(HEADER + VERTEX, HEADER + edges_source()),
                gl_program(HEADER + VERTEX, HEADER + canopy_source()),
                gl_program(HEADER + VERTEX, HEADER + dress_source()),
            )
            shader = cls(programs, projection, shape, light, heights)
            shader._relief.fit(1, 1)
            return shader
        except Exception:  # noqa: BLE001 - any GL failure means falling back to the CPU ground
            return None

    def set_cells(self, info: npt.NDArray[np.uint8], version: int) -> None:
        """Takes what the shader needs of every cell (cell_info)."""
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._cells)
        rows, cols = info.shape[:2]
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_RGBA8UI,
            cols,
            rows,
            0,
            GL.GL_RGBA_INTEGER,
            GL.GL_UNSIGNED_BYTE,
            np.ascontiguousarray(info),
        )
        self.cells_version = version

    def set_bed(self, bed: tuple[FloatArray, FloatArray] | None, version: int) -> None:
        """Takes the bed graded for track (Elevation.bed), or None where no track is, with the
        number of times the ground has been graded."""
        hills = self._heights.hills
        beds = np.zeros((*hills.shape, 4), dtype=np.float32)
        beds[..., 0] = hills
        if bed is not None:
            beds[..., 1], beds[..., 2] = bed
        if self._beds:
            GL.glDeleteTextures(1, [self._beds])
        self._beds = _floats(beds)
        self.bed_version = version

    def set_tints(self, colours: npt.NDArray[np.uint8], version: object) -> None:
        """Takes each cell's tint as (rows, cols, 4) RGBA bytes, clear where it has none."""
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._tints)
        rows, cols = colours.shape[:2]
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_RGBA8,
            cols,
            rows,
            0,
            GL.GL_RGBA,
            GL.GL_UNSIGNED_BYTE,
            np.ascontiguousarray(colours),
        )
        self.tints_version = version

    def tinted(
        self,
        size: tuple[int, int],
        flat: tuple[float, float, float, float, float, float],
        raised: tuple[View, Screen] | None = None,
    ) -> int:
        """The texture of a map area `size` device pixels big with every pixel tinted by the cell
        it shows (set_tints): the map point at device pixel (x, y) from its top left corner being
        (a x + b y + e, c x + d y + f) for `flat` (a, b, c, d, e, f), or, with `raised`, the
        ground raised up to it where the ground picture of that view and screen shows it."""
        key = (size, flat, raised, self.tints_version, self.bed_version)
        if self._tinted.fit(*size) or self._tinted_key != key:
            a, b, c, d, e, f = flat
            uniforms = self._plane_uniforms() | {
                "lifted": ("1i", (int(raised is not None),)),
                "screen_to_plane": ("m2", (a, c, b, d)),
                "plane_origin": ("2f", (e, f)),
                "metres_per_pixel": ("1f", (self._heights.metres_per_pixel,)),
                "cells_size": ("2i", (self._shape[1], self._shape[0])),
                "tints": ("1i", (0,)),
                "relief": ("1i", (1,)),
            }
            if raised is not None:
                view, screen = raised
                self.art(view)
                uniforms |= {
                    "origin": ("2f", screen.origin),
                    "scale": ("1f", (screen.scale,)),
                    "relief_origin": ("2i", (view.left - 1, view.top - 1)),
                }
            inputs = (self._tints, self._relief.texture)
            self._run(self._tinted, self._programs.tint, uniforms, inputs)
            self._tinted_key = key
        return self._tinted.texture

    def art(self, view: View) -> int:
        """The texture holding the ground's art pixels in `view`, one texel each, drawn again
        only if the view, the cells or the bed for track changed since."""
        drawn = (view, self.cells_version, self.bed_version)
        if self._art.fit(view.width, view.height) or self._drawn != drawn:
            self._draw_raised(view)
            self._drawn = drawn
        return self._art.texture

    def _draw_raised(self, view: View) -> None:
        reach = relief.MOST_LIFT + 2
        relief_origin = (view.left - 1, view.top - 1)
        self._relief.fit(view.width + 2, view.height + reach + 2)
        self._lit.fit(view.width, view.height + reach)
        plane = self._plane_uniforms()
        rows, cols = self._heights.hills.shape
        uniforms = plane | {
            "art_origin": ("2i", relief_origin),
            "raster_size": ("2i", (cols, rows)),
            "table": ("1i", (0,)),
            "heights": ("1i", (1,)),
            "beds": ("1i", (2,)),
        }
        inputs = (self._table, self._rasters, self._beds)
        self._run(self._relief, self._programs.relief, uniforms, inputs)
        uniforms = self._ground_uniforms(view) | {"relief_origin": ("2i", relief_origin)}
        self._run(self._lit, self._programs.ground, uniforms, self._inputs())
        uniforms = {
            "art_origin": ("2i", (view.left, view.top)),
            "ground_origin": ("2i", (view.left, view.top)),
            "relief_origin": ("2i", relief_origin),
            "metres_per_pixel": ("1f", (self._heights.metres_per_pixel,)),
            "ground": ("1i", (0,)),
            "relief": ("1i", (1,)),
        }
        inputs = (self._lit.texture, self._relief.texture)
        self._run(self._art, self._programs.raised, uniforms, inputs)

    def picture(self, view: View, screen: Screen, blend: bool = False) -> int:
        """The texture holding the ground's art pixels in `view` blown up for `screen`, each a
        crisp square, or with `blend` blended between their middles, as the ground is up close;
        drawn again only if any of these or the cells changed since."""
        art = self.art(view)
        shown = (view, screen, self.cells_version, self.bed_version, blend)
        if self._screen.fit(screen.width, screen.height) or self._shown != shown:
            uniforms = {
                "origin": ("2f", screen.origin),
                "scale": ("1f", (screen.scale,)),
                "art_origin": ("2i", (view.left, view.top)),
                "blend": ("1i", (int(blend),)),
                "premultiply": ("1i", (0,)),
                "art": ("1i", (0,)),
            }
            self._run(self._screen, self._programs.blow_up, uniforms, (art,))
            self._shown = shown
        return self._screen.texture

    def blow_up_onto(self, target: RenderTarget, pixels: npt.NDArray[np.uint8], times: int) -> None:
        """Draws a chunk's RGBA pixels into `target`, premultiplied by their alpha, each
        repeated `times` times each way."""
        small = new_texture()
        rows, cols = pixels.shape[:2]
        data = np.ascontiguousarray(pixels)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, cols, rows, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, data
        )
        uniforms = {
            "origin": ("2f", (0.0, 0.0)),
            "scale": ("1f", (float(times),)),
            "art_origin": ("2i", (0, 0)),
            "blend": ("1i", (0,)),
            "premultiply": ("1i", (1,)),
            "art": ("1i", (0,)),
        }
        self._run(target, self._programs.blow_up, uniforms, (small,))
        GL.glDeleteTextures(1, [small])

    def set_surfaces(self, ground: cpu.Ground, version: object) -> None:
        """Takes what is drawn over the ground from `ground`: each cell's say in the woods and
        whether it is land, where trees stand, whose shade lies on the grass round them, where
        logging camps clear the woods round them, the fields and groves and the buildings' yards."""
        GL.glDeleteTextures(1, [self._woodland])
        self._woodland = _floats(np.pad(ground.woodland, ((0, 0), (0, 0), (0, 2))))
        self._trees = np.stack(ground.trees, axis=-1).astype(np.float32).reshape(-1, 2)
        xs, feet, reaches = (np.asarray(values, dtype=np.float64) for values in ground.clearings)
        ys = feet + cpu.CLEARING_DROP
        seeds = (xs * 7 + ys * 13).astype(np.int64) & 1023
        self._clearings = np.stack([xs, ys, reaches, seeds], axis=-1).astype(np.float32)
        self._plots = ground.plot_table
        self._yards = ground.yard_table
        self.surfaces_version = version

    def lay_surfaces(self, chunk: tuple[int, int], k: int) -> None:
        """Works out what lies on a chunk's ground, `k` pixels to an art pixel (no finer than
        FINE): its raised ground (RELIEF), where woods cover it (WOODS) and how far round them
        their crowns reach (EDGES), and the woods, the shade of trees, fields and yards over it
        (CANOPY), for `dress_onto` to lay and for what stands on it to read (`chunk_inputs`)."""
        drawn = min(k, FINE)
        left, top = chunk[0] * CHUNK, chunk[1] * CHUNK
        rows = CHUNK + relief.MOST_LIFT + 2
        relief_origin = (left - 1, top - 1)
        woods_origin = (left - WOODS_MARGIN, top - WOODS_MARGIN)
        self._chunk_relief.fit(CHUNK + 2, rows + 2)
        self._wooded.fit(CHUNK + 2 * WOODS_MARGIN, rows + 2 * WOODS_MARGIN)
        self._edges.fit(*self._wooded.size)
        self._canopy.fit(CHUNK * drawn, rows * drawn)
        plane = self._plane_uniforms()
        uniforms = plane | {
            "art_origin": ("2i", relief_origin),
            "raster_size": ("2i", self._heights.hills.shape[::-1]),
            "table": ("1i", (0,)),
            "heights": ("1i", (1,)),
            "beds": ("1i", (2,)),
        }
        inputs = (self._table, self._rasters, self._beds)
        self._run(self._chunk_relief, self._programs.relief, uniforms, inputs)
        size = self._wooded.size
        trees = _near(self._trees, woods_origin, size, TREE_SHADE_REACH)
        mills = self._clearings
        mills = _near(mills, woods_origin, size, float(mills[:, 2].max(initial=0.0)) + 1)
        uniforms = plane | {
            "art_origin": ("2i", woods_origin),
            "cells_size": ("2i", (self._shape[1], self._shape[0])),
            "tree_count": ("1i", (len(trees),)),
            "mill_count": ("1i", (len(mills),)),
            "table": ("1i", (0,)),
            "cells": ("1i", (1,)),
            "woods": ("1i", (2,)),
            "heights": ("1i", (3,)),
            "trees": ("1i", (4,)),
            "mills": ("1i", (5,)),
        }
        listed = (_floats(_row(trees)), _floats(_row(mills)))
        inputs = (self._table, self._cells, self._woodland, self._rasters, *listed)
        self._run(self._wooded, self._programs.woods, uniforms, inputs)
        GL.glDeleteTextures(2, list(listed))
        uniforms = {"fine": ("1i", (int(drawn > 1),)), "wooded": ("1i", (0,))}
        self._run(self._edges, self._programs.edges, uniforms, (self._wooded.texture,))
        chunk_size = (CHUNK, rows)
        plots = _near(self._plots, (left, top), chunk_size, PLOT_REACH)[:, 2:]
        yards = self._yards
        yards = _near(yards, (left, top), chunk_size, float(yards[:, 2].max(initial=0.0)) + 1)
        uniforms = self._ground_uniforms(View(left, top, CHUNK, rows)) | {
            "chunk_origin": ("2i", (left, top)),
            "k": ("1i", (drawn,)),
            "woods_origin": ("2i", woods_origin),
            "relief_origin": ("2i", relief_origin),
            "plot_count": ("1i", (len(plots),)),
            "yard_count": ("1i", (len(yards),)),
            "wooded": ("1i", (4,)),
            "edges": ("1i", (5,)),
            "plots": ("1i", (6,)),
            "yards": ("1i", (7,)),
        }
        listed = (_floats(_columns(plots)), _floats(_row(yards)))
        textures = (self._wooded.texture, self._edges.texture, *listed)
        inputs = (self._cells, self._light, self._table, self._chunk_relief.texture, *textures)
        self._run(self._canopy, self._programs.canopy, uniforms, inputs)
        GL.glDeleteTextures(2, list(listed))

    def dress_onto(self, target: RenderTarget, chunk: tuple[int, int], k: int, times: int) -> None:
        """Draws what `lay_surfaces` worked out for a chunk into `target`, premultiplied by its
        alpha, raised with the ground where it rises and laid with the ground's texture finely,
        crops out of the reach of crowns with blades: `k` pixels to an art pixel, blown up
        `times` times each way (DRESS)."""
        left, top = chunk[0] * CHUNK, chunk[1] * CHUNK
        uniforms = {
            "chunk_origin": ("2i", (left, top)),
            "k": ("1i", (k,)),
            "drawn": ("1i", (min(k, FINE),)),
            "times": ("1i", (times,)),
            "relief_origin": ("2i", (left - 1, top - 1)),
            "metres_per_pixel": ("1f", (self._heights.metres_per_pixel,)),
            "woods_origin": ("2i", (left - WOODS_MARGIN, top - WOODS_MARGIN)),
            "surface": ("1i", (0,)),
            "relief": ("1i", (1,)),
            "edges": ("1i", (2,)),
        }
        inputs = (self._canopy.texture, self._chunk_relief.texture, self._edges.texture)
        self._run(target, self._programs.dress, uniforms, inputs)

    def chunk_inputs(self, chunk: tuple[int, int]) -> ChunkInputs:
        """What `lay_surfaces` last worked out that what stands on a chunk reads: its raised
        ground and whether each art pixel lies on the map, with where each starts."""
        left, top = chunk[0] * CHUNK, chunk[1] * CHUNK
        return ChunkInputs(
            (left, top),
            self._chunk_relief.texture,
            (left - 1, top - 1),
            self._wooded.texture,
            (left - WOODS_MARGIN, top - WOODS_MARGIN),
            self._heights.metres_per_pixel,
        )

    def _inputs(self) -> tuple[int, ...]:
        return self._cells, self._light, self._table, self._relief.texture

    def _plane_uniforms(self) -> dict[str, tuple[str, tuple]]:
        (a, b), (c, d) = np.linalg.inv(self._projection.matrix)
        return {
            "to_plane": ("m2", (a, c, b, d)),
            "projection_origin": ("2f", tuple(map(float, self._projection.origin))),
        }

    def _ground_uniforms(self, view: View) -> dict[str, tuple[str, tuple]]:
        (m00, m01), (m10, m11) = self._projection.matrix
        return self._plane_uniforms() | {
            "art_origin": ("2i", (view.left, view.top)),
            "slope": ("4f", (m00, m10, m01, m11)),
            "metres_per_unit": ("1f", (self._heights.metres_per_unit,)),
            "cells_size": ("2i", (self._shape[1], self._shape[0])),
            "light_size": ("2i", (self._light_shape[1], self._light_shape[0])),
            "cells": ("1i", (0,)),
            "light": ("1i", (1,)),
            "table": ("1i", (2,)),
            "relief": ("1i", (3,)),
        }

    def _run(
        self,
        target: RenderTarget | FrameBuffer,
        program: int,
        uniforms: dict[str, tuple[str, tuple]],
        inputs: tuple[int, ...],
    ) -> None:
        """Draws `program` over the whole of `target`, reading `inputs` from texture units 0 on."""
        saved = saved_gl_state()
        try:
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, target.framebuffer)
            GL.glViewport(0, 0, *target.size)
            GL.glDisable(GL.GL_BLEND)
            GL.glDisable(GL.GL_SCISSOR_TEST)
            use(program, uniforms, inputs)
            GL.glBindVertexArray(self._vao)
            GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)
        finally:
            restore_gl_state(saved)


def _near(
    table: npt.NDArray[np.float32], origin: tuple[int, int], size: tuple[int, int], reach: float
) -> npt.NDArray[np.float32]:
    """The rows of `table`, each starting with a place in the picture, within `reach` art pixels
    of the art pixels from `origin` of `size`."""
    left, top = origin[0] - reach, origin[1] - reach
    right, bottom = origin[0] + size[0] + reach, origin[1] + size[1] + reach
    xs, ys = table[:, 0], table[:, 1]
    return table[(xs > left) & (xs < right) & (ys > top) & (ys < bottom)]


def _columns(values: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Eight values of each of a list as a column of two texels of four floats (at least one
    column) for a texture."""
    columns = np.zeros((2, max(len(values), 1), 4), dtype=np.float32)
    columns[0, : len(values)] = values[:, :4]
    columns[1, : len(values)] = values[:, 4:]
    return columns


def _row(values: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Values of a list, a few to each, as one row of four floats each (at least one) for a
    texture."""
    row = np.zeros((1, max(len(values), 1), 4), dtype=np.float32)
    row[0, : len(values), : values.shape[1]] = values
    return row


def _floats(values: npt.NDArray[np.floating]) -> int:
    """A texture of one float per texel, or four where `values` has four to a place, read with
    texelFetch."""
    texture = new_texture()
    data = np.ascontiguousarray(values, dtype=np.float32)
    rows, cols = data.shape[:2]
    internal, form = (GL.GL_RGBA32F, GL.GL_RGBA) if data.ndim == 3 else (GL.GL_R32F, GL.GL_RED)
    GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, internal, cols, rows, 0, form, GL.GL_FLOAT, data)
    return texture
