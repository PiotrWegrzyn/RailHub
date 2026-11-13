"""The ground of the pixel map drawn by the graphics card: a shader works out every screen pixel
from the art pixel under it the way railroads.ui.pixel.ground does on the CPU (Ground.render):
which hex cell it lies in, the cell's biome and its ramp of shades picked by noise, meadows and
savanna grass with blades and flowers, tufts, rippled water with sandy shores and foam, the light
on the slopes, the glow where buildings are served and the signs of deposits. The noise reads the
same table of random numbers as the CPU's, so both draw the same ground.

It draws one pixel per art pixel over the part of the map on screen, then blows that up into a
texture the size of the map on screen, every art pixel a crisp square, which the map shows like
any picture under the chunks the CPU draws (railroads.ui.pixel.layer). Chunks where nothing rises are then
drawn without their ground (PixelWorld.render with `over_ground`), with only the fields, woods,
track and buildings to lay over it; chunks where hills or mountains rise still bring their own
lifted ground. A picture is drawn again only when the view or the cells change.

The same blowing up makes the chunks' textures where a chunk's pixel covers several screen pixels
(GroundShader.blow_up): imgui samples every texture smoothly, so a texture must hold a texel for
every screen pixel to show crisp squares, and blowing a chunk up here gives the very texels the CPU
would (np.repeat) from an upload a fraction of the size.

On the desktop it reaches OpenGL through PyOpenGL; in the browser build, through the GL that
imgui draws with there (railroads.ui.pixel.webgl), as WebGL 2. Without either, or if the graphics
card refuses the shader, GroundShader.make returns None and the map keeps its smooth ground
picture under whole chunks.
"""

import ctypes
import math
import sys
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from railroads.sim.tilemap import Deposit, TileMap
from railroads.ui.biomes import Biome
from railroads.ui.pixel import ground as cpu
from railroads.ui.pixel.noise import _TABLE, TABLE_SIZE
from railroads.ui.pixel.projection import FloatArray, PixelProjection

if sys.platform == "emscripten":
    from railroads.ui.pixel import webgl as GL

    HEADER = "#version 300 es\nprecision highp float;\nprecision highp int;\n"
else:
    try:
        from OpenGL import GL
    except ImportError:
        GL = None
    HEADER = "#version 330 core\n"

# Each cell's glow is handed to the graphics card in steps of this much.
GLOW_STEPS = 250

VERTEX = """layout(location = 0) in vec2 corner;
void main() {
    gl_Position = vec4(corner, 0.0, 1.0);
}
"""

FRAGMENT = """out vec4 colour;
uniform ivec2 art_origin;
uniform mat2 to_plane;
uniform vec2 projection_origin;
uniform ivec2 cells_size;
uniform ivec2 light_size;
uniform highp usampler2D cells;
uniform highp sampler2D light;
uniform highp sampler2D table;

const float HEX_RADIUS = 0.5773502691896258;
const float SQRT3 = 1.7320508075688772;
const int MASK = @MASK@;
const vec3 RAMPS[@RAMP_COUNT@] = vec3[@RAMP_COUNT@](@RAMPS@);
const float BAYER[16] = float[16](@BAYER@);
const int TUFT_ROWS[6] = int[6](@TUFT_ROWS@);
const vec3 FLOWERS[4] = vec3[4](@FLOWERS@);
const vec3 SPECKS[@SPECK_COUNT@] = vec3[@SPECK_COUNT@](@SPECKS@);

int fdiv(int a, int b) {
    return a >= 0 ? a / b : -((-a + b - 1) / b);
}

int fmod(int a, int b) {
    return a - b * fdiv(a, b);
}

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

vec2 plane(ivec2 art) {
    return to_plane * (vec2(art) + 0.5 + projection_origin);
}

uvec4 cell_info(ivec2 cell) {
    return texelFetch(cells, clamp(cell, ivec2(0), cells_size - 1), 0);
}

bool is_water(uint biome) {
    return biome <= @LAST_WATER@u;
}

bool in_set(uint biome, uint set) {
    return ((set >> biome) & 1u) != 0u;
}

float tbl(int i, int j) {
    return texelFetch(table, ivec2(i & MASK, j & MASK), 0).r;
}

float hash(int a, int b, int seed) {
    return tbl(a * 13 + seed * 71, b * 7 + seed * 131);
}

float noise(vec2 p, float inv, float off_u, float off_v) {
    float u = p.x * inv + off_u;
    float v = p.y * inv + off_v;
    float fi = floor(u), fj = floor(v);
    int i0 = int(fi) & MASK, j0 = int(fj) & MASK;
    int i1 = (i0 + 1) & MASK, j1 = (j0 + 1) & MASK;
    float fu = u - fi, fv = v - fj;
    fu = fu * fu * (3.0 - 2.0 * fu);
    fv = fv * fv * (3.0 - 2.0 * fv);
    float top = tbl(i0, j0) * (1.0 - fu) + tbl(i1, j0) * fu;
    float bottom = tbl(i0, j1) * (1.0 - fu) + tbl(i1, j1) * fu;
    return top * (1.0 - fv) + bottom * fv;
}

float slope_light(vec2 p) {
    vec2 at = clamp(p * @STEPS@ - 0.5, vec2(0.0), vec2(light_size - 1));
    ivec2 corner = min(ivec2(at), light_size - 2);
    vec2 t = at - vec2(corner);
    float a = texelFetch(light, corner, 0).r;
    float b = texelFetch(light, corner + ivec2(1, 0), 0).r;
    float c = texelFetch(light, corner + ivec2(0, 1), 0).r;
    float d = texelFetch(light, corner + ivec2(1, 1), 0).r;
    return mix(a, b, t.x) * (1.0 - t.y) + mix(c, d, t.x) * t.y;
}

vec3 ramp(float t, vec3 c0, vec3 c1, vec3 c2) {
    t = clamp(t, 0.0, 1.0);
    return t < 0.5 ? mix(c0, c1, t * 2.0) : mix(c1, c2, (t - 0.5) * 2.0);
}

vec3 grass(vec2 p, int ix, int iy, float lit, int kind_of_grass) {
    vec3 s0, s1, s2, t0, t1, t2, b0, b1, b2;
    float rarity;
    if (kind_of_grass == 0) {
        @MEADOW_RAMPS@
    } else {
        @SAVANNA_RAMPS@
    }
    float kind = @N_55_81@ + 0.25 * (@N_15_85@ - 0.5);
    float t = 0.5 + 0.36 * (@N_24_41@ - 0.5) + 0.14 * (@N_4_42@ - 0.5) + lit / 3.0;
    float rough = clamp((kind - 0.55) * 3.5, 0.0, 1.0);
    float dry = clamp((@N_45_82@ - 0.74) * 3.5, 0.0, 1.0) * (1.0 - rough);
    float pasture = 1.0 - rough - dry;
    vec3 rgb = ramp(t, s0, s1, s2) * pasture + ramp(t, t0, t1, t2) * rough
        + ramp(t, b0, b1, b2) * dry;
    bool even = fmod(iy, 2) == 0;
    float blade = (even && hash(ix, fdiv(iy, 2), 83) > 0.72) ? 1.0 : 0.0;
    float tip = (!even && hash(ix, fdiv(iy + 1, 2), 83) > 0.72) ? 1.0 : 0.0;
    rgb *= 1.0 - 0.14 * rough * blade + 0.1 * rough * tip;
    rgb += (hash(ix, iy, 84) - 0.5) * 8.0 + (@N_03_86@ - 0.5) * 10.0;
    if (@N_5_91@ > 0.6 && hash(ix, iy, 92) > rarity) {
        rgb = FLOWERS[int(hash(ix, iy, 93) * 4.0)];
    }
    return rgb;
}

bool water_at(ivec2 art) {
    return is_water(cell_info(cell_at(plane(art))).r);
}

void main() {
    // The texture's first row is the picture's top row, as imgui shows it.
    ivec2 art = ivec2(floor(gl_FragCoord.xy)) + art_origin;
    int ix = art.x, iy = art.y;
    vec2 p = plane(art);
    ivec2 cell = cell_at(p);
    if (cell.x < 0 || cell.y < 0 || cell.x >= cells_size.x || cell.y >= cells_size.y) {
        colour = vec4(0.0);
        return;
    }
    uvec4 info = cell_info(cell);
    uint biome = info.r;
    bool water = is_water(biome);
    float lit = slope_light(p);
    float value = 1.45 + 1.6 * (@N_24_41@ - 0.5) + 0.8 * (@N_75_42@ - 0.5);
    value += 0.45 * (@N_22_43@ - 0.5) + 0.45 * BAYER[fmod(iy, 4) * 4 + fmod(ix, 4)];
    value += lit + @HILL_LIFT@ * float(info.b);
    if (water) {
        bool ripple = hash(fdiv(ix, 3), iy, 13) > 0.94;
        value = ripple ? 3.0 : 1.3 + 0.6 * (hash(ix, iy, 11) - 0.5);
    }
    int shade = int(clamp(roundEven(value), 0.0, 3.0));
    vec3 rgb = RAMPS[int(biome) * 4 + shade];
    if (!water && in_set(biome, @MEADOWS@u)) {
        rgb = grass(p, ix, iy, lit, 0);
    } else if (!water && biome == @SAVANNA@u) {
        rgb = grass(p, ix, iy, lit, 1);
    } else if (!water && in_set(biome, @GRASSY@u)) {
        bool tuft = hash(fdiv(ix, 6), fdiv(iy, 6), 51) > @TUFT_ABOVE@
            && ((TUFT_ROWS[fmod(iy, 6)] >> fmod(ix, 6)) & 1) != 0;
        if (tuft) {
            rgb = RAMPS[int(biome) * 4 + max(shade - 1, 0)] * 0.92;
        }
    }
    if (water) {
        if (!water_at(art + ivec2(1, 0)) || !water_at(art - ivec2(1, 0))
                || !water_at(art + ivec2(0, 1)) || !water_at(art - ivec2(0, 1))) {
            rgb = @FOAM@;
        }
    } else if (water_at(art + ivec2(1, 0)) || water_at(art - ivec2(1, 0))
            || water_at(art + ivec2(0, 1)) || water_at(art - ivec2(0, 1))
            || water_at(art + ivec2(2, 0)) || water_at(art - ivec2(2, 0))) {
        rgb = @SAND@;
    }
    if (!water) {
        float glow = float(info.a) / @GLOW_STEPS@;
        rgb = rgb * (1.0 - glow) + @GLOW@ * glow;
    }
    uint deposit = info.g;
    if (!water && deposit > 0u && deposit != @GARRISON@u
            && hash(fdiv(ix, 12), fdiv(iy, 12), 71) > @SIGNS_ABOVE@) {
        int lx = fmod(ix, 12), ly = fmod(iy, 12);
        if (deposit == @WOOL@u) {
            if (lx >= 4 && lx <= 7 && ly >= 5 && ly <= 7) rgb = @SHEEP@;
            if (lx == 3 && ly == 5) rgb = @SHEEP_DARK@;
            if ((lx == 4 || lx == 7) && ly == 8) rgb = @SHEEP_DARK@;
            if (lx >= 4 && lx <= 7 && ly == 9) rgb *= 0.7;
        } else {
            if (lx >= 4 && lx <= 7 && ly >= 6 && ly <= 8 && !(lx == 7 && ly == 6)) {
                rgb = @ORE_ROCK@;
            }
            if (lx == 5 && ly == 7) rgb = SPECKS[deposit];
            if (lx >= 4 && lx <= 8 && ly == 9) rgb *= 0.72;
        }
    }
    colour = vec4(clamp(rgb, 0.0, 255.0) / 255.0, 1.0);
}
"""


BLOW_UP = """out vec4 colour;
uniform vec2 origin;
uniform float scale;
uniform ivec2 art_origin;
uniform highp sampler2D art;
void main() {
    ivec2 at = ivec2(floor((gl_FragCoord.xy - origin) / scale)) - art_origin;
    colour = texelFetch(art, at, 0);
}
"""


def _number(value: float) -> str:
    """A GLSL float literal for exactly the float32 the CPU works with."""
    text = repr(float(np.float32(value)))
    return text if "." in text or "e" in text else f"{text}.0"


def _vec3(rgb: object) -> str:
    r, g, b = (float(v) for v in np.asarray(rgb, dtype=np.float32))
    return f"vec3({_number(r)}, {_number(g)}, {_number(b)})"


def _noise(scale: float, seed: int) -> str:
    """The call for railroads.ui.pixel.noise.noise at `scale` and `seed`, with its numbers worked
    out in float32 as there."""
    inverse = np.float32(1 / scale)
    off_u, off_v = np.float32(seed * 37.1 + 500.0), np.float32(seed * 11.3 + 500.0)
    return f"noise(p, {_number(inverse)}, {_number(off_u)}, {_number(off_v)})"


def _bits(biomes: tuple[Biome, ...]) -> int:
    return sum(1 << int(biome) for biome in set(biomes))


def _grass_ramps(name: str) -> str:
    short, tall, bare, rarity = cpu.GRASSES[name]
    lines = []
    for prefix, stops in (("s", short), ("t", tall), ("b", bare)):
        assert [at for at, _ in stops] == [0.0, 0.5, 1.0]
        lines += [f"{prefix}{i} = {_vec3(colour)};" for i, (_, colour) in enumerate(stops)]
    lines.append(f"rarity = {_number(rarity)};")
    return " ".join(lines)


def fragment_source() -> str:
    """The fragment shader with the CPU ground's tables and numbers written in."""
    ramps = [_vec3(cpu.RAMPS[biome][shade]) for biome in Biome for shade in range(4)]
    tuft_rows = [sum(1 << x for x in range(6) if cpu.TUFT[y, x]) for y in range(6)]
    names = {
        "MASK": str(TABLE_SIZE - 1),
        "RAMP_COUNT": str(len(ramps)),
        "RAMPS": ", ".join(ramps),
        "BAYER": ", ".join(_number(v) for v in cpu.BAYER.ravel()),
        "TUFT_ROWS": ", ".join(str(row) for row in tuft_rows),
        "FLOWERS": ", ".join(_vec3(f) for f in cpu.FLOWERS),
        "SPECK_COUNT": str(len(cpu.DEPOSIT_SPECKS)),
        "SPECKS": ", ".join(_vec3(s) for s in cpu.DEPOSIT_SPECKS),
        "LAST_WATER": str(max(int(b) for b in cpu.WATER)),
        "STEPS": _number(cpu.RELIEF_STEPS_PER_CELL),
        "MEADOW_RAMPS": _grass_ramps("meadow"),
        "SAVANNA_RAMPS": _grass_ramps("savanna"),
        "HILL_LIFT": _number(cpu.HILL_LIFT),
        "MEADOWS": str(_bits(cpu.MEADOWS)),
        "GRASSY": str(_bits(cpu.GRASSY)),
        "SAVANNA": str(int(Biome.SAVANNA)),
        "TUFT_ABOVE": _number(1 - cpu.TUFT_SHARE),
        "FOAM": _vec3(cpu.FOAM),
        "SAND": _vec3(cpu.SAND),
        "GLOW_STEPS": _number(GLOW_STEPS),
        "GLOW": _vec3(cpu.GLOW),
        "GARRISON": str(int(Deposit.GARRISON)),
        "WOOL": str(int(Deposit.WOOL)),
        "SIGNS_ABOVE": _number(1 - cpu.DEPOSIT_SIGNS),
        "SHEEP": _vec3(cpu.SHEEP),
        "SHEEP_DARK": _vec3(cpu.SHEEP_DARK),
        "ORE_ROCK": _vec3(cpu.ORE_ROCK),
        "N_24_41": _noise(2.4, 41),
        "N_75_42": _noise(0.75, 42),
        "N_22_43": _noise(0.22, 43),
        "N_55_81": _noise(0.55, 81),
        "N_15_85": _noise(0.15, 85),
        "N_4_42": _noise(0.4, 42),
        "N_45_82": _noise(0.45, 82),
        "N_03_86": _noise(0.03, 86),
        "N_5_91": _noise(0.5, 91),
    }
    source = FRAGMENT
    for name, value in names.items():
        source = source.replace(f"@{name}@", value)
    assert "@" not in source
    return source


def cell_info(
    biomes: npt.NDArray[np.integer],
    deposits: npt.NDArray[np.integer] | None,
    hills: npt.NDArray[np.bool_],
    glow: FloatArray,
) -> npt.NDArray[np.uint8]:
    """What the shader needs of each cell, as (rows, cols, 4) bytes: its biome, its deposit, 1 on
    hills, and how strongly it glows in GLOW_STEPS."""
    info = np.zeros((*biomes.shape, 4), dtype=np.uint8)
    info[..., 0] = biomes
    if deposits is not None:
        info[..., 1] = deposits
    info[..., 2] = hills
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
class _Into:
    """A framebuffer the shaders draw into, `size` device pixels."""

    framebuffer: int
    size: tuple[int, int]


# Textures of chunks let go of since GL was last reached, deleted then: they can be let go of by
# whichever thread drops the last hold on them, and GL is reached from the frames only.
_UNWANTED: list[int] = []


class ChunkTexture:
    """A chunk's texture blown up by the graphics card (GroundShader.blow_up), shown like any
    texture imgui made and deleted once nothing holds it."""

    def __init__(self, texture: int) -> None:
        self._texture = texture

    def texture_id(self) -> int:
        return self._texture

    def __del__(self) -> None:
        _UNWANTED.append(self._texture)


def free_textures() -> None:
    """Deletes the chunks' textures nothing holds any more; called from the frames."""
    if _UNWANTED:
        unwanted = [_UNWANTED.pop() for _ in range(len(_UNWANTED))]
        GL.glDeleteTextures(len(unwanted), unwanted)


class _Target:
    """A texture the shaders draw into, with its framebuffer."""

    def __init__(self) -> None:
        self.texture = 0
        self.framebuffer = GL.glGenFramebuffers(1)
        self.size = (0, 0)

    def fit(self, width: int, height: int) -> bool:
        """Makes the texture this size; returns whether it had to be made anew."""
        if self.size == (width, height):
            return False
        if self.texture:
            GL.glDeleteTextures(1, [self.texture])
        self.texture = _texture()
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
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.framebuffer)
        GL.glFramebufferTexture2D(
            GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, self.texture, 0
        )
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, int(saved))
        self.size = (width, height)
        return True


class GroundShader:
    """The compiled shaders, what they read on the graphics card (the cells, the slopes' light
    and the noise table), the art pixels last drawn and the picture last blown up from them."""

    def __init__(
        self,
        programs: tuple[int, int],
        projection: PixelProjection,
        tile_map: TileMap,
        light: FloatArray,
    ) -> None:
        self._ground, self._blow_up = programs
        self._projection = projection
        self._shape = tile_map.grid.shape
        self._light_shape = light.shape
        self._vao = GL.glGenVertexArrays(1)
        self._buffer = GL.glGenBuffers(1)
        GL.glBindVertexArray(self._vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._buffer)
        corners = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype=np.float32)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, corners.nbytes, corners, GL.GL_STATIC_DRAW)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, GL.GL_FALSE, 0, ctypes.c_void_p(0))
        GL.glBindVertexArray(0)
        self._cells = _texture()
        self._light = _floats(light)
        self._table = _floats(_TABLE)
        self._art = _Target()
        self._screen = _Target()
        self._drawn: tuple[View, int] | None = None
        self._shown: tuple[View, Screen, int] | None = None
        self.cells_version = -1
        self._chunks = GL.glGenFramebuffers(1)

    @classmethod
    def make(
        cls, projection: PixelProjection, tile_map: TileMap, light: FloatArray
    ) -> GroundShader | None:
        """The shaders for this map, whose slopes catch `light` (Ground's raster of it), or None
        where OpenGL can't run them."""
        if GL is None:
            return None
        try:
            ground = _program(HEADER + VERTEX, HEADER + fragment_source())
            blow_up = _program(HEADER + VERTEX, HEADER + BLOW_UP)
            return cls((ground, blow_up), projection, tile_map, light)
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

    def art(self, view: View) -> int:
        """The texture holding the ground's art pixels in `view`, one texel each, drawn again
        only if the view or the cells changed since."""
        if self._art.fit(view.width, view.height) or self._drawn != (view, self.cells_version):
            self._run(self._art, self._ground, self._ground_uniforms(view), self._inputs())
            self._drawn = (view, self.cells_version)
        return self._art.texture

    def picture(self, view: View, screen: Screen) -> int:
        """The texture holding the ground's art pixels in `view` blown up for `screen`, drawn
        again only if either or the cells changed since."""
        art = self.art(view)
        shown = (view, screen, self.cells_version)
        if self._screen.fit(screen.width, screen.height) or self._shown != shown:
            uniforms = {
                "origin": ("2f", screen.origin),
                "scale": ("1f", (screen.scale,)),
                "art_origin": ("2i", (view.left, view.top)),
                "art": ("1i", (0,)),
            }
            self._run(self._screen, self._blow_up, uniforms, (art,))
            self._shown = shown
        return self._screen.texture

    def blow_up(self, pixels: npt.NDArray[np.uint8], times: int) -> ChunkTexture:
        """A texture of a chunk's RGBA pixels with each repeated `times` times each way."""
        free_textures()
        rows, cols = pixels.shape[:2]
        size = (cols * times, rows * times)
        small = _texture()
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_RGBA8,
            cols,
            rows,
            0,
            GL.GL_RGBA,
            GL.GL_UNSIGNED_BYTE,
            np.ascontiguousarray(pixels),
        )
        big = _texture()
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, *size, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None
        )
        saved = GL.glGetIntegerv(GL.GL_FRAMEBUFFER_BINDING)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._chunks)
        GL.glFramebufferTexture2D(
            GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, big, 0
        )
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, int(saved))
        uniforms = {
            "origin": ("2f", (0.0, 0.0)),
            "scale": ("1f", (float(times),)),
            "art_origin": ("2i", (0, 0)),
            "art": ("1i", (0,)),
        }
        self._run(_Into(self._chunks, size), self._blow_up, uniforms, (small,))
        GL.glDeleteTextures(1, [small])
        return ChunkTexture(big)

    def _inputs(self) -> tuple[int, ...]:
        return self._cells, self._light, self._table

    def _ground_uniforms(self, view: View) -> dict[str, tuple[str, tuple]]:
        (a, b), (c, d) = np.linalg.inv(self._projection.matrix)
        return {
            "art_origin": ("2i", (view.left, view.top)),
            "to_plane": ("m2", (a, c, b, d)),
            "projection_origin": ("2f", tuple(map(float, self._projection.origin))),
            "cells_size": ("2i", (self._shape[1], self._shape[0])),
            "light_size": ("2i", (self._light_shape[1], self._light_shape[0])),
            "cells": ("1i", (0,)),
            "light": ("1i", (1,)),
            "table": ("1i", (2,)),
        }

    def _run(
        self,
        target: _Target | _Into,
        program: int,
        uniforms: dict[str, tuple[str, tuple]],
        inputs: tuple[int, ...],
    ) -> None:
        """Draws `program` over the whole of `target`, reading `inputs` from texture units 0 on."""
        saved = _saved_state()
        try:
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, target.framebuffer)
            GL.glViewport(0, 0, *target.size)
            GL.glDisable(GL.GL_BLEND)
            GL.glDisable(GL.GL_SCISSOR_TEST)
            GL.glUseProgram(program)
            for name, (kind, values) in uniforms.items():
                at = GL.glGetUniformLocation(program, name)
                match kind:
                    case "1i":
                        GL.glUniform1i(at, *values)
                    case "2i":
                        GL.glUniform2i(at, *values)
                    case "1f":
                        GL.glUniform1f(at, *values)
                    case "2f":
                        GL.glUniform2f(at, *values)
                    case "m2":
                        matrix = np.array(values, np.float32)
                        GL.glUniformMatrix2fv(at, 1, GL.GL_FALSE, matrix)
            for unit, texture in enumerate(inputs):
                GL.glActiveTexture(GL.GL_TEXTURE0 + unit)
                GL.glBindTexture(GL.GL_TEXTURE_2D, texture)
            GL.glBindVertexArray(self._vao)
            GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)
        finally:
            _restore(saved)


def _texture() -> int:
    texture = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, texture)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
    return texture


def _floats(values: npt.NDArray[np.floating]) -> int:
    """A texture of one float per texel, read with texelFetch."""
    texture = _texture()
    data = np.ascontiguousarray(values, dtype=np.float32)
    rows, cols = data.shape
    GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_R32F, cols, rows, 0, GL.GL_RED, GL.GL_FLOAT, data)
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
    for unit in (2, 1):
        GL.glActiveTexture(GL.GL_TEXTURE0 + unit)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
    GL.glActiveTexture(active)
    if blend:
        GL.glEnable(GL.GL_BLEND)
    if scissor:
        GL.glEnable(GL.GL_SCISSOR_TEST)
