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
any picture under the chunks the CPU draws (railroads.ui.pixel.layer). Chunks are then drawn
without their ground (PixelWorld.render with `over_ground`), with only the fields, woods, track
and buildings to lay over it, lifted with the ground where it rises. A picture is drawn again only
when the view, the cells or the bed graded for track change.

The same blowing up makes the chunks' textures where a chunk's pixel covers several screen pixels
(GroundShader.blow_up): imgui samples every texture smoothly, so a texture must hold a texel for
every screen pixel to show crisp squares, and blowing a chunk up here gives the very texels the CPU
would (np.repeat) from an upload a fraction of the size.

The woods, and the shade trees cast on the grass, are drawn here too, a chunk at a time as its texture
is made (GroundShader.dress), so the CPU draws chunks without them: where woods cover each art
pixel round the chunk (WOODS), how far crowns reach round them (EDGES), then the crowns, the floor
between them, trunks and shadows the way railroads.ui.pixel.ground paints them (Ground.canopy),
finely where an art pixel covers several of the chunk's pixels (CANOPY), and last the chunk laid
over them, raised with the ground and with its texture (DRESS). The crowns differ from the CPU's
only by the grain of that texture, as the ground does.

On the desktop it reaches OpenGL through PyOpenGL; in the browser build, through the GL that
imgui draws with there (railroads.ui.pixel.webgl), as WebGL 2. Without either, or if the graphics
card refuses the shaders or can't draw into textures of floats, GroundShader.make returns None and
the map keeps its smooth ground picture under whole chunks.
"""

import ctypes
import math
import sys
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from railroads.sim.grids import FloatArray
from railroads.sim.tilemap import Deposit
from railroads.ui.biomes import Biome
from railroads.ui.iso_camera import TILT
from railroads.ui.pixel import ground as cpu
from railroads.ui.pixel import relief
from railroads.ui.pixel.noise import _TABLE, TABLE_SIZE
from railroads.ui.pixel.projection import CHUNK, PixelProjection
from railroads.ui.pixel.relief import LIGHT, Heights
from railroads.ui.pixel.sheet import FINE

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
# The woods are worked out this many art pixels round a chunk, as far as anything in it looks for
# them: the crowns and shadows of trees standing there, and where crowns reach.
WOODS_MARGIN = 16
# How far from a chunk the trees whose shade, and the middles of the cells whose fields, can reach
# into it lie, in art pixels.
TREE_SHADE_REACH = 12
PLOT_REACH = 32

VERTEX = """layout(location = 0) in vec2 corner;
void main() {
    gl_Position = vec4(corner, 0.0, 1.0);
}
"""

NOISE = """uniform highp sampler2D table;
const int MASK = @MASK@;

float tbl(int i, int j) {
    return texelFetch(table, ivec2(i & MASK, j & MASK), 0).r;
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

float crest(float n) {
    float r = 1.0 - abs(2.0 * n - 1.0);
    return r * r;
}
"""

HEX = """const float HEX_RADIUS = 0.5773502691896258;
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
"""

# Each screen pixel of the map tinted by the cell it shows (cells' colours in `tints`): the map
# point under it where the map is flat (`flat`), or in pixel art the ground raised up to it, as
# RAISE finds it.
TINT = """out vec4 colour;
uniform int lifted;
uniform mat2 screen_to_plane;
uniform vec2 plane_origin;
uniform vec2 origin;
uniform float scale;
uniform mat2 to_plane;
uniform vec2 projection_origin;
uniform ivec2 relief_origin;
uniform float metres_per_pixel;
uniform ivec2 cells_size;
uniform highp sampler2D tints;
uniform highp sampler2D relief;
@HEX@
float lift_at(ivec2 art) {
    float rise = texelFetch(relief, art - relief_origin, 0).x;
    return clamp(rise / metres_per_pixel * @EXAGGERATION@ * @TILT@, 0.0, @MOST_LIFT@);
}

void main() {
    vec2 p;
    if (lifted == 1) {
        vec2 at = (gl_FragCoord.xy - origin) / scale;
        ivec2 art = ivec2(floor(at));
        int offset = 0;
        for (int d = 1; d <= @MOST_LIFT_ROWS@; d++) {
            if (floor(lift_at(art + ivec2(0, d))) >= float(d)) offset = d;
        }
        p = to_plane * (at + vec2(0.0, float(offset)) + projection_origin);
    } else {
        p = screen_to_plane * gl_FragCoord.xy + plane_origin;
    }
    ivec2 cell = cell_at(p);
    if (cell.x < 0 || cell.y < 0 || cell.x >= cells_size.x || cell.y >= cells_size.y) {
        colour = vec4(0.0);
        return;
    }
    colour = texelFetch(tints, cell, 0);
}
"""

# The raised ground of each art pixel from the rasters of railroads.ui.pixel.relief.Elevation, as
# Elevation._rise works it out: how many metres it rises above its valley, how mountainous and
# how hilly it is, and the valley floor's height.
RELIEF = """out vec4 colour;
uniform ivec2 art_origin;
uniform mat2 to_plane;
uniform vec2 projection_origin;
uniform ivec2 raster_size;
uniform highp sampler2D heights;
uniform highp sampler2D beds;
@NOISE@
float ridges(vec2 p) {
    vec2 w = vec2(p.x + 0.9 * (@W_25_11@ - 0.5), p.y + 0.9 * (@W_25_12@ - 0.5));
    float big = crest(@R_26_1@);
    return 0.55 * big + 0.28 * big * crest(@R_1_2@) + 0.12 * crest(@R_42_3@)
        + 0.05 * crest(@R_16_6@);
}

void main() {
    ivec2 art = ivec2(floor(gl_FragCoord.xy)) + art_origin;
    vec2 p = to_plane * (vec2(art) + 0.5 + projection_origin);
    vec2 at = clamp(p * @STEPS@ - 0.5, vec2(0.0), vec2(raster_size - 1));
    ivec2 corner = min(ivec2(at), raster_size - 2);
    vec2 t = at - vec2(corner);
    vec4 a = texelFetch(heights, corner, 0) * (1.0 - t.x)
        + texelFetch(heights, corner + ivec2(1, 0), 0) * t.x;
    vec4 b = texelFetch(heights, corner + ivec2(0, 1), 0) * (1.0 - t.x)
        + texelFetch(heights, corner + ivec2(1, 1), 0) * t.x;
    vec4 here = a * (1.0 - t.y) + b * t.y;
    a = texelFetch(beds, corner, 0) * (1.0 - t.x) + texelFetch(beds, corner + ivec2(1, 0), 0) * t.x;
    b = texelFetch(beds, corner + ivec2(0, 1), 0) * (1.0 - t.x)
        + texelFetch(beds, corner + ivec2(1, 1), 0) * t.x;
    vec4 bed = a * (1.0 - t.y) + b * t.y;
    float m = max(clamp(here.z * @MOUNTAIN_GAIN@ - @MOUNTAIN_FROM@, 0.0, 1.0),
                  clamp(here.w * @PEAK_GAIN@ - @PEAK_FROM@, 0.0, 1.0));
    float h = bed.x;
    float rise = 0.0;
    if (m > 0.0 || h > 0.0) {
        rise = clamp(m + h, 0.0, 1.0) * max(here.x - here.y, 0.0) * 0.5;
        if (m > 0.0) {
            rise += pow(m, 1.6) * (@MOUNTAIN_FLOOR@ + @MOUNTAIN_RIDGES@ * ridges(p));
        }
        float rolls = h * (1.0 - m) * @HILL_ROLL@;
        if (rolls != 0.0) {
            rise += rolls * (0.6 * @N_9_4@ + 0.4 * @N_35_5@);
        }
        rise = rise * (1.0 - bed.y) + bed.z;
    }
    colour = vec4(rise, m, h, here.y);
}
"""

# The ground raised as railroads.ui.pixel.relief.raycast and raise_picture raise it: each art
# pixel shows the furthest ground below it whose lift reaches up to it, blended towards the row
# after by where the slope passes through it (crossings). Where that ground lies off the map, the
# ground under the pixel itself shows through, as it does under a chunk raised on the CPU.
RAISE = """out vec4 colour;
uniform ivec2 art_origin;
uniform ivec2 ground_origin;
uniform ivec2 relief_origin;
uniform highp sampler2D ground;
uniform float metres_per_pixel;
uniform highp sampler2D relief;

float lift_at(ivec2 art) {
    float rise = texelFetch(relief, art - relief_origin, 0).x;
    return clamp(rise / metres_per_pixel * @EXAGGERATION@ * @TILT@, 0.0, @MOST_LIFT@);
}

void main() {
    ivec2 art = ivec2(floor(gl_FragCoord.xy)) + art_origin;
    int offset = 0;
    for (int d = 1; d <= @MOST_LIFT_ROWS@; d++) {
        if (floor(lift_at(art + ivec2(0, d))) >= float(d)) offset = d;
    }
    float here = lift_at(art + ivec2(0, offset)) - float(offset);
    float drop = here - (lift_at(art + ivec2(0, offset + 1)) - float(offset) - 1.0);
    float share = drop > 0.0 ? clamp(here / drop, 0.0, 1.0) : 0.0;
    vec4 near = texelFetch(ground, art + ivec2(0, offset) - ground_origin, 0);
    vec4 far = texelFetch(ground, art + ivec2(0, offset + 1) - ground_origin, 0);
    vec4 shown = near + (far - near) * share;
    vec4 under = texelFetch(ground, art - ground_origin, 0);
    shown += under * (1.0 - shown.a);
    colour = shown.a > 0.0 ? vec4(clamp(shown.rgb / shown.a, 0.0, 1.0), shown.a) : vec4(0.0);
}
"""

# The light on raised ground (railroads.ui.pixel.relief.lighting): how much the slope at an art
# pixel faces the light and how steep it is, from the heights of the relief round it, and how much
# brighter or darker that makes it.
SUN = """float height_at(ivec2 art) {
    vec4 raised = texelFetch(relief, art - relief_origin, 0);
    return raised.w + raised.x;
}

vec2 sun(ivec2 art) {
    float per_col = (height_at(art + ivec2(1, 0)) - height_at(art - ivec2(1, 0))) * 0.5;
    float per_row = (height_at(art + ivec2(0, 1)) - height_at(art - ivec2(0, 1))) * 0.5;
    float gx = (per_col * slope.x + per_row * slope.y) / metres_per_unit;
    float gy = (per_col * slope.z + per_row * slope.w) / metres_per_unit;
    float nz = 1.0 / sqrt(1.0 + gx * gx + gy * gy);
    float s = clamp((-gx * @LIGHT_X@ - gy * @LIGHT_Y@ + @LIGHT_Z@) * nz, 0.0, 1.0);
    return vec2(s, 1.0 - nz);
}

float lit_by(float s, float m, float h) {
    float lit = clamp(1.0 + 1.3 * (s - @LIGHT_Z@), 0.55, 1.3);
    return 1.0 + (lit - 1.0) * clamp((m + h) * 4.0, 0.0, 1.0);
}
"""

# How the colours of plain land are multiplied for the light on its swells, the water and the
# mountains near it (Ground._natural); `around` is the raster of `_surroundings` there.
NATURAL = """float rolling(vec2 q) {
    return @R_16_301@ + 0.45 * @R_55_302@ + 0.12 * @R_18_303@;
}

vec3 natural(vec2 p, vec4 around) {
    float here = rolling(p);
    float e = @ROLL_STEP@;
    float gx = (rolling(vec2(p.x + e, p.y)) - here) / e;
    float gy = (rolling(vec2(p.x, p.y + e)) - here) / e;
    float facing = -(gx * @LIGHT_X@ + gy * @LIGHT_Y@);
    float deep = @ROLL_LEAST@ + (@ROLL_MOST@ - @ROLL_LEAST@) * @N_25_306@;
    float roll = clamp(1.0 + @ROLL_LIGHT@ * deep * facing, @ROLL_LOW@, @ROLL_HIGH@);
    float region = @N_3_305@ * 2.0 - 1.0;
    vec3 plains = region < 0.0 ? @GREENER_PLAINS@ : @DRIER_PLAINS@;
    float lush = clamp(around.g * @WET_GAIN@, 0.0, 1.0) * @LUSH@;
    float foot = clamp(around.b * @FOOT_GAIN@, 0.0, 1.0) * @FOOT_SHADE@;
    vec3 tint = 1.0 + (plains - 1.0) * (abs(region) * @PLAINS_TINT@);
    tint *= 1.0 + (@LUSH_TINT@ - 1.0) * lush;
    tint *= 1.0 + (@FOOT_TINT@ - 1.0) * foot;
    return tint * roll;
}
"""

FRAGMENT = """out vec4 colour;
uniform ivec2 art_origin;
uniform mat2 to_plane;
uniform vec2 projection_origin;
uniform ivec2 cells_size;
uniform ivec2 light_size;
uniform ivec2 relief_origin;
uniform highp usampler2D cells;
uniform vec4 slope;
uniform float metres_per_unit;
uniform highp sampler2D light;
uniform highp sampler2D relief;
@NOISE@
@HEX@
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

float hash(int a, int b, int seed) {
    return tbl(a * 13 + seed * 71, b * 7 + seed * 131);
}

@SUN@
// Raised ground lit and coloured as railroads.ui.pixel.relief.Elevation.colour does: by the
// light on its slope, and with foothills, bare rock and snow where it is mountainous.
vec3 raised_colour(vec3 rgb, vec4 raised, ivec2 art, vec2 p, bool rocky) {
    float above = raised.x, m = raised.y, h = raised.z;
    vec2 facing = sun(art);
    float s = facing.x, st = facing.y;
    float lit = lit_by(s, m, h);
    rgb *= lit;
    float under = max(m, rocky ? clamp(m * 3.0, 0.0, 1.0) : 0.0);
    if (under <= 0.0) return rgb;
    vec3 col = rgb * (1.0 - under) + @FOOTHILL@ * (0.9 + 0.2 * @F_3_13@) * under * lit;
    if ((above - @TREELINE@) / 200.0 + st * 1.4 - 0.3 > 0.0) {
        float treeline = @TREELINE@ + @TREELINE_SPREAD@ * @F_7_8@;
        float cover = clamp((above - treeline) / 200.0 + st * 1.4 - 0.3, 0.0, 1.0) * m;
        vec3 rock = @ROCK_DARK@ + (@ROCK_LIT@ - @ROCK_DARK@) * clamp(s * 1.3 - 0.2, 0.0, 1.0);
        rock *= 0.9 + 0.2 * @F_05_7@;
        if (st > 0.55) rock *= 0.85;
        col = col * (1.0 - cover) + rock * cover;
    }
    float lying = clamp(1.8 - st * 2.0, 0.0, 1.0);
    if ((above - (@SNOWLINE@ - 140.0 * (1.0 - s))) / 140.0 > 0.0 && lying > 0.0) {
        float snowline = @SNOWLINE@ + @SNOWLINE_SPREAD@ * @F_6_9@ - 140.0 * (1.0 - s);
        float snow = clamp((above - snowline) / 140.0, 0.0, 1.0) * lying * m;
        if (st > 0.3 && crest(@RIBS@) > 0.55) snow *= 0.25;
        vec3 white = @SNOW_DARK@ + (@SNOW_LIT@ - @SNOW_DARK@) * clamp(s * 1.5 - 0.35, 0.0, 1.0);
        col = col * (1.0 - snow) + white * snow;
    }
    float dark = clamp((0.45 - s) * 2.0, 0.0, 1.0) * (m + 0.3 * h);
    return col * (1.0 - dark * (1.0 - @SHADOW_TINT@) * 0.6);
}

vec4 surroundings(vec2 p) {
    vec2 at = clamp(p * @STEPS@ - 0.5, vec2(0.0), vec2(light_size - 1));
    ivec2 corner = min(ivec2(at), light_size - 2);
    vec2 t = at - vec2(corner);
    vec4 a = texelFetch(light, corner, 0);
    vec4 b = texelFetch(light, corner + ivec2(1, 0), 0);
    vec4 c = texelFetch(light, corner + ivec2(0, 1), 0);
    vec4 d = texelFetch(light, corner + ivec2(1, 1), 0);
    return mix(a, b, t.x) * (1.0 - t.y) + mix(c, d, t.x) * t.y;
}

@NATURAL@
vec2 wound(vec2 q) {
    float wx = q.x + @WARP@ * 2.0 * (@W_7_211@ - 0.5) + @WARP_FINE@ * 2.0 * (@W_2_213@ - 0.5);
    float wy = q.y + @WARP@ * 2.0 * (@W_7_212@ - 0.5) + @WARP_FINE@ * 2.0 * (@W_2_214@ - 0.5);
    wx += @WARP_BROAD@ * 2.0 * (@W_18_215@ - 0.5);
    wy += @WARP_BROAD@ * 2.0 * (@W_18_216@ - 0.5);
    return vec2(wx, wy);
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
    } else if (kind_of_grass == 1) {
        @SAVANNA_RAMPS@
    } else if (kind_of_grass == 2) {
        @TUNDRA_RAMPS@
    } else {
        @DESERT_RAMPS@
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

bool land_at(ivec2 art) {
    return !water_at(art);
}

float dune_height(vec2 q) {
    vec2 s = vec2(q.x + 0.3 * @D_8_311@, q.y);
    float n = 1.0 - abs(2.0 * @D_45_312@ - 1.0);
    return n * n + 0.3 * @D_15_313@;
}

float dunes(vec2 p) {
    float e = @DUNE_STEP@;
    float here = dune_height(p);
    float gx = (dune_height(vec2(p.x + e, p.y)) - here) / e;
    float gy = (dune_height(vec2(p.x, p.y + e)) - here) / e;
    return -(gx * @LIGHT_X@ + gy * @LIGHT_Y@) * @DUNE_LIGHT@;
}

float waves(vec2 q) {
    return @V_1_401@ + 0.5 * @V_045_402@;
}

vec3 water_colour(vec2 p, int ix, int iy, float share) {
    float depth = clamp((share - @DEPTH_FROM@) / @DEPTH_SPAN@, 0.0, 1.0);
    vec3 rgb = depth < 0.5
        ? @SHALLOW@ + (@OPEN_WATER@ - @SHALLOW@) * (depth * 2.0)
        : @OPEN_WATER@ + (@DEEP_WATER@ - @OPEN_WATER@) * (depth * 2.0 - 1.0);
    float e = @WAVE_STEP@;
    float here = waves(p);
    float gx = (waves(vec2(p.x + e, p.y)) - here) / e;
    float gy = (waves(vec2(p.x, p.y + e)) - here) / e;
    float facing = -(gx * @LIGHT_X@ + gy * @LIGHT_Y@) * @WAVE_LIGHT@;
    rgb *= clamp(1.0 + facing, @WAVE_LOW@, @WAVE_HIGH@);
    if (facing > @SPARKLE_FACING@ && hash(ix, iy, 403) > @SPARKLE_ABOVE@) {
        rgb = rgb * (1.0 - @SPARKLE_MIX@) + @SPARKLE@ * @SPARKLE_MIX@;
    }
    return rgb;
}

bool beside_land(ivec2 art) {
    return land_at(art + ivec2(1, 0)) || land_at(art - ivec2(1, 0))
        || land_at(art + ivec2(0, 1)) || land_at(art - ivec2(0, 1));
}

bool beside_water(ivec2 art) {
    return water_at(art + ivec2(1, 0)) || water_at(art - ivec2(1, 0))
        || water_at(art + ivec2(0, 1)) || water_at(art - ivec2(0, 1));
}

vec3 shore(vec3 rgb, ivec2 art, bool water) {
    if (water) {
        if (beside_land(art)) {
            return rgb * (1.0 - @SHORE_NEAR@) + @SHORE_WATER@ * @SHORE_NEAR@;
        }
        if (land_at(art + ivec2(2, 0)) || land_at(art - ivec2(2, 0))
                || land_at(art + ivec2(0, 2)) || land_at(art - ivec2(0, 2))) {
            rgb = rgb * (1.0 - @SHORE_FAR@) + @SHORE_WATER@ * @SHORE_FAR@;
        }
        if (land_at(art - ivec2(1, 1)) || land_at(art - ivec2(1, 2))
                || land_at(art - ivec2(2, 2))) {
            rgb *= @BANK_SHADE@;
        }
        return rgb;
    }
    if (beside_water(art)) {
        return rgb * (1.0 - @BEACH_NEAR@) + @SAND@ * @BEACH_NEAR@;
    }
    if (water_at(art + ivec2(2, 0)) || water_at(art - ivec2(2, 0))
            || water_at(art + ivec2(0, 2)) || water_at(art - ivec2(0, 2))
            || water_at(art + ivec2(1, 1)) || water_at(art - ivec2(1, 1))
            || water_at(art + ivec2(1, -1)) || water_at(art - ivec2(1, -1))) {
        return rgb * (1.0 - @BEACH_FAR@) + @SAND@ * @BEACH_FAR@;
    }
    return rgb;
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
    bool rocky = (info.b & 2u) != 0u;
    if (!water) {
        uvec4 warped = cell_info(cell_at(wound(p)));
        if (!is_water(warped.r)) {
            biome = warped.r;
            rocky = (warped.b & 2u) != 0u;
        }
    }
    vec4 around = surroundings(p);
    float lit = around.r;
    float value = 1.45 + 1.6 * (@N_24_41@ - 0.5) + 0.8 * (@N_75_42@ - 0.5);
    value += 0.45 * (@N_22_43@ - 0.5) + 0.45 * BAYER[fmod(iy, 4) * 4 + fmod(ix, 4)];
    value += lit + @HILL_LIFT@ * float(info.b & 1u);
    int shade = int(clamp(roundEven(value), 0.0, 3.0));
    vec3 rgb = RAMPS[int(biome) * 4 + shade];
    if (water) {
        rgb = water_colour(p, ix, iy, around.g);
    } else if (in_set(biome, @MEADOWS@u)) {
        rgb = grass(p, ix, iy, lit, 0);
    } else if (biome == @SAVANNA@u) {
        rgb = grass(p, ix, iy, lit, 1);
    } else if (biome == @TUNDRA@u) {
        rgb = grass(p, ix, iy, lit, 2);
    } else if (biome == @DESERT@u) {
        rgb = grass(p, ix, iy, lit + dunes(p), 3);
    } else if (in_set(biome, @GRASSY@u)) {
        bool tuft = hash(fdiv(ix, 6), fdiv(iy, 6), 51) > @TUFT_ABOVE@
            && ((TUFT_ROWS[fmod(iy, 6)] >> fmod(ix, 6)) & 1) != 0;
        if (tuft) {
            rgb = RAMPS[int(biome) * 4 + max(shade - 1, 0)] * 0.92;
        }
    }
    if (!water) {
        rgb *= natural(p, around);
    }
    rgb = shore(rgb, art, water);
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
    vec4 raised = texelFetch(relief, art - relief_origin, 0);
    if (raised.y > 0.0 || raised.z > 0.0) {
        rgb = raised_colour(rgb, raised, art, p, rocky);
    }
    colour = vec4(clamp(rgb, 0.0, 255.0) / 255.0, 1.0);
}
"""


# The texture of the ground up close, as railroads.ui.pixel.ground_texture lays it.
TEXTURE = """float hash(vec2 q) {
    return fract(sin(q.x * 127.1 + q.y * 311.7) * 43758.5453);
}
float layer(vec2 p, vec3 wave) {
    vec2 q = p * wave.xy + wave.z;
    vec2 period = 64.0 * wave.xy;
    vec2 i = floor(q);
    vec2 t = q - i;
    t = t * t * (3.0 - 2.0 * t);
    vec2 i0 = mod(i, period);
    vec2 i1 = mod(i + 1.0, period);
    float top = mix(hash(i0), hash(vec2(i1.x, i0.y)), t.x);
    float bottom = mix(hash(vec2(i0.x, i1.y)), hash(i1), t.x);
    return mix(top, bottom, t.y);
}
vec3 textured(vec3 rgb, vec2 p) {
    float green = rgb.g - max(rgb.r, rgb.b);
    float wet = rgb.b - max(rgb.r, rgb.g);
    vec3 tufts = vec3(1.5, 1.5, 0.0);
    float grain = layer(p, vec3(4.0, 4.0, 0.0));
    float bump = layer(p + vec2(0.3, 0.0), tufts) - layer(p - vec2(0.3, 0.0), tufts);
    float light = clamp(1.0 - bump * 0.5, 0.78, 1.22);
    float shade_by = (0.76 + 0.1 * grain + 0.14 * layer(p, tufts) + 0.1 * layer(p, vec3(0.625, 0.625, 0.0)))
        * (1.0 + 0.12 * smoothstep(0.75, 0.9, layer(p, vec3(3.5, 3.5, 41.0))));
    float factor = shade_by * light;
    if (green > 0.0) {
        factor *= 1.0 - 0.2 * smoothstep(0.62, 0.8, layer(p, vec3(5.0, 1.125, 17.0)));
    } else {
        factor = 1.0 + (factor - 1.0) * 0.7;
    }
    if (wet > 0.02) {
        factor = 1.0 + (factor - 1.0) * 0.25;
    }
    return clamp(rgb * factor, 0.0, 1.0);
}
"""

BLOW_UP = """out vec4 colour;
uniform vec2 origin;
uniform float scale;
uniform ivec2 art_origin;
uniform int blend;
uniform highp sampler2D art;
@TEXTURE@
void main() {
    vec2 p = (gl_FragCoord.xy - origin) / scale;
    if (blend == 0) {
        colour = texelFetch(art, ivec2(floor(p)) - art_origin, 0);
        return;
    }
    vec2 q = p - 0.5;
    vec2 f = fract(q);
    ivec2 last = textureSize(art, 0) - 1;
    ivec2 at = ivec2(floor(q)) - art_origin;
    vec4 a = texelFetch(art, clamp(at, ivec2(0), last), 0);
    vec4 b = texelFetch(art, clamp(at + ivec2(1, 0), ivec2(0), last), 0);
    vec4 c = texelFetch(art, clamp(at + ivec2(0, 1), ivec2(0), last), 0);
    vec4 d = texelFetch(art, clamp(at + ivec2(1, 1), ivec2(0), last), 0);
    colour = mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
    colour.rgb = textured(colour.rgb, p);
}
"""

# A raster over the map's plane, `steps` points to a cell unit, read at plane point `p` blended
# from its four nearest points, as railroads.ui.pixel.relief.sample reads it.
RASTER = """vec4 blended(highp sampler2D raster, vec2 p, float steps) {
    ivec2 size = textureSize(raster, 0);
    vec2 at = clamp(p * steps - 0.5, vec2(0.0), vec2(size - 1));
    ivec2 corner = min(ivec2(at), size - 2);
    vec2 t = at - vec2(corner);
    vec4 a = texelFetch(raster, corner, 0) * (1.0 - t.x)
        + texelFetch(raster, corner + ivec2(1, 0), 0) * t.x;
    vec4 b = texelFetch(raster, corner + ivec2(0, 1), 0) * (1.0 - t.x)
        + texelFetch(raster, corner + ivec2(1, 1), 0) * t.x;
    return a * (1.0 - t.y) + b * t.y;
}

float table_hash(int a, int b, int seed) {
    return tbl(a * 13 + seed * 71, b * 7 + seed * 131);
}
"""

# Which art pixels woods cover, as Ground._wooded and Ground.canopy work it out: land where the
# woods' share of the cells round it, frayed by noise, is over half, but for the clearings round
# lumber mills and bare mountainsides; then how deeply trees shade the grass there
# (Ground._tree_shade) and whether the art pixel lies on the map.
WOODS = """out vec4 colour;
uniform ivec2 art_origin;
uniform mat2 to_plane;
uniform vec2 projection_origin;
uniform ivec2 cells_size;
uniform highp usampler2D cells;
uniform highp sampler2D woods;
uniform highp sampler2D heights;
uniform highp sampler2D trees;
uniform int tree_count;
uniform highp sampler2D mills;
uniform int mill_count;
@NOISE@
@HEX@
@RASTER@
const vec2 AROUND[@AROUND_COUNT@] = vec2[@AROUND_COUNT@](@AROUND@);

ivec2 held(ivec2 cell) {
    return clamp(cell, ivec2(0), cells_size - 1);
}

void main() {
    ivec2 art = ivec2(floor(gl_FragCoord.xy)) + art_origin;
    vec2 p = to_plane * (vec2(art) + 0.5 + projection_origin);
    ivec2 cell = cell_at(p);
    bool inside = all(greaterThanEqual(cell, ivec2(0))) && all(lessThan(cell, cells_size));
    float share = 0.0;
    for (int i = 0; i < @AROUND_COUNT@; i++) {
        share += texelFetch(woods, held(cell_at(p + AROUND[i])), 0).r;
    }
    share /= @AROUND_FLOAT@;
    bool land = texelFetch(woods, held(cell), 0).g > 0.5;
    bool wooded = land && share >= 1.0;
    if (land && share > 0.0 && share < 1.0) {
        float edge = @N_55_61@ * 0.55 + @N_16_62@ * 0.3;
        wooded = share + (edge - 0.42) * 0.9 > 0.5;
    }
    for (int i = 0; i < mill_count && wooded; i++) {
        vec4 mill = texelFetch(mills, ivec2(i, 0), 0);
        vec2 off = (vec2(art) + 0.5 - mill.xy) / vec2(mill.z, mill.z * 0.5);
        if (length(off) + table_hash(art.x, art.y, int(mill.w)) * 0.35 < 1.0) wooded = false;
    }
    if (wooded) {
        vec4 here = blended(heights, p, @STEPS@);
        float m = max(clamp(here.z * @MOUNTAIN_GAIN@ - @MOUNTAIN_FROM@, 0.0, 1.0),
                      clamp(here.w * @PEAK_GAIN@ - @PEAK_FROM@, 0.0, 1.0));
        wooded = m < @BARE_ABOVE@;
    }
    float shade = 0.0;
    uint biome = texelFetch(cells, held(cell), 0).r;
    if (inside && ((@GRASS@u >> biome) & 1u) != 0u) {
        for (int i = 0; i < tree_count; i++) {
            vec2 d = vec2(art) - floor(texelFetch(trees, ivec2(i, 0), 0).xy);
            if (d.x >= -6.0 && d.x <= 10.0 && d.y >= -4.0 && d.y <= 4.0) {
                vec2 e = (d - vec2(2.0, 1.0)) / vec2(@SHADE_X@, @SHADE_Y@);
                shade = max(shade, clamp(1.2 - e.x * e.x - e.y * e.y, 0.0, 1.0));
            }
        }
    }
    colour = vec4(wooded ? 1.0 : 0.0, @TREE_SHADE@ * shade, inside ? 1.0 : 0.0, 0.0);
}
"""

# Round the woods: which art pixels their crowns may reach over (`near`: up to OVERHANG art
# pixels round them and OVERHANG + 3 below, finer than an art pixel to a pixel, as _spread finds
# them; just the woods otherwise), and how much of the three by three round each the woods cover.
EDGES = """out vec4 colour;
uniform int fine;
uniform highp sampler2D wooded;

float woods(ivec2 at) {
    ivec2 size = textureSize(wooded, 0);
    if (any(lessThan(at, ivec2(0))) || any(greaterThanEqual(at, size))) return 0.0;
    return texelFetch(wooded, at, 0).r;
}

void main() {
    ivec2 at = ivec2(floor(gl_FragCoord.xy));
    float near = woods(at);
    if (fine == 1) {
        for (int ox = -@OVERHANG@; ox <= @OVERHANG@ && near < 0.5; ox++) {
            int side = abs(ox);
            for (int oy = side - @BELOW@; oy <= @OVERHANG@ - side; oy++) {
                if (woods(at + ivec2(ox, oy)) > 0.5) {
                    near = 1.0;
                    break;
                }
            }
        }
    }
    float around = 0.0;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) around += woods(at + ivec2(dx, dy));
    }
    colour = vec4(near, around / 9.0, 0.0, 0.0);
}
"""

# Fields, groves and the ground round buildings, as Ground.fields and Ground.yards paint them,
# `k` pixels to an art pixel: the plots near the chunk in `plots`, a column each holding its cell
# (row, column), the cell's middle on the plane, which way it faces, how wide it spreads and its
# kind (its place in FIELD_KINDS); the buildings in `yards`, where each stands in the picture,
# how far its yard reaches and which kind of yard it is.
FIELDS = """const vec3 CROP_PLANT[@CROP_COUNT@] = vec3[@CROP_COUNT@](@CROP_PLANT@);
const vec3 CROP_SOIL[@CROP_COUNT@] = vec3[@CROP_COUNT@](@CROP_SOIL@);
const vec3 GROVE_GROUND[@GROVE_COUNT@] = vec3[@GROVE_COUNT@](@GROVE_GROUND@);
const vec3 GROVE_TREE[@GROVE_COUNT@] = vec3[@GROVE_COUNT@](@GROVE_TREE@);
const vec3 GROVE_FRUIT[@FRUIT_COUNT@] = vec3[@FRUIT_COUNT@](@GROVE_FRUIT@);
const int GROVE_FRUIT_FROM[@GROVE_COUNT@] = int[@GROVE_COUNT@](@GROVE_FRUIT_FROM@);
const int GROVE_FRUITS[@GROVE_COUNT@] = int[@GROVE_COUNT@](@GROVE_FRUITS@);
const float GROVE_ROWS[@GROVE_COUNT@] = float[@GROVE_COUNT@](@GROVE_ROWS@);
const float GROVE_CROWN[@GROVE_COUNT@] = float[@GROVE_COUNT@](@GROVE_CROWN@);
const vec3 YARD_COLOURS[@YARD_COUNT@] = vec3[@YARD_COUNT@](@YARD_COLOURS@);

bool in_plot(vec2 p, vec4 at, vec4 shape) {
    if (clamp(cell_at(p), ivec2(0), cells_size - 1) != ivec2(int(at.y), int(at.x))) return false;
    vec2 off = p - at.zw;
    return shape.x * off.x + shape.y * off.y >= shape.z * length(off);
}

// How fully a plot covers the ground at `texel`, from 0 at its frayed edge to 1 FIELD_FEATHER
// art pixels in (Ground._feathered): how many rings of pixels round it lie in the plot too.
float feathered(ivec2 texel, vec4 at, vec4 shape, vec2 p) {
    int most = @FIELD_FEATHER@ * k;
    int depth = 0;
    for (int ring = 1; ring <= most; ring++) {
        bool whole = true;
        for (int dy = -ring; dy <= ring && whole; dy++) {
            for (int dx = -ring; dx <= ring; dx++) {
                if (max(abs(dx), abs(dy)) != ring) continue;
                vec2 there = vec2(chunk_origin) + (vec2(texel + ivec2(dx, dy)) + 0.5) / float(k);
                if (!in_plot(to_plane * (there + projection_origin), at, shape)) {
                    whole = false;
                    break;
                }
            }
        }
        if (!whole) break;
        depth = ring;
    }
    float fray = (@N_FRAY@ - 0.5) * @FIELD_FRAY@ * float(k);
    return clamp((float(depth) + 0.5 + fray) / float(most), 0.0, 1.0);
}

// A field of crops in rows (Ground._sow): wheat on a farm (`crop` below 0), or another crop.
vec3 sow(vec2 p, ivec2 pixel, vec4 at, int crop) {
    int pick = int(at.x) * 7 + int(at.y) * 13;
    vec2 off = p - at.zw;
    vec2 rows = pick % 2 != 0 ? off : off.yx;
    float u = rows.x * @FIELD_STRIPES@, w = rows.y * @FIELD_STRIPES@;
    float grain = table_hash(pixel.x, pixel.y, 71) - 0.5;
    float line = floor(u);
    float fu = u - line;
    vec3 sown;
    if (crop < 0) {
        sown = mod(line, 2.0) == 0.0 ? @WHEAT@ : @WHEAT_DARK@;
        if (fu < 0.2) sown = @WHEAT_LIGHT@;
        if (fu > 0.88) sown = @WHEAT_DARK@ * 0.86;
        sown *= 1.0 + 0.14 * grain;
        if (table_hash(pixel.x, pixel.y, 72) > 0.72 && fu > 0.3 && fu < 0.85) sown = @WHEAT_EARS@;
    } else {
        vec3 plant = CROP_PLANT[crop];
        vec3 light = min(plant * 1.18, vec3(255.0));
        sown = CROP_SOIL[crop] * (1.0 + 0.16 * grain);
        if (fu < 0.12) sown = @SOIL_DARK@;
        float lump = w * 2.2;
        bool bushes = table_hash(int(floor(lump)), int(line), 81) > 0.18;
        if (bushes && length(vec2(fu - 0.5, (lump - floor(lump) - 0.5) * 0.7)) < 0.34) {
            sown = plant * (1.0 + 0.1 * grain);
            if (fu < 0.42) sown = light;
            if (fu > 0.66) sown = plant * 0.7;
            if (table_hash(pixel.x, pixel.y, 82) > 0.86) sown = light * 1.08;
        }
    }
    float grey = (sown.r + sown.g + sown.b) / 3.0;
    return grey + (sown - grey) * @FIELD_SATURATION@;
}

// A grove of trees in rows (Ground._plant), with how fully it covers the ground.
vec4 plant(vec2 p, ivec2 pixel, vec4 at, int grove, float fade) {
    vec2 uv = (p - at.zw) * GROVE_ROWS[grove] + 0.5;
    vec2 d = uv - roundEven(uv);
    float crown = GROVE_CROWN[grove];
    float grain = 1.0 + 0.12 * (table_hash(pixel.x, pixel.y, 23) - 0.5);
    vec3 colour = GROVE_GROUND[grove] * grain;
    float cover = fade * @GROVE_FLOOR@;
    if (length(d - vec2(0.12, 0.1)) < crown) {
        colour = @SHADOW_COLOUR@;
        cover = fade * @GROVE_SHADOW@;
    }
    if (length(d) < crown) {
        float lit = clamp(0.82 + 0.3 * (-(d.x + d.y) / max(crown, 1e-6)), 0.62, 1.22);
        colour = clamp(GROVE_TREE[grove] * (lit * grain), 0.0, 255.0);
        cover = min(fade * 3.0, 1.0);
        if (table_hash(pixel.x, pixel.y, 21) > 0.82) {
            int pick = int(table_hash(pixel.x, pixel.y, 22) * float(GROVE_FRUITS[grove]));
            colour = GROVE_FRUIT[GROVE_FRUIT_FROM[grove] + pick];
        }
    }
    return vec4(colour, cover);
}

void lay(inout vec4 layer, vec3 colour, float cover) {
    layer.rgb = layer.rgb * (1.0 - cover) + colour * cover;
    layer.a = layer.a * (1.0 - cover) + cover;
}

// Lays the field or grove at picture point `q`, if any, lit by the slope it lies on (`lit`).
void lay_fields(vec2 q, ivec2 texel, float lit, inout vec4 layer) {
    vec2 p = to_plane * (q + projection_origin);
    for (int i = 0; i < plot_count; i++) {
        vec4 at = texelFetch(plots, ivec2(i, 0), 0);
        vec4 shape = texelFetch(plots, ivec2(i, 1), 0);
        if (!in_plot(p, at, shape)) continue;
        ivec2 pixel = chunk_origin * k + texel;
        float fade = feathered(texel, at, shape, p);
        int kind = int(shape.w);
        vec4 sown = kind < @GROVES_FROM@ ? vec4(sow(p, pixel, at, kind - 1), fade)
            : plant(p, pixel, at, kind - @GROVES_FROM@, fade);
        vec3 colour = sown.rgb * natural(p, blended(light, p, @RELIEF_STEPS@)) * lit;
        if (sown.a > 0.0) lay(layer, clamp(colour, 0.0, 255.0) / 255.0, sown.a);
        return;
    }
}

// Lays the ground round each building near the chunk over `texel` (Ground._yard), worked out
// from the chunk's corner as there.
void lay_yards(ivec2 texel, inout vec4 layer) {
    vec2 f = vec2(texel) + 0.5;
    for (int i = 0; i < yard_count; i++) {
        vec4 yard = texelFetch(yards, ivec2(i, 0), 0);
        vec2 c = (yard.xy - vec2(chunk_origin)) * float(k);
        float rx = yard.z * float(k), ry = rx / 2.0;
        ivec2 low = ivec2(int(c.x - rx) - 1, int(c.y - ry) - 1);
        ivec2 high = ivec2(int(c.x + rx) + 2, int(c.y + ry) + 2);
        if (any(lessThan(texel, low)) || any(greaterThanEqual(texel, high))) continue;
        int seed = int(c.x * 7.0 + c.y * 13.0) & 1023;
        float d = length((f - c) / vec2(rx, ry));
        float fray = table_hash(texel.x / k, texel.y / k, seed) * 0.35;
        float cover = clamp((1.0 - d - fray * 0.6) * 4.0, 0.0, 1.0);
        int kind = int(yard.w);
        vec3 colour = YARD_COLOURS[kind];
        if (kind == @COTTAGE@) {
            colour *= 0.9 + 0.2 * table_hash(texel.x, texel.y, 31);
            vec2 uv = (f - c) / float(k);
            if (uv.x > 1.5 && uv.x < 6.5 && uv.y > -3.4 && uv.y < -0.6) {
                colour = @SOIL@ * 0.95;
                bool row = mod(floor((uv.y + 3.4) * 1.4), 2.0) == 0.0;
                if (row && table_hash(texel.x, texel.y, 32) > 0.25) {
                    colour = @GARDEN_LEAVES@ * (0.9 + 0.3 * table_hash(texel.x, texel.y, 33));
                }
                cover = 1.0;
            }
        } else if (kind == @PAVED@ || kind == @FLAGSTONES@) {
            int stone = (kind == @PAVED@ ? 2 : 3) * k;
            colour *= 0.92 + 0.12 * table_hash(texel.x / stone, texel.y / k, 34);
            if ((texel.x / stone + texel.y / k) % 3 == 0 && table_hash(texel.x, texel.y, 35) > 0.5) {
                colour *= 0.88;
            }
        } else if (kind == @SPOIL@) {
            colour *= 0.8 + 0.35 * table_hash(texel.x, texel.y, 38);
            if (table_hash(texel.x / k, texel.y / k, 39) > 0.9) colour = vec3(150.0, 144.0, 134.0);
        } else {
            colour *= 0.86 + 0.24 * table_hash(texel.x, texel.y, 36);
        }
        lay(layer, colour / 255.0, cover);
    }
}
"""

# The woods over a chunk, unraised, as Ground.canopy paints them over the shade of trees, `k`
# pixels to an art pixel (one or FINE): a roof of lit crowns, the dark side of the trees along its
# lower edge and their shadow on the ground beside them; finer than an art pixel to a pixel,
# crowns show their clumps of leaves and reach past the woods' edge, and the front row its
# trunks. RGBA multiplied by alpha.
CANOPY = """out vec4 colour;
uniform ivec2 chunk_origin;
uniform int k;
uniform mat2 to_plane;
uniform vec2 projection_origin;
uniform ivec2 cells_size;
uniform ivec2 woods_origin;
uniform highp usampler2D cells;
uniform highp sampler2D light;
uniform highp sampler2D wooded;
uniform highp sampler2D edges;
uniform highp sampler2D plots;
uniform int plot_count;
uniform highp sampler2D yards;
uniform int yard_count;
uniform ivec2 relief_origin;
uniform vec4 slope;
uniform float metres_per_unit;
uniform highp sampler2D relief;
@NOISE@
@HEX@
@RASTER@
@SUN@
@NATURAL@
@FIELDS@
const float TAU = 6.283185307179586;
const vec3 CROWN_LIGHT = @CROWN_LIGHT@;

struct Dome {
    float height;
    vec2 d;
    float radius;
    ivec2 square;
};

vec4 woods_at(ivec2 art) {
    ivec2 at = clamp(art - woods_origin, ivec2(0), textureSize(wooded, 0) - 1);
    return texelFetch(wooded, at, 0);
}

bool rooted(ivec2 art) {
    return woods_at(art).r > 0.5;
}

vec4 edges_at(ivec2 art) {
    ivec2 at = clamp(art - woods_origin, ivec2(0), textureSize(edges, 0) - 1);
    return texelFetch(edges, at, 0);
}

vec2 crown_spot(ivec2 g, float gap, int seed) {
    return vec2((float(g.x) + 0.15 + 0.7 * table_hash(g.x, g.y, seed)) * gap,
                (float(g.y) + 0.15 + 0.7 * table_hash(g.x, g.y, seed + 1)) * gap);
}

float crown_radius(ivec2 g, vec2 j) {
    float size = @CROWN_SIZE_0@ + @CROWN_SIZE_1@ * table_hash(g.x, g.y, 53);
    float age = @YOUNG@ + @AGE_SPAN@ * @N_STAND@;
    float thin = clamp((@N_THIN@ - 0.62) * 2.2, 0.0, @THINNED@);
    return table_hash(g.x, g.y, 54) < thin ? 0.01 : size * age;
}

float clump_radius(ivec2 g) {
    return @CLUMP_GAP@ * (@CLUMP_SIZE_0@ + @CLUMP_SIZE_1@ * table_hash(g.x, g.y, 155));
}

// The dome standing highest over picture point `q`, domes standing one to a square `gap` art
// pixels a side, as _tallest_dome finds it: crowns, or with `clumps` clumps of leaves.
Dome tallest(vec2 q, float gap, int seed, bool clumps) {
    ivec2 home = ivec2(floor(q / gap));
    Dome best = Dome(-1e30, vec2(0.0), 1.0, home);
    for (int ox = -1; ox <= 1; ox++) {
        for (int oy = -1; oy <= 1; oy++) {
            ivec2 g = home + ivec2(ox, oy);
            vec2 j = crown_spot(g, gap, seed);
            float r = clumps ? clump_radius(g) : crown_radius(g, j);
            vec2 d = q - j;
            float h = 1.0 - (d.x * d.x + d.y * d.y) / (r * r);
            if (h > best.height) best = Dome(h, d, r, g);
        }
    }
    return best;
}

// Whether `q` shows the trunk of a tree in the front row of the woods, below its crown, and its
// bark there (_trunks).
bool trunk(vec2 q, out vec3 bark) {
    ivec2 home = ivec2(floor(q / @CROWN_GAP@));
    bool on = false;
    bark = vec3(0.0);
    for (int ox = -1; ox <= 1; ox++) {
        for (int oy = -2; oy <= 0; oy++) {
            ivec2 g = home + ivec2(ox, oy);
            vec2 j = crown_spot(g, @CROWN_GAP@, 51);
            float r = crown_radius(g, j);
            ivec2 foot = ivec2(floor(j));
            int behind = foot.y + int(ceil(r * 0.7));
            if (!rooted(foot) || rooted(ivec2(foot.x, behind)) || r <= 0.5) continue;
            float grown = @TRUNK_LENGTH_0@ + @TRUNK_LENGTH_1@ * table_hash(g.x, g.y, 59);
            float off = q.x - j.x, reach = q.y - j.y;
            if (abs(off) < @TRUNK_HALF@ && reach > r * 0.55 && reach < r + grown) {
                float out_ = clamp((reach - r) / grown, 0.0, 1.0);
                bark = @TRUNK_BARK@ * ((off < 0.0 ? 1.15 : 0.8) * (0.55 + 0.45 * out_));
                on = true;
            }
        }
    }
    return on;
}

float slope_light(vec2 p) {
    return blended(light, p, @RELIEF_STEPS@).r;
}

float swell(vec2 p) {
    float here = @N_SWELL@;
    float gx = (@N_SWELL_X@ - here) / 0.05;
    float gy = (@N_SWELL_Y@ - here) / 0.05;
    return -(gx * CROWN_LIGHT.x + gy * CROWN_LIGHT.y) * @SWELL@;
}

// Paints the crown standing highest over picture point `q` in art pixel `art`, or the floor
// between crowns inside the woods, or below the woods a trunk; returns whether it painted.
bool crowns(vec2 q, ivec2 art, inout vec4 layer) {
    bool fine = k > 1;
    Dome crown = tallest(q, @CROWN_GAP@, 51, false);
    float height = crown.height, radius = crown.radius;
    vec2 d = crown.d;
    ivec2 g = crown.square;
    ivec2 foot = ivec2(floor(q - d));
    float lowest = fine ? -(@LUMPS@ / 2.0 + @CLUMP_EDGE@) : -@LUMPS@ / 2.0;
    bool inside;
    if (fine) {
        vec2 at = q - 0.5 - vec2(woods_origin);
        ivec2 c0 = ivec2(floor(at));
        vec2 t = at - vec2(c0);
        float a = edges_at(c0 + woods_origin).g * (1.0 - t.x)
            + edges_at(c0 + ivec2(1, 0) + woods_origin).g * t.x;
        float b = edges_at(c0 + ivec2(0, 1) + woods_origin).g * (1.0 - t.x)
            + edges_at(c0 + ivec2(1, 1) + woods_origin).g * t.x;
        inside = a * (1.0 - t.y) + b * t.y > @FLOOR_FROM@;
    } else {
        inside = rooted(art + ivec2(0, -1)) && rooted(art + ivec2(0, 1))
            && rooted(art + ivec2(-1, 0)) && rooted(art + ivec2(1, 0));
    }
    if (!inside && !(height > lowest && rooted(foot))) {
        if (!fine || rooted(art)) return false;
        bool below = false;
        for (int up = 1; up <= @TRUNK_BELOW@; up++) below = below || rooted(art - ivec2(0, up));
        vec3 bark;
        if (below && trunk(q, bark)) {
            layer = vec4(bark / 255.0, 1.0);
            return true;
        }
        return false;
    }
    vec2 p = to_plane * (q + projection_origin);
    vec2 centre = to_plane * (vec2(art) + 0.5 + projection_origin);
    uint biome = texelFetch(cells, clamp(cell_at(centre), ivec2(0), cells_size - 1), 0).r;
    height += @LUMPS@ * (@N_GRAIN@ - 0.5);
    float conifers = biome == @TAIGA@u ? 0.9 : 0.0;
    if (biome != @TAIGA@u && biome != @JUNGLE@u) {
        conifers = 0.05 + 0.85 * clamp((@N_STANDS@ - 0.5) * 4.0 + 0.5, 0.0, 1.0);
    }
    bool conifer = table_hash(g.x, g.y, 63) < conifers;
    float tone = table_hash(g.x, g.y, 66) * @CROWN_TONES@;
    vec3 first = conifer ? @CONIFER_0@ : @LEAFY_0@;
    vec3 green = first + ((conifer ? @CONIFER_1@ : @LEAFY_1@) - first) * tone;
    green *= 0.9 + 0.2 * @N_CROWNS@;
    float region = @N_REGION@ * 2.0 - 1.0;
    green *= 1.0 + ((region < 0.0 ? @DARK_GREENS@ : @GOLDEN_GREENS@) - 1.0) * abs(region);
    float towards = (d.x * CROWN_LIGHT.x + d.y * CROWN_LIGHT.y) / radius * @CROWN_ROUND@;
    float up = sqrt(clamp(1.0 - (1.0 - height) * @CROWN_ROUND@ * @CROWN_ROUND@, 0.0, 1.0));
    float facing = towards + up * CROWN_LIGHT.z;
    bool solid = height > 0.0;
    bool hollow = false, branch = false;
    float depth = 1.0;
    if (fine) {
        float grain = table_hash(int(floor(q.x * 4.0)), int(floor(q.y * 4.0)), 59);
        bool leafy = !conifer && height > -@CLUMP_EDGE@ * 0.7;
        float h = max(height, 0.01);
        solid = false;
        facing = 0.0;
        if (leafy) {
            Dome clump = tallest(q, @CLUMP_GAP@, 151, true);
            float bump = sqrt(clamp(clump.height, 0.0, 1.0));
            float crown_up = sqrt(clamp(1.0 - (1.0 - h) * @CROWN_ROUND@ * @CROWN_ROUND@, 0.0, 1.0));
            float crown_facing = (d.x * CROWN_LIGHT.x + d.y * CROWN_LIGHT.y) / radius * @CROWN_ROUND@
                + crown_up * CROWN_LIGHT.z;
            float clump_facing = (clump.d.x * CROWN_LIGHT.x + clump.d.y * CROWN_LIGHT.y) / clump.radius
                + bump * CROWN_LIGHT.z;
            facing = (1.0 - @CLUMP_LIGHT@) * crown_facing + @CLUMP_LIGHT@ * clump_facing
                + 0.05 * (grain - 0.5);
            solid = h + @CLUMP_EDGE@ * (clamp(clump.height, -0.6, 1.0) - 0.3) > 0.0;
            depth = 0.82 + 0.18 * bump;
            hollow = solid && clump.height < (@HOLLOWS@ - h) * 0.9;
            if (hollow) {
                float twist = table_hash(g.x, g.y, 57);
                float turn = atan(d.y, d.x) / TAU;
                float ray = (roundEven(turn * @BRANCHES@ + twist) - twist) * (TAU / @BRANCHES@);
                branch = abs(d.x * sin(ray) - d.y * cos(ray)) < @BRANCH_WIDTH@;
            }
        }
        if (conifer) {
            float far = length(d) + 1e-6;
            float points = atan(d.y, d.x) / TAU * @CONIFER_POINTS@ + table_hash(g.x, g.y, 58);
            float notch = abs(points - roundEven(points)) * 2.0;
            float cone = far / (radius * @CONIFER_SLIM@ * (1.0 - 0.25 * notch));
            float tier = fract(cone * @CONIFER_TIERS@);
            facing = (d.x * CROWN_LIGHT.x + d.y * CROWN_LIGHT.y) / far * 0.75
                + 0.66 * CROWN_LIGHT.z + 0.3 * (0.4 - tier) + 0.16 * (grain - 0.5);
            solid = cone < 1.0;
            depth = 0.92 - 0.12 * tier;
            h = 1.0 - cone;
        }
        height = h;
    }
    facing += swell(p) + @CROWN_SLOPES@ * slope_light(p);
    float s = clamp(facing, 0.0, 1.0);
    float lit = clamp(1.0 + 1.3 * (s - CROWN_LIGHT.z), 0.5, 1.35);
    float dark = clamp((0.45 - s) * 2.0, 0.0, 1.0);
    float deep = @CROWN_DEPTH@ + (1.0 - @CROWN_DEPTH@) * sqrt(clamp(height * 2.0, 0.0, 1.0));
    float rim = height < @CROWN_RIM_0@ ? @CROWN_RIM_1@ : 1.0;
    float shine = 1.0 + clamp(s - @CROWN_SHINE_0@, 0.0, 1.0) * @CROWN_SHINE_1@ * 5.0;
    float edge;
    if (fine) {
        rim = (height < @CROWN_RIM_0@ ? (1.0 + @CROWN_RIM_1@) / 2.0 : 1.0) * depth;
        bool front = !rooted(ivec2(foot.x, foot.y + int(ceil(radius * 0.7))));
        edge = front ? clamp(d.y / radius * 1.5 + 0.3, 0.0, 1.0) : 0.0;
    } else {
        bool side = rooted(art) && !(rooted(art + ivec2(0, 1)) && rooted(art + ivec2(0, 2)));
        edge = side ? 1.0 : 0.0;
    }
    float cool = max(dark, edge);
    float lighting = lit * (1.0 - dark * 0.35) * deep * rim * shine
        * (1.0 - (1.0 - @WOODS_SIDE@) * edge);
    vec3 leaves = green * max(lighting, @CROWN_DARKEST@);
    leaves *= 1.0 - cool * 0.6 * (1.0 - @SHADOW_TINT@);
    if (hollow) leaves *= @HOLLOW_DARK@;
    if (branch) {
        leaves = @BRANCH@ * (0.8 + 0.3 * s) * (1.0 - 0.5 * (1.0 - @SHADOW_TINT@));
    }
    if (!solid) leaves = @WOODS_FLOOR@ * (0.8 + 0.4 * @N_UNDERGROWTH@);
    if (!(inside || solid)) return false;
    layer = vec4(leaves / 255.0, 1.0);
    return true;
}

// Lays the woods' shadow, SHADOW_REACH art pixels down and to the right of the crowns, as
// Ground._cast_shadow does.
void cast_shadow(vec2 q, inout vec4 layer) {
    vec2 from = q - @SHADOW_REACH@;
    Dome crown = tallest(from, @CROWN_GAP@, 51, false);
    float strength;
    if (k > 1) {
        bool standing = rooted(ivec2(floor(from - crown.d)));
        strength = standing ? clamp(crown.height * @SHADOW_SOFT@, 0.0, 1.0) : 0.0;
    } else {
        strength = crown.height > 0.0 ? 1.0 : 0.0;
    }
    if (strength <= 0.0) return;
    strength *= @EDGE_SHADOW@;
    layer.rgb = layer.rgb * (1.0 - strength) + @SHADOW_COLOUR@ / 255.0 * strength;
    layer.a = layer.a * (1.0 - strength) + strength;
}

void main() {
    ivec2 texel = ivec2(floor(gl_FragCoord.xy));
    ivec2 art = chunk_origin + texel / k;
    vec2 q = vec2(chunk_origin) + (vec2(texel) + 0.5) / float(k);
    vec4 here = woods_at(art);
    if (here.b < 0.5) {
        colour = vec4(0.0);
        return;
    }
    vec4 layer = vec4(0.0, 0.0, 0.0, here.g);
    vec4 raised = texelFetch(relief, art - relief_origin, 0);
    lay_fields(q, texel, lit_by(sun(art).x, raised.y, raised.z), layer);
    lay_yards(texel, layer);
    bool painted = false;
    if (edges_at(art).r > 0.5) painted = crowns(q, art, layer);
    if (!painted && edges_at(art - @SHADOW_STEP@).r > 0.5) cast_shadow(q, layer);
    colour = layer;
}
"""

# A chunk's texture: the woods (CANOPY) blown up to the chunk's `k` pixels to an art pixel,
# blended between their pixels where finer, raised as RAISE raises the ground and laid with the
# ground's texture, with the chunk the CPU drew over them, every pixel blown up `times` each way.
DRESS = """out vec4 colour;
uniform ivec2 chunk_origin;
uniform int k;
uniform int drawn;
uniform int times;
uniform ivec2 relief_origin;
uniform float metres_per_pixel;
uniform highp sampler2D pixels;
uniform highp sampler2D surface;
uniform highp sampler2D relief;
@TEXTURE@
float lift_at(ivec2 art) {
    float rise = texelFetch(relief, art - relief_origin, 0).x;
    return clamp(rise / metres_per_pixel * @EXAGGERATION@ * @TILT@, 0.0, @MOST_LIFT@);
}

vec4 surface_at(vec2 at) {
    ivec2 last = textureSize(surface, 0) - 1;
    if (drawn == k) return texelFetch(surface, min(ivec2(floor(at * float(k))), last), 0);
    vec2 q = clamp(at * float(drawn) - 0.5, vec2(0.0), vec2(last));
    ivec2 c0 = ivec2(floor(q));
    ivec2 c1 = min(c0 + 1, last);
    vec2 t = q - vec2(c0);
    vec4 a = mix(texelFetch(surface, c0, 0), texelFetch(surface, ivec2(c1.x, c0.y), 0), t.x);
    vec4 b = mix(texelFetch(surface, ivec2(c0.x, c1.y), 0), texelFetch(surface, c1, 0), t.x);
    return mix(a, b, t.y);
}

void main() {
    ivec2 pixel = ivec2(floor(gl_FragCoord.xy)) / times;
    ivec2 art = chunk_origin + pixel / k;
    int offset = 0;
    for (int d = 1; d <= @MOST_LIFT_ROWS@; d++) {
        if (floor(lift_at(art + ivec2(0, d))) >= float(d)) offset = d;
    }
    float here = lift_at(art + ivec2(0, offset)) - float(offset);
    float drop = here - (lift_at(art + ivec2(0, offset + 1)) - float(offset) - 1.0);
    float share = drop > 0.0 ? clamp(here / drop, 0.0, 1.0) : 0.0;
    vec2 at = (vec2(pixel) + 0.5) / float(k);
    vec4 shown = surface_at(at + vec2(0.0, float(offset)));
    if (share > 0.0) {
        shown += (surface_at(at + vec2(0.0, float(offset + 1))) - shown) * share;
    }
    if (k > 1 && shown.a > 0.0) {
        shown.rgb = textured(shown.rgb / shown.a, at + vec2(chunk_origin)) * shown.a;
    }
    vec4 over = texelFetch(pixels, pixel, 0);
    float alpha = over.a + shown.a * (1.0 - over.a);
    vec3 rgb = over.rgb * over.a + shown.rgb * (1.0 - over.a);
    colour = alpha > 0.0 ? vec4(clamp(rgb / alpha, 0.0, 1.0), alpha) : vec4(0.0);
}
"""


def _number(value: float) -> str:
    """A GLSL float literal for exactly the float32 the CPU works with."""
    text = repr(float(np.float32(value)))
    return text if "." in text or "e" in text else f"{text}.0"


def _vec3(rgb: object) -> str:
    r, g, b = (float(v) for v in np.asarray(rgb, dtype=np.float32))
    return f"vec3({_number(r)}, {_number(g)}, {_number(b)})"


def _noise(scale: float, seed: int, at: str = "p") -> str:
    """The call for railroads.ui.pixel.noise.noise at point `at`, `scale` and `seed`, with its
    numbers worked out in float32 as there."""
    inverse = np.float32(1 / scale)
    off_u, off_v = np.float32(seed * 37.1 + 500.0), np.float32(seed * 11.3 + 500.0)
    return f"noise({at}, {_number(inverse)}, {_number(off_u)}, {_number(off_v)})"


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


def _shared_names() -> dict[str, str]:
    """The numbers of the snippets the ground and what lies on it share (SUN, NATURAL)."""
    return {
        "R_16_301": _noise(1.6, 301, "q"),
        "R_55_302": _noise(0.55, 302, "q"),
        "R_18_303": _noise(0.18, 303, "q"),
        "ROLL_STEP": _number(cpu.ROLL_STEP),
        "N_25_306": _noise(cpu.ROLL_SPAN, 306),
        "N_3_305": _noise(cpu.PLAINS_SPAN, 305),
        "ROLL_LEAST": _number(cpu.ROLL_LEAST),
        "ROLL_MOST": _number(cpu.ROLL_MOST),
        "GREENER_PLAINS": _vec3(cpu.GREENER_PLAINS),
        "DRIER_PLAINS": _vec3(cpu.DRIER_PLAINS),
        "PLAINS_TINT": _number(cpu.PLAINS_TINT),
        "LIGHT_X": _number(LIGHT[0]),
        "LIGHT_Y": _number(LIGHT[1]),
        "ROLL_LIGHT": _number(cpu.ROLL_LIGHT),
        "ROLL_LOW": _number(cpu.ROLL_RANGE[0]),
        "ROLL_HIGH": _number(cpu.ROLL_RANGE[1]),
        "WET_GAIN": _number(cpu.WET_GAIN),
        "FOOT_GAIN": _number(cpu.FOOT_GAIN),
        "LUSH": _number(cpu.LUSH),
        "FOOT_SHADE": _number(cpu.FOOT_SHADE),
        "LUSH_TINT": _vec3(cpu.LUSH_TINT),
        "FOOT_TINT": _vec3(cpu.FOOT_TINT),
        "LIGHT_Z": _number(relief.LIGHT[2]),
    }


def fragment_source() -> str:
    """The fragment shader with the CPU ground's tables and numbers written in."""
    ramps = [_vec3(cpu.RAMPS[biome][shade]) for biome in Biome for shade in range(4)]
    tuft_rows = [sum(1 << x for x in range(6) if cpu.TUFT[y, x]) for y in range(6)]
    names = _shared_names() | {
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
        "TUNDRA_RAMPS": _grass_ramps("tundra"),
        "DESERT_RAMPS": _grass_ramps("desert"),
        "TUNDRA": str(int(Biome.TUNDRA)),
        "DESERT": str(int(Biome.DESERT)),
        "D_8_311": _noise(0.8, 311, "q"),
        "D_45_312": _noise(0.45, 312, "s"),
        "D_15_313": _noise(0.15, 313, "q"),
        "DUNE_STEP": _number(cpu.DUNE_STEP),
        "DUNE_LIGHT": _number(cpu.DUNE_LIGHT),
        "V_1_401": _noise(0.1, 401, "q"),
        "V_045_402": _noise(0.045, 402, "q"),
        "DEPTH_FROM": _number(cpu.DEPTH_FROM),
        "DEPTH_SPAN": _number(cpu.DEPTH_SPAN),
        "SHALLOW": _vec3(cpu.SHALLOW),
        "OPEN_WATER": _vec3(cpu.OPEN_WATER),
        "DEEP_WATER": _vec3(cpu.DEEP_WATER),
        "WAVE_STEP": _number(cpu.WAVE_STEP),
        "WAVE_LIGHT": _number(cpu.WAVE_LIGHT),
        "WAVE_LOW": _number(cpu.WAVE_RANGE[0]),
        "WAVE_HIGH": _number(cpu.WAVE_RANGE[1]),
        "SPARKLE_FACING": _number(cpu.SPARKLE_FACING),
        "SPARKLE_ABOVE": _number(1 - cpu.SPARKLE_SHARE),
        "SPARKLE_MIX": _number(cpu.SPARKLE_MIX),
        "SPARKLE": _vec3(cpu.SPARKLE),
        "SHORE_WATER": _vec3(cpu.SHORE_WATER),
        "SHORE_NEAR": _number(cpu.SHORE_MIX[0]),
        "SHORE_FAR": _number(cpu.SHORE_MIX[1]),
        "BANK_SHADE": _vec3(cpu.BANK_SHADE),
        "BEACH_NEAR": _number(cpu.BEACH[0]),
        "BEACH_FAR": _number(cpu.BEACH[1]),
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
        "W_7_211": _noise(0.7, 211, "q"),
        "W_7_212": _noise(0.7, 212, "q"),
        "W_2_213": _noise(0.2, 213, "q"),
        "W_2_214": _noise(0.2, 214, "q"),
        "WARP": _number(cpu.WARP),
        "WARP_FINE": _number(cpu.WARP_FINE),
        "WARP_BROAD": _number(cpu.WARP_BROAD),
        "W_18_215": _noise(1.8, 215, "q"),
        "W_18_216": _noise(1.8, 216, "q"),
        "FOOTHILL": _vec3(relief.FOOTHILL),
        "F_3_13": _noise(0.3, 13),
        "TREELINE": _number(relief.TREELINE),
        "TREELINE_SPREAD": _number(relief.TREELINE_SPREAD),
        "F_7_8": _noise(0.7, 8),
        "ROCK_DARK": _vec3(relief.ROCK_DARK),
        "ROCK_LIT": _vec3(relief.ROCK_LIT),
        "F_05_7": _noise(0.05, 7),
        "SNOWLINE": _number(relief.SNOWLINE),
        "SNOWLINE_SPREAD": _number(relief.SNOWLINE_SPREAD),
        "F_6_9": _noise(0.6, 9),
        "RIBS": _noise(0.12, 14, "vec2(p.x * 3.0, p.y * 0.8)"),
        "SNOW_DARK": _vec3(relief.SNOW_DARK),
        "SNOW_LIT": _vec3(relief.SNOW_LIT),
        "SHADOW_TINT": _vec3(relief.SHADOW_TINT),
    }
    return _filled(FRAGMENT, names)


def relief_source() -> str:
    """The shader working out the raised ground (RELIEF), with Elevation's numbers written in."""
    names = {
        "STEPS": _number(relief.STEPS),
        "MOUNTAIN_GAIN": _number(relief.MOUNTAIN_GAIN),
        "MOUNTAIN_FROM": _number(relief.MOUNTAIN_FROM),
        "PEAK_GAIN": _number(relief.PEAK_GAIN),
        "PEAK_FROM": _number(relief.PEAK_FROM),
        "MOUNTAIN_FLOOR": _number(relief.MOUNTAIN_FLOOR),
        "MOUNTAIN_RIDGES": _number(relief.MOUNTAIN_RIDGES),
        "HILL_ROLL": _number(relief.HILL_ROLL),
        "W_25_11": _noise(2.5, 11),
        "W_25_12": _noise(2.5, 12),
        "R_26_1": _noise(2.6, 1, "w"),
        "R_1_2": _noise(1.0, 2, "w"),
        "R_42_3": _noise(0.42, 3, "w"),
        "R_16_6": _noise(0.16, 6, "w"),
        "N_9_4": _noise(0.9, 4),
        "N_35_5": _noise(0.35, 5),
    }
    return _filled(RELIEF, names)


def raise_source() -> str:
    """The shader raising the ground (RAISE), with Elevation's numbers written in."""
    names = {
        "EXAGGERATION": _number(relief.EXAGGERATION),
        "TILT": _number(TILT),
        "MOST_LIFT": _number(relief.MOST_LIFT),
        "MOST_LIFT_ROWS": str(relief.MOST_LIFT),
    }
    return _filled(RAISE, names)


def tint_source() -> str:
    """The shader tinting cells (TINT), with Elevation's numbers written in."""
    names = {
        "EXAGGERATION": _number(relief.EXAGGERATION),
        "TILT": _number(TILT),
        "MOST_LIFT": _number(relief.MOST_LIFT),
        "MOST_LIFT_ROWS": str(relief.MOST_LIFT),
    }
    return _filled(TINT, names)


def woods_source() -> str:
    """The shader finding where woods are and how deeply trees shade the grass (WOODS)."""
    names = {
        "AROUND_COUNT": str(len(cpu.AROUND)),
        "AROUND": ", ".join(f"vec2({_number(x)}, {_number(y)})" for x, y in cpu.AROUND),
        "AROUND_FLOAT": _number(len(cpu.AROUND)),
        "N_55_61": _noise(0.55, 61),
        "N_16_62": _noise(0.16, 62),
        "STEPS": _number(relief.STEPS),
        "MOUNTAIN_GAIN": _number(relief.MOUNTAIN_GAIN),
        "MOUNTAIN_FROM": _number(relief.MOUNTAIN_FROM),
        "PEAK_GAIN": _number(relief.PEAK_GAIN),
        "PEAK_FROM": _number(relief.PEAK_FROM),
        "BARE_ABOVE": _number(cpu.BARE_ABOVE),
        "GRASS": str(sum(1 << i for i, grass in enumerate(cpu._GRASS) if grass)),
        "SHADE_X": _number(cpu.SHADE_REACH[0]),
        "SHADE_Y": _number(cpu.SHADE_REACH[1]),
        "TREE_SHADE": _number(cpu.TREE_SHADE),
    }
    return _filled(WOODS, names)


def edges_source() -> str:
    """The shader finding where crowns reach round the woods (EDGES)."""
    return _filled(EDGES, {"OVERHANG": str(cpu.OVERHANG), "BELOW": str(cpu.OVERHANG + 3)})


def _fields_names() -> dict[str, str]:
    """The numbers and tables of fields, groves and yards (FIELDS)."""
    groves = list(cpu.GROVES.values())
    fruit = [colour for grove in groves for colour in grove.fruit]
    yards = list(cpu.YARD_COLOURS)
    return {
        "CROP_COUNT": str(len(cpu.CROP_ROWS)),
        "CROP_PLANT": ", ".join(_vec3(plant) for plant, _ in cpu.CROP_ROWS.values()),
        "CROP_SOIL": ", ".join(
            _vec3(np.array(ground, np.float32) * 0.8 + cpu.SOIL * 0.2)
            for _, ground in cpu.CROP_ROWS.values()
        ),
        "GROVE_COUNT": str(len(groves)),
        "GROVE_GROUND": ", ".join(_vec3(grove.ground) for grove in groves),
        "GROVE_TREE": ", ".join(_vec3(grove.tree) for grove in groves),
        "FRUIT_COUNT": str(len(fruit)),
        "GROVE_FRUIT": ", ".join(_vec3(colour) for colour in fruit),
        "GROVE_FRUIT_FROM": ", ".join(
            str(sum(len(g.fruit) for g in groves[:i])) for i in range(len(groves))
        ),
        "GROVE_FRUITS": ", ".join(str(len(grove.fruit)) for grove in groves),
        "GROVE_ROWS": ", ".join(_number(grove.rows) for grove in groves),
        "GROVE_CROWN": ", ".join(_number(grove.crown) for grove in groves),
        "GROVES_FROM": str(1 + len(cpu.CROP_ROWS)),
        "YARD_COUNT": str(len(yards)),
        "YARD_COLOURS": ", ".join(_vec3(colour) for colour in cpu.YARD_COLOURS.values()),
        "COTTAGE": str(yards.index("cottage")),
        "PAVED": str(yards.index("paved")),
        "FLAGSTONES": str(yards.index("flagstones")),
        "SPOIL": str(yards.index("spoil")),
        "FIELD_FEATHER": str(cpu.FIELD_FEATHER),
        "FIELD_FRAY": _number(cpu.FIELD_FRAY),
        "N_FRAY": _noise(cpu.FIELD_FRAY_SIZE, 74),
        "FIELD_STRIPES": _number(cpu.FIELD_STRIPES),
        "FIELD_SATURATION": _number(cpu.FIELD_SATURATION),
        "WHEAT": _vec3(cpu.WHEAT),
        "WHEAT_DARK": _vec3(cpu.WHEAT_DARK),
        "WHEAT_LIGHT": _vec3(cpu.WHEAT_LIGHT),
        "WHEAT_EARS": _vec3(cpu.WHEAT_EARS),
        "SOIL": _vec3(cpu.SOIL),
        "SOIL_DARK": _vec3(cpu.SOIL_DARK),
        "GROVE_FLOOR": _number(cpu.GROVE_FLOOR),
        "GROVE_SHADOW": _number(cpu.GROVE_SHADOW),
        "GARDEN_LEAVES": _vec3(cpu.GARDEN_LEAVES),
    }


def canopy_source() -> str:
    """The shader painting the woods (CANOPY), with Ground.canopy's numbers written in."""
    light = cpu.CROWN_LIGHT
    names = (
        _shared_names()
        | _fields_names()
        | {
            "CROWN_LIGHT": f"vec3({_number(light[0])}, {_number(light[1])}, {_number(light[2])})",
            "CROWN_GAP": _number(cpu.CROWN_GAP),
            "CROWN_SIZE_0": _number(cpu.CROWN_SIZE[0]),
            "CROWN_SIZE_1": _number(cpu.CROWN_SIZE[1]),
            "YOUNG": _number(cpu.YOUNG),
            "AGE_SPAN": _number(cpu.OLD - cpu.YOUNG),
            "N_STAND": _noise(cpu.STAND_SPAN, 70, "j"),
            "N_THIN": _noise(cpu.THIN_SPAN, 71, "j"),
            "THINNED": _number(cpu.THINNED),
            "CLUMP_GAP": _number(cpu.CLUMP_GAP),
            "CLUMP_SIZE_0": _number(cpu.CLUMP_SIZE[0]),
            "CLUMP_SIZE_1": _number(cpu.CLUMP_SIZE[1]),
            "TRUNK_LENGTH_0": _number(cpu.TRUNK_LENGTH[0]),
            "TRUNK_LENGTH_1": _number(cpu.TRUNK_LENGTH[1]),
            "TRUNK_HALF": _number(cpu.TRUNK_HALF),
            "TRUNK_BARK": _vec3(cpu.TRUNK_BARK),
            "TRUNK_BELOW": str(cpu.TRUNK_BELOW),
            "RELIEF_STEPS": _number(cpu.RELIEF_STEPS_PER_CELL),
            "N_SWELL": _noise(cpu.CANOPY_SWELL_SIZE, 67),
            "N_SWELL_X": _noise(cpu.CANOPY_SWELL_SIZE, 67, "p + vec2(0.05, 0.0)"),
            "N_SWELL_Y": _noise(cpu.CANOPY_SWELL_SIZE, 67, "p + vec2(0.0, 0.05)"),
            "SWELL": _number(cpu.CANOPY_SWELL * cpu.CANOPY_SWELL_SIZE),
            "LUMPS": _number(cpu.LUMPS),
            "CLUMP_EDGE": _number(cpu.CLUMP_EDGE),
            "FLOOR_FROM": _number(cpu.FLOOR_FROM),
            "N_GRAIN": _noise(cpu.LEAF_GRAIN, 68),
            "TAIGA": str(int(Biome.TAIGA)),
            "JUNGLE": str(int(Biome.JUNGLE)),
            "N_STANDS": _noise(cpu.CONIFER_STANDS, 64),
            "CROWN_TONES": _number(cpu.CROWN_TONES),
            "CONIFER_0": _vec3(cpu.CONIFER_CROWNS[0]),
            "CONIFER_1": _vec3(cpu.CONIFER_CROWNS[1]),
            "LEAFY_0": _vec3(cpu.LEAFY_CROWNS[0]),
            "LEAFY_1": _vec3(cpu.LEAFY_CROWNS[1]),
            "N_CROWNS": _noise(0.9, 65),
            "N_REGION": _noise(cpu.REGION_SPAN, 72),
            "DARK_GREENS": _vec3(cpu.DARK_GREENS),
            "GOLDEN_GREENS": _vec3(cpu.GOLDEN_GREENS),
            "CROWN_ROUND": _number(cpu.CROWN_ROUND),
            "CLUMP_LIGHT": _number(cpu.CLUMP_LIGHT),
            "HOLLOWS": _number(cpu.HOLLOWS),
            "BRANCHES": _number(cpu.BRANCHES),
            "BRANCH_WIDTH": _number(cpu.BRANCH_WIDTH),
            "CONIFER_POINTS": _number(cpu.CONIFER_POINTS),
            "CONIFER_SLIM": _number(cpu.CONIFER_SLIM),
            "CONIFER_TIERS": _number(cpu.CONIFER_TIERS),
            "CROWN_SLOPES": _number(cpu.CROWN_SLOPES),
            "CROWN_DEPTH": _number(cpu.CROWN_DEPTH),
            "CROWN_RIM_0": _number(cpu.CROWN_RIM[0]),
            "CROWN_RIM_1": _number(cpu.CROWN_RIM[1]),
            "CROWN_SHINE_0": _number(cpu.CROWN_SHINE[0]),
            "CROWN_SHINE_1": _number(cpu.CROWN_SHINE[1]),
            "WOODS_SIDE": _number(cpu.WOODS_SIDE),
            "CROWN_DARKEST": _number(cpu.CROWN_DARKEST),
            "SHADOW_TINT": _vec3(relief.SHADOW_TINT),
            "HOLLOW_DARK": _number(cpu.HOLLOW_DARK),
            "BRANCH": _vec3(cpu.BRANCH),
            "WOODS_FLOOR": _vec3(cpu.WOODS_FLOOR),
            "N_UNDERGROWTH": _noise(cpu.LEAF_GRAIN * 1.5, 73),
            "SHADOW_REACH": f"vec2({_number(cpu.SHADOW_REACH[0])}, {_number(cpu.SHADOW_REACH[1])})",
            "SHADOW_STEP": f"ivec2({cpu.SHADOW_REACH[0]}, {cpu.SHADOW_REACH[1]})",
            "SHADOW_SOFT": _number(cpu.SHADOW_SOFT),
            "EDGE_SHADOW": _number(cpu.EDGE_SHADOW),
            "SHADOW_COLOUR": _vec3(cpu.SHADOW_COLOUR),
        }
    )
    return _filled(CANOPY, names)


def dress_source() -> str:
    """The shader laying a chunk over its woods (DRESS), with Elevation's numbers written in."""
    names = {
        "EXAGGERATION": _number(relief.EXAGGERATION),
        "TILT": _number(TILT),
        "MOST_LIFT": _number(relief.MOST_LIFT),
        "MOST_LIFT_ROWS": str(relief.MOST_LIFT),
    }
    return _filled(DRESS, names)


def _filled(source: str, names: dict[str, str]) -> str:
    source = source.replace("@NOISE@", NOISE).replace("@HEX@", HEX)
    source = source.replace("@RASTER@", RASTER).replace("@TEXTURE@", TEXTURE)
    source = source.replace("@SUN@", SUN).replace("@NATURAL@", NATURAL)
    source = source.replace("@FIELDS@", FIELDS)
    for name, value in (names | {"MASK": str(TABLE_SIZE - 1)}).items():
        source = source.replace(f"@{name}@", value)
    assert "@" not in source
    return source


def cell_info(
    biomes: npt.NDArray[np.integer],
    deposits: npt.NDArray[np.integer] | None,
    hills: npt.NDArray[np.bool_],
    glow: FloatArray,
    mountains: npt.NDArray[np.bool_] | None = None,
) -> npt.NDArray[np.uint8]:
    """What the shader needs of each cell, as (rows, cols, 4) bytes: its biome, its deposit, 1 on
    hills and 2 on mountains, and how strongly it glows in GLOW_STEPS."""
    info = np.zeros((*biomes.shape, 4), dtype=np.uint8)
    info[..., 0] = biomes
    if deposits is not None:
        info[..., 1] = deposits
    info[..., 2] = hills
    if mountains is not None:
        info[..., 2] |= mountains.astype(np.uint8) << 1
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
    """A texture the shaders draw into, with its framebuffer: of bytes, or of floats where
    `floats`, for what is worked on before it is shown."""

    def __init__(self, floats: bool = False) -> None:
        self.texture = 0
        self.framebuffer = GL.glGenFramebuffers(1)
        self.size = (0, 0)
        self._form = (GL.GL_RGBA32F, GL.GL_FLOAT) if floats else (GL.GL_RGBA8, GL.GL_UNSIGNED_BYTE)

    def fit(self, width: int, height: int) -> bool:
        """Makes the texture this size; returns whether it had to be made anew."""
        if self.size == (width, height):
            return False
        if self.texture:
            GL.glDeleteTextures(1, [self.texture])
        self.texture = _texture()
        internal, kind = self._form
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, internal, width, height, 0, GL.GL_RGBA, kind, None)
        saved = GL.glGetIntegerv(GL.GL_FRAMEBUFFER_BINDING)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.framebuffer)
        GL.glFramebufferTexture2D(
            GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, self.texture, 0
        )
        complete = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER) == GL.GL_FRAMEBUFFER_COMPLETE
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, int(saved))
        if not complete:
            raise RuntimeError("The graphics card can't draw into this texture")
        self.size = (width, height)
        return True


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
        self._cells = _texture()
        self._light = _floats(light)
        self._table = _floats(_TABLE)
        self._rasters = _floats(heights.rasters)
        self._beds = 0
        self._relief = _Target(floats=True)
        self._lit = _Target(floats=True)
        self._art = _Target()
        self._screen = _Target()
        self._tinted = _Target()
        self._tints = _texture()
        self.tints_version: object = None
        self._tinted_key: tuple | None = None
        self._drawn: tuple[View, int, int] | None = None
        self._shown: tuple[View, Screen, int, int, bool] | None = None
        self.cells_version = -1
        self.bed_version = -1
        self.set_bed(None, -1)
        self._chunks = GL.glGenFramebuffers(1)
        self._woodland = _texture()
        self._trees = np.zeros((0, 2), dtype=np.float32)
        self._clearings = np.zeros((0, 4), dtype=np.float32)
        self._plots = np.zeros((0, 10), dtype=np.float32)
        self._yards = np.zeros((0, 4), dtype=np.float32)
        self.surfaces_version: object = None
        self._chunk_relief = _Target(floats=True)
        self._wooded = _Target(floats=True)
        self._edges = _Target(floats=True)
        self._canopy = _Target(floats=True)

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
                _program(HEADER + VERTEX, HEADER + fragment_source()),
                _program(HEADER + VERTEX, HEADER + relief_source()),
                _program(HEADER + VERTEX, HEADER + raise_source()),
                _program(HEADER + VERTEX, HEADER + _filled(BLOW_UP, {})),
                _program(HEADER + VERTEX, HEADER + tint_source()),
                _program(HEADER + VERTEX, HEADER + woods_source()),
                _program(HEADER + VERTEX, HEADER + edges_source()),
                _program(HEADER + VERTEX, HEADER + canopy_source()),
                _program(HEADER + VERTEX, HEADER + dress_source()),
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
                "art": ("1i", (0,)),
            }
            self._run(self._screen, self._programs.blow_up, uniforms, (art,))
            self._shown = shown
        return self._screen.texture

    def blow_up(self, pixels: npt.NDArray[np.uint8], times: int) -> ChunkTexture:
        """A texture of a chunk's RGBA pixels with each repeated `times` times each way."""
        free_textures()
        small, big, size = self._chunk_textures(pixels, times)
        uniforms = {
            "origin": ("2f", (0.0, 0.0)),
            "scale": ("1f", (float(times),)),
            "art_origin": ("2i", (0, 0)),
            "blend": ("1i", (0,)),
            "art": ("1i", (0,)),
        }
        self._run(_Into(self._chunks, size), self._programs.blow_up, uniforms, (small,))
        GL.glDeleteTextures(1, [small])
        return ChunkTexture(big)

    def set_surfaces(self, ground: cpu.Ground, version: object) -> None:
        """Takes what is drawn over the ground from `ground`: each cell's say in the woods and
        whether it is land, where trees stand, whose shade lies on the grass round them, where
        lumber mills clear the woods round them, the fields and groves and the buildings' yards."""
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

    def dress(
        self, pixels: npt.NDArray[np.uint8], times: int, chunk: tuple[int, int]
    ) -> ChunkTexture:
        """A texture of a chunk drawn without its woods (PixelWorld.render without `woods`) laid
        over the woods and the shade of trees, which are drawn here (WOODS, EDGES, CANOPY), raised
        with the ground where it rises; every pixel repeated `times` times each way."""
        free_textures()
        k = pixels.shape[1] // CHUNK
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
        small, big, size = self._chunk_textures(pixels, times)
        uniforms = {
            "chunk_origin": ("2i", (left, top)),
            "k": ("1i", (k,)),
            "drawn": ("1i", (drawn,)),
            "times": ("1i", (times,)),
            "relief_origin": ("2i", relief_origin),
            "metres_per_pixel": ("1f", (self._heights.metres_per_pixel,)),
            "pixels": ("1i", (0,)),
            "surface": ("1i", (1,)),
            "relief": ("1i", (2,)),
        }
        inputs = (small, self._canopy.texture, self._chunk_relief.texture)
        self._run(_Into(self._chunks, size), self._programs.dress, uniforms, inputs)
        GL.glDeleteTextures(1, [small])
        return ChunkTexture(big)

    def _chunk_textures(self, pixels: npt.NDArray[np.uint8], times: int) -> tuple[int, int, tuple]:
        """A texture of a chunk's pixels, and one `times` as big each way for the chunk's
        framebuffer to draw into, with its size."""
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
        return small, big, size

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
                    case "4f":
                        GL.glUniform4f(at, *values)
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
    texture = _texture()
    data = np.ascontiguousarray(values, dtype=np.float32)
    rows, cols = data.shape[:2]
    internal, form = (GL.GL_RGBA32F, GL.GL_RGBA) if data.ndim == 3 else (GL.GL_R32F, GL.GL_RED)
    GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, internal, cols, rows, 0, form, GL.GL_FLOAT, data)
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
    for unit in range(7, 0, -1):
        GL.glActiveTexture(GL.GL_TEXTURE0 + unit)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
    GL.glActiveTexture(active)
    if blend:
        GL.glEnable(GL.GL_BLEND)
    if scissor:
        GL.glEnable(GL.GL_SCISSOR_TEST)
