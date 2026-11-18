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
from railroads.ui.pixel.projection import PixelProjection
from railroads.ui.pixel.relief import LIGHT, Heights

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
const float HEX_RADIUS = 0.5773502691896258;
const float SQRT3 = 1.7320508075688772;
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

float hash(int a, int b, int seed) {
    return tbl(a * 13 + seed * 71, b * 7 + seed * 131);
}

float height_at(ivec2 art) {
    vec4 raised = texelFetch(relief, art - relief_origin, 0);
    return raised.w + raised.x;
}

// Raised ground lit and coloured as railroads.ui.pixel.relief.Elevation.colour does: by the
// light on its slope, and with foothills, bare rock and snow where it is mountainous.
vec3 raised_colour(vec3 rgb, vec4 raised, ivec2 art, vec2 p, bool rocky) {
    float above = raised.x, m = raised.y, h = raised.z;
    float per_col = (height_at(art + ivec2(1, 0)) - height_at(art - ivec2(1, 0))) * 0.5;
    float per_row = (height_at(art + ivec2(0, 1)) - height_at(art - ivec2(0, 1))) * 0.5;
    float gx = (per_col * slope.x + per_row * slope.y) / metres_per_unit;
    float gy = (per_col * slope.z + per_row * slope.w) / metres_per_unit;
    float nz = 1.0 / sqrt(1.0 + gx * gx + gy * gy);
    float s = clamp((-gx * @LIGHT_X@ - gy * @LIGHT_Y@ + @LIGHT_Z@) * nz, 0.0, 1.0);
    float st = 1.0 - nz;
    float lit = clamp(1.0 + 1.3 * (s - @LIGHT_Z@), 0.55, 1.3);
    lit = 1.0 + (lit - 1.0) * clamp((m + h) * 4.0, 0.0, 1.0);
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

float rolling(vec2 q) {
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


BLOW_UP = """out vec4 colour;
uniform vec2 origin;
uniform float scale;
uniform ivec2 art_origin;
uniform int blend;
uniform highp sampler2D art;
float hash(vec2 q) {
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


def fragment_source() -> str:
    """The fragment shader with the CPU ground's tables and numbers written in."""
    ramps = [_vec3(cpu.RAMPS[biome][shade]) for biome in Biome for shade in range(4)]
    tuft_rows = [sum(1 << x for x in range(6) if cpu.TUFT[y, x]) for y in range(6)]
    names = {
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
        "R_16_301": _noise(1.6, 301, "q"),
        "R_55_302": _noise(0.55, 302, "q"),
        "R_18_303": _noise(0.18, 303, "q"),
        "W_7_211": _noise(0.7, 211, "q"),
        "W_7_212": _noise(0.7, 212, "q"),
        "W_2_213": _noise(0.2, 213, "q"),
        "W_2_214": _noise(0.2, 214, "q"),
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
        "WARP": _number(cpu.WARP),
        "WARP_FINE": _number(cpu.WARP_FINE),
        "WARP_BROAD": _number(cpu.WARP_BROAD),
        "W_18_215": _noise(1.8, 215, "q"),
        "W_18_216": _noise(1.8, 216, "q"),
        "LIGHT_Z": _number(relief.LIGHT[2]),
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


def _filled(source: str, names: dict[str, str]) -> str:
    source = source.replace("@NOISE@", NOISE)
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
        self._drawn: tuple[View, int, int] | None = None
        self._shown: tuple[View, Screen, int, int, bool] | None = None
        self.cells_version = -1
        self.bed_version = -1
        self.set_bed(None, -1)
        self._chunks = GL.glGenFramebuffers(1)

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
                _program(HEADER + VERTEX, HEADER + BLOW_UP),
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
            "blend": ("1i", (0,)),
            "art": ("1i", (0,)),
        }
        self._run(_Into(self._chunks, size), self._programs.blow_up, uniforms, (small,))
        GL.glDeleteTextures(1, [small])
        return ChunkTexture(big)

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
    for unit in (3, 2, 1):
        GL.glActiveTexture(GL.GL_TEXTURE0 + unit)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
    GL.glActiveTexture(active)
    if blend:
        GL.glEnable(GL.GL_BLEND)
    if scissor:
        GL.glEnable(GL.GL_SCISSOR_TEST)
