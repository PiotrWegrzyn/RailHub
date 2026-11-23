"""The GLSL the ground shader is built from, and the sources of its programs with the numbers of railroads.ui.pixel.ground written into them (fragment_source and the rest), so the GPU draws the ground as the CPU does."""

import numpy as np

from railroads.sim.tilemap import Deposit
from railroads.ui.biomes import Biome
from railroads.ui.iso_camera import TILT
from railroads.ui.pixel import ground as cpu
from railroads.ui.pixel import relief, streams
from railroads.ui.pixel.noise import TABLE_SIZE
from railroads.ui.pixel.relief import LIGHT

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


# How far a plane point in a cell lies from the minor river through it, along the links the cell
# keeps in its flags (railroads.ui.pixel.streams.stream_distance); for shaders that read `cells`.
STREAM = """const vec2 STREAM_STEPS[6] = vec2[6](@STREAM_STEPS@);

float stream_gap(vec2 p, ivec2 cell, uint links) {
    if (links == 0u) return @STREAM_NONE@;
    vec2 centre = vec2(float(cell.x) + 0.5 + 0.5 * float(cell.y & 1),
                       HEX_RADIUS + float(cell.y) * 1.5 * HEX_RADIUS);
    vec2 q = vec2(p.x + @MEANDER@ * sin(p.y * 7.3 + p.x * 2.1),
                  p.y + @MEANDER@ * sin(p.x * 6.1 - p.y * 1.7));
    float nearest = @STREAM_NONE@;
    for (int i = 0; i < 6; i++) {
        if (((links >> uint(i)) & 1u) == 0u) continue;
        vec2 d = STREAM_STEPS[i];
        float along = clamp(dot(q - centre, d) / dot(d, d), 0.0, 1.0);
        nearest = min(nearest, length(q - centre - along * d));
    }
    return nearest;
}

// Whether plane point `p` lies on a minor river or its banks, which fields leave showing.
bool on_stream(vec2 p) {
    ivec2 cell = clamp(cell_at(p), ivec2(0), cells_size - 1);
    return stream_gap(p, cell, texelFetch(cells, cell, 0).b >> 2u) < @STREAM_BANK_WIDTH@;
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

@STREAM@

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
        float gap = stream_gap(p, cell, info.b >> 2u);
        if (gap < @STREAM_BANK_WIDTH@) rgb *= @STREAM_BANK@;
        if (gap < @STREAM_HALF_WIDTH@) {
            rgb = (water_colour(p, ix, iy, around.g) + @STREAM_WATER@) * 0.5;
        }
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
uniform int premultiply;
uniform highp sampler2D art;
@TEXTURE@
void main() {
    vec2 p = (gl_FragCoord.xy - origin) / scale;
    if (blend == 0) {
        colour = texelFetch(art, ivec2(floor(p)) - art_origin, 0);
        if (premultiply == 1) colour.rgb *= colour.a;
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
# logging camps and bare mountainsides; then how deeply trees shade the grass there
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
FIELDS = """@STREAM@
const vec3 CROP_PLANT[@CROP_COUNT@] = vec3[@CROP_COUNT@](@CROP_PLANT@);
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
    if (on_stream(p)) return;
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
    colour = shown;
}
"""


def glsl_number(value: float) -> str:
    """A GLSL float literal for exactly the float32 the CPU works with."""
    text = repr(float(np.float32(value)))
    return text if "." in text or "e" in text else f"{text}.0"


def _vec3(rgb: object) -> str:
    r, g, b = (float(v) for v in np.asarray(rgb, dtype=np.float32))
    return f"vec3({glsl_number(r)}, {glsl_number(g)}, {glsl_number(b)})"


def _noise(scale: float, seed: int, at: str = "p") -> str:
    """The call for railroads.ui.pixel.noise.noise at point `at`, `scale` and `seed`, with its
    numbers worked out in float32 as there."""
    inverse = np.float32(1 / scale)
    off_u, off_v = np.float32(seed * 37.1 + 500.0), np.float32(seed * 11.3 + 500.0)
    return f"noise({at}, {glsl_number(inverse)}, {glsl_number(off_u)}, {glsl_number(off_v)})"


def _bits(biomes: tuple[Biome, ...]) -> int:
    return sum(1 << int(biome) for biome in set(biomes))


def _grass_ramps(name: str) -> str:
    short, tall, bare, rarity = cpu.GRASSES[name]
    lines = []
    for prefix, stops in (("s", short), ("t", tall), ("b", bare)):
        assert [at for at, _ in stops] == [0.0, 0.5, 1.0]
        lines += [f"{prefix}{i} = {_vec3(colour)};" for i, (_, colour) in enumerate(stops)]
    lines.append(f"rarity = {glsl_number(rarity)};")
    return " ".join(lines)


def _shared_names() -> dict[str, str]:
    """The numbers of the snippets the ground and what lies on it share (SUN, NATURAL, STREAM)."""
    return {
        "STREAM_STEPS": ", ".join(
            f"vec2({glsl_number(x)}, {glsl_number(y)})" for x, y in streams.STEPS
        ),
        "STREAM_NONE": glsl_number(streams.NONE_NEAR),
        "MEANDER": glsl_number(streams.MEANDER),
        "STREAM_BANK_WIDTH": glsl_number(streams.BANK_WIDTH),
        "STREAM_HALF_WIDTH": glsl_number(streams.HALF_WIDTH),
        "STREAM_BANK": glsl_number(streams.STREAM_BANK),
        "STREAM_WATER": _vec3(streams.STREAM_WATER),
        "R_16_301": _noise(1.6, 301, "q"),
        "R_55_302": _noise(0.55, 302, "q"),
        "R_18_303": _noise(0.18, 303, "q"),
        "ROLL_STEP": glsl_number(cpu.ROLL_STEP),
        "N_25_306": _noise(cpu.ROLL_SPAN, 306),
        "N_3_305": _noise(cpu.PLAINS_SPAN, 305),
        "ROLL_LEAST": glsl_number(cpu.ROLL_LEAST),
        "ROLL_MOST": glsl_number(cpu.ROLL_MOST),
        "GREENER_PLAINS": _vec3(cpu.GREENER_PLAINS),
        "DRIER_PLAINS": _vec3(cpu.DRIER_PLAINS),
        "PLAINS_TINT": glsl_number(cpu.PLAINS_TINT),
        "LIGHT_X": glsl_number(LIGHT[0]),
        "LIGHT_Y": glsl_number(LIGHT[1]),
        "ROLL_LIGHT": glsl_number(cpu.ROLL_LIGHT),
        "ROLL_LOW": glsl_number(cpu.ROLL_RANGE[0]),
        "ROLL_HIGH": glsl_number(cpu.ROLL_RANGE[1]),
        "WET_GAIN": glsl_number(cpu.WET_GAIN),
        "FOOT_GAIN": glsl_number(cpu.FOOT_GAIN),
        "LUSH": glsl_number(cpu.LUSH),
        "FOOT_SHADE": glsl_number(cpu.FOOT_SHADE),
        "LUSH_TINT": _vec3(cpu.LUSH_TINT),
        "FOOT_TINT": _vec3(cpu.FOOT_TINT),
        "LIGHT_Z": glsl_number(relief.LIGHT[2]),
    }


def fragment_source() -> str:
    """The fragment shader with the CPU ground's tables and numbers written in."""
    ramps = [_vec3(cpu.RAMPS[biome][shade]) for biome in Biome for shade in range(4)]
    tuft_rows = [sum(1 << x for x in range(6) if cpu.TUFT[y, x]) for y in range(6)]
    names = _shared_names() | {
        "RAMP_COUNT": str(len(ramps)),
        "RAMPS": ", ".join(ramps),
        "BAYER": ", ".join(glsl_number(v) for v in cpu.BAYER.ravel()),
        "TUFT_ROWS": ", ".join(str(row) for row in tuft_rows),
        "FLOWERS": ", ".join(_vec3(f) for f in cpu.FLOWERS),
        "SPECK_COUNT": str(len(cpu.DEPOSIT_SPECKS)),
        "SPECKS": ", ".join(_vec3(s) for s in cpu.DEPOSIT_SPECKS),
        "LAST_WATER": str(max(int(b) for b in cpu.WATER)),
        "STEPS": glsl_number(cpu.RELIEF_STEPS_PER_CELL),
        "MEADOW_RAMPS": _grass_ramps("meadow"),
        "SAVANNA_RAMPS": _grass_ramps("savanna"),
        "HILL_LIFT": glsl_number(cpu.HILL_LIFT),
        "MEADOWS": str(_bits(cpu.MEADOWS)),
        "GRASSY": str(_bits(cpu.GRASSY)),
        "SAVANNA": str(int(Biome.SAVANNA)),
        "TUFT_ABOVE": glsl_number(1 - cpu.TUFT_SHARE),
        "TUNDRA_RAMPS": _grass_ramps("tundra"),
        "DESERT_RAMPS": _grass_ramps("desert"),
        "TUNDRA": str(int(Biome.TUNDRA)),
        "DESERT": str(int(Biome.DESERT)),
        "D_8_311": _noise(0.8, 311, "q"),
        "D_45_312": _noise(0.45, 312, "s"),
        "D_15_313": _noise(0.15, 313, "q"),
        "DUNE_STEP": glsl_number(cpu.DUNE_STEP),
        "DUNE_LIGHT": glsl_number(cpu.DUNE_LIGHT),
        "V_1_401": _noise(0.1, 401, "q"),
        "V_045_402": _noise(0.045, 402, "q"),
        "DEPTH_FROM": glsl_number(cpu.DEPTH_FROM),
        "DEPTH_SPAN": glsl_number(cpu.DEPTH_SPAN),
        "SHALLOW": _vec3(cpu.SHALLOW),
        "OPEN_WATER": _vec3(cpu.OPEN_WATER),
        "DEEP_WATER": _vec3(cpu.DEEP_WATER),
        "WAVE_STEP": glsl_number(cpu.WAVE_STEP),
        "WAVE_LIGHT": glsl_number(cpu.WAVE_LIGHT),
        "WAVE_LOW": glsl_number(cpu.WAVE_RANGE[0]),
        "WAVE_HIGH": glsl_number(cpu.WAVE_RANGE[1]),
        "SPARKLE_FACING": glsl_number(cpu.SPARKLE_FACING),
        "SPARKLE_ABOVE": glsl_number(1 - cpu.SPARKLE_SHARE),
        "SPARKLE_MIX": glsl_number(cpu.SPARKLE_MIX),
        "SPARKLE": _vec3(cpu.SPARKLE),
        "SHORE_WATER": _vec3(cpu.SHORE_WATER),
        "SHORE_NEAR": glsl_number(cpu.SHORE_MIX[0]),
        "SHORE_FAR": glsl_number(cpu.SHORE_MIX[1]),
        "BANK_SHADE": _vec3(cpu.BANK_SHADE),
        "BEACH_NEAR": glsl_number(cpu.BEACH[0]),
        "BEACH_FAR": glsl_number(cpu.BEACH[1]),
        "SAND": _vec3(cpu.SAND),
        "GLOW_STEPS": glsl_number(GLOW_STEPS),
        "GLOW": _vec3(cpu.GLOW),
        "GARRISON": str(int(Deposit.GARRISON)),
        "WOOL": str(int(Deposit.WOOL)),
        "SIGNS_ABOVE": glsl_number(1 - cpu.DEPOSIT_SIGNS),
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
        "WARP": glsl_number(cpu.WARP),
        "WARP_FINE": glsl_number(cpu.WARP_FINE),
        "WARP_BROAD": glsl_number(cpu.WARP_BROAD),
        "W_18_215": _noise(1.8, 215, "q"),
        "W_18_216": _noise(1.8, 216, "q"),
        "FOOTHILL": _vec3(relief.FOOTHILL),
        "F_3_13": _noise(0.3, 13),
        "TREELINE": glsl_number(relief.TREELINE),
        "TREELINE_SPREAD": glsl_number(relief.TREELINE_SPREAD),
        "F_7_8": _noise(0.7, 8),
        "ROCK_DARK": _vec3(relief.ROCK_DARK),
        "ROCK_LIT": _vec3(relief.ROCK_LIT),
        "F_05_7": _noise(0.05, 7),
        "SNOWLINE": glsl_number(relief.SNOWLINE),
        "SNOWLINE_SPREAD": glsl_number(relief.SNOWLINE_SPREAD),
        "F_6_9": _noise(0.6, 9),
        "RIBS": _noise(0.12, 14, "vec2(p.x * 3.0, p.y * 0.8)"),
        "SNOW_DARK": _vec3(relief.SNOW_DARK),
        "SNOW_LIT": _vec3(relief.SNOW_LIT),
        "SHADOW_TINT": _vec3(relief.SHADOW_TINT),
    }
    return fill_names(FRAGMENT, names)


def relief_source() -> str:
    """The shader working out the raised ground (RELIEF), with Elevation's numbers written in."""
    names = {
        "STEPS": glsl_number(relief.STEPS),
        "MOUNTAIN_GAIN": glsl_number(relief.MOUNTAIN_GAIN),
        "MOUNTAIN_FROM": glsl_number(relief.MOUNTAIN_FROM),
        "PEAK_GAIN": glsl_number(relief.PEAK_GAIN),
        "PEAK_FROM": glsl_number(relief.PEAK_FROM),
        "MOUNTAIN_FLOOR": glsl_number(relief.MOUNTAIN_FLOOR),
        "MOUNTAIN_RIDGES": glsl_number(relief.MOUNTAIN_RIDGES),
        "HILL_ROLL": glsl_number(relief.HILL_ROLL),
        "W_25_11": _noise(2.5, 11),
        "W_25_12": _noise(2.5, 12),
        "R_26_1": _noise(2.6, 1, "w"),
        "R_1_2": _noise(1.0, 2, "w"),
        "R_42_3": _noise(0.42, 3, "w"),
        "R_16_6": _noise(0.16, 6, "w"),
        "N_9_4": _noise(0.9, 4),
        "N_35_5": _noise(0.35, 5),
    }
    return fill_names(RELIEF, names)


def raise_source() -> str:
    """The shader raising the ground (RAISE), with Elevation's numbers written in."""
    names = {
        "EXAGGERATION": glsl_number(relief.EXAGGERATION),
        "TILT": glsl_number(TILT),
        "MOST_LIFT": glsl_number(relief.MOST_LIFT),
        "MOST_LIFT_ROWS": str(relief.MOST_LIFT),
    }
    return fill_names(RAISE, names)


def tint_source() -> str:
    """The shader tinting cells (TINT), with Elevation's numbers written in."""
    names = {
        "EXAGGERATION": glsl_number(relief.EXAGGERATION),
        "TILT": glsl_number(TILT),
        "MOST_LIFT": glsl_number(relief.MOST_LIFT),
        "MOST_LIFT_ROWS": str(relief.MOST_LIFT),
    }
    return fill_names(TINT, names)


def woods_source() -> str:
    """The shader finding where woods are and how deeply trees shade the grass (WOODS)."""
    names = {
        "AROUND_COUNT": str(len(cpu.AROUND)),
        "AROUND": ", ".join(f"vec2({glsl_number(x)}, {glsl_number(y)})" for x, y in cpu.AROUND),
        "AROUND_FLOAT": glsl_number(len(cpu.AROUND)),
        "N_55_61": _noise(0.55, 61),
        "N_16_62": _noise(0.16, 62),
        "STEPS": glsl_number(relief.STEPS),
        "MOUNTAIN_GAIN": glsl_number(relief.MOUNTAIN_GAIN),
        "MOUNTAIN_FROM": glsl_number(relief.MOUNTAIN_FROM),
        "PEAK_GAIN": glsl_number(relief.PEAK_GAIN),
        "PEAK_FROM": glsl_number(relief.PEAK_FROM),
        "BARE_ABOVE": glsl_number(cpu.BARE_ABOVE),
        "GRASS": str(sum(1 << i for i, grass in enumerate(cpu.GRASS_BIOMES) if grass)),
        "SHADE_X": glsl_number(cpu.SHADE_REACH[0]),
        "SHADE_Y": glsl_number(cpu.SHADE_REACH[1]),
        "TREE_SHADE": glsl_number(cpu.TREE_SHADE),
    }
    return fill_names(WOODS, names)


def edges_source() -> str:
    """The shader finding where crowns reach round the woods (EDGES)."""
    return fill_names(EDGES, {"OVERHANG": str(cpu.OVERHANG), "BELOW": str(cpu.OVERHANG + 3)})


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
        "GROVE_ROWS": ", ".join(glsl_number(grove.rows) for grove in groves),
        "GROVE_CROWN": ", ".join(glsl_number(grove.crown) for grove in groves),
        "GROVES_FROM": str(1 + len(cpu.CROP_ROWS)),
        "YARD_COUNT": str(len(yards)),
        "YARD_COLOURS": ", ".join(_vec3(colour) for colour in cpu.YARD_COLOURS.values()),
        "COTTAGE": str(yards.index("cottage")),
        "PAVED": str(yards.index("paved")),
        "FLAGSTONES": str(yards.index("flagstones")),
        "SPOIL": str(yards.index("spoil")),
        "FIELD_FEATHER": str(cpu.FIELD_FEATHER),
        "FIELD_FRAY": glsl_number(cpu.FIELD_FRAY),
        "N_FRAY": _noise(cpu.FIELD_FRAY_SIZE, 74),
        "FIELD_STRIPES": glsl_number(cpu.FIELD_STRIPES),
        "FIELD_SATURATION": glsl_number(cpu.FIELD_SATURATION),
        "WHEAT": _vec3(cpu.WHEAT),
        "WHEAT_DARK": _vec3(cpu.WHEAT_DARK),
        "WHEAT_LIGHT": _vec3(cpu.WHEAT_LIGHT),
        "WHEAT_EARS": _vec3(cpu.WHEAT_EARS),
        "SOIL": _vec3(cpu.SOIL),
        "SOIL_DARK": _vec3(cpu.SOIL_DARK),
        "GROVE_FLOOR": glsl_number(cpu.GROVE_FLOOR),
        "GROVE_SHADOW": glsl_number(cpu.GROVE_SHADOW),
        "GARDEN_LEAVES": _vec3(cpu.GARDEN_LEAVES),
    }


def canopy_source() -> str:
    """The shader painting the woods (CANOPY), with Ground.canopy's numbers written in."""
    light = cpu.CROWN_LIGHT
    names = (
        _shared_names()
        | _fields_names()
        | {
            "CROWN_LIGHT": f"vec3({glsl_number(light[0])}, {glsl_number(light[1])}, {glsl_number(light[2])})",
            "CROWN_GAP": glsl_number(cpu.CROWN_GAP),
            "CROWN_SIZE_0": glsl_number(cpu.CROWN_SIZE[0]),
            "CROWN_SIZE_1": glsl_number(cpu.CROWN_SIZE[1]),
            "YOUNG": glsl_number(cpu.YOUNG),
            "AGE_SPAN": glsl_number(cpu.OLD - cpu.YOUNG),
            "N_STAND": _noise(cpu.STAND_SPAN, 70, "j"),
            "N_THIN": _noise(cpu.THIN_SPAN, 71, "j"),
            "THINNED": glsl_number(cpu.THINNED),
            "CLUMP_GAP": glsl_number(cpu.CLUMP_GAP),
            "CLUMP_SIZE_0": glsl_number(cpu.CLUMP_SIZE[0]),
            "CLUMP_SIZE_1": glsl_number(cpu.CLUMP_SIZE[1]),
            "TRUNK_LENGTH_0": glsl_number(cpu.TRUNK_LENGTH[0]),
            "TRUNK_LENGTH_1": glsl_number(cpu.TRUNK_LENGTH[1]),
            "TRUNK_HALF": glsl_number(cpu.TRUNK_HALF),
            "TRUNK_BARK": _vec3(cpu.TRUNK_BARK),
            "TRUNK_BELOW": str(cpu.TRUNK_BELOW),
            "RELIEF_STEPS": glsl_number(cpu.RELIEF_STEPS_PER_CELL),
            "N_SWELL": _noise(cpu.CANOPY_SWELL_SIZE, 67),
            "N_SWELL_X": _noise(cpu.CANOPY_SWELL_SIZE, 67, "p + vec2(0.05, 0.0)"),
            "N_SWELL_Y": _noise(cpu.CANOPY_SWELL_SIZE, 67, "p + vec2(0.0, 0.05)"),
            "SWELL": glsl_number(cpu.CANOPY_SWELL * cpu.CANOPY_SWELL_SIZE),
            "LUMPS": glsl_number(cpu.LUMPS),
            "CLUMP_EDGE": glsl_number(cpu.CLUMP_EDGE),
            "FLOOR_FROM": glsl_number(cpu.FLOOR_FROM),
            "N_GRAIN": _noise(cpu.LEAF_GRAIN, 68),
            "TAIGA": str(int(Biome.TAIGA)),
            "JUNGLE": str(int(Biome.JUNGLE)),
            "N_STANDS": _noise(cpu.CONIFER_STANDS, 64),
            "CROWN_TONES": glsl_number(cpu.CROWN_TONES),
            "CONIFER_0": _vec3(cpu.CONIFER_CROWNS[0]),
            "CONIFER_1": _vec3(cpu.CONIFER_CROWNS[1]),
            "LEAFY_0": _vec3(cpu.LEAFY_CROWNS[0]),
            "LEAFY_1": _vec3(cpu.LEAFY_CROWNS[1]),
            "N_CROWNS": _noise(0.9, 65),
            "N_REGION": _noise(cpu.REGION_SPAN, 72),
            "DARK_GREENS": _vec3(cpu.DARK_GREENS),
            "GOLDEN_GREENS": _vec3(cpu.GOLDEN_GREENS),
            "CROWN_ROUND": glsl_number(cpu.CROWN_ROUND),
            "CLUMP_LIGHT": glsl_number(cpu.CLUMP_LIGHT),
            "HOLLOWS": glsl_number(cpu.HOLLOWS),
            "BRANCHES": glsl_number(cpu.BRANCHES),
            "BRANCH_WIDTH": glsl_number(cpu.BRANCH_WIDTH),
            "CONIFER_POINTS": glsl_number(cpu.CONIFER_POINTS),
            "CONIFER_SLIM": glsl_number(cpu.CONIFER_SLIM),
            "CONIFER_TIERS": glsl_number(cpu.CONIFER_TIERS),
            "CROWN_SLOPES": glsl_number(cpu.CROWN_SLOPES),
            "CROWN_DEPTH": glsl_number(cpu.CROWN_DEPTH),
            "CROWN_RIM_0": glsl_number(cpu.CROWN_RIM[0]),
            "CROWN_RIM_1": glsl_number(cpu.CROWN_RIM[1]),
            "CROWN_SHINE_0": glsl_number(cpu.CROWN_SHINE[0]),
            "CROWN_SHINE_1": glsl_number(cpu.CROWN_SHINE[1]),
            "WOODS_SIDE": glsl_number(cpu.WOODS_SIDE),
            "CROWN_DARKEST": glsl_number(cpu.CROWN_DARKEST),
            "SHADOW_TINT": _vec3(relief.SHADOW_TINT),
            "HOLLOW_DARK": glsl_number(cpu.HOLLOW_DARK),
            "BRANCH": _vec3(cpu.BRANCH),
            "WOODS_FLOOR": _vec3(cpu.WOODS_FLOOR),
            "N_UNDERGROWTH": _noise(cpu.LEAF_GRAIN * 1.5, 73),
            "SHADOW_REACH": f"vec2({glsl_number(cpu.SHADOW_REACH[0])}, {glsl_number(cpu.SHADOW_REACH[1])})",
            "SHADOW_STEP": f"ivec2({cpu.SHADOW_REACH[0]}, {cpu.SHADOW_REACH[1]})",
            "SHADOW_SOFT": glsl_number(cpu.SHADOW_SOFT),
            "EDGE_SHADOW": glsl_number(cpu.EDGE_SHADOW),
            "SHADOW_COLOUR": _vec3(cpu.SHADOW_COLOUR),
        }
    )
    return fill_names(CANOPY, names)


def dress_source() -> str:
    """The shader laying a chunk over its woods (DRESS), with Elevation's numbers written in."""
    names = {
        "EXAGGERATION": glsl_number(relief.EXAGGERATION),
        "TILT": glsl_number(TILT),
        "MOST_LIFT": glsl_number(relief.MOST_LIFT),
        "MOST_LIFT_ROWS": str(relief.MOST_LIFT),
    }
    return fill_names(DRESS, names)


def fill_names(source: str, names: dict[str, str]) -> str:
    source = source.replace("@NOISE@", NOISE).replace("@HEX@", HEX)
    source = source.replace("@RASTER@", RASTER).replace("@TEXTURE@", TEXTURE)
    source = source.replace("@SUN@", SUN).replace("@NATURAL@", NATURAL)
    source = source.replace("@FIELDS@", FIELDS).replace("@STREAM@", STREAM)
    for name, value in (names | {"MASK": str(TABLE_SIZE - 1)}).items():
        source = source.replace(f"@{name}@", value)
    assert "@" not in source
    return source
