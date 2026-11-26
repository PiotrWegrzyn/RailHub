"""The fine texture laid over the ground at zooms where an art pixel covers several screen
pixels, so the ground reads as a lit surface of grass and earth instead of soft squares: grain,
tufts and patches of light and shade, blade strokes on grass, light flecks and a bump lit from
the left. It is a function of where on the map a point lies and repeats every PERIOD art pixels,
so the CPU (railroads.ui.pixel.world) can work it out once and tile it, and the ground shader
(railroads.ui.pixel.ground_shader) draws the same pattern.

The bare ground has finer detail still, one model of it at every zoom: blades of grass and
flowers in it, pebbles and crumbs in the earth, ripples on the water, all sized in art pixels.
Each shows as far as the pixels on screen can draw it and fades into its average where they are
too coarse (`visible`), so the closer the view, the more of it shows, while from further off
the ground keeps its colour."""

from functools import cache

import numpy as np
import numpy.typing as npt

# How many art pixels the pattern repeats over, each way; chunks are a whole number of these.
PERIOD = 64
# Each layer's frequency in waves per art pixel, across and down, and the lattice offset that
# keeps the layers apart; 64 times each frequency is a whole number, so every layer repeats.
GRAIN = (4.0, 4.0, 0.0)
TUFTS = (1.5, 1.5, 0.0)
PATCHES = (0.625, 0.625, 0.0)
STROKES = (5.0, 1.125, 17.0)
FLECKS = (3.5, 3.5, 41.0)
CRUMBS = (8.0, 8.0, 29.0)
SWELLS = (0.5, 3.0, 7.0)

# Blades of grass, two sizes of them: how wide and how high the square each grows from is in
# art pixels, how strongly they light and shade the grass, and the seed of their random numbers.
BLADES = ((0.5, 1.0, 0.5, 3), (0.25, 0.5, 0.4, 13))
# How much darker the ground between blades is than their roots.
BLADE_GAP = 0.3
# Flowers in the grass: one in this share of the squares FLOWER_GAP art pixels a side, its
# head FLOWER_SIZE art pixels across at most, in one of FLOWER_COLOURS with a FLOWER_EYE eye.
FLOWER_GAP = 2.0
FLOWER_SHARE = 0.09
FLOWER_SIZE = 0.42
FLOWER_COLOURS = (
    (0.93, 0.93, 0.88),
    (0.96, 0.84, 0.29),
    (0.77, 0.63, 0.89),
    (0.89, 0.44, 0.47),
    (0.67, 0.77, 0.94),
)
FLOWER_EYE = (0.95, 0.72, 0.2)
# Pebbles in the earth, one to a square PEBBLE_GAP art pixels a side in PEBBLE_SHARE of them.
PEBBLE_GAP = 0.5
PEBBLE_SHARE = 0.55
# Ripples on water: one to a box RIPPLE_BOX art pixels wide and high in RIPPLE_SHARE of them, a
# crest RIPPLE_CREST high.
RIPPLE_BOX = (4.0, 0.5)
RIPPLE_SHARE = 0.7
RIPPLE_CREST = 0.06
# Flowers grow only on grass at least this much greener than red or blue.
FLOWER_GREEN = 0.04
# Crops are ripe where their red and green are both at least this much over their blue.
RIPE = 0.3

_MASK = 0xFFFFFFFF

type Floats = npt.NDArray[np.float64]
type Ints = npt.NDArray[np.int64]


def _hash(i: Floats, j: Floats) -> Floats:
    value = np.sin(i * 127.1 + j * 311.7) * 43758.5453
    return value - np.floor(value)


def noise(x: Floats, y: Floats, layer: tuple[float, float, float]) -> Floats:
    """Smooth value noise of one layer at art points (x, y), repeating every PERIOD."""
    fx, fy, offset = layer
    qx, qy = x * fx + offset, y * fy + offset
    period_x, period_y = PERIOD * fx, PERIOD * fy
    ix, iy = np.floor(qx), np.floor(qy)
    tx, ty = qx - ix, qy - iy
    tx, ty = tx * tx * (3 - 2 * tx), ty * ty * (3 - 2 * ty)
    x0, x1 = np.mod(ix, period_x), np.mod(ix + 1, period_x)
    y0, y1 = np.mod(iy, period_y), np.mod(iy + 1, period_y)
    top = _hash(x0, y0) * (1 - tx) + _hash(x1, y0) * tx
    bottom = _hash(x0, y1) * (1 - tx) + _hash(x1, y1) * tx
    return top * (1 - ty) + bottom * ty


def square_hash(i: Ints, j: Ints, size: tuple[float, float], seed: int) -> Floats:
    """A random number from 0 to 1 for square (i, j) of a lattice of squares `size` art pixels
    big, the same every PERIOD, worked out in 32-bit whole numbers as the shader does."""
    i = np.mod(i, round(PERIOD / size[0])).astype(np.uint64)
    j = np.mod(j, round(PERIOD / size[1])).astype(np.uint64)
    h = ((i * 73856093) & _MASK) ^ ((j * 19349663) & _MASK) ^ ((seed * 83492791) & _MASK)
    h ^= h >> np.uint64(16)
    h = (h * 0x7FEB352D) & _MASK
    h ^= h >> np.uint64(15)
    h = (h * 0x846CA68B) & _MASK
    h ^= h >> np.uint64(16)
    return (h & 0xFFFFFF).astype(np.float64) / 16777216.0


def _smoothstep(low: float, high: float, value: Floats) -> Floats:
    t = np.clip((value - low) / (high - low), 0, 1)
    return t * t * (3 - 2 * t)


def visible(size: float, pixels: float) -> float:
    """How much of something `size` art pixels big shows at `pixels` screen pixels to an art
    pixel: none below a pixel, all from three."""
    t = min(max((size * pixels - 1.0) / 2.0, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def blades(x: Floats, y: Floats, width: float, height: float, seed: int) -> Floats:
    """How light the grass is at art points (x, y) from blades growing one from each square
    `width` by `height` art pixels, leaning and lit from the left: the frontmost blade over a
    point, dark at its root and light at its tip, or BLADE_GAP between them."""
    size = (width, height)
    ci, cj = np.floor(x / width).astype(np.int64), np.floor(y / height).astype(np.int64)
    value = np.full(np.shape(x), BLADE_GAP)
    front = np.full(np.shape(x), -np.inf)
    for dj in (0, 1, 2):
        for di in (-1, 0, 1):
            gi, gj = ci + di, cj + dj
            root_x = (gi + square_hash(gi, gj, size, seed)) * width
            root_y = (gj + square_hash(gi, gj, size, seed + 1)) * height
            tall = height * (1.0 + 0.5 * square_hash(gi, gj, size, seed + 2))
            t = (root_y - y) / tall
            lean = (square_hash(gi, gj, size, seed + 3) - 0.5) * 1.6 * width
            half = width * 0.34 * (1.0 - 0.8 * t)
            off = (x - root_x - lean * t * t) / np.maximum(half, 1e-6)
            hit = (t >= 0) & (t <= 1) & (np.abs(off) < 1) & (root_y > front)
            tone = square_hash(gi, gj, size, seed + 4) - 0.5
            lit = 0.5 + 0.55 * t + 0.25 * tone - 0.12 * off
            value = np.where(hit, lit, value)
            front = np.where(hit, root_y, front)
    return value


def flowers(x: Floats, y: Floats, pixels: float) -> tuple[Floats, Floats]:
    """The flowers in the grass at art points (x, y), seen at `pixels` screen pixels to an art
    pixel: their colour and how fully they cover the grass, none where too small to see."""
    size = (FLOWER_GAP, FLOWER_GAP)
    gi, gj = np.floor(x / FLOWER_GAP).astype(np.int64), np.floor(y / FLOWER_GAP).astype(np.int64)
    there = square_hash(gi, gj, size, 61) < FLOWER_SHARE
    cx = (gi + 0.25 + 0.5 * square_hash(gi, gj, size, 62)) * FLOWER_GAP
    cy = (gj + 0.25 + 0.5 * square_hash(gi, gj, size, 63)) * FLOWER_GAP
    radius = FLOWER_SIZE / 2 * (0.7 + 0.3 * square_hash(gi, gj, size, 64))
    pick = np.minimum(
        (square_hash(gi, gj, size, 65) * len(FLOWER_COLOURS)).astype(np.int64),
        len(FLOWER_COLOURS) - 1,
    )
    d = np.hypot(x - cx, (y - cy) * 1.4) / radius
    edge = 1.0 / (radius * max(pixels, 1e-6))
    cover = np.clip((1.0 - d) / edge, 0, 1) * there * visible(FLOWER_SIZE, pixels)
    colour = np.asarray(FLOWER_COLOURS)[pick]
    lit = 1.08 - 0.25 * np.clip((x - cx + y - cy) / radius, -1, 1)[..., None]
    colour = np.where((d < 0.38)[..., None], FLOWER_EYE, colour * lit)
    return colour, cover


def pebbles(x: Floats, y: Floats) -> Floats:
    """How much pebbles light (above 0) or shade the earth at art points (x, y), lit from the
    top left and casting a shadow to the bottom right."""
    size = (PEBBLE_GAP, PEBBLE_GAP)
    gi, gj = np.floor(x / PEBBLE_GAP).astype(np.int64), np.floor(y / PEBBLE_GAP).astype(np.int64)
    there = square_hash(gi, gj, size, 71) < PEBBLE_SHARE
    cx = (gi + 0.3 + 0.4 * square_hash(gi, gj, size, 72)) * PEBBLE_GAP
    cy = (gj + 0.3 + 0.4 * square_hash(gi, gj, size, 73)) * PEBBLE_GAP
    radius = 0.06 + 0.08 * square_hash(gi, gj, size, 74)
    tone = np.where(square_hash(gi, gj, size, 75) > 0.5, 0.7, -0.5)
    dx, dy = (x - cx) / radius, (y - cy) / radius
    inside = dx * dx + dy * dy < 1
    shadow = (dx - 0.45) ** 2 + (dy - 0.45) ** 2 < 1
    stone = tone - 0.35 * np.clip(dx + dy, -1.5, 1.5)
    return there * np.where(inside, stone, np.where(shadow, -0.6, 0.0))


def ripples(x: Floats, y: Floats) -> Floats:
    """How much the ripples light (above 0) or shade water at art points (x, y): short crests
    across the water, each with a dark trough below it, tapering to their ends."""
    width, height = RIPPLE_BOX
    ci, cj = np.floor(x / width).astype(np.int64), np.floor(y / height).astype(np.int64)
    value = np.zeros(np.shape(x))
    for dj in (-1, 0):
        for di in (-1, 0, 1):
            gi, gj = ci + di, cj + dj
            cx = (gi + square_hash(gi, gj, RIPPLE_BOX, 81)) * width
            cy = (gj + 0.2 + 0.6 * square_hash(gi, gj, RIPPLE_BOX, 82)) * height
            length = 0.5 + 1.2 * square_hash(gi, gj, RIPPLE_BOX, 83)
            there = square_hash(gi, gj, RIPPLE_BOX, 84) < RIPPLE_SHARE
            taper = np.maximum(1.0 - ((x - cx) / length) ** 2, 0.0) * there
            dy = y - cy
            crest = np.abs(dy) < RIPPLE_CREST
            trough = (dy >= RIPPLE_CREST) & (dy < 3 * RIPPLE_CREST)
            value += taper * np.where(crest, 1.0, np.where(trough, -0.6, 0.0))
    return value


@cache
def _means() -> tuple[float, ...]:
    """The average of each detail over the ground: what it fades into where too small to see."""
    rng = np.random.default_rng(1)
    x, y = rng.uniform(0, PERIOD, 40_000), rng.uniform(0, PERIOD, 40_000)
    found = [float(blades(x, y, w, h, seed).mean()) for w, h, _, seed in BLADES]
    return (*found, float(pebbles(x, y).mean()), float(ripples(x, y).mean()))


def blade_means() -> tuple[float, ...]:
    """The average lightness of each size of BLADES."""
    return _means()[: len(BLADES)]


def pebble_mean() -> float:
    return _means()[len(BLADES)]


def ripple_mean() -> float:
    return _means()[len(BLADES) + 1]


def _blades_over(x: Floats, y: Floats, pixels: float) -> Floats:
    """How much the BLADES light and shade plants at art points (x, y)."""
    more = np.ones_like(x)
    for (width, height, strength, seed), mean in zip(BLADES, blade_means(), strict=True):
        shown = visible(width, pixels)
        if shown > 0:
            more += strength * shown * (blades(x, y, width, height, seed) - mean)
    return more


@cache
def tile(k: int, bare: bool = False, crops: bool = False) -> dict[str, Floats]:
    """The pattern over one PERIOD of art pixels, `k` pixels for each: how much to shade grass,
    earth, and anything else by at each pixel; on `bare` ground or `crops` how much the blades
    of plants shade them by, and on `bare` ground how much pebbles shade earth and ripples water
    and the flowers' colours and cover."""
    at = (np.arange(PERIOD * k) + 0.5) / k
    x, y = np.meshgrid(at, at)
    grain, tufts, patches = noise(x, y, GRAIN), noise(x, y, TUFTS), noise(x, y, PATCHES)
    strokes, flecks = noise(x, y, STROKES), noise(x, y, FLECKS)
    bump = noise(x + 0.3, y, TUFTS) - noise(x - 0.3, y, TUFTS)
    light = np.clip(1 - bump * 0.5, 0.78, 1.22)
    shade = (0.76 + 0.1 * grain + 0.14 * tufts + 0.1 * patches) * (
        1 + 0.12 * _smoothstep(0.75, 0.9, flecks)
    )
    shade *= 1 + 0.12 * visible(1 / CRUMBS[0], k) * (noise(x, y, CRUMBS) - 0.5)
    pattern = {
        "grass": shade * (1 - 0.2 * _smoothstep(0.62, 0.8, strokes)) * light,
        "earth": shade * light,
    }
    if bare or crops:
        pattern["blades"] = _blades_over(x, y, k)
    if bare:
        pattern["earth"] = pattern["earth"] * (
            1.0 + 0.22 * visible(0.2, k) * (pebbles(x, y) - pebble_mean())
        )
        water = 1.0 + 0.1 * visible(1 / SWELLS[1], k) * (noise(x, y, SWELLS) - 0.5)
        shown = visible(2 * RIPPLE_CREST, k)
        if shown > 0:
            water = water + 0.3 * shown * (ripples(x, y) - ripple_mean())
        pattern["water"] = water
        pattern["flower"], pattern["bloom"] = flowers(x, y, k)
    return pattern


def _tiled(pattern: Floats, left: int, top: int, k: int, shape: tuple[int, ...]) -> Floats:
    rows, cols = shape[:2]
    span = PERIOD * k
    oy, ox = (top % PERIOD) * k, (left % PERIOD) * k
    reps = (-(-(rows + oy) // span), -(-(cols + ox) // span)) + (1,) * (pattern.ndim - 2)
    return np.tile(pattern, reps)[oy : oy + rows, ox : ox + cols]


def textured(
    canvas: Floats, left: int, top: int, k: int, bare: bool = False, crops: bool = False
) -> None:
    """Lays the texture over a chunk's ground, `canvas` premultiplied RGBA with `k` pixels for
    every art pixel from (`left`, `top`), which lie on whole PERIODs; grass takes it all, other
    ground most of it and water little. On `bare` ground, not under fields, yards or woods, its
    finer detail too: blades and flowers in the grass, pebbles in the earth, ripples on water;
    on `crops`, fields and yards out of the woods, blades on all that grows, green or ripe."""
    alpha = np.maximum(canvas[..., 3], 1e-6)
    r, g, b = (canvas[..., c] / alpha for c in range(3))
    green = g - np.maximum(r, b)
    wet = b - np.maximum(r, g)
    pattern = {
        name: _tiled(values, left, top, k, canvas.shape)
        for name, values in tile(k, bare, crops).items()
    }
    factor = np.where(green > 0, pattern["grass"], 1 + (pattern["earth"] - 1) * 0.7)
    if bare:
        factor = np.where(green > 0, factor * pattern["blades"], factor)
    if crops:
        grows = (green > 0) | (np.minimum(r, g) - b > RIPE)
        factor = np.where(grows, factor * pattern["blades"], factor)
    factor = np.where(wet > 0.02, 1 + (factor - 1) * 0.25, factor)
    if bare:
        factor = np.where(wet > 0.02, factor * pattern["water"], factor)
    canvas[..., :3] *= factor[..., None]
    if bare:
        cover = (pattern["bloom"] * (green > FLOWER_GREEN))[..., None]
        canvas[..., :3] += (pattern["flower"] * canvas[..., 3:] - canvas[..., :3]) * cover
