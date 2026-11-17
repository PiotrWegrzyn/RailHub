"""The fine texture laid over the ground at zooms where an art pixel covers several screen
pixels, so the ground reads as a lit surface of grass and earth instead of soft squares: grain,
tufts and patches of light and shade, blade strokes on grass, light flecks and a bump lit from
the left. It is a function of where on the map a point lies and repeats every PERIOD art pixels,
so the CPU (railroads.ui.pixel.world) can work it out once and tile it, and the ground shader
(railroads.ui.pixel.ground_shader) draws the same pattern."""

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

type Floats = npt.NDArray[np.float64]


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


def _smoothstep(low: float, high: float, value: Floats) -> Floats:
    t = np.clip((value - low) / (high - low), 0, 1)
    return t * t * (3 - 2 * t)


@cache
def tile(k: int) -> tuple[Floats, Floats, Floats]:
    """The pattern over one PERIOD of art pixels, `k` pixels for each: how much to shade grass,
    earth, and anything else by at each pixel."""
    at = (np.arange(PERIOD * k) + 0.5) / k
    x, y = np.meshgrid(at, at)
    grain, tufts, patches = noise(x, y, GRAIN), noise(x, y, TUFTS), noise(x, y, PATCHES)
    strokes, flecks = noise(x, y, STROKES), noise(x, y, FLECKS)
    bump = noise(x + 0.3, y, TUFTS) - noise(x - 0.3, y, TUFTS)
    light = np.clip(1 - bump * 0.5, 0.78, 1.22)
    shade = (0.76 + 0.1 * grain + 0.14 * tufts + 0.1 * patches) * (
        1 + 0.12 * _smoothstep(0.75, 0.9, flecks)
    )
    grass = shade * (1 - 0.2 * _smoothstep(0.62, 0.8, strokes)) * light
    earth = shade * light
    return grass, earth, light


def textured(canvas: Floats, left: int, top: int, k: int) -> None:
    """Lays the texture over a chunk's ground, `canvas` premultiplied RGBA with `k` pixels for
    every art pixel from (`left`, `top`), which lie on whole PERIODs; grass takes it all, other
    ground most of it and water little."""
    rows, cols = canvas.shape[:2]
    grass, earth, _ = tile(k)
    span = PERIOD * k
    oy, ox = (top % PERIOD) * k, (left % PERIOD) * k
    reps = (-(-(rows + oy) // span), -(-(cols + ox) // span))
    grass = np.tile(grass, reps)[oy : oy + rows, ox : ox + cols]
    earth = np.tile(earth, reps)[oy : oy + rows, ox : ox + cols]
    alpha = np.maximum(canvas[..., 3], 1e-6)
    r, g, b = (canvas[..., c] / alpha for c in range(3))
    green = g - np.maximum(r, b)
    wet = b - np.maximum(r, g)
    factor = np.where(green > 0, grass, 1 + (earth - 1) * 0.7)
    factor = np.where(wet > 0.02, 1 + (factor - 1) * 0.25, factor)
    canvas[..., :3] *= factor[..., None]
