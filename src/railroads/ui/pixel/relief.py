"""Hills and mountains as raised, lit ground in the pixel map.

Every point of the map has a height: the map's own heights, smoothed into broad valleys, with
ridges and valleys carved in where the land is mountainous (ridged noise, twisted by more noise so
the ridges wind) and gentle rolling ground on hills. Only hills and mountains rise. The picture
lifts every point by its height, EXAGGERATION times as tall as true, so peaks stand up and hide
what lies behind them: each pixel shows the nearest point raised up to it (raycast).

Raised ground is lit by its slopes from LIGHT, low in the top left, and coloured by its height
above the valley and its steepness: green foothills, bare rock above the treeline, and snow above a
wavy snowline, broken by ribs of rock on steep slopes. All of it is worked out at one pixel per art
pixel and blown up with the rest of the ground.
"""

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from railroads.sim.tilemap import Relief, Terrain, TileMap
from railroads.ui.iso_camera import TILT
from railroads.ui.pixel.noise import noise, ridged
from railroads.ui.pixel.projection import FloatArray, PixelProjection

Floats = npt.NDArray[np.float32]

# Heights are kept on a raster this many steps per cell, smoothed over SMOOTHING steps, and the
# valley floor under them over VALLEY_SMOOTHING steps.
STEPS = 4
SMOOTHING = 6
# How far, in STEPS, every mountain cell's own peak is spread, so that a lone mountain cell still
# rises to a peak in its cell rather than spreading thin over its neighbours.
PEAK_SMOOTHING = 1
VALLEY_SMOOTHING = 24
EXAGGERATION = 3.2
# The most anything is lifted, in art pixels.
MOST_LIFT = 64
# How many points' lift Elevation.lift_at keeps before it forgets them all.
LIFTS_KEPT = 1 << 17
# How mountains rise above the valley, in metres: at least MOUNTAIN_FLOOR, up to MOUNTAIN_RIDGES
# more along the ridges; hills roll up to HILL_ROLL.
MOUNTAIN_FLOOR = 200.0
MOUNTAIN_RIDGES = 1700.0
HILL_ROLL = 140.0
LIGHT = np.array([-0.68, -0.38, 0.5]) / math.sqrt(0.68**2 + 0.38**2 + 0.5**2)
ROCK_LIT = np.array((150, 132, 112), dtype=np.float32)
ROCK_DARK = np.array((92, 84, 80), dtype=np.float32)
SNOW_LIT = np.array((246, 248, 252), dtype=np.float32)
SNOW_DARK = np.array((150, 168, 198), dtype=np.float32)
FOOTHILL = np.array((74, 98, 54), dtype=np.float32)
SHADOW_TINT = np.array((0.78, 0.84, 1.0), dtype=np.float32)
# Metres above the valley where bare rock starts and snow lies, each varying by its spread.
TREELINE = 300.0
TREELINE_SPREAD = 160.0
SNOWLINE = 950.0
SNOWLINE_SPREAD = 260.0


def smooth(values: FloatArray, radius: int) -> FloatArray:
    """Two passes of a box blur `radius` steps either side, along both axes."""
    if radius < 1:
        return values
    for axis in (0, 1, 0, 1):
        padded = np.pad(
            values, [(radius + 1, radius) if a == axis else (0, 0) for a in (0, 1)], mode="edge"
        )
        sums = np.cumsum(padded, axis=axis)
        width = 2 * radius + 1
        upper = sums[width:] if axis == 0 else sums[:, width:]
        lower = sums[:-width] if axis == 0 else sums[:, :-width]
        values = (upper - lower) / (2 * radius + 1)
    return values


def sample(raster: FloatArray, x: FloatArray, y: FloatArray) -> FloatArray:
    """The raster at fractional raster positions, blended from its four nearest steps."""
    fx, fy = np.clip(x - 0.5, 0, raster.shape[1] - 1), np.clip(y - 0.5, 0, raster.shape[0] - 1)
    x0, y0 = (
        np.minimum(fx.astype(np.int64), raster.shape[1] - 2),
        np.minimum(fy.astype(np.int64), raster.shape[0] - 2),
    )
    tx, ty = fx - x0, fy - y0
    top = raster[y0, x0] * (1 - tx) + raster[y0, x0 + 1] * tx
    bottom = raster[y0 + 1, x0] * (1 - tx) + raster[y0 + 1, x0 + 1] * tx
    return top * (1 - ty) + bottom * ty


@dataclass(frozen=True)
class Raised:
    """The raised ground over a block of art pixels: how far each is lifted, how high it rises
    above the valley, how mountainous and how hilly it is, how much light its slope catches and
    how steep it is."""

    lift: Floats
    rise: Floats
    mountains: Floats
    hills: Floats
    shade: Floats
    steep: Floats

    def top(self, rows: int) -> Raised:
        """The same over only the first `rows` rows."""
        return Raised(*(getattr(self, name)[:rows] for name in self.__dataclass_fields__))


def lighting(raised: Raised) -> Floats:
    """How much brighter or darker each pixel of raised ground is for the way its slope faces
    the light, as a factor: 1 on flat ground."""
    m, h, s = raised.mountains, raised.hills, raised.shade
    lit = np.clip(1 + 1.3 * (s - np.float32(LIGHT[2])), 0.55, 1.3)
    return 1 + (lit - 1) * np.clip((m + h) * 4, 0, 1)


class Elevation:
    """The height of every point of one map, and how its hills and mountains are drawn."""

    def __init__(self, tile_map: TileMap, projection: PixelProjection) -> None:
        self.projection = projection
        grid = tile_map.grid
        width, height = grid.size()
        ys, xs = np.mgrid[0 : math.ceil(height * STEPS), 0 : math.ceil(width * STEPS)]
        rows, cols = grid.cell_at((xs + 0.5) / STEPS, (ys + 0.5) / STEPS)
        rows, cols = np.clip(rows, 0, grid.rows - 1), np.clip(cols, 0, grid.cols - 1)
        land = tile_map.terrain[rows, cols] == Terrain.LAND
        relief = tile_map.relief[rows, cols]
        heights = np.where(land, tile_map.height_m[rows, cols], 0).astype(np.float32)
        self._base = smooth(heights, SMOOTHING).astype(np.float32)
        self._valley = smooth(self._base, VALLEY_SMOOTHING).astype(np.float32)
        mountains = (land & (relief == Relief.MOUNTAINS)).astype(np.float32)
        self._mountains = smooth(mountains, SMOOTHING).astype(np.float32)
        self._peaks = smooth(mountains, PEAK_SMOOTHING).astype(np.float32)
        self._hills = smooth((land & (relief == Relief.HILLS)).astype(np.float32), 3)
        self._hills = self._hills.astype(np.float32)
        self._metres_per_pixel = grid.km_per_cell * 1000 / float(np.hypot(*projection.matrix[:, 0]))
        self._metres_per_unit = grid.km_per_cell * 1000
        self._lifts: dict[tuple[float, float], float] = {}

    def shares(self, x: FloatArray, y: FloatArray) -> tuple[Floats, Floats]:
        """How mountainous and how hilly the ground is at plane points, from 0 to 1."""
        m = np.clip(sample(self._mountains, x * STEPS, y * STEPS) * 2.4 - 0.15, 0, 1)
        peak = np.clip(sample(self._peaks, x * STEPS, y * STEPS) * 2.2 - 0.25, 0, 1)
        m = np.maximum(m, peak)
        h = sample(self._hills, x * STEPS, y * STEPS)
        return m.astype(np.float32), h.astype(np.float32)

    def rise(self, x: npt.ArrayLike, y: npt.ArrayLike) -> tuple[Floats, Floats, Floats]:
        """How many metres plane points rise above their valley, and how mountainous and hilly
        they are; only hills and mountains rise."""
        x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        m, h = self.shares(x, y)
        rise = np.zeros(m.shape, dtype=np.float32)
        raised = (m > 0) | (h > 0)
        if raised.any():
            rx, ry, rm, rh = x[raised], y[raised], m[raised], h[raised]
            base = sample(self._base, rx * STEPS, ry * STEPS)
            valley = sample(self._valley, rx * STEPS, ry * STEPS)
            wx = rx + 0.9 * (noise(rx, ry, 2.5, 11) - 0.5)
            wy = ry + 0.9 * (noise(rx, ry, 2.5, 12) - 0.5)
            big = ridged(wx, wy, 2.6, 1)
            ridges = 0.55 * big + 0.28 * big * ridged(wx, wy, 1.0, 2)
            ridges = ridges + 0.12 * ridged(wx, wy, 0.42, 3) + 0.05 * ridged(wx, wy, 0.16, 6)
            rolling = 0.6 * noise(rx, ry, 0.9, 4) + 0.4 * noise(rx, ry, 0.35, 5)
            lifted = np.clip(rm + rh, 0, 1) * np.maximum(base - valley, 0) * 0.5
            lifted = lifted + rm**1.6 * (MOUNTAIN_FLOOR + MOUNTAIN_RIDGES * ridges)
            rise[raised] = lifted + rh * (1 - rm) * HILL_ROLL * rolling
        return rise, m, h

    def to_lift(self, rise: Floats) -> Floats:
        """How many art pixels ground that rises so many metres is lifted in the picture."""
        lift = rise / self._metres_per_pixel * EXAGGERATION * TILT
        return np.clip(lift, 0, MOST_LIFT).astype(np.float32)

    def lift(self, x: npt.ArrayLike, y: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """How many art pixels plane points are lifted in the picture."""
        rise, _, _ = self.rise(x, y)
        return self.to_lift(rise).astype(np.float64)

    def lift_at(self, x: float, y: float) -> float:
        """How many art pixels one plane point is lifted, remembered for the points marks on the
        map ask about frame after frame, up to LIFTS_KEPT of them."""
        lift = self._lifts.get((x, y))
        if lift is None:
            if len(self._lifts) >= LIFTS_KEPT:
                self._lifts.clear()
            lift = self._lifts[(x, y)] = float(self.lift(x, y))
        return lift

    def area(
        self, left: int, top: int, width: int, rows: int, resolution: int = 1
    ) -> Raised | None:
        """The raised ground over `width` by `rows` art pixels from (`left`, `top`), at the
        middle of each of `resolution` pixels a side in every art pixel, with its lift in those
        pixels; None where nothing there rises."""
        k = resolution
        py, px = np.mgrid[0 : rows * k, 0 : width * k]
        x, y = self.projection.to_plane(left + (px + 0.5) / k, top + (py + 0.5) / k)
        rise, m, h = self.rise(x, y)
        if not (m > 0).any() and not (h > 0).any():
            return None
        height = sample(self._valley, x * STEPS, y * STEPS).astype(np.float32) + rise
        shade, steep = self._light(height, k)
        return Raised(self.to_lift(rise) * k, rise, m, h, shade, steep)

    def rises_in(self, left: int, top: int, width: int, rows: int) -> bool:
        """Whether any ground over these art pixels rises, from a sample every few pixels."""
        py, px = np.mgrid[0:rows:4, 0:width:4]
        x, y = self.projection.to_plane(left + px + 0.5, top + py + 0.5)
        m, h = self.shares(x, y)
        return bool((m > 0).any() or (h > 0).any())

    def _light(self, height: Floats, resolution: int = 1) -> tuple[Floats, Floats]:
        """How much light ground of these heights, `resolution` to an art pixel each way,
        catches, and how steep it is (0 flat, nearing 1 sheer)."""
        per_row, per_col = np.gradient(height)
        per_row, per_col = per_row * resolution, per_col * resolution
        # A plane step moves the pixel by the projection's columns, so height per plane unit is
        # height per pixel along them.
        m = self.projection.matrix
        gx = (per_col * m[0, 0] + per_row * m[1, 0]) / self._metres_per_unit
        gy = (per_col * m[0, 1] + per_row * m[1, 1]) / self._metres_per_unit
        nz = 1.0 / np.sqrt(1 + gx**2 + gy**2)
        shade = np.clip((-gx * LIGHT[0] - gy * LIGHT[1] + LIGHT[2]) * nz, 0, 1)
        return shade.astype(np.float32), (1 - nz).astype(np.float32)

    def colour(
        self,
        rgb: Floats,
        raised: Raised,
        x: FloatArray,
        y: FloatArray,
        rocky: npt.NDArray[np.bool_],
    ) -> None:
        """Lights and colours raised ground in place: `rgb` is the flat ground's colour over the
        same art pixels, at plane points `x`, `y`; `rocky` marks mountain cells, which take the
        mountains' colours fully once their mountains start to rise, so a lone low mountain cell
        does not show its edges."""
        m, h, s, steep = raised.mountains, raised.hills, raised.shade, raised.steep
        lit = lighting(raised)
        rgb *= lit[..., None]
        under = np.maximum(m, rocky * np.clip(m * 3, 0, 1))
        chosen = under > 0
        if not chosen.any():
            return
        s, st, above, mm, hh = s[chosen], steep[chosen], raised.rise[chosen], m[chosen], h[chosen]
        xs, ys = x[chosen], y[chosen]
        u = under[chosen][:, None]
        foothill = FOOTHILL * (0.9 + 0.2 * noise(xs, ys, 0.3, 13))[:, None]
        col = rgb[chosen] * (1 - u) + foothill * u * lit[chosen][:, None]
        grain = noise(xs, ys, 0.05, 7)[:, None]
        rock = ROCK_DARK + (ROCK_LIT - ROCK_DARK) * np.clip(s * 1.3 - 0.2, 0, 1)[:, None]
        rock = rock * (0.9 + 0.2 * grain)
        rock = np.where((st > 0.55)[:, None], rock * 0.85, rock)
        treeline = TREELINE + TREELINE_SPREAD * noise(xs, ys, 0.7, 8)
        bare = (np.clip((above - treeline) / 200 + st * 1.4 - 0.3, 0, 1) * mm)[:, None]
        col = col * (1 - bare) + rock * bare
        snowline = SNOWLINE + SNOWLINE_SPREAD * noise(xs, ys, 0.6, 9) - 140 * (1 - s)
        snow = np.clip((above - snowline) / 140, 0, 1) * np.clip(1.8 - st * 2.0, 0, 1) * mm
        ribs = (ridged(xs * 3.0, ys * 0.8, 0.12, 14) > 0.55) & (st > 0.3)
        snow = np.where(ribs, snow * 0.25, snow)[:, None]
        snowy = SNOW_DARK + (SNOW_LIT - SNOW_DARK) * np.clip(s * 1.5 - 0.35, 0, 1)[:, None]
        col = col * (1 - snow) + snowy * snow
        dark = (np.clip((0.45 - s) * 2, 0, 1) * (mm + 0.3 * hh))[:, None]
        rgb[chosen] = col * (1 - dark * (1 - SHADOW_TINT) * 0.6)

    def ground_at(self, px: float, py: float) -> tuple[float, float]:
        """The plane point that shows at an art pixel: the nearest one raised up to it."""
        rows = py + np.arange(MOST_LIFT + 1)
        x, y = self.projection.to_plane(np.full(rows.shape, px), rows)
        reach = self.lift(x, y) >= np.arange(MOST_LIFT + 1)
        d = int(np.flatnonzero(reach)[-1]) if reach.any() else 0
        return float(x[d]), float(y[d])


def raycast(lift: Floats, rows: int, cols: int) -> npt.NDArray[np.int64]:
    """How many rows below each of `rows` by `cols` pixels lies the point that shows there: the
    nearest one whose lift reaches up to it. `lift` covers at least the highest lift's rows more."""
    offset = np.zeros((rows, cols), dtype=np.int64)
    found = np.zeros((rows, cols), dtype=bool)
    for d in range(int(lift.max()), -1, -1):
        reach = lift[d : d + rows, :cols] >= d
        offset[reach & ~found] = d
        found |= reach
    return offset


def raise_picture(
    picture: FloatArray, offset: npt.NDArray[np.int64], resolution: int
) -> FloatArray:
    """The picture as seen with its ground lifted: each pixel taken from the rows `offset` art
    pixels below it, `picture` holding `resolution` pixels for every art pixel and enough rows."""
    k = resolution
    rows, cols = offset.shape[0] * k, offset.shape[1] * k
    fine = np.repeat(np.repeat(offset, k, axis=0), k, axis=1) * k
    shown = np.minimum(np.arange(rows)[:, None] + fine, picture.shape[0] - 1)
    return picture[shown, np.arange(cols)[None, :]]
