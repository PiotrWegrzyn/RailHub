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
from railroads.ui.pixel.lattice import Lattice
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
# How mountainous ground is from its share of mountain cells round it: that share times the gain
# less where it starts, or more where the peak of a single mountain cell stands.
MOUNTAIN_GAIN = 2.4
MOUNTAIN_FROM = 0.15
PEAK_GAIN = 2.2
PEAK_FROM = 0.25
# How far over the most a bound on how mountainous ground is reaches, far more than rounding in
# blending four steps of a raster can add.
BOUND_MARGIN = 1e-6
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
    return Blend.at(raster.shape, x, y).read(raster)


class Blend:
    """Where fractional positions fall on rasters of one shape and how much each of the four
    steps round them counts, worked out once to read several rasters at the same positions:
    the flat index of the step before and above each, and how far past it each lies."""

    def __init__(
        self, shape: tuple[int, int], corner: npt.NDArray[np.int64], tx: FloatArray, ty: FloatArray
    ) -> None:
        self.shape = shape
        self._corner, self._tx, self._ty = corner, tx, ty
        self._sx, self._sy = 1 - tx, 1 - ty

    @classmethod
    def at(cls, shape: tuple[int, ...], x: npt.ArrayLike, y: npt.ArrayLike) -> Blend:
        """The blend at raster positions `x`, `y` on rasters of this shape."""
        height, width = shape
        fx = np.asarray(np.clip(np.subtract(x, 0.5), 0, width - 1))
        fy = np.asarray(np.clip(np.subtract(y, 0.5), 0, height - 1))
        x0 = np.minimum(fx.astype(np.int64), width - 2)
        y0 = np.minimum(fy.astype(np.int64), height - 2)
        corner = np.asarray(y0 * width + x0)
        return cls((height, width), corner, np.asarray(fx - x0), np.asarray(fy - y0))

    def where(self, chosen: npt.NDArray[np.bool_]) -> Blend:
        """The same positions, only those chosen."""
        return Blend(self.shape, self._corner[chosen], self._tx[chosen], self._ty[chosen])

    def read(self, raster: FloatArray) -> FloatArray:
        """The raster at these positions."""
        if raster.shape != self.shape:
            raise ValueError("A blend reads only rasters of the shape it was worked out for")
        flat = raster.ravel()
        corner = self._corner
        top = flat[corner] * self._sx
        top += flat[corner + 1] * self._tx
        below = corner + self.shape[1]
        bottom = flat[below] * self._sx
        bottom += flat[below + 1] * self._tx
        top *= self._sy
        bottom *= self._ty
        top += bottom
        return top


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
        return self._shares(Blend.at(self._mountains.shape, x * STEPS, y * STEPS))

    def mountainous(self, x: FloatArray, y: FloatArray) -> Floats:
        """How mountainous the ground is at plane points, from 0 to 1, as `shares` says."""
        return self._mountainous(Blend.at(self._mountains.shape, x * STEPS, y * STEPS))

    def _mountainous(self, at: Blend) -> Floats:
        m = np.clip(at.read(self._mountains) * MOUNTAIN_GAIN - MOUNTAIN_FROM, 0, 1)
        peak = np.clip(at.read(self._peaks) * PEAK_GAIN - PEAK_FROM, 0, 1)
        return np.maximum(m, peak).astype(np.float32)

    def _shares(self, at: Blend) -> tuple[Floats, Floats]:
        return self._mountainous(at), at.read(self._hills).astype(np.float32)

    def most(self, left: float, top: float, width: float, rows: float) -> tuple[float, float]:
        """At least as mountainous and as hilly as any ground over these art pixels is, from
        the most of each that the steps of the rasters read there hold: a cheap bound, a hair
        over, that lets a block where nothing can rise skip working out its points. Where the
        bound is 0 or less, no ground there is mountainous or hilly at all."""
        px = np.array([left, left + width, left, left + width], dtype=np.float64)
        py = np.array([top, top, top + rows, top + rows], dtype=np.float64)
        x, y = self.projection.to_plane(px, py)
        height, width_steps = self._mountains.shape
        steps = (
            _steps_read(float(y.min()) * STEPS, float(y.max()) * STEPS, height),
            _steps_read(float(x.min()) * STEPS, float(x.max()) * STEPS, width_steps),
        )
        mountains = float(self._mountains[steps].max()) * MOUNTAIN_GAIN - MOUNTAIN_FROM
        peaks = float(self._peaks[steps].max()) * PEAK_GAIN - PEAK_FROM
        return max(mountains, peaks) + BOUND_MARGIN, float(self._hills[steps].max())

    def may_rise(self, left: float, top: float, width: float, rows: float) -> bool:
        """Whether any ground over these art pixels may rise, by `most`: False only where none
        can."""
        mountains, hills = self.most(left, top, width, rows)
        return mountains > 0 or hills > 0

    def rise(self, x: npt.ArrayLike, y: npt.ArrayLike) -> tuple[Floats, Floats, Floats]:
        """How many metres plane points rise above their valley, and how mountainous and hilly
        they are; only hills and mountains rise."""
        x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        return self._rise(x, y, Blend.at(self._base.shape, x * STEPS, y * STEPS))

    def _rise(
        self, x: FloatArray, y: FloatArray, at: Blend, valley: FloatArray | None = None
    ) -> tuple[Floats, Floats, Floats]:
        """`rise` at points `at` blends, with the valley floor there if it is read already."""
        m, h = self._shares(at)
        rise = np.zeros(m.shape, dtype=np.float32)
        raised = (m > 0) | (h > 0)
        if raised.any():
            rx, ry, rm, rh = x[raised], y[raised], m[raised], h[raised]
            there = at.where(raised)
            base = there.read(self._base)
            valley = there.read(self._valley) if valley is None else valley[raised]
            lifted = np.clip(rm + rh, 0, 1) * np.maximum(base - valley, 0) * 0.5
            # Ridges lift only mountains and rolling ground only hills; elsewhere each adds
            # nothing, so neither is worked out there.
            mountains = rm > 0
            if mountains.any():
                ridges = _ridges(rx[mountains], ry[mountains])
                lifted[mountains] += rm[mountains] ** 1.6 * (
                    MOUNTAIN_FLOOR + MOUNTAIN_RIDGES * ridges
                )
            rolls = rh * (1 - rm) * HILL_ROLL
            rolling = rolls != 0
            if rolling.any():
                fx, fy = rx[rolling].astype(np.float32), ry[rolling].astype(np.float32)
                ground = 0.6 * noise(fx, fy, 0.9, 4) + 0.4 * noise(fx, fy, 0.35, 5)
                lifted[rolling] += rolls[rolling] * ground
            rise[raised] = lifted
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
        self,
        left: int,
        top: int,
        width: int,
        rows: int,
        resolution: int = 1,
        points: Lattice | None = None,
    ) -> Raised | None:
        """The raised ground over `width` by `rows` art pixels from (`left`, `top`), at the
        middle of each of `resolution` pixels a side in every art pixel, with its lift in those
        pixels; None where nothing there rises. `points` may hold their plane points already."""
        if not self.may_rise(left, top, width, rows):
            return None
        k = resolution
        if points is None or points.k != k or not points.holds(left, top, width, rows):
            points = Lattice(self.projection, left, top, width, rows, k)
        x, y = points.plane(left, top, width, rows)
        at = Blend.at(self._base.shape, x * STEPS, y * STEPS)
        valley = at.read(self._valley)
        rise, m, h = self._rise(x, y, at, valley)
        if not (m > 0).any() and not (h > 0).any():
            return None
        height = valley.astype(np.float32) + rise
        shade, steep = self._light(height, k)
        return Raised(self.to_lift(rise) * k, rise, m, h, shade, steep)

    def rises_in(self, left: int, top: int, width: int, rows: int) -> bool:
        """Whether any ground over these art pixels rises, from a sample every few pixels."""
        if not self.may_rise(left, top, width, rows):
            return False
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
        under = mountain_cover(m, rocky)
        chosen = under > 0
        if not chosen.any():
            return
        s, st, above, mm, hh = s[chosen], steep[chosen], raised.rise[chosen], m[chosen], h[chosen]
        xs, ys = x[chosen], y[chosen]
        fx, fy = xs.astype(np.float32), ys.astype(np.float32)
        u = under[chosen][:, None]
        foothill = FOOTHILL * (0.9 + 0.2 * noise(fx, fy, 0.3, 13))[:, None]
        col = rgb[chosen] * (1 - u) + foothill * u * lit[chosen][:, None]
        # Bare rock and snow cover ground only above the treeline and the snowline, which never
        # lie below TREELINE and SNOWLINE less 140 metres, and ground they don't cover keeps its
        # colour exactly; so their noise is worked out only where they may show.
        bare = (above - np.float64(TREELINE)) / 200 + st * 1.4 - 0.3 > 0
        if bare.any():
            at = np.flatnonzero(bare)
            treeline = TREELINE + TREELINE_SPREAD * noise(fx[at], fy[at], 0.7, 8)
            cover = np.clip((above[at] - treeline) / 200 + st[at] * 1.4 - 0.3, 0, 1) * mm[at]
            grain = noise(fx[at], fy[at], 0.05, 7)[:, None]
            rock = ROCK_DARK + (ROCK_LIT - ROCK_DARK) * np.clip(s[at] * 1.3 - 0.2, 0, 1)[:, None]
            rock = rock * (0.9 + 0.2 * grain)
            rock = np.where((st[at] > 0.55)[:, None], rock * 0.85, rock)
            col[at] = col[at] * (1 - cover[:, None]) + rock * cover[:, None]
        lying = np.clip(1.8 - st * 2.0, 0, 1)
        snowy = ((above - (np.float64(SNOWLINE) - 140 * (1 - s))) / 140 > 0) & (lying > 0)
        if snowy.any():
            at = np.flatnonzero(snowy)
            snowline = (
                SNOWLINE + SNOWLINE_SPREAD * noise(fx[at], fy[at], 0.6, 9) - 140 * (1 - s[at])
            )
            snow = np.clip((above[at] - snowline) / 140, 0, 1) * lying[at] * mm[at]
            ribs = st[at] > 0.3
            ribs[ribs] = ridged(xs[at][ribs] * 3.0, ys[at][ribs] * 0.8, 0.12, 14) > 0.55
            snow = np.where(ribs, snow * 0.25, snow)[:, None]
            white = SNOW_DARK + (SNOW_LIT - SNOW_DARK) * np.clip(s[at] * 1.5 - 0.35, 0, 1)[:, None]
            col[at] = col[at] * (1 - snow) + white * snow
        dark = (np.clip((0.45 - s) * 2, 0, 1) * (mm + 0.3 * hh))[:, None]
        rgb[chosen] = col * (1 - dark * (1 - SHADOW_TINT) * 0.6)

    def ground_at(self, px: float, py: float) -> tuple[float, float]:
        """The plane point that shows at an art pixel: the nearest one raised up to it."""
        rows = py + np.arange(MOST_LIFT + 1)
        x, y = self.projection.to_plane(np.full(rows.shape, px), rows)
        reach = self.lift(x, y) >= np.arange(MOST_LIFT + 1)
        d = int(np.flatnonzero(reach)[-1]) if reach.any() else 0
        return float(x[d]), float(y[d])


def mountain_cover(mountains: Floats, rocky: npt.NDArray[np.bool_]) -> Floats:
    """How fully raised ground takes the mountains' own colours over its ground's, from 0 to 1:
    as far as it is mountainous, and fully on mountain cells (`rocky`) once their mountains start
    to rise. Where it is 1 the ground's own colour does not show at all."""
    return np.maximum(mountains, rocky * np.clip(mountains * 3, 0, 1))


def _ridges(x: FloatArray, y: FloatArray) -> FloatArray:
    """How high mountain ridges, winding by noise, stand at plane points, from 0 to 1."""
    # Noise turns its points to float32; turned once, they serve every scale.
    fx, fy = x.astype(np.float32), y.astype(np.float32)
    wx = x + 0.9 * (noise(fx, fy, 2.5, 11) - 0.5)
    wy = y + 0.9 * (noise(fx, fy, 2.5, 12) - 0.5)
    fx, fy = wx.astype(np.float32), wy.astype(np.float32)
    big = ridged(fx, fy, 2.6, 1)
    ridges = 0.55 * big + 0.28 * big * ridged(fx, fy, 1.0, 2)
    return ridges + 0.12 * ridged(fx, fy, 0.42, 3) + 0.05 * ridged(fx, fy, 0.16, 6)


def _steps_read(low: float, high: float, size: int) -> slice:
    """The steps of a raster `size` steps long that blending reads at any position from `low`
    to `high`, and one more each way for positions rounded a hair outside them."""
    first = min(int(np.clip(low - 0.5, 0, size - 1)), size - 2)
    last = min(int(np.clip(high - 0.5, 0, size - 1)), size - 2) + 1
    return slice(max(first - 1, 0), min(last + 1, size - 1) + 1)


def raycast(lift: Floats, rows: int, cols: int) -> npt.NDArray[np.int64]:
    """How many rows below each of `rows` by `cols` pixels lies the point that shows there: the
    nearest one whose lift reaches up to it. `lift` covers at least the highest lift's rows more."""
    highest = int(lift.max())
    # Rows further down that reach a pixel overwrite those nearer, leaving the furthest; a lift
    # reaches a whole number of rows exactly when its whole part does.
    whole = np.floor(lift).astype(np.int16 if highest < 1 << 15 else np.int64)
    offset = np.zeros((rows, cols), dtype=whole.dtype)
    for d in range(1, highest + 1):
        np.copyto(offset, d, where=whole[d : d + rows, :cols] >= d)
    return offset.astype(np.int64)


def raise_picture(
    picture: FloatArray, offset: npt.NDArray[np.int64], resolution: int
) -> FloatArray:
    """The picture as seen with its ground lifted: each pixel taken from the rows `offset` art
    pixels below it, `picture` holding `resolution` pixels for every art pixel and enough rows."""
    k = resolution
    rows, cols = offset.shape[0] * k, offset.shape[1] * k
    art_rows, art_cols = offset.shape
    whole = picture.shape[0] % k == 0 and picture.shape[1] == cols
    if whole and (art_rows + int(offset.max())) * k <= picture.shape[0]:
        # No pixel is taken from below the picture, so each art pixel's rows of pixels come
        # whole from the art pixel its offset below.
        run = k * math.prod(picture.shape[2:])
        runs = picture.reshape(picture.shape[0], art_cols, run)
        below = (np.arange(art_rows)[:, None] + offset) * k
        raised = np.empty((art_rows, k, art_cols, run), dtype=picture.dtype)
        for row in range(k):
            raised[:, row] = runs[below + row, np.arange(art_cols)[None, :]]
        return raised.reshape(rows, cols, *picture.shape[2:])
    fine = np.repeat(np.repeat(offset, k, axis=0), k, axis=1) * k
    shown = np.minimum(np.arange(rows)[:, None] + fine, picture.shape[0] - 1)
    shown *= picture.shape[1]
    shown += np.arange(cols)[None, :]
    return picture.reshape(-1, *picture.shape[2:])[shown]
