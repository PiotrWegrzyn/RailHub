"""The ground of the pixel map, every pixel coloured from its cell with a little texture: a few shades per biome, sandy shores, rippled water, bevels where the land steps up, slopes lit from the top left by the land's real heights, a warm glow where buildings are served, specks of deposits, striped fields under farms and rows of trees under orchards. Textures come from a hash of the pixel's place, so chunks meet without seams and look the same each time they are drawn."""

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from railroads.sim.constructions import BuildingKind, Constructions
from railroads.sim.economy import Satisfaction
from railroads.sim.tilemap import Deposit, Relief, Terrain, TileMap
from railroads.ui.biomes import Biome, biome_grid
from railroads.ui.buildings import Plot, plots_of
from railroads.ui.pixel.projection import CHUNK, RGB, FloatArray, IntArray, PixelProjection, hash01

# Ground is worked out this far past a chunk's edges, so shores and bevels meet across chunks.
APRON = 2


# Four shades per biome, dark to light; the texture picks between them pixel by pixel.
RAMPS: dict[Biome, list[RGB]] = {
    Biome.SEA: [(26, 56, 110), (34, 72, 136), (44, 90, 160), (110, 150, 200)],
    Biome.LAKE: [(30, 70, 130), (40, 88, 156), (52, 106, 178), (120, 164, 214)],
    Biome.RIVER: [(30, 70, 130), (40, 88, 156), (52, 106, 178), (120, 164, 214)],
    Biome.SNOW: [(176, 188, 204), (206, 214, 226), (228, 234, 242), (250, 252, 255)],
    Biome.TUNDRA: [(130, 132, 100), (150, 152, 116), (168, 170, 132), (188, 190, 152)],
    Biome.TAIGA: [(36, 64, 44), (46, 78, 54), (58, 92, 64), (76, 110, 80)],
    Biome.BADLANDS: [(150, 126, 88), (170, 146, 104), (188, 164, 120), (206, 184, 140)],
    Biome.GRASSLAND: [(70, 116, 42), (88, 138, 50), (106, 156, 58), (134, 178, 74)],
    Biome.FOREST: [(36, 70, 30), (46, 86, 36), (58, 102, 42), (76, 120, 50)],
    Biome.DESERT: [(186, 156, 98), (206, 176, 114), (222, 194, 132), (238, 214, 156)],
    Biome.SAVANNA: [(150, 144, 70), (170, 162, 82), (188, 180, 96), (206, 198, 116)],
    Biome.JUNGLE: [(22, 70, 32), (30, 86, 40), (40, 102, 48), (56, 120, 60)],
    Biome.MOUNTAIN: [(86, 84, 82), (110, 106, 100), (136, 132, 124), (172, 168, 158)],
}


WATER = (Biome.SEA, Biome.LAKE, Biome.RIVER)


SAND: RGB = (200, 184, 128)


FOAM: RGB = (110, 150, 200)


# Field colours for farms, in pairs of furrow and ridge.
FIELDS: list[tuple[RGB, RGB]] = [
    ((150, 108, 64), (120, 84, 48)),
    ((214, 186, 90), (186, 156, 70)),
    ((140, 164, 70), (112, 138, 56)),
]


FIELD_STRIPES = 9


# Groves of trees in rows over the ground of the cell: the ground, the trees, the colours of the
# fruit some of them bear, how many rows a cell each way and how big each crown is.
@dataclass(frozen=True)
class Grove:
    ground: RGB
    tree: RGB
    fruit: tuple[RGB, ...]
    rows: int
    crown: float


GROVES: dict[BuildingKind, Grove] = {
    BuildingKind.ORCHARD: Grove(
        (146, 178, 84), (58, 104, 46), ((204, 60, 46), (226, 150, 52)), 6, 0.32
    ),
    BuildingKind.DATE_PALMS: Grove((214, 192, 134), (78, 122, 56), ((110, 62, 36),), 5, 0.3),
    BuildingKind.BANANAS: Grove((76, 122, 50), (44, 100, 38), ((236, 206, 64),), 7, 0.4),
    BuildingKind.ORANGE_FARM: Grove((150, 176, 84), (50, 104, 44), ((244, 142, 36),), 6, 0.36),
    BuildingKind.VINEYARD: Grove((150, 150, 90), (70, 110, 50), ((104, 50, 124),), 8, 0.3),
    BuildingKind.COFFEE_PLANTATION: Grove((96, 120, 62), (40, 86, 44), ((190, 52, 44),), 7, 0.36),
    BuildingKind.RUBBER_PLANTATION: Grove((70, 104, 56), (36, 72, 40), ((226, 224, 210),), 5, 0.38),
    BuildingKind.SHEEP_FARM: Grove((140, 182, 86), (238, 234, 222), ((60, 56, 54),), 4, 0.2),
    BuildingKind.CATTLE_RANCH: Grove((132, 176, 80), (120, 78, 48), ((236, 232, 224),), 4, 0.24),
    BuildingKind.HOP_GARDEN: Grove((150, 140, 96), (72, 132, 50), ((170, 214, 110),), 7, 0.34),
}


# Crops in rows, like a farm's furrows.
CROP_ROWS: dict[BuildingKind, tuple[RGB, RGB]] = {
    BuildingKind.SUGAR: ((156, 200, 100), (118, 162, 74)),
    BuildingKind.COTTON_FIELD: ((238, 236, 226), (150, 172, 100)),
    BuildingKind.POTATO_FARM: ((150, 120, 80), (96, 140, 64)),
    BuildingKind.FLAX_FIELD: ((140, 166, 226), (130, 160, 90)),
}


FIELD_KINDS = (BuildingKind.FARM, *CROP_ROWS, *GROVES)


# The specks showing each deposit on the ground, in the order of Deposit.
DEPOSIT_SPECKS = np.array(
    [
        (0, 0, 0),
        (172, 96, 70),
        (206, 130, 76),
        (255, 214, 60),
        (196, 240, 255),
        (24, 22, 28),
        (240, 240, 244),
        (120, 176, 210),
        (236, 230, 212),
        (116, 58, 136),
        (248, 248, 244),
        (92, 58, 38),
        (58, 62, 54),
        (0, 0, 0),
        (226, 196, 128),
    ],
    dtype=np.float64,
)


DEPOSIT_SPECK_SHARE = 0.08


BAYER = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]) / 16 - 0.5


HILL_LIFT = 0.3


# Slopes are lit from the top left of the screen, from the land's real heights. Heights are
# smoothed over a raster this many steps per cell wide, so slopes run across cell edges.
RELIEF_STEPS_PER_CELL = 4


RELIEF_SMOOTHING_CELLS = 0.75


# Shades of brightening per unit of grade facing the light (0.05 is a 5% grade), and the most
# a slope can brighten or darken.
SLOPE_LIGHT = 16.0


MAX_SLOPE_SHADES = 1.5


EDGE_SHADE = 0.94


# Land whose buildings are content or thriving is washed with a warm glow: how strongly, by
# their satisfaction.
GLOW: RGB = (255, 214, 110)


SERVED_GLOW = {
    Satisfaction.UNSATISFIED: 0.0,
    Satisfaction.CONTENT: 0.12,
    Satisfaction.THRIVING: 0.24,
}


BEVEL_LIGHT = 1.18


BEVEL_SHADE = 0.78


def glow_strength(levels: npt.NDArray[np.int8]) -> FloatArray:
    """How strongly each cell glows for its Satisfaction; -1 (nothing trading) does not."""
    table = np.array([0.0, *(SERVED_GLOW[level] for level in Satisfaction)])
    return table[levels.astype(np.int64) + 1]


def _smooth(values: FloatArray, radius: int) -> FloatArray:
    """Two passes of a box blur `radius` steps either side, along both axes."""
    if radius < 1:
        return values
    for axis in (0, 1, 0, 1):
        padded = np.pad(
            values, [(radius + 1, radius) if a == axis else (0, 0) for a in (0, 1)], mode="edge"
        )
        sums = np.cumsum(padded, axis=axis)
        upper = np.take(sums, np.arange(2 * radius + 1, sums.shape[axis]), axis=axis)
        lower = np.take(sums, np.arange(0, sums.shape[axis] - 2 * radius - 1), axis=axis)
        values = (upper - lower) / (2 * radius + 1)
    return values


def _sample(raster: FloatArray, x: FloatArray, y: FloatArray) -> FloatArray:
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


def _near(mask: npt.NDArray[np.bool_], shifts: list[tuple[int, int]]) -> npt.NDArray[np.bool_]:
    near = np.zeros_like(mask)
    for shift in shifts:
        near |= np.roll(mask, shift, axis=(0, 1))
    return near


class Ground:
    """The ground of one map, drawn a chunk at a time, and what it is drawn from: each cell's
    biome and level, the slopes' light, the fields and groves, and how satisfied each cell is."""

    def __init__(
        self,
        tile_map: TileMap,
        constructions: Constructions,
        projection: PixelProjection,
        angle_deg: float,
    ) -> None:
        self.tile_map = tile_map
        self.constructions = constructions
        self.projection = projection
        self._biomes = biome_grid(tile_map)
        land = tile_map.terrain == Terrain.LAND
        self._levels = np.where(land, tile_map.relief.astype(np.int64), -1)
        # Each cell's Satisfaction, or -1 where nothing trades.
        self.satisfaction = np.full(tile_map.grid.shape, -1, dtype=np.int8)
        self._hills = land & (tile_map.relief == Relief.HILLS)
        self._relief_light = self._light_slopes(angle_deg)
        self.ramps = np.zeros((len(Biome), 4, 3))
        for biome, ramp in RAMPS.items():
            self.ramps[biome] = ramp
        self._plots = plots_of(constructions, FIELD_KINDS)

    def refresh(self) -> None:
        """Works out the biomes and fields again after buildings or forests changed."""
        self._biomes = biome_grid(self.tile_map)
        self._plots = plots_of(self.constructions, FIELD_KINDS)

    def render(self, left: int, top: int) -> tuple[FloatArray, FloatArray]:
        grid = self.tile_map.grid
        size = CHUNK + 2 * APRON
        iy, ix = np.mgrid[top - APRON : top - APRON + size, left - APRON : left - APRON + size]
        x, y = self.projection.to_plane(ix + 0.5, iy + 0.5)
        rows, cols = grid.cell_at(x, y)
        inside = grid.contains(rows, cols)
        rows = np.clip(rows, 0, grid.rows - 1)
        cols = np.clip(cols, 0, grid.cols - 1)
        biomes = self._biomes[rows, cols]
        water = np.isin(biomes, WATER)

        grain = hash01(ix, iy, 11)
        cell_tone = hash01(rows, cols, 12)
        value = 1.4 + 0.9 * (grain - 0.5) + 0.5 * (cell_tone - 0.5) + 0.9 * BAYER[iy % 4, ix % 4]
        value += HILL_LIFT * self._hills[rows, cols]
        value += _sample(self._relief_light, x * RELIEF_STEPS_PER_CELL, y * RELIEF_STEPS_PER_CELL)
        ripple = hash01(ix // 3, iy, 13) > 0.94
        value = np.where(water, np.where(ripple, 3, 1.3 + 0.6 * (grain - 0.5)), value)
        shade = np.clip(np.rint(value), 0, 3).astype(np.int64)
        rgb = self.ramps[biomes, shade]

        near_water = _near(water, [(0, 1), (0, -1), (1, 0), (-1, 0), (0, 2), (0, -2)])
        near_land = _near(~water, [(0, 1), (0, -1), (1, 0), (-1, 0)])
        rgb[~water & near_water & inside] = SAND
        rgb[water & near_land & inside] = FOAM

        cell_id = rows * grid.cols + cols
        edge = (cell_id != np.roll(cell_id, -1, axis=1)) | (cell_id != np.roll(cell_id, -1, axis=0))
        rgb[edge & ~water] *= EDGE_SHADE
        levels = self._levels[rows, cols]
        lit = (levels > np.roll(levels, (1, 1), axis=(0, 1))) | (
            levels > np.roll(levels, (2, 2), axis=(0, 1))
        )
        shaded = (levels > np.roll(levels, (-1, -1), axis=(0, 1))) | (
            levels > np.roll(levels, (-2, -2), axis=(0, 1))
        )
        rgb[lit & ~water] *= BEVEL_LIGHT
        rgb[shaded & ~water] *= BEVEL_SHADE
        strength = glow_strength(self.satisfaction)[rows, cols][..., None] * ~water[..., None]
        rgb = rgb * (1 - strength) + np.array(GLOW) * strength
        if self.tile_map.deposits is not None:
            deposit = self.tile_map.deposits[rows, cols]
            shown = (deposit > 0) & (deposit != Deposit.GARRISON)
            speck = shown & (hash01(ix, iy, 31) < DEPOSIT_SPECK_SHARE)
            rgb[speck] = DEPOSIT_SPECKS[deposit[speck]]
        self._plough(rgb, x, y, rows * grid.cols + cols)
        self._plant_orchards(rgb, x, y, rows * grid.cols + cols, ix, iy)

        inner = (slice(APRON, APRON + CHUNK), slice(APRON, APRON + CHUNK))
        alpha = inside[inner].astype(np.float64)
        return np.clip(rgb[inner], 0, 255) * alpha[..., None], alpha

    def _light_slopes(self, angle_deg: float) -> FloatArray:
        """How many shades lighter (or darker, below 0) the ground is where it faces towards (or
        away from) the light, on a raster RELIEF_STEPS_PER_CELL steps per cell. Water is level."""
        grid = self.tile_map.grid
        width, height = grid.size()
        steps = RELIEF_STEPS_PER_CELL
        ys, xs = np.mgrid[0 : math.ceil(height * steps), 0 : math.ceil(width * steps)]
        rows, cols = grid.cell_at((xs + 0.5) / steps, (ys + 0.5) / steps)
        rows = np.clip(rows, 0, grid.rows - 1)
        cols = np.clip(cols, 0, grid.cols - 1)
        heights = self.tile_map.height_m[rows, cols].astype(np.float64)
        water = self.tile_map.terrain[rows, cols] != Terrain.LAND
        heights = _smooth(heights, round(RELIEF_SMOOTHING_CELLS * steps))
        km_per_step = grid.km_per_cell / steps
        rise_y, rise_x = np.gradient(heights, km_per_step * 1000)
        light_x, light_y = self._towards_light(angle_deg)
        facing = -(rise_x * light_x + rise_y * light_y)
        light = np.clip(SLOPE_LIGHT * facing, -MAX_SLOPE_SHADES, MAX_SLOPE_SHADES)
        return np.where(water, 0.0, light)

    def _towards_light(self, angle_deg: float) -> tuple[float, float]:
        """The direction on the map, as a unit vector, towards the top left of the screen."""
        x, y = np.linalg.inv(self.projection.matrix) @ np.array([-1.0, -1.0])
        length = math.hypot(x, y)
        return x / length, y / length

    def _plough(self, rgb: FloatArray, x: FloatArray, y: FloatArray, cells: IntArray) -> None:
        """Lays striped fields over the half of a cell each farm or crop in rows takes up."""
        grid = self.tile_map.grid
        x0, x1, y0, y1 = x.min(), x.max(), y.min(), y.max()
        fields: list[tuple[Plot, BuildingKind | None]] = [
            (plot, None) for plot in self._plots[BuildingKind.FARM]
        ]
        fields += [(plot, kind) for kind in CROP_ROWS for plot in self._plots[kind]]
        for plot, crop in fields:
            row, col = plot.cell
            fx, fy = grid.center(row, col)
            if not (x0 - 1 < fx < x1 + 1 and y0 - 1 < fy < y1 + 1):
                continue
            field = (cells == row * grid.cols + col) & plot.covers(x - fx, y - fy)
            if not field.any():
                continue
            pick = row * 7 + col * 13
            across = x - fx if pick % 2 else y - fy
            stripe = np.floor(across * FIELD_STRIPES) % 2 == 0
            furrow, ridge = CROP_ROWS[crop] if crop is not None else FIELDS[pick % len(FIELDS)]
            rgb[field] = np.where(stripe[field][:, None], furrow, ridge)

    def _plant_orchards(
        self,
        rgb: FloatArray,
        x: FloatArray,
        y: FloatArray,
        cells: IntArray,
        ix: IntArray,
        iy: IntArray,
    ) -> None:
        """Lays groves of trees in rows over the half of a cell each orchard, date palms or
        plantation takes up."""
        x0, x1, y0, y1 = x.min(), x.max(), y.min(), y.max()
        for kind, grove in GROVES.items():
            self._plant(rgb, x, y, cells, ix, iy, grove, self._plots[kind], (x0, x1, y0, y1))

    def _plant(
        self,
        rgb: FloatArray,
        x: FloatArray,
        y: FloatArray,
        cells: IntArray,
        ix: IntArray,
        iy: IntArray,
        grove: Grove,
        planted: list[Plot],
        box: tuple[float, float, float, float],
    ) -> None:
        grid = self.tile_map.grid
        x0, x1, y0, y1 = box
        for plot in planted:
            row, col = plot.cell
            fx, fy = grid.center(row, col)
            if not (x0 - 1 < fx < x1 + 1 and y0 - 1 < fy < y1 + 1):
                continue
            ground = (cells == row * grid.cols + col) & plot.covers(x - fx, y - fy)
            if not ground.any():
                continue
            u = (x[ground] - fx) * grove.rows + 0.5
            v = (y[ground] - fy) * grove.rows + 0.5
            near = np.hypot(u - np.round(u), v - np.round(v))
            colors = np.broadcast_to(np.array(grove.ground, dtype=np.float64), (u.size, 3)).copy()
            tree = near < grove.crown
            colors[tree] = grove.tree
            fruit = tree & (hash01(ix[ground], iy[ground], 21) > 0.82)
            pick = (hash01(ix[ground], iy[ground], 22) * len(grove.fruit)).astype(np.int64)
            for index, color in enumerate(grove.fruit):
                colors[fruit & (pick == index)] = color
            rgb[ground] = colors
