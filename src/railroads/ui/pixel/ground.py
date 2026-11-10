"""The ground of the pixel map, worked out at one pixel per art pixel: broad soft patches of a few
shades per biome with tufts of grass, sandy shores, rippled water, slopes lit from the top left by
the land's real heights, a warm glow where buildings are served, small signs of deposits (sheep
grazing wool land, rocks showing their ore elsewhere), fields of crops behind hedges under farms
and rows of trees under orchards. Hills and mountains are raised, lit and coloured by
railroads.ui.pixel.relief. Textures come from noise over the map's plane and from each pixel's
place, so chunks meet without seams and look the same each time they are drawn."""

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from railroads.sim.constructions import BuildingKind, Constructions
from railroads.sim.economy import Satisfaction
from railroads.sim.tilemap import Deposit, Relief, Terrain, TileMap
from railroads.ui.biomes import Biome, biome_grid
from railroads.ui.buildings import Plot, plots_of
from railroads.ui.pixel.noise import noise, table_hash
from railroads.ui.pixel.projection import CHUNK, RGB, FloatArray, IntArray, PixelProjection
from railroads.ui.pixel.relief import Elevation, Raised, sample, smooth

# Ground is worked out this far past a chunk's edges, so shores meet across chunks.
APRON = 2


# Four shades per biome, dark to light; broad patches of noise pick between them.
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


# The colour of each deposit's ore, in the order of Deposit, shown in the rocks marking it.
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


# Land whose buildings are content or thriving is washed with a warm glow: how strongly, by
# their satisfaction.
GLOW: RGB = (255, 214, 110)


SERVED_GLOW = {
    Satisfaction.UNSATISFIED: 0.0,
    Satisfaction.CONTENT: 0.12,
    Satisfaction.THRIVING: 0.24,
}


# Farms grow one of these, chosen by their cell, with the colours of their soil and plants.
FARM_CROPS = ("potatoes", "wheat", "cabbages")
SOIL = np.array((128, 88, 54), dtype=np.float32)
SOIL_DARK = np.array((100, 68, 42), dtype=np.float32)
WHEAT = np.array((218, 184, 86), dtype=np.float32)
WHEAT_DARK = np.array((200, 166, 72), dtype=np.float32)
WHEAT_LIGHT = np.array((238, 210, 116), dtype=np.float32)
WHEAT_EARS = np.array((170, 134, 54), dtype=np.float32)
POTATO = np.array((84, 138, 52), dtype=np.float32)
POTATO_LIGHT = np.array((116, 168, 66), dtype=np.float32)
CABBAGE = np.array((118, 170, 82), dtype=np.float32)
CABBAGE_HEART = np.array((172, 208, 124), dtype=np.float32)
CABBAGE_LEAVES = np.array((78, 128, 54), dtype=np.float32)
HEDGE = np.array((64, 104, 42), dtype=np.float32)


# A tuft of grass in a 6 by 6 block: three blades, in a TUFT_SHARE of the blocks on grassy land.
TUFT = np.zeros((6, 6), dtype=bool)
for _dx, _dy in ((0, 3), (0, 4), (1, 2), (2, 1), (2, 2), (2, 3), (2, 4), (3, 2), (4, 3), (4, 4)):
    TUFT[_dy, _dx] = True
TUFT_SHARE = 0.2
GRASSY = (Biome.GRASSLAND, Biome.SAVANNA, Biome.TUNDRA, Biome.FOREST, Biome.TAIGA, Biome.JUNGLE)


# Deposits show in one in DEPOSIT_SIGNS of the 12 by 12 blocks over them: a sheep on wool land,
# elsewhere a rock with a vein of its ore.
DEPOSIT_SIGNS = 0.18
SHEEP = np.array((240, 236, 226), dtype=np.float32)
SHEEP_DARK = np.array((40, 36, 34), dtype=np.float32)
ORE_ROCK = np.array((112, 106, 98), dtype=np.float32)


def glow_strength(levels: npt.NDArray[np.int8]) -> FloatArray:
    """How strongly each cell glows for its Satisfaction; -1 (nothing trading) does not."""
    table = np.array([0.0, *(SERVED_GLOW[level] for level in Satisfaction)])
    return table[levels.astype(np.int64) + 1]


class Ground:
    """The ground of one map, drawn a block of art pixels at a time, and what it is drawn from:
    each cell's biome, the slopes' light, the fields and groves, and how satisfied each cell is."""

    def __init__(
        self,
        tile_map: TileMap,
        constructions: Constructions,
        projection: PixelProjection,
        angle_deg: float,
        elevation: Elevation,
    ) -> None:
        self.tile_map = tile_map
        self.constructions = constructions
        self.projection = projection
        self.elevation = elevation
        self._biomes = biome_grid(tile_map)
        land = tile_map.terrain == Terrain.LAND
        # Each cell's Satisfaction, or -1 where nothing trades.
        self.satisfaction = np.full(tile_map.grid.shape, -1, dtype=np.int8)
        self._hills = land & (tile_map.relief == Relief.HILLS)
        self._relief_light = self._light_slopes(angle_deg)
        self.ramps = np.zeros((len(Biome), 4, 3), dtype=np.float32)
        for biome, ramp in RAMPS.items():
            self.ramps[biome] = ramp
        self.refresh()

    def refresh(self) -> None:
        """Works out the biomes and fields again after buildings or forests changed."""
        self._biomes = biome_grid(self.tile_map)
        self._plots = plots_of(self.constructions, FIELD_KINDS)
        grid = self.tile_map.grid
        # Where each kind's fields sit in the picture, to find those near a block quickly.
        self._plot_pixels: dict[BuildingKind, tuple[FloatArray, FloatArray]] = {}
        for kind, plots in self._plots.items():
            cells = np.array([plot.cell for plot in plots], dtype=np.int64).reshape(-1, 2)
            px, py = self.projection.to_pixels(*grid.center(cells[:, 0], cells[:, 1]))
            self._plot_pixels[kind] = (np.atleast_1d(px), np.atleast_1d(py))

    def render(
        self, left: int, top: int, rows: int, raised: Raised | None = None
    ) -> tuple[FloatArray, FloatArray]:
        """The ground over CHUNK art pixels across and `rows` down from (`left`, `top`), as RGB
        from 0 to 255 already multiplied by its alpha, and its alpha: 1 on the map, 0 off it.
        `raised` is the raised ground over at least these pixels, if any rises there."""
        grid = self.tile_map.grid
        a = APRON
        iy, ix = np.mgrid[top - a : top + rows + a, left - a : left + CHUNK + a]
        x, y = self.projection.to_plane(ix + 0.5, iy + 0.5)
        r, c = grid.cell_at(x, y)
        inside = grid.contains(r, c)
        r, c = np.clip(r, 0, grid.rows - 1), np.clip(c, 0, grid.cols - 1)
        biomes = self._biomes[r, c]
        water = np.isin(biomes, WATER)
        value = 1.45 + 1.6 * (noise(x, y, 2.4, 41) - 0.5) + 0.8 * (noise(x, y, 0.75, 42) - 0.5)
        value += 0.45 * (noise(x, y, 0.22, 43) - 0.5) + 0.45 * BAYER[iy % 4, ix % 4]
        value += HILL_LIFT * self._hills[r, c]
        value += sample(self._relief_light, x * RELIEF_STEPS_PER_CELL, y * RELIEF_STEPS_PER_CELL)
        ripple = table_hash(ix // 3, iy, 13) > 0.94
        calm = 1.3 + 0.6 * (table_hash(ix, iy, 11) - 0.5)
        value = np.where(water, np.where(ripple, 3, calm), value)
        shade = np.clip(np.rint(value), 0, 3).astype(np.int64)
        rgb = self.ramps[biomes, shade]
        grassy = np.isin(biomes, GRASSY) & ~water
        tuft = grassy & (table_hash(ix // 6, iy // 6, 51) > 1 - TUFT_SHARE) & TUFT[iy % 6, ix % 6]
        rgb[tuft] = self.ramps[biomes[tuft], np.maximum(shade[tuft] - 1, 0)] * 0.92
        near_water = _near(water, [(0, 1), (0, -1), (1, 0), (-1, 0), (0, 2), (0, -2)])
        near_land = _near(~water, [(0, 1), (0, -1), (1, 0), (-1, 0)])
        rgb[~water & near_water & inside] = SAND
        rgb[water & near_land & inside] = FOAM
        glow = glow_strength(self.satisfaction)[r, c]
        if glow.any():
            strength = (glow * ~water)[..., None].astype(np.float32)
            rgb = rgb * (1 - strength) + np.array(GLOW, dtype=np.float32) * strength
        if self.tile_map.deposits is not None:
            deposit = self.tile_map.deposits[r, c]
            if deposit.any():
                _sign_deposits(rgb, deposit, ix, iy, water)
        cells = r * grid.cols + c
        origin = (left - a, top - a)
        for crop in (None, *CROP_ROWS):
            kind = BuildingKind.FARM if crop is None else crop
            for plot, patch in self._patches(kind, origin, rgb.shape[:2]):
                args = (rgb[patch], x[patch], y[patch], cells[patch], ix[patch], iy[patch])
                self._sow(*args, plot, crop)
        for kind, grove in GROVES.items():
            for plot, patch in self._patches(kind, origin, rgb.shape[:2]):
                args = (rgb[patch], x[patch], y[patch], cells[patch], ix[patch], iy[patch])
                self._plant(*args, plot, grove)
        inner = (slice(a, a + rows), slice(a, a + CHUNK))
        rgb, alpha = rgb[inner], inside[inner].astype(np.float32)
        if raised is not None:
            rocky = self.tile_map.relief[r[inner], c[inner]] == Relief.MOUNTAINS
            self.elevation.colour(rgb, raised.top(rows), x[inner], y[inner], rocky)
        return np.clip(rgb, 0, 255) * alpha[..., None], alpha

    def _patches(
        self, kind: BuildingKind, origin: tuple[int, int], shape: tuple[int, ...]
    ) -> list[tuple[Plot, tuple[slice, slice]]]:
        """The fields of a kind near a block of pixels starting at `origin`, each with the slice
        of the block its cell can cover."""
        px, py = self._plot_pixels[kind]
        left, top = origin
        rows, cols = shape
        near = np.flatnonzero((px > left - 24) & (px < left + cols + 24))
        near = near[(py[near] > top - 16) & (py[near] < top + rows + 16)]
        patches = []
        for i in near.tolist():
            x0, y0 = int(px[i] - left - 20), int(py[i] - top - 12)
            box = (slice(max(y0, 0), min(y0 + 24, rows)), slice(max(x0, 0), min(x0 + 40, cols)))
            if box[0].start < box[0].stop and box[1].start < box[1].stop:
                patches.append((self._plots[kind][i], box))
        return patches

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
        heights = smooth(heights, round(RELIEF_SMOOTHING_CELLS * steps))
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

    def _sow(
        self,
        rgb: FloatArray,
        x: FloatArray,
        y: FloatArray,
        cells: IntArray,
        ix: IntArray,
        iy: IntArray,
        plot: Plot,
        crop: BuildingKind | None,
    ) -> None:
        """Lays a field of crops in rows over the half of a cell a farm or crop takes up, behind
        a hedge: a farm grows one of FARM_CROPS, chosen by its cell."""
        grid = self.tile_map.grid
        row, col = plot.cell
        fx, fy = grid.center(row, col)
        field = (cells == row * grid.cols + col) & plot.covers(x - fx, y - fy)
        if not field.any():
            return
        pick = row * 7 + col * 13
        across, along = (x - fx, y - fy) if pick % 2 else (y - fy, x - fx)
        u, w = across[field] * FIELD_STRIPES, along[field] * FIELD_STRIPES
        line = np.floor(u).astype(np.int64)
        fu, fw = u - line, w - np.floor(w)
        px, py = ix[field], iy[field]
        kind = FARM_CROPS[pick % len(FARM_CROPS)] if crop is None else None
        out = np.empty((u.size, 3), dtype=np.float32)
        if kind == "wheat":
            out[:] = np.where((line % 2 == 0)[:, None], WHEAT, WHEAT_DARK)
            out[fu < 0.2] = WHEAT_LIGHT
            out[(table_hash(px, py, 72) > 0.7) & (fu > 0.35)] = WHEAT_EARS
        elif kind == "cabbages":
            out[:] = SOIL
            d = np.hypot(fu - 0.5, (fw - 0.5) * 0.9)
            out[d < 0.38] = CABBAGE_LEAVES
            out[d < 0.3] = CABBAGE
            out[d < 0.14] = CABBAGE_HEART
        else:
            if crop is None:
                soil, plant, light = SOIL, POTATO, POTATO_LIGHT
            else:
                plant_rgb, ground_rgb = CROP_ROWS[crop]
                soil = np.array(ground_rgb, dtype=np.float32) * 0.8 + SOIL * 0.2
                plant = np.array(plant_rgb, dtype=np.float32)
                light = np.minimum(plant * 1.18, 255)
            out[:] = soil
            out[fu < 0.12] = SOIL_DARK
            bushes = table_hash(np.floor(w * 2.2).astype(np.int64), line, 81) > 0.18
            bush = (np.abs(fu - 0.5) < 0.3) & bushes
            out[bush] = plant
            out[bush & (fu < 0.42)] = light
            out[bush & (fu > 0.68)] = plant * 0.72
        inner = field & np.roll(field, 1, 0) & np.roll(field, -1, 0)
        inner &= np.roll(field, 1, 1) & np.roll(field, -1, 1)
        edge = (field & ~inner)[field]
        shade = 0.85 + 0.3 * table_hash(px[edge] // 2, py[edge] // 2, 91)
        out[edge] = HEDGE * shade[:, None]
        rgb[field] = out

    def _plant(
        self,
        rgb: FloatArray,
        x: FloatArray,
        y: FloatArray,
        cells: IntArray,
        ix: IntArray,
        iy: IntArray,
        plot: Plot,
        grove: Grove,
    ) -> None:
        """Lays a grove of trees in rows over the half of a cell an orchard, date palms or
        plantation takes up."""
        grid = self.tile_map.grid
        row, col = plot.cell
        fx, fy = grid.center(row, col)
        ground = (cells == row * grid.cols + col) & plot.covers(x - fx, y - fy)
        if not ground.any():
            return
        u = (x[ground] - fx) * grove.rows + 0.5
        v = (y[ground] - fy) * grove.rows + 0.5
        near = np.hypot(u - np.round(u), v - np.round(v))
        colors = np.broadcast_to(np.array(grove.ground, dtype=np.float32), (u.size, 3)).copy()
        tree = near < grove.crown
        colors[tree] = grove.tree
        fruit = tree & (table_hash(ix[ground], iy[ground], 21) > 0.82)
        pick = (table_hash(ix[ground], iy[ground], 22) * len(grove.fruit)).astype(np.int64)
        for index, color in enumerate(grove.fruit):
            colors[fruit & (pick == index)] = color
        rgb[ground] = colors


def _sign_deposits(
    rgb: FloatArray,
    deposit: npt.NDArray[np.uint8],
    ix: IntArray,
    iy: IntArray,
    water: npt.NDArray[np.bool_],
) -> None:
    """Marks deposits with a few small signs: sheep grazing wool land, rocks with a vein of their
    ore elsewhere."""
    shown = (deposit > 0) & (deposit != Deposit.GARRISON) & ~water
    lx, ly = ix % 12, iy % 12
    here = shown & (table_hash(ix // 12, iy // 12, 71) > 1 - DEPOSIT_SIGNS)
    sheep = here & (deposit == Deposit.WOOL)
    rgb[sheep & (lx >= 4) & (lx <= 7) & (ly >= 5) & (ly <= 7)] = SHEEP
    rgb[sheep & (lx == 3) & (ly == 5)] = SHEEP_DARK
    rgb[sheep & ((lx == 4) | (lx == 7)) & (ly == 8)] = SHEEP_DARK
    rgb[sheep & (lx >= 4) & (lx <= 7) & (ly == 9)] *= 0.7
    rock = here & (deposit != Deposit.WOOL)
    stone = rock & (lx >= 4) & (lx <= 7) & (ly >= 6) & (ly <= 8) & ~((lx == 7) & (ly == 6))
    rgb[stone] = ORE_ROCK
    vein = rock & (lx == 5) & (ly == 7)
    rgb[vein] = DEPOSIT_SPECKS[deposit[vein]]
    rgb[rock & (lx >= 4) & (lx <= 8) & (ly == 9)] *= 0.72


def _near(mask: npt.NDArray[np.bool_], shifts: list[tuple[int, int]]) -> npt.NDArray[np.bool_]:
    near = np.zeros_like(mask)
    for shift in shifts:
        near |= np.roll(mask, shift, axis=(0, 1))
    return near
