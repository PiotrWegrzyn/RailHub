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

from railroads.sim.constructions import Building, BuildingKind, Constructions
from railroads.sim.economy import Satisfaction
from railroads.sim.grids import Grid
from railroads.sim.tilemap import Deposit, Relief, Terrain, TileMap
from railroads.ui.biomes import Biome, biome_grid
from railroads.ui.building_spots import Plot, plots_of
from railroads.ui.pixel.lattice import Lattice
from railroads.ui.pixel.noise import noise, table_hash
from railroads.ui.pixel.projection import CHUNK, RGB, FloatArray, IntArray, PixelProjection
from railroads.ui.pixel.relief import Elevation, Raised, mountain_cover, sample, smooth

type BoolArray = npt.NDArray[np.bool_]

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
    BuildingKind.PIG_FARM: Grove((140, 150, 90), (120, 92, 60), ((226, 160, 156),), 4, 0.24),
    BuildingKind.CHICKEN_FARM: Grove((150, 176, 90), (196, 170, 110), ((244, 240, 230),), 5, 0.2),
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


# Farms grow wheat, which is what they make; other crops grow in rows (CROP_ROWS) over soil.
SOIL = np.array((128, 88, 54), dtype=np.float32)
SOIL_DARK = np.array((100, 68, 42), dtype=np.float32)
WHEAT = np.array((218, 184, 86), dtype=np.float32)
WHEAT_DARK = np.array((200, 166, 72), dtype=np.float32)
WHEAT_LIGHT = np.array((238, 210, 116), dtype=np.float32)
WHEAT_EARS = np.array((170, 134, 54), dtype=np.float32)
HEDGE = np.array((64, 104, 42), dtype=np.float32)
HEADLAND = np.array((150, 140, 96), dtype=np.float32)


# A tuft of grass in a 6 by 6 block: three blades, in a TUFT_SHARE of the blocks on grassy land.
TUFT = np.zeros((6, 6), dtype=bool)
for _dx, _dy in ((0, 3), (0, 4), (1, 2), (2, 1), (2, 2), (2, 3), (2, 4), (3, 2), (4, 3), (4, 4)):
    TUFT[_dy, _dx] = True
TUFT_SHARE = 0.2
GRASSY = (Biome.GRASSLAND, Biome.SAVANNA, Biome.TUNDRA, Biome.FOREST, Biome.TAIGA, Biome.JUNGLE)

# Grassland is drawn as meadow, and so is the ground under mountains, which shows where they rise
# only a little, and under woods, which shows in their clearings.
MEADOWS = (Biome.GRASSLAND, Biome.MOUNTAIN, Biome.FOREST, Biome.TAIGA, Biome.JUNGLE)
# Meadows are smooth grass of three kinds side by side, each a ramp of colours from shaded to lit:
# short pasture, darker rough grass whose blades show, and dry yellowing patches.
PASTURE = ((0.0, (62, 106, 36)), (0.5, (94, 146, 54)), (1.0, (140, 180, 80)))
ROUGH = ((0.0, (54, 94, 34)), (0.5, (80, 126, 46)), (1.0, (120, 160, 66)))
DRY = ((0.0, (104, 124, 50)), (0.5, (136, 152, 66)), (1.0, (170, 178, 92)))
# Savanna is drawn the same way in its own three kinds: golden grass, greener grass where it is
# damper, and patches of bare red earth.
GOLDEN = ((0.0, (150, 132, 62)), (0.5, (194, 174, 90)), (1.0, (224, 208, 132)))
DAMP = ((0.0, (132, 124, 56)), (0.5, (168, 160, 78)), (1.0, (202, 194, 112)))
EARTH = ((0.0, (138, 98, 62)), (0.5, (172, 128, 84)), (1.0, (202, 164, 116)))
# Each grassland, and how rare wild flowers are in it (1 for none).
GRASSES = {
    "meadow": (PASTURE, ROUGH, DRY, 0.997),
    "savanna": (GOLDEN, DAMP, EARTH, 1.0),
}
# The colours of wild flowers in meadows.
FLOWERS = np.array([(244, 240, 232), (246, 214, 80), (196, 130, 210), (226, 90, 70)], np.float32)
# Woods are drawn as a canopy seen from above (Ground.canopy): crowns about CROWN_GAP art pixels
# apart, each lit on the side towards the light, conifers in their own darker greens, over a dark
# floor showing between them. Their edge follows the forest cells but frays, curving round
# clearings where something is built, and they throw a shadow on the ground beside them.
WOODS = (Biome.FOREST, Biome.TAIGA, Biome.JUNGLE)
CROWN_GAP = 3.1
# How many art pixels above and left of a block the woods are worked out, for their shadow.
CANOPY_REACH = 4
CONIFER_CROWNS = np.array([(22, 48, 34), (32, 66, 44), (52, 92, 58)], dtype=np.float32)
LEAFY_CROWNS = np.array([(38, 76, 30), (62, 112, 40), (106, 152, 58)], dtype=np.float32)
WOODS_FLOOR = np.array((30, 52, 28), dtype=np.float32) * 0.7
EDGE_SHADOW = 0.75
# How mountainous ground may be before woods give way to the mountain's own bare slopes, and how
# far below that a bound on it (Elevation.most) must stay to show none of the woods are bare.
BARE_ABOVE = 0.3
BARE_MARGIN = 1e-4
CLEARING = -4.0
# Lumber mills stand in the woods: their cells keep their trees, but for a small clearing round
# the mill itself, MILL_CLEARING of its yard's reach across and half that down, its edge frayed,
# its middle CLEARING_DROP art pixels in front of the mill so its yard shows below it.
IN_THE_WOODS = (BuildingKind.LUMBER_MILL,)
MILL_CLEARING = 1.2
CLEARING_DROP = 3.0
# Where the woods' share of the ground is worked out around each point: the point itself and six
# around it, in cells.
AROUND = (
    (0.0, 0.0),
    (0.45, 0.0),
    (-0.45, 0.0),
    (0.22, 0.39),
    (-0.22, 0.39),
    (0.22, -0.39),
    (-0.22, -0.39),
)
# The ground round buildings, so they stand in it rather than on untouched grass: trodden earth
# round a cottage with a vegetable garden beside it and round a farmstead at the back of its
# field, paving round town buildings and flagstones round churches and landmarks, gravel round
# works, spoil round mines and pits, and sand under palms. Each is an oval its building's reach
# across and half that down, its edge frayed; a cottage's and a farmstead's reach is fixed.
YARD_REACH = {"cottage": 7.5, "farmyard": 11.0}
YARD_COLOURS = {
    "cottage": (150, 128, 88),
    "farmyard": (150, 128, 88),
    "paved": (168, 164, 152),
    "flagstones": (182, 176, 160),
    "gravel": (148, 142, 128),
    "spoil": (118, 102, 84),
    "sand": (214, 192, 134),
}
GARDEN_LEAVES = np.array((84, 132, 52), dtype=np.float32)
# How dark the grass in a tree's shade gets, and how far the shade reaches from its foot, in art
# pixels across and down.
TREE_SHADE = 0.24
SHADE_REACH = (7.0, 3.5)


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
        self._land = land
        # The cells round each cell that points AROUND a point in it can lie in, and where cells
        # round a cell lie off the map, which never change.
        self._round, self._off_round = _cells_round(tile_map.grid)
        # Each cell's Satisfaction, or -1 where nothing trades.
        self.satisfaction = np.full(tile_map.grid.shape, -1, dtype=np.int8)
        self._hills = land & (tile_map.relief == Relief.HILLS)
        self._relief_light = self._light_slopes(angle_deg)
        # Where trees stand in the picture, which shade the grass round them, and where buildings
        # stand, what ground lies round each and how far it reaches, both set by the world.
        self.trees: tuple[FloatArray, FloatArray] = (np.zeros(0), np.zeros(0))
        self.buildings: tuple[FloatArray, FloatArray, npt.NDArray[np.str_], FloatArray] = (
            np.zeros(0),
            np.zeros(0),
            np.zeros(0, dtype=str),
            np.zeros(0),
        )
        # Where lumber mills stand in the picture and how far the clearing round each reaches.
        self.clearings: tuple[FloatArray, FloatArray, FloatArray] = (
            np.zeros(0),
            np.zeros(0),
            np.zeros(0),
        )
        self.ramps = np.zeros((len(Biome), 4, 3), dtype=np.float32)
        for biome, ramp in RAMPS.items():
            self.ramps[biome] = ramp
        self.refresh()

    def refresh(self) -> None:
        """Works out the biomes and fields again after buildings or forests changed."""
        self._biomes = biome_grid(self.tile_map)
        built = np.zeros(self.tile_map.grid.shape, dtype=bool)
        woodland = np.zeros(self.tile_map.grid.shape, dtype=bool)
        for construction in self.constructions:
            if isinstance(construction, Building) and construction.kind in IN_THE_WOODS:
                woodland[construction.cell] = True
            else:
                built[construction.cell] = True
        # Each cell's say in whether woods cover a point near it: 1 for woods and lumber mills,
        # 0 for open land, and CLEARING where anything else is built, so the woods give it room.
        woods = np.isin(self._biomes, WOODS) | woodland
        self._woods = np.where(built, CLEARING, woods).astype(np.float32)
        grid = self.tile_map.grid
        self._plots = plots_of(self.constructions, FIELD_KINDS)
        # Where each kind's fields sit in the picture, to find those near a block quickly.
        self._plot_pixels: dict[BuildingKind, tuple[FloatArray, FloatArray]] = {}
        for kind, plots in self._plots.items():
            cells = np.array([plot.cell for plot in plots], dtype=np.int64).reshape(-1, 2)
            px, py = self.projection.to_pixels(*grid.center(cells[:, 0], cells[:, 1]))
            self._plot_pixels[kind] = (np.atleast_1d(px), np.atleast_1d(py))

    def render(
        self,
        left: int,
        top: int,
        rows: int,
        raised: Raised | None = None,
        points: Lattice | None = None,
    ) -> tuple[FloatArray, FloatArray]:
        """The ground over CHUNK art pixels across and `rows` down from (`left`, `top`), as RGB
        from 0 to 255 already multiplied by its alpha, and its alpha: 1 on the map, 0 off it.
        `raised` is the raised ground over at least these pixels, if any rises there; `points`
        may hold their plane points and cells already, and those of the APRON round them.

        Each pixel is coloured by the one kind of ground it shows: water, meadow, savanna or
        the plain ramp of its biome, each worked out only at its own pixels. Where raised ground
        takes the mountains' colours fully (relief.mountain_cover), its own colour is not
        worked out at all."""
        a = APRON
        around = (left - a, top - a, CHUNK + 2 * a, rows + 2 * a)
        points = _lattice(points, self.projection, self.tile_map.grid, around)
        r, c, inside = points.cells(*around)
        water = _WATER[self._biomes[r, c]]
        # Only shores look at the apron: the colours are worked out inside it.
        inner = (slice(a, a + rows), slice(a, a + CHUNK))
        r, c, inside, wet = r[inner], c[inner], inside[inner], water[inner]
        biomes = self._biomes[r, c]
        x, y = points.plane(left, top, CHUNK, rows)
        rocky = self.tile_map.relief[r, c] == Relief.MOUNTAINS
        seen = np.ones(x.shape, dtype=bool)
        if raised is not None:
            seen = mountain_cover(raised.mountains[:rows], rocky) < 1
        meadow = _MEADOW[biomes] & ~wet
        savanna = (biomes == Biome.SAVANNA) & ~wet
        plain = ~(wet | meadow | savanna) & seen
        meadow &= seen
        savanna &= seen
        dry = ~wet
        lit = meadow | savanna | plain
        light = np.zeros(x.shape, dtype=np.float64)
        light[lit] = sample(
            self._relief_light, x[lit] * RELIEF_STEPS_PER_CELL, y[lit] * RELIEF_STEPS_PER_CELL
        )
        rgb = np.zeros((*x.shape, 3), dtype=np.float32)
        shade = np.zeros(x.shape, dtype=np.int64)
        down, across = np.nonzero(wet & seen)
        if len(down):
            ix, iy = left + across, top + down
            ripple = table_hash(ix // 3, iy, 13) > 0.94
            calm = 1.3 + 0.6 * (table_hash(ix, iy, 11) - 0.5)
            shade[down, across] = np.clip(np.rint(np.where(ripple, 3, calm)), 0, 3).astype(np.int64)
        down, across = np.nonzero(plain)
        if len(down):
            ix, iy = left + across, top + down
            fx, fy = x[plain].astype(np.float32), y[plain].astype(np.float32)
            value = 1.45 + 1.6 * (noise(fx, fy, 2.4, 41) - 0.5)
            value += 0.8 * (noise(fx, fy, 0.75, 42) - 0.5)
            value += 0.45 * (noise(fx, fy, 0.22, 43) - 0.5) + 0.45 * BAYER[iy % 4, ix % 4]
            value += light[plain] + HILL_LIFT * self._hills[r[plain], c[plain]]
            shade[plain] = np.clip(np.rint(value), 0, 3).astype(np.int64)
        ramped = (wet & seen) | plain
        rgb[ramped] = self.ramps[biomes[ramped], shade[ramped]]
        if meadow.any() or savanna.any():
            shaded = self._tree_shade(left, top, x.shape)
            for grass, kind in ((meadow, "meadow"), (savanna, "savanna")):
                down, across = np.nonzero(grass)
                if not len(down):
                    continue
                args = (x[grass], y[grass], left + across, top + down, light[grass])
                colour = _meadow(*args, kind).astype(np.float32)
                colour *= (1 - TREE_SHADE * shaded[grass])[:, None]
                rgb[grass] = colour
        tufted = plain & _GRASSY[biomes]
        down, across = np.nonzero(tufted)
        if len(down):
            ix, iy = left + across, top + down
            tuft = (table_hash(ix // 6, iy // 6, 51) > 1 - TUFT_SHARE) & TUFT[iy % 6, ix % 6]
            down, across = down[tuft], across[tuft]
            lower = np.maximum(shade[down, across] - 1, 0)
            rgb[down, across] = self.ramps[biomes[down, across], lower] * 0.92
        near_water = _near(water, a, rows, [(0, 1), (0, -1), (1, 0), (-1, 0), (0, 2), (0, -2)])
        near_land = _near(~water, a, rows, [(0, 1), (0, -1), (1, 0), (-1, 0)])
        rgb[~wet & near_water & inside] = SAND
        rgb[wet & near_land & inside] = FOAM
        # Ground that does not glow keeps its colour exactly, so only glowing ground is blended.
        strength = glow_strength(self.satisfaction[r, c])
        glowing = (strength > 0) & dry
        if glowing.any():
            served = strength[glowing].astype(np.float32)[:, None]
            rgb[glowing] = rgb[glowing] * (1 - served) + np.array(GLOW, np.float32) * served
        if self.tile_map.deposits is not None:
            _sign_deposits(rgb, self.tile_map.deposits[r, c], left, top, wet)
        alpha = inside.astype(np.float32)
        if raised is not None:
            self.elevation.colour(rgb, raised.top(rows), x, y, rocky)
        return np.clip(rgb, 0, 255) * alpha[..., None], alpha

    @property
    def biomes(self) -> npt.NDArray[np.integer]:
        """Each cell's biome."""
        return self._biomes

    @property
    def hills(self) -> BoolArray:
        """Where land is hilly."""
        return self._hills

    @property
    def slopes_light(self) -> FloatArray:
        """How many shades lighter or darker the slopes are, RELIEF_STEPS_PER_CELL steps a cell."""
        return self._relief_light

    def under(
        self, left: int, top: int, rows: int, points: Lattice | None = None
    ) -> tuple[BoolArray, FloatArray]:
        """For the CHUNK art pixels across and `rows` down from (`left`, `top`): which lie on
        the map, and how deeply trees shade the grass there, as the cover of a black layer. For
        a chunk drawn over ground the graphics card draws, which leaves the shade out. `points`
        may hold their plane points and cells already."""
        block = (left, top, CHUNK, rows)
        points = _lattice(points, self.projection, self.tile_map.grid, block)
        r, c, inside = points.cells(*block)
        grass = _GRASS[self._biomes[r, c]] & inside
        cover = np.zeros(inside.shape, dtype=np.float32)
        if grass.any():
            cover[grass] = TREE_SHADE * self._tree_shade(left, top, inside.shape)[grass]
        return inside, cover

    def fields(
        self,
        canvas: FloatArray,
        left: int,
        top: int,
        k: int,
        lit: FloatArray | None = None,
        shown: BoolArray | None = None,
        points: Lattice | None = None,
    ) -> None:
        """Paints farms' fields and groves over a picture of the ground from (`left`, `top`),
        `k` pixels to an art pixel, RGB from 0 to 1 multiplied by its alpha, worked out at that
        resolution so they keep up with the buildings drawn as finely; `lit` brightens or
        darkens each art pixel with the slope it lies on. They go only where the picture shows the
        map: where `shown`, or else where it isn't clear. Only the patches round fields are
        worked out, from `points` where they hold them at this resolution."""
        rows, a = canvas.shape[0] // k, APRON
        shape = (rows + 2 * a, CHUNK + 2 * a)
        origin = (left - a, top - a)
        jobs = [
            (plot, patch, crop, None)
            for crop in (None, *CROP_ROWS)
            for plot, patch in self._patches(
                BuildingKind.FARM if crop is None else crop, origin, shape, k
            )
        ]
        jobs += [
            (plot, patch, None, grove)
            for kind, grove in GROVES.items()
            for plot, patch in self._patches(kind, origin, shape, k)
        ]
        if not jobs:
            return
        # Patches are slices of a picture of the ground with the apron round it; paint covers
        # just the box round all of them.
        down = min(patch[0].start for _, patch, _, _ in jobs)
        across = min(patch[1].start for _, patch, _, _ in jobs)
        bottom = max(patch[0].stop for _, patch, _, _ in jobs)
        right = max(patch[1].stop for _, patch, _, _ in jobs)
        paint = np.empty((bottom - down, right - across, 3), dtype=np.float32)
        painted = np.zeros(paint.shape[:2], dtype=bool)
        # Each patch works out the plane points and cells of its own pixels, unless patches
        # overlap so much that working them out once over the box round all of them is less.
        boxes = {(p[0].start, p[0].stop, p[1].start, p[1].stop) for _, p, _, _ in jobs}
        shared = None
        if sum((b[1] - b[0]) * (b[3] - b[2]) for b in boxes) > paint.shape[0] * paint.shape[1]:
            shared = self._patch_points((down, bottom, across, right), left - a, top - a, k, points)
        worked: dict[tuple[int, int, int, int], tuple] = {}
        for plot, patch, crop, grove in jobs:
            box = (patch[0].start, patch[0].stop, patch[1].start, patch[1].stop)
            place = (slice(box[0] - down, box[1] - down), slice(box[2] - across, box[3] - across))
            if box not in worked:
                worked[box] = (
                    self._patch_points(box, left - a, top - a, k, points)
                    if shared is None
                    else tuple(values[place] for values in shared)
                )
            args = (paint[place], *worked[box])
            if grove is None:
                painted[place] |= self._sow(*args, plot, crop, k)
            else:
                painted[place] |= self._plant(*args, plot, grove)
        top_row, left_col = max(down, a * k), max(across, a * k)
        bottom_row, right_col = min(bottom, (a + rows) * k), min(right, (a + CHUNK) * k)
        if top_row >= bottom_row or left_col >= right_col:
            return
        inside = (
            slice(top_row - down, bottom_row - down),
            slice(left_col - across, right_col - across),
        )
        inner = paint[inside]
        region = (
            slice(top_row - a * k, bottom_row - a * k),
            slice(left_col - a * k, right_col - a * k),
        )
        under = canvas[region]
        laid = painted[inside] & (under[..., 3] > 0 if shown is None else shown[region])
        if not laid.any():
            return
        colour = inner[laid]
        if lit is not None:
            art_rows, art_cols = np.nonzero(laid)
            art_rows, art_cols = (
                (art_rows + region[0].start) // k,
                (art_cols + region[1].start) // k,
            )
            colour = colour * lit[art_rows, art_cols][:, None]
        under[laid, :3] = np.clip(colour, 0, 255) / 255
        under[laid, 3] = 1.0

    def _patch_points(
        self,
        box: tuple[int, int, int, int],
        left: int,
        top: int,
        k: int,
        points: Lattice | None,
    ) -> tuple[FloatArray, FloatArray, IntArray, IntArray, IntArray]:
        """The plane points, cells (as flat indices), and pixel places of the pixels in rows and
        columns `box` of a picture from art pixel (`left`, `top`), `k` pixels to an art pixel."""
        grid = self.tile_map.grid
        first_row, last_row, first_col, last_col = box
        fy = np.arange(first_row, last_row)[:, None]
        fx = np.arange(first_col, last_col)[None, :]
        shape = (last_row - first_row, last_col - first_col)
        block = (left + first_col, top + first_row, shape[1], shape[0])
        if k == 1 and points is not None and points.k == 1 and points.holds(*block):
            x, y = points.plane(*block)
            r, c, _ = points.cells(*block)
        else:
            x, y = self.projection.to_plane(left + (fx + 0.5) / k, top + (fy + 0.5) / k)
            r, c = grid.cell_at(x, y)
            r, c = np.clip(r, 0, grid.rows - 1), np.clip(c, 0, grid.cols - 1)
        ix = np.broadcast_to(left * k + fx, shape)
        iy = np.broadcast_to(top * k + fy, shape)
        return x, y, r * grid.cols + c, ix, iy

    def yards(
        self, canvas: FloatArray, left: int, top: int, k: int, shown: BoolArray | None = None
    ) -> None:
        """Paints the ground round buildings over a picture of the ground from (`left`, `top`),
        `k` pixels to an art pixel, RGB from 0 to 1 multiplied by its alpha (YARD_REACH), where
        the picture shows the map, as `fields` says."""
        xs, ys, kinds, reaches = self.buildings
        rows, cols = canvas.shape[0] // k, canvas.shape[1] // k
        reach = float(reaches.max()) if reaches.size else 0.0
        near = (xs > left - reach) & (xs < left + cols + reach)
        near &= (ys > top - reach) & (ys < top + rows + reach)
        if not near.any():
            return
        shown = _on_map(canvas, shown)
        for x0, y0, kind, out in zip(
            xs[near].tolist(),
            ys[near].tolist(),
            kinds[near].tolist(),
            reaches[near].tolist(),
            strict=True,
        ):
            self._yard(canvas, shown, (x0 - left) * k, (y0 - top) * k, kind, out, k)

    def _yard(
        self,
        canvas: FloatArray,
        shown: npt.NDArray[np.bool_],
        cx: float,
        cy: float,
        kind: str,
        reach: float,
        k: int,
    ) -> None:
        rx = reach * k
        ry = rx / 2
        x0, x1 = max(int(cx - rx) - 1, 0), min(int(cx + rx) + 2, canvas.shape[1])
        y0, y1 = max(int(cy - ry) - 1, 0), min(int(cy + ry) + 2, canvas.shape[0])
        if x0 >= x1 or y0 >= y1:
            return
        fy, fx = np.mgrid[y0:y1, x0:x1]
        seed = int(cx * 7 + cy * 13) & 1023
        d = np.hypot((fx + 0.5 - cx) / rx, (fy + 0.5 - cy) / ry)
        fray = table_hash(fx // max(k, 1), fy // max(k, 1), seed) * 0.35
        cover = np.clip((1.0 - d - fray * 0.6) * 4, 0, 1)
        base = np.array(YARD_COLOURS[kind], dtype=np.float32)
        if kind == "cottage":
            colour = base * (0.9 + 0.2 * table_hash(fx, fy, 31))[..., None]
            u = (fx + 0.5 - cx) / k
            v = (fy + 0.5 - cy) / k
            garden = (u > 1.5) & (u < 6.5) & (v > -3.4) & (v < -0.6)
            colour[garden] = SOIL * 0.95
            rows = garden & (np.floor((v + 3.4) * 1.4) % 2 == 0)
            leaves = rows & (table_hash(fx, fy, 32) > 0.25)
            colour[leaves] = (
                GARDEN_LEAVES * (0.9 + 0.3 * table_hash(fx[leaves], fy[leaves], 33))[:, None]
            )
            cover = np.maximum(cover, garden.astype(np.float32))
        elif kind in ("paved", "flagstones"):
            size = 2 if kind == "paved" else 3
            colour = base * (0.92 + 0.12 * table_hash(fx // (size * k), fy // k, 34))[..., None]
            joints = ((fx // (size * k)) + (fy // k)) % 3 == 0
            colour[joints & (table_hash(fx, fy, 35) > 0.5)] *= 0.88
        elif kind == "spoil":
            colour = base * (0.8 + 0.35 * table_hash(fx, fy, 38))[..., None]
            stones = table_hash(fx // max(k, 1), fy // max(k, 1), 39) > 0.9
            colour[stones] = np.array((150, 144, 134), dtype=np.float32)
        else:
            colour = base * (0.86 + 0.24 * table_hash(fx, fy, 36))[..., None]
        cover *= shown[y0:y1, x0:x1]
        _over(canvas[y0:y1, x0:x1], colour / 255, cover)

    def canopy(
        self,
        canvas: FloatArray,
        left: int,
        top: int,
        k: int,
        shown: BoolArray | None = None,
        points: Lattice | None = None,
    ) -> None:
        """Paints the woods over a picture of the ground from (`left`, `top`), `k` pixels to an
        art pixel, RGB from 0 to 1 multiplied by its alpha: a roof of lit crowns with the dark
        floor between them, and their shadow on the ground beside them, worked out an art pixel
        at a time like the rest of the ground, where the picture shows the map, as `fields`
        says. `points` may hold the plane points and cells of the art pixels already."""
        rows, cols = canvas.shape[0] // k, canvas.shape[1] // k
        reach = CANOPY_REACH
        block = (left - reach, top - reach, cols + reach, rows + reach)
        points = _lattice(points, self.projection, self.tile_map.grid, block)
        r, c, _ = points.cells(*block)
        if not (self._woods[r.min() : r.max() + 2, c.min() : c.max() + 2] > 0).any():
            return
        woods = self._wooded(points, block, r, c)
        if woods.any():
            self._clear_round_mills(woods, block)
        if woods.any() and self.elevation.most(*block)[0] >= BARE_ABOVE - BARE_MARGIN:
            x, y = points.plane(*block)
            woods[woods] = self.elevation.mountainous(x[woods], y[woods]) < BARE_ABOVE
        shadow = woods[: rows + reach - 3, : cols + reach - 2]
        shadow = shadow[reach - 3 :, reach - 2 :] & ~woods[reach:, reach:]
        woods = woods[reach:, reach:]
        shown = canvas[::k, ::k, 3] > 0 if shown is None else shown[::k, ::k]
        blocks = art_pixels(canvas, k)
        down, across = np.nonzero(shadow & shown)
        if len(down):
            shaded = blocks[down, :, across]
            shaded[..., :3] *= EDGE_SHADOW
            shaded[..., 3] = shaded[..., 3] * EDGE_SHADOW + (1 - EDGE_SHADOW)
            blocks[down, :, across] = shaded
        down, across = np.nonzero(woods & shown)
        if not len(down):
            return
        x, y = points.plane(*block)
        x, y = x[down + reach, across + reach], y[down + reach, across + reach]
        fx, fy = x.astype(np.float32), y.astype(np.float32)
        biomes = self._biomes[r[down + reach, across + reach], c[down + reach, across + reach]]
        d, dx, dy, gx, gy = _nearest_crown(left + across + 0.5, top + down + 0.5, CROWN_GAP)
        radius = 1.25 + 0.75 * table_hash(gx, gy, 53)
        conifers = np.where(biomes == Biome.TAIGA, 0.9, 0.0)
        broadleaved = (biomes != Biome.TAIGA) & (biomes != Biome.JUNGLE)
        conifers[broadleaved] = 0.2 + 0.6 * noise(fx[broadleaved], fy[broadleaved], 1.1, 64)
        conifer = table_hash(gx, gy, 63) < conifers
        towards = (-dx * 0.7 - dy * 0.7) / np.maximum(radius, 0.1)
        part = np.where(towards > 0.25, 2, np.where(towards < -0.45, 0, 1))
        leaves = np.where(conifer[:, None], CONIFER_CROWNS[part], LEAFY_CROWNS[part])
        leaves *= (0.88 + 0.28 * noise(fx, fy, 0.9, 65))[:, None]
        colour = np.where((d < radius)[:, None], leaves, WOODS_FLOOR) / 255
        blocks[down, :, across, :, :3] = colour[:, None, None, :]
        blocks[down, :, across, :, 3] = 1.0

    def _clear_round_mills(self, woods: BoolArray, block: tuple[int, int, int, int]) -> None:
        """Takes the woods out of the clearing round each lumber mill in the block, an oval
        frayed at its edge like a yard's."""
        left, top = block[0], block[1]
        rows, cols = woods.shape
        xs, ys, reaches = self.clearings
        reach = float(reaches.max()) + CLEARING_DROP if reaches.size else 0.0
        near = (xs > left - reach) & (xs < left + cols + reach)
        near &= (ys > top - reach) & (ys < top + rows + reach)
        for cx, foot, rx in zip(
            xs[near].tolist(), ys[near].tolist(), reaches[near].tolist(), strict=True
        ):
            cy, ry = foot + CLEARING_DROP, rx / 2
            x0, x1 = max(int(cx - rx) - 1, left), min(int(cx + rx) + 2, left + cols)
            y0, y1 = max(int(cy - ry) - 1, top), min(int(cy + ry) + 2, top + rows)
            if x0 >= x1 or y0 >= y1:
                continue
            fy, fx = np.mgrid[y0:y1, x0:x1]
            seed = int(cx * 7 + cy * 13) & 1023
            d = np.hypot((fx + 0.5 - cx) / rx, (fy + 0.5 - cy) / ry)
            clear = d + table_hash(fx, fy, seed) * 0.35 < 1.0
            woods[y0 - top : y1 - top, x0 - left : x1 - left] &= ~clear

    def _wooded(
        self, points: Lattice, block: tuple[int, int, int, int], r: IntArray, c: IntArray
    ) -> BoolArray:
        """Which art pixels of the block woods cover, before bare mountainsides are taken out:
        land where the woods' share of the ground round it, frayed by noise, is over half.

        A pixel takes the share of the cells AROUND it, which all lie in its own cell or the
        cells round it, so where those cells all have the same say as its own, the share is its
        own cell's say, and where none has woods there are no woods, without looking. Woods
        cover all of a share of 1 whatever the noise, and none of a share of 0 or less, so only
        shares between need the noise."""
        grid = self.tile_map.grid
        share = self._woods[r, c]
        unsure = self._unsure_woods(r, c)
        if unsure.any():
            x, y = points.plane(*block)
            x, y = x[unsure], y[unsure]
            around = np.zeros(x.shape, dtype=np.float32)
            for dx, dy in AROUND:
                sr, sc = grid.cell_at(x + dx, y + dy)
                sr, sc = np.clip(sr, 0, grid.rows - 1), np.clip(sc, 0, grid.cols - 1)
                around += self._woods[sr, sc]
            around /= len(AROUND)
            share[unsure] = around
        land = self._land[r, c]
        woods = land & (share >= 1)
        frayed = land & (share > 0) & (share < 1)
        if frayed.any():
            x, y = points.plane(*block)
            fx, fy = x[frayed].astype(np.float32), y[frayed].astype(np.float32)
            edge = noise(fx, fy, 0.55, 61) * 0.55 + noise(fx, fy, 0.16, 62) * 0.3
            woods[frayed] = share[frayed] + (edge - 0.42) * 0.9 > 0.5
        return woods

    def _unsure_woods(self, r: IntArray, c: IntArray) -> BoolArray:
        """Which of these cells have woods among the cells round them and another say than
        their own, or lie where cells round them are off the map: where the woods' share of the
        ground at a point can't be told from its own cell alone. Worked out for just the cells
        between the first and last row and column asked about."""
        top, left = int(r.min()), int(c.min())
        block = (slice(top, int(r.max()) + 1), slice(left, int(c.max()) + 1))
        here = self._woods[block]
        there = self._woods.ravel()[self._round[(slice(None), *block)]]
        unlike = self._off_round[block] | (there != here).any(axis=0)
        wooded = (here > 0) | (there > 0).any(axis=0)
        return (unlike & wooded)[r - top, c - left]

    def _tree_shade(self, left: int, top: int, shape: tuple[int, ...]) -> FloatArray:
        """How deep in a tree's shade each pixel of a block from (`left`, `top`) lies, from 0 to
        1: a soft oval reaching out from the foot of each tree, mostly to its right."""
        shade = np.zeros(shape, dtype=np.float32)
        xs, ys = self.trees
        rows, cols = shape
        reach_x, reach_y = SHADE_REACH
        near = (xs > left - 12) & (xs < left + cols + 12) & (ys > top - 6) & (ys < top + rows + 6)
        dy, dx = np.mgrid[-4:5, -6:11]
        stamp = np.clip(1.2 - ((dx - 2) / reach_x) ** 2 - ((dy - 1) / reach_y) ** 2, 0, 1)
        for x0, y0 in zip(xs[near].tolist(), ys[near].tolist(), strict=True):
            px, py = int(x0) - left - 6, int(y0) - top - 4
            sy = slice(max(py, 0), min(py + stamp.shape[0], rows))
            sx = slice(max(px, 0), min(px + stamp.shape[1], cols))
            if sy.start < sy.stop and sx.start < sx.stop:
                part = stamp[sy.start - py : sy.stop - py, sx.start - px : sx.stop - px]
                np.maximum(shade[sy, sx], part, out=shade[sy, sx])
        return shade

    def _patches(
        self, kind: BuildingKind, origin: tuple[int, int], shape: tuple[int, ...], k: int = 1
    ) -> list[tuple[Plot, tuple[slice, slice]]]:
        """The fields of a kind near a block of art pixels starting at `origin`, each with the
        slice of the block, drawn `k` pixels to an art pixel, its cell can cover."""
        px, py = self._plot_pixels[kind]
        left, top = origin
        rows, cols = shape[0] * k, shape[1] * k
        near = np.flatnonzero((px > left - 24) & (px < left + shape[1] + 24))
        near = near[(py[near] > top - 16) & (py[near] < top + shape[0] + 16)]
        patches = []
        for i in near.tolist():
            x0, y0 = int(px[i] - left - 20) * k, int(py[i] - top - 12) * k
            box = (
                slice(max(y0, 0), min(y0 + 24 * k, rows)),
                slice(max(x0, 0), min(x0 + 40 * k, cols)),
            )
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
        k: int = 1,
    ) -> BoolArray:
        """Lays a field of crops in rows over the half of a cell a farm or crop takes up, `k`
        pixels to an art pixel: a farm grows wheat, other crops grow in rows. Plants are
        lit on the side towards the light and shaded on the other, the soil between rows is
        darker in the furrow, a strip of bare headland runs round inside a lumpy hedge.
        Returns which pixels the field covers."""
        grid = self.tile_map.grid
        row, col = plot.cell
        fx, fy = grid.center(row, col)
        field = (cells == row * grid.cols + col) & plot.covers(x - fx, y - fy)
        if not field.any():
            return field
        pick = row * 7 + col * 13
        across, along = (x - fx, y - fy) if pick % 2 else (y - fy, x - fx)
        u, w = across[field] * FIELD_STRIPES, along[field] * FIELD_STRIPES
        line = np.floor(u).astype(np.int64)
        fu = u - line
        px, py = ix[field], iy[field]
        grain = (table_hash(px, py, 71) - 0.5)[:, None]
        out = np.empty((u.size, 3), dtype=np.float32)
        if crop is None:
            out[:] = np.where((line % 2 == 0)[:, None], WHEAT, WHEAT_DARK)
            out[fu < 0.2] = WHEAT_LIGHT
            out[fu > 0.88] = WHEAT_DARK * 0.86
            out *= 1 + 0.14 * grain
            out[(table_hash(px, py, 72) > 0.72) & (fu > 0.3) & (fu < 0.85)] = WHEAT_EARS
        else:
            plant_rgb, ground_rgb = CROP_ROWS[crop]
            soil = np.array(ground_rgb, dtype=np.float32) * 0.8 + SOIL * 0.2
            plant = np.array(plant_rgb, dtype=np.float32)
            light = np.minimum(plant * 1.18, 255)
            out[:] = soil * (1 + 0.16 * grain)
            out[fu < 0.12] = SOIL_DARK
            lump = w * 2.2
            bushes = table_hash(np.floor(lump).astype(np.int64), line, 81) > 0.18
            round_ = np.hypot(fu - 0.5, (lump - np.floor(lump) - 0.5) * 0.7) < 0.34
            bush = round_ & bushes
            out[bush] = plant * (1 + 0.1 * grain[bush])
            out[bush & (fu < 0.42)] = light
            out[bush & (fu > 0.66)] = plant * 0.7
            leaf = bush & (table_hash(px, py, 82) > 0.86)
            out[leaf] = light * 1.08
        depth = _depths(field, 2 * k)
        hedge = depth < k
        headland = (depth >= k) & (depth < 2 * k)
        lumps = 0.78 + 0.4 * table_hash(px[hedge] // (2 * k), py[hedge] // (2 * k), 91)
        out[headland] = HEADLAND * (1 + 0.16 * grain[headland])
        out[hedge] = HEDGE * lumps[:, None]
        dark = (hedge & ~np.roll(field, -k, 0)[field]) if k > 1 else np.zeros_like(hedge)
        out[dark] *= 0.72
        rgb[field] = out
        return field

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
    ) -> BoolArray:
        """Lays a grove of trees in rows over the half of a cell an orchard, date palms or
        plantation takes up: each crown lit on the side towards the light, with its shadow on
        the ground beside it, and some bearing fruit. Returns which pixels the grove covers."""
        grid = self.tile_map.grid
        row, col = plot.cell
        fx, fy = grid.center(row, col)
        ground = (cells == row * grid.cols + col) & plot.covers(x - fx, y - fy)
        if not ground.any():
            return ground
        u = (x[ground] - fx) * grove.rows + 0.5
        v = (y[ground] - fy) * grove.rows + 0.5
        du, dv = u - np.round(u), v - np.round(v)
        near = np.hypot(du, dv)
        px, py = ix[ground], iy[ground]
        grain = 1 + 0.12 * (table_hash(px, py, 23) - 0.5)
        floor = np.array(grove.ground, dtype=np.float32)
        colors = (floor * grain[:, None]).astype(np.float32)
        shadow = np.hypot(du - 0.12, dv - 0.1) < grove.crown
        colors[shadow] *= 0.74
        tree = near < grove.crown
        leaves = np.array(grove.tree, dtype=np.float32)
        towards = -(du + dv) / max(grove.crown, 1e-6)
        lit = np.clip(0.82 + 0.3 * towards, 0.62, 1.22)
        colors[tree] = np.clip(leaves * (lit[tree] * grain[tree])[:, None], 0, 255)
        fruit = tree & (table_hash(px, py, 21) > 0.82)
        pick = (table_hash(px, py, 22) * len(grove.fruit)).astype(np.int64)
        for index, color in enumerate(grove.fruit):
            colors[fruit & (pick == index)] = color
        rgb[ground] = colors
        return ground


def _depths(field: BoolArray, most: int) -> npt.NDArray[np.int16]:
    """How many steps in from its edge each pixel of a field lies, up to `most`, in the order
    of the field's pixels; the edges of the block wrap round to its other side, as np.roll's do.
    The field shrinks with every step, so a pixel's depth is how many steps it stays in. A field
    clear of the block's edges is shrunk in just the box round it and a pixel more, where the
    wrapping brings in nothing but clear pixels."""
    rows, cols = np.flatnonzero(field.any(axis=1)), np.flatnonzero(field.any(axis=0))
    top, bottom, left, right = rows[0], rows[-1], cols[0], cols[-1]
    if top > 0 and left > 0 and bottom < field.shape[0] - 1 and right < field.shape[1] - 1:
        field = field[top - 1 : bottom + 2, left - 1 : right + 2]
    depth = np.zeros(field.shape, dtype=np.int16)
    inside = field
    for _ in range(most):
        inside = _shrunk(inside)
        depth += inside
    return depth[field]


def _shrunk(inside: BoolArray) -> BoolArray:
    """`inside` less every pixel with one beside it, above, below, left or right, that is not:
    first up and down, then across, the block's edges wrapping round as np.roll's do."""
    rows = inside.copy()
    rows[1:] &= inside[:-1]
    rows[:1] &= inside[-1:]
    rows[:-1] &= inside[1:]
    rows[-1:] &= inside[:1]
    both = rows.copy()
    both[:, 1:] &= rows[:, :-1]
    both[:, :1] &= rows[:, -1:]
    both[:, :-1] &= rows[:, 1:]
    both[:, -1:] &= rows[:, :1]
    return both


def _on_map(canvas: FloatArray, shown: BoolArray | None) -> BoolArray:
    """Where a picture shows the map: `shown` if given, else wherever it isn't clear."""
    return canvas[..., 3] > 0 if shown is None else shown


def _over(under: FloatArray, colour: FloatArray, cover: FloatArray) -> None:
    """Lays `colour` (RGB from 0 to 1) over a block of a premultiplied RGBA picture, covering
    each pixel by `cover`: on opaque ground it blends the colours, on a clear picture it leaves
    what to lay over the ground later."""
    cover = cover[..., None]
    under[..., :3] = under[..., :3] * (1 - cover) + colour * cover
    under[..., 3:] = under[..., 3:] * (1 - cover) + cover


def _sign_deposits(
    rgb: FloatArray,
    deposit: npt.NDArray[np.uint8],
    left: int,
    top: int,
    water: npt.NDArray[np.bool_],
) -> None:
    """Marks deposits with a few small signs: sheep grazing wool land, rocks with a vein of their
    ore elsewhere; `rgb` is the ground from art pixel (`left`, `top`)."""
    down, across = np.nonzero((deposit > 0) & (deposit != Deposit.GARRISON) & ~water)
    if not len(down):
        return
    ix, iy = left + across, top + down
    deposit = deposit[down, across]
    colour = rgb[down, across]
    lx, ly = ix % 12, iy % 12
    here = table_hash(ix // 12, iy // 12, 71) > 1 - DEPOSIT_SIGNS
    sheep = here & (deposit == Deposit.WOOL)
    colour[sheep & (lx >= 4) & (lx <= 7) & (ly >= 5) & (ly <= 7)] = SHEEP
    colour[sheep & (lx == 3) & (ly == 5)] = SHEEP_DARK
    colour[sheep & ((lx == 4) | (lx == 7)) & (ly == 8)] = SHEEP_DARK
    colour[sheep & (lx >= 4) & (lx <= 7) & (ly == 9)] *= 0.7
    rock = here & (deposit != Deposit.WOOL)
    stone = rock & (lx >= 4) & (lx <= 7) & (ly >= 6) & (ly <= 8) & ~((lx == 7) & (ly == 6))
    colour[stone] = ORE_ROCK
    vein = rock & (lx == 5) & (ly == 7)
    colour[vein] = DEPOSIT_SPECKS[deposit[vein]]
    colour[rock & (lx >= 4) & (lx <= 8) & (ly == 9)] *= 0.72
    rgb[down, across] = colour


def art_pixels(picture: FloatArray, k: int) -> FloatArray:
    """The picture seen as art pixels of `k` by `k` pixels, by row, row of pixels, column, column
    of pixels and channel, sharing the picture's memory, so painting an art pixel paints all of
    its pixels."""
    rows, cols = picture.shape[0] // k, picture.shape[1] // k
    return picture.reshape(rows, k, cols, k, picture.shape[2], copy=False)


def _nearest_crown(
    px: FloatArray, py: FloatArray, gap: float
) -> tuple[FloatArray, FloatArray, FloatArray, IntArray, IntArray]:
    """The nearest crown to each picture point, crowns standing one to a square `gap` pixels a
    side at a random spot in it: how far, which way, and which square. Where the crown of every
    square round the points stands is worked out once a square; ties go to the square met
    first, going down each column of the three by three round a point's own."""
    cx, cy = np.floor(px / gap).astype(np.int64), np.floor(py / gap).astype(np.int64)
    first_x, first_y = int(cx.min()) - 1, int(cy.min()) - 1
    gx = np.arange(first_x, int(cx.max()) + 2)[None, :]
    gy = np.arange(first_y, int(cy.max()) + 2)[:, None]
    jx = ((gx + 0.15 + 0.7 * table_hash(gx, gy, 51)) * gap).ravel()
    jy = ((gy + 0.15 + 0.7 * table_hash(gx, gy, 52)) * gap).ravel()
    width = gx.shape[1]
    home = (cy - first_y) * width + (cx - first_x)
    best = np.full(px.shape, np.inf, dtype=np.float32)
    nearest = home.copy()
    for ox in (-1, 0, 1):
        for oy in (-1, 0, 1):
            square = home + (oy * width + ox)
            dx, dy = (px - jx[square]).astype(np.float32), (py - jy[square]).astype(np.float32)
            dx *= dx
            dy *= dy
            dx += dy
            closer = dx < best
            np.copyto(best, dx, where=closer)
            np.copyto(nearest, square, where=closer)
    dx, dy = (px - jx[nearest]).astype(np.float32), (py - jy[nearest]).astype(np.float32)
    return np.sqrt(best), dx, dy, nearest % width + first_x, nearest // width + first_y


def _ramp(t: FloatArray, stops: tuple[tuple[float, RGB], ...]) -> FloatArray:
    """Colours along a smooth ramp of (t, colour) stops for t from 0 to 1, held to that range
    already."""
    marks = np.array([at for at, _ in stops], dtype=np.float32)
    colours = np.array([colour for _, colour in stops], dtype=np.float32)
    rgb = np.empty((*np.shape(t), 3), dtype=np.float64)
    for k in range(3):
        rgb[..., k] = np.interp(t, marks, colours[:, k])
    return rgb


def _meadow(
    x: FloatArray,
    y: FloatArray,
    ix: IntArray,
    iy: IntArray,
    light: FloatArray,
    grass: str = "meadow",
) -> FloatArray:
    """Grass at these points, by GRASSES: for a meadow pasture, rough grass and dry patches
    blended softly, lit with the slopes, with a fine grain, blades in the rough grass and a few
    wild flowers; savanna the same in its own kinds."""
    short, tall, bare, rarity = GRASSES[grass]
    x, y = np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32)
    kind = noise(x, y, 0.55, 81) + 0.25 * (noise(x, y, 0.15, 85) - 0.5)
    t = 0.5 + 0.36 * (noise(x, y, 2.4, 41) - 0.5) + 0.14 * (noise(x, y, 0.4, 42) - 0.5) + light / 3
    rough = np.clip((kind - 0.55) * 3.5, 0, 1)
    dry = np.clip((noise(x, y, 0.45, 82) - 0.74) * 3.5, 0, 1) * (1 - rough)
    pasture = 1 - rough - dry
    t = np.clip(t, 0, 1)
    rgb = (
        _ramp(t, short) * pasture[:, None]
        + _ramp(t, tall) * rough[:, None]
        + _ramp(t, bare) * dry[:, None]
    )
    # Blades stand in even rows and their tips in the odd rows below, two rows to a hash.
    odd = iy % 2 == 1
    tall_grass = table_hash(ix, (iy + odd) // 2, 83) > 0.72
    blade, tip = tall_grass & ~odd, tall_grass & odd
    rgb *= (1 - 0.14 * rough * blade + 0.1 * rough * tip)[:, None]
    rgb += ((table_hash(ix, iy, 84) - 0.5) * 8 + (noise(x, y, 0.03, 86) - 0.5) * 10)[:, None]
    bloom = table_hash(ix, iy, 92) > rarity
    if bloom.any():
        bloom[bloom] = noise(x[bloom], y[bloom], 0.5, 91) > 0.6
        flower = (table_hash(ix[bloom], iy[bloom], 93) * len(FLOWERS)).astype(np.int64)
        rgb[bloom] = FLOWERS[flower]
    return rgb


def _near(
    mask: npt.NDArray[np.bool_], apron: int, rows: int, shifts: list[tuple[int, int]]
) -> npt.NDArray[np.bool_]:
    """Where `mask` holds at any of these shifts of each pixel inside the apron round a block
    of `rows` by CHUNK pixels."""
    near = np.zeros((rows, CHUNK), dtype=bool)
    for down, across in shifts:
        near |= mask[apron - down : apron - down + rows, apron - across : apron - across + CHUNK]
    return near


def _lattice(
    points: Lattice | None,
    projection: PixelProjection,
    grid: Grid,
    block: tuple[int, int, int, int],
) -> Lattice:
    """`points` if they hold the plane points and cells of a block of art pixels, or else a
    lattice of just the block."""
    if points is not None and points.k == 1 and points.grid is grid and points.holds(*block):
        return points
    return Lattice(projection, *block, grid=grid)


def _biome_table(biomes: tuple[Biome, ...]) -> BoolArray:
    """Which biomes are among these, by biome."""
    table = np.zeros(len(Biome), dtype=bool)
    table[list(biomes)] = True
    return table


def _cells_round(grid: Grid) -> tuple[npt.NDArray[np.int32], BoolArray]:
    """The flat index of each cell round every cell in each direction of `_round`, held to the
    map, and the cells on the map's edge or with a cell round them off it."""
    rows, cols = np.indices(grid.shape)
    x, y = grid.center(rows, cols)
    off = np.zeros(grid.shape, dtype=bool)
    off[[0, -1], :] = True
    off[:, [0, -1]] = True
    round_ = []
    for dx, dy in _round(grid):
        r, c = grid.cell_at(x + dx, y + dy)
        off |= ~grid.contains(r, c)
        r, c = np.clip(r, 0, grid.rows - 1), np.clip(c, 0, grid.cols - 1)
        round_.append((r * grid.cols + c).astype(np.int32))
    return np.stack(round_), off


def _round(grid: Grid) -> FloatArray:
    """The steps from a cell's centre to the cells round it that points AROUND a point in it can
    lie in: all those no further than the farthest of AROUND and two of the cell's corners."""
    centre = np.array(grid.center(0, 0), dtype=np.float64)
    corner = max(float(np.hypot(*(np.array(at) - centre))) for at in grid.corners(0, 0))
    reach = max(math.hypot(dx, dy) for dx, dy in AROUND) + 2 * corner
    steps = grid.track_steps()
    return steps[np.hypot(steps[:, 0], steps[:, 1]) <= reach]


_WATER = _biome_table(WATER)
_MEADOW = _biome_table(MEADOWS)
_GRASSY = _biome_table(GRASSY)
_GRASS = _biome_table((*MEADOWS, Biome.SAVANNA))
