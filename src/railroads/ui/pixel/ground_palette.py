"""The colours and tuning numbers the pixel ground is drawn with, on the CPU
(railroads.ui.pixel.ground) and on the graphics card (railroads.ui.pixel.ground_glsl): water and
shores, relief, fields, grass, woods, clearings and deposits."""

import math
from dataclasses import dataclass

import numpy as np

from railroads.sim.constructions import BuildingKind
from railroads.sim.goods import HOMES, MAKES, RECIPES
from railroads.ui.biomes import Biome
from railroads.ui.pixel.art.parts import METRE as SPRITE_METRE
from railroads.ui.pixel.art.parts import SCALE as SPRITE_SCALE
from railroads.ui.pixel.projection import RGB
from railroads.ui.pixel.relief import (
    SHADOW_TINT,
)

# Ground is worked out this far past a chunk's edges, so shores meet across chunks.
APRON = 2


# Four shades per biome, dark to light; broad patches of noise pick between them.
RAMPS: dict[Biome, list[RGB]] = {
    Biome.SEA: [(26, 56, 110), (34, 72, 136), (44, 90, 160), (110, 150, 200)],
    Biome.LAKE: [(30, 70, 130), (40, 88, 156), (52, 106, 178), (120, 164, 214)],
    Biome.RIVER: [(30, 70, 130), (40, 88, 156), (52, 106, 178), (120, 164, 214)],
    Biome.SNOW: [(174, 186, 188), (202, 212, 210), (223, 232, 226), (243, 250, 239)],
    Biome.TUNDRA: [(124, 128, 99), (143, 147, 115), (160, 165, 130), (180, 185, 149)],
    Biome.TAIGA: [(38, 59, 42), (48, 72, 52), (60, 86, 62), (77, 103, 77)],
    Biome.BADLANDS: [(138, 124, 91), (157, 143, 107), (175, 161, 122), (192, 180, 141)],
    Biome.GRASSLAND: [(69, 105, 48), (86, 125, 57), (103, 142, 66), (129, 164, 83)],
    Biome.FOREST: [(37, 63, 32), (47, 77, 39), (59, 92, 46), (75, 109, 55)],
    Biome.DESERT: [(171, 152, 104), (190, 171, 120), (205, 189, 136), (222, 209, 158)],
    Biome.SAVANNA: [(138, 137, 78), (157, 154, 91), (174, 172, 105), (192, 190, 123)],
    Biome.JUNGLE: [(26, 62, 32), (35, 76, 41), (45, 91, 49), (60, 108, 61)],
    Biome.MOUNTAIN: [(83, 83, 78), (106, 105, 95), (131, 130, 118), (165, 166, 150)],
}


WATER = (Biome.SEA, Biome.LAKE, Biome.RIVER)


SAND: RGB = (200, 184, 128)


# Water is lit like the land: shallow and light by the shore, deeper and darker away from it,
# by how much water lies round it (Ground._surroundings) from DEPTH_FROM to DEPTH_FROM and
# DEPTH_SPAN more; low waves brighten or darken it by WAVE_LIGHT a unit of slope towards LIGHT,
# within WAVE_RANGE, and a SPARKLE_SHARE of the pixels on waves facing it more than
# SPARKLE_FACING glint, SPARKLE_MIX of the way to SPARKLE.
SHALLOW = np.array((64, 128, 146), dtype=np.float32)
OPEN_WATER = np.array((38, 86, 134), dtype=np.float32)
DEEP_WATER = np.array((24, 56, 104), dtype=np.float32)
DEPTH_FROM = 0.45
DEPTH_SPAN = 0.5
WAVE_LIGHT = 0.012
WAVE_RANGE = (0.82, 1.2)
WAVE_STEP = 0.01
SPARKLE = np.array((206, 224, 232), dtype=np.float32)
SPARKLE_FACING = 0.17
SPARKLE_SHARE = 0.15
SPARKLE_MIX = 0.55
# Shores are soft: sand covers the land BEACH[0] fully beside the water and BEACH[1] a pixel
# further, the water is SHORE_MIX of the way to SHORE_WATER beside the land and a pixel
# further, and the bank shades the water below and right of it by BANK_SHADE.
BEACH = (0.8, 0.4)
SHORE_WATER = np.array((110, 160, 168), dtype=np.float32)
SHORE_MIX = (0.6, 0.25)
BANK_SHADE = SHADOW_TINT * 0.85
# The pixels round a pixel that a shore looks at: beside it, a pixel further, and up and left of
# it for the bank's shade, as (down, across) steps back to them.
BESIDE = [(0, 1), (0, -1), (1, 0), (-1, 0)]
FURTHER = [(0, 2), (0, -2), (2, 0), (-2, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]
WATER_FURTHER = [(0, 2), (0, -2), (2, 0), (-2, 0)]
UP_LEFT = [(1, 1), (2, 1), (2, 2)]


FIELD_STRIPES = 9


# A metre in art pixels, as the sprites measure it by the house: railroads.ui.pixel.art.parts.METRE
# painter units, each SPRITE_SCALE art pixels in a building's sprite.
ART_METRE = SPRITE_METRE * SPRITE_SCALE


# Groves of trees in rows over the ground of the cell: the ground, the trees, the colours of the
# fruit some of them bear, how many rows a cell each way and how big each crown is. A herd is a
# grove of animals instead, at their true size: each `herd` metres long with its head in the
# colour of its fruit. They gather in groups, one to a square `flock` metres a side, on a patch
# of `trodden` ground HERD_PATCH of that square across round a hay rack `shelter` metres long
# with a roof of the `roof` colour, that marks from afar what grazes there. In the patch the
# animals stand one to a square `gap` metres a side in HERD_SHARE of them, anywhere in its middle
# half and facing either way.
@dataclass(frozen=True)
class Grove:
    ground: RGB
    tree: RGB
    fruit: tuple[RGB, ...]
    rows: int
    crown: float
    herd: float = 0.0
    gap: float = 0.0
    flock: float = 0.0
    trodden: RGB = (0, 0, 0)
    shelter: float = 0.0
    roof: RGB = (0, 0, 0)


HERD_SHARE = 0.6
HERD_PATCH = 0.6


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
    BuildingKind.SHEEP_FARM: Grove(
        (140, 182, 86),
        (238, 234, 222),
        ((60, 56, 54),),
        4,
        0.2,
        herd=1.3,
        gap=2.4,
        flock=21.0,
        trodden=(156, 164, 100),
        shelter=3.0,
        roof=(206, 176, 92),
    ),
    BuildingKind.HOP_GARDEN: Grove((150, 140, 96), (72, 132, 50), ((170, 214, 110),), 7, 0.34),
    BuildingKind.TEA_PLANTATION: Grove((128, 150, 80), (56, 120, 52), ((120, 176, 80),), 6, 0.3),
}


# Crops in rows, like a farm's furrows.
CROP_ROWS: dict[BuildingKind, tuple[RGB, RGB]] = {
    BuildingKind.SUGAR: ((156, 200, 100), (118, 162, 74)),
    BuildingKind.COTTON_FIELD: ((238, 236, 226), (150, 172, 100)),
    BuildingKind.POTATO_FARM: ((150, 120, 80), (96, 140, 64)),
    BuildingKind.RICE_PADDY: ((112, 184, 72), (104, 150, 170)),
    BuildingKind.SOY_FIELD: ((104, 156, 62), (134, 108, 72)),
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


# Plains are lit and coloured like the mountains beside them (railroads.ui.pixel.relief): they
# roll in low swells that catch the light from LIGHT, brightening or darkening by up to
# ROLL_RANGE, are greener near water and lie in a cool shade at the mountains' foot. How far
# the water and the mountains reach, in cells, how strongly near they count, and how much
# greener and darker they make the ground at the most.
ROLL_LIGHT = 0.14
# The plains differ from place to place too: their swells are low in some and deeper in others,
# from ROLL_LEAST to ROLL_MOST times ROLL_LIGHT over ROLL_SPAN cells, and their grass is drier or
# greener over PLAINS_SPAN cells, by up to these tints.
ROLL_SPAN = 2.5
ROLL_LEAST, ROLL_MOST = 0.3, 1.7
PLAINS_SPAN = 3.0
DRIER_PLAINS = np.array((1.06, 1.02, 0.88), dtype=np.float32)
GREENER_PLAINS = np.array((0.92, 1.0, 0.94), dtype=np.float32)
PLAINS_TINT = 0.6
ROLL_RANGE = (0.72, 1.24)
ROLL_STEP = 0.04
WET_REACH = 1.5
FOOT_REACH = 2.5
WET_GAIN = 2.5
FOOT_GAIN = 2.2
LUSH = 0.4
LUSH_TINT = np.array((0.72, 1.0, 0.7), dtype=np.float32)
FOOT_SHADE = 0.45
FOOT_TINT = SHADOW_TINT * 0.8
# Biomes and mountains on land meet along a winding line instead of the cells' edges: each pixel
# takes the biome and mountains of the cell under a point near it, pushed by noise up to WARP
# cells, WARP_FINE more on a smaller scale and WARP_BROAD more on a broader one, unless that
# cell is water. Together they push a point less than two rings of cells away.
WARP = 0.32
WARP_FINE = 0.12
WARP_BROAD = 0.3


# Farms grow wheat, which is what they make; other crops grow in rows (CROP_ROWS) over soil.
SOIL = np.array((128, 88, 54), dtype=np.float32)
SOIL_DARK = np.array((100, 68, 42), dtype=np.float32)
WHEAT = np.array((218, 184, 86), dtype=np.float32)
WHEAT_DARK = np.array((200, 166, 72), dtype=np.float32)
WHEAT_LIGHT = np.array((238, 210, 116), dtype=np.float32)
WHEAT_EARS = np.array((170, 134, 54), dtype=np.float32)
# Fields are ploughed into the ground rather than laid on it: their colours are
# FIELD_SATURATION as strong, and they fade into the grass over FIELD_FEATHER art pixels at an
# edge frayed by noise FIELD_FRAY art pixels deep over FIELD_FRAY_SIZE cells. A grove's own
# ground covers the grass GROVE_FLOOR fully, and its trees' shadows GROVE_SHADOW.
FIELD_SATURATION = 0.88
FIELD_FEATHER = 2
FIELD_FRAY = 2.0
FIELD_FRAY_SIZE = 0.08
GROVE_FLOOR = 0.45
GROVE_SHADOW = 0.4


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
PASTURE = ((0.0, (62, 95, 42)), (0.5, (92, 132, 62)), (1.0, (134, 166, 88)))
ROUGH = ((0.0, (54, 84, 38)), (0.5, (79, 114, 53)), (1.0, (115, 147, 74)))
DRY = ((0.0, (98, 115, 57)), (0.5, (127, 142, 75)), (1.0, (159, 168, 100)))
# Savanna is drawn the same way in its own three kinds: golden grass, greener grass where it is
# damper, and patches of bare red earth.
GOLDEN = ((0.0, (137, 126, 71)), (0.5, (178, 167, 100)), (1.0, (208, 201, 138)))
DAMP = ((0.0, (121, 118, 64)), (0.5, (155, 152, 87)), (1.0, (188, 186, 119)))
EARTH = ((0.0, (124, 97, 67)), (0.5, (156, 127, 90)), (1.0, (186, 162, 120)))
# Tundra is drawn the same way as moss, brownish heath and pale lichen, and desert as sand,
# rippled sand and reddish gravel, lit along its dunes.
MOSS = ((0.0, (80, 92, 60)), (0.5, (112, 122, 82)), (1.0, (146, 152, 110)))
HEATH = ((0.0, (86, 80, 58)), (0.5, (118, 108, 78)), (1.0, (150, 140, 106)))
LICHEN = ((0.0, (106, 110, 98)), (0.5, (142, 146, 132)), (1.0, (178, 182, 168)))
SAND_FLAT = ((0.0, (166, 140, 98)), (0.5, (202, 178, 130)), (1.0, (228, 210, 162)))
SAND_RIPPLED = ((0.0, (150, 122, 84)), (0.5, (186, 158, 112)), (1.0, (216, 192, 146)))
GRAVEL = ((0.0, (128, 98, 74)), (0.5, (160, 126, 96)), (1.0, (190, 158, 124)))
# Each grassland, and how rare wild flowers are in it (1 for none).
GRASSES = {
    "meadow": (PASTURE, ROUGH, DRY, 0.997),
    "savanna": (GOLDEN, DAMP, EARTH, 1.0),
    "tundra": (MOSS, HEATH, LICHEN, 0.998),
    "desert": (SAND_FLAT, SAND_RIPPLED, GRAVEL, 1.0),
}
# Dunes in the desert catch the light this much more for the way they face it.
DUNE_LIGHT = 0.11
DUNE_STEP = 0.02
# The colours of wild flowers in meadows.
FLOWERS = np.array([(244, 240, 232), (246, 214, 80), (196, 130, 210), (226, 90, 70)], np.float32)
# Woods are drawn as a canopy seen from above (Ground.canopy), lit the way the mountains are
# (railroads.ui.pixel.relief): crowns about CROWN_GAP art pixels apart, each a dome CROWN_SIZE
# art pixels across at most, conifers in their own darker greens, the roof they make rolling
# gently in broad swells, the whole brightened on slopes facing CROWN_LIGHT and tinted cool in
# its shade. Their edge follows the forest cells but frays, curving round clearings where
# something is built; their lower edge shows the dark side of the trees, and they throw a cool
# shadow on the ground beside them.
WOODS = (Biome.FOREST, Biome.TAIGA, Biome.JUNGLE)
CROWN_GAP = 4.4
CROWN_SIZE = (2.2, 1.8)
CROWN_LIGHT = np.array((-0.75, -0.5, 0.6)) / math.hypot(0.75, 0.5, 0.6)
CANOPY_SWELL = 0.7
CANOPY_SWELL_SIZE = 0.6
CROWN_SLOPES = 0.15
# Woods differ from stand to stand, so no two look alike: old stands of big crowns grown into
# each other and young ones of small crowns with gaps between (by noise over STAND_SPAN art
# pixels, crowns from YOUNG to OLD times their size), thin patches where up to THINNED of the
# crowns are missing (over THIN_SPAN art pixels), stands of conifers among broadleaved trees
# (over CONIFER_STANDS cells), and their greens turning darker or more golden over REGION_SPAN
# cells, by these tints.
STAND_SPAN = 110.0
YOUNG, OLD = 0.7, 1.25
THIN_SPAN = 70.0
THINNED = 0.7
CONIFER_STANDS = 1.3
REGION_SPAN = 2.2
DARK_GREENS = np.array((0.8, 0.9, 0.94), dtype=np.float32)
GOLDEN_GREENS = np.array((1.14, 1.06, 0.84), dtype=np.float32)
# How far round towards their rims crowns curve, how dark the hollows between them are, how
# fine the grain of their leaves, how far in from its edge a crown shows a dark rim and how
# dark, and from where and how much brighter the tops catching the light most are.
CROWN_ROUND = 0.9
CROWN_DEPTH = 0.5
LEAF_GRAIN = 0.04
LUMPS = 0.5
CROWN_RIM = (0.2, 0.7)
CROWN_SHINE = (0.85, 0.3)
# The darkest crowns get, as a share of their colour in full light.
CROWN_DARKEST = 0.32
CROWN_TONES = 0.5
# How many art pixels above and left of a block the woods are worked out, for their shadow.
CANOPY_REACH = 4
CONIFER_CROWNS = np.array([(58, 86, 58), (64, 90, 56)], dtype=np.float32)
LEAFY_CROWNS = np.array([(80, 108, 54), (90, 114, 56)], dtype=np.float32)
WOODS_FLOOR = np.array((54, 70, 40), dtype=np.float32)
WOODS_SIDE = 0.7
EDGE_SHADOW = 0.28
# How far right and down of the crowns their shadow falls, in art pixels.
SHADOW_REACH = (2, 3)
SHADOW_COLOUR = np.array((18, 26, 46), dtype=np.float32)
# Drawn two or more pixels to an art pixel, woods show more: broadleaved crowns heaped from
# clumps of leaves about CLUMP_GAP art pixels apart, CLUMP_SIZE of that wide, each lit on its
# own by CLUMP_LIGHT, their outline scalloped by CLUMP_EDGE, with dark hollows between them where
# the crown is lower than HOLLOWS, showing BRANCHES branches; conifers as stars of CONIFER_POINTS
# drooping points, CONIFER_SLIM as wide as a broadleaved crown, in CONIFER_TIERS tiers. Crowns
# reach up to OVERHANG art pixels past the woods' edge on their own round outline, the trunks of
# the front row show below them, TRUNK_LENGTH art pixels long, and their shadow fades at its
# edge.
CLUMP_GAP = 1.5
CLUMP_SIZE = (0.62, 0.3)
CLUMP_LIGHT = 0.5
CLUMP_EDGE = 0.35
HOLLOWS = 0.4
HOLLOW_DARK = 0.5
BRANCHES = 6
BRANCH_WIDTH = 0.28
BRANCH = np.array((74, 58, 40), dtype=np.float32)
CONIFER_POINTS = 9
CONIFER_SLIM = 1.05
CONIFER_TIERS = 3.0
OVERHANG = 3
TRUNK_LENGTH = (1.0, 1.4)
TRUNK_HALF = 0.5
TRUNK_BELOW = 5
TRUNK_BARK = np.array((88, 66, 46), dtype=np.float32)
SHADOW_SOFT = 3.0
FLOOR_FROM = 0.8
# How mountainous ground may be before woods give way to the mountain's own bare slopes, and how
# far below that a bound on it (Elevation.most) must stay to show none of the woods are bare.
BARE_ABOVE = 0.3
BARE_MARGIN = 1e-4
CLEARING = -4.0
# Logging camps stand in the woods: their cells keep their trees, but for a small clearing round
# the mill itself, MILL_CLEARING of its yard's reach across and half that down, its edge frayed,
# its middle CLEARING_DROP art pixels in front of the mill so its yard shows below it.
IN_THE_WOODS = (BuildingKind.LOGGING_CAMP,)
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
# field, paving round town buildings and flagstones round churches and landmarks, spoil round
# mines and pits, sand under palms, and worn earth round works and every other building. Each is
# an oval its building's reach across and half that down, its edge frayed; a cottage's and a
# farmstead's reach is fixed.
YARD_REACH = {"cottage": 6.0, "farmyard": 11.0}
YARD_COLOURS = {
    "cottage": (150, 128, 88),
    "farmyard": (150, 128, 88),
    "paved": (168, 164, 152),
    "flagstones": (182, 176, 160),
    "worn": (142, 122, 88),
    "spoil": (118, 102, 84),
    "sand": (214, 192, 134),
}
# A building stands on the land it works with: homes, and what takes raw goods straight from the
# land (fields, farms, groves, herds, mines, pits, quarries, wells, logging camps, fisheries),
# keep their cell's own ground. Every other building, works making goods from goods, shops,
# churches, landmarks, ports and bases, lays town ground over its whole cell: setts in rows
# TOWN_STONE art pixels across and down, laid half a stone along each row, each stone its own
# shade, joints TOWN_JOINT of an art pixel wide showing once the stones are big enough, patches of
# packed earth where the TOWN_EARTH noise is high and grass through the holes where the
# TOWN_HOLES noise is. It covers where the share of town cells AROUND a point, drawn in to
# TOWN_BLEND of its reach, frayed by noise TOWN_FRAY deep, is over TOWN_FROM, so it fills its cell and gives out into the grass at its edge, and runs on
# unbroken into a town cell beside it.
ON_THE_LAND = HOMES | {kind for kind in MAKES if kind not in RECIPES}
TOWN_SETTS = np.array((132, 126, 114), dtype=np.float32)
TOWN_JOINTS = np.array((96, 90, 80), dtype=np.float32)
TOWN_EARTH_COLOUR = np.array((138, 118, 86), dtype=np.float32)
TOWN_STONE = (2.0, 1.0)
TOWN_JOINT = 0.18
TOWN_FROM = 0.3
TOWN_BLEND = 0.4
TOWN_FRAY = 0.5
TOWN_SOFT = 5.0
# How far town ground may reach from the middle of its cell, in art pixels: past its corners.
TOWN_REACH = 24.0
TOWN_EARTH = (0.32, 0.6)
TOWN_HOLES = (0.07, 0.84)
GARDEN_LEAVES = np.array((84, 132, 52), dtype=np.float32)
# Where a cottage's vegetable garden lies from the middle of its yard, in art pixels: from the
# first to the second across, and from the third to the fourth down; and how many rows of
# leaves it holds to an art pixel down.
GARDEN = (1.0, 4.6, -2.4, -0.4)
GARDEN_ROWS = 2.0
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
