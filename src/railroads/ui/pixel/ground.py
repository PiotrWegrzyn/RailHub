"""The ground of the pixel map, worked out at one pixel per art pixel: broad soft patches of a few
shades per biome, meadow, savanna, tundra and desert blended from their own kinds of ground,
water shallow by the shore and deep away from it with lit waves, soft sandy shores, slopes lit from
the top left by the land's real heights, small signs of deposits (sheep grazing wool land, rocks showing their ore elsewhere), fields of crops ploughed
into the ground under farms and rows of trees under orchards. Hills and mountains are raised, lit
and coloured by railroads.ui.pixel.relief. Textures come from noise over the map's plane and from
each pixel's place, so chunks meet without seams and look the same each time they are drawn."""

import math
from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from railroads.sim.constructions import Building, BuildingKind, Constructions
from railroads.sim.grids import Cell, FloatArray, Grid, HexGrid, IntArray
from railroads.sim.tilemap import Deposit, Relief, Terrain, TileMap
from railroads.ui.biomes import Biome, biome_grid
from railroads.ui.building_spots import Plot, plots_of
from railroads.ui.pixel.ground_palette import (
    APRON,
    AROUND,
    ART_METRE,
    BANK_SHADE,
    BARE_ABOVE,
    BARE_MARGIN,
    BAYER,
    BEACH,
    BESIDE,
    BRANCH,
    BRANCH_WIDTH,
    BRANCHES,
    CANOPY_REACH,
    CANOPY_SWELL,
    CANOPY_SWELL_SIZE,
    CLEARING,
    CLEARING_DROP,
    CLUMP_EDGE,
    CLUMP_GAP,
    CLUMP_LIGHT,
    CLUMP_SIZE,
    CONIFER_CROWNS,
    CONIFER_POINTS,
    CONIFER_SLIM,
    CONIFER_STANDS,
    CONIFER_TIERS,
    CROP_ROWS,
    CROWN_DARKEST,
    CROWN_DEPTH,
    CROWN_GAP,
    CROWN_LIGHT,
    CROWN_RIM,
    CROWN_ROUND,
    CROWN_SHINE,
    CROWN_SIZE,
    CROWN_SLOPES,
    CROWN_TONES,
    DARK_GREENS,
    DEEP_WATER,
    DEPOSIT_SIGNS,
    DEPOSIT_SPECKS,
    DEPTH_FROM,
    DEPTH_SPAN,
    DRIER_PLAINS,
    DUNE_LIGHT,
    DUNE_STEP,
    EDGE_SHADOW,
    FIELD_FEATHER,
    FIELD_FRAY,
    FIELD_FRAY_SIZE,
    FIELD_KINDS,
    FIELD_SATURATION,
    FIELD_STRIPES,
    FLOOR_FROM,
    FLOWERS,
    FOOT_GAIN,
    FOOT_REACH,
    FOOT_SHADE,
    FOOT_TINT,
    FURTHER,
    GARDEN,
    GARDEN_LEAVES,
    GARDEN_ROWS,
    GOLDEN_GREENS,
    GRASSES,
    GRASSY,
    GREENER_PLAINS,
    GROVE_FLOOR,
    GROVE_SHADOW,
    GROVES,
    HERD_PATCH,
    HERD_SHARE,
    HILL_LIFT,
    HOLLOW_DARK,
    HOLLOWS,
    IN_THE_WOODS,
    LEAF_GRAIN,
    LEAFY_CROWNS,
    LUMPS,
    LUSH,
    LUSH_TINT,
    MAX_SLOPE_SHADES,
    MEADOWS,
    OLD,
    ON_THE_LAND,
    OPEN_WATER,
    ORE_ROCK,
    OVERHANG,
    PLAINS_SPAN,
    PLAINS_TINT,
    RAMPS,
    REGION_SPAN,
    RELIEF_SMOOTHING_CELLS,
    RELIEF_STEPS_PER_CELL,
    ROLL_LEAST,
    ROLL_LIGHT,
    ROLL_MOST,
    ROLL_RANGE,
    ROLL_SPAN,
    ROLL_STEP,
    SAND,
    SHADE_REACH,
    SHADOW_COLOUR,
    SHADOW_REACH,
    SHADOW_SOFT,
    SHALLOW,
    SHEEP,
    SHEEP_DARK,
    SHORE_MIX,
    SHORE_WATER,
    SLOPE_LIGHT,
    SOIL,
    SOIL_DARK,
    SPARKLE,
    SPARKLE_FACING,
    SPARKLE_MIX,
    SPARKLE_SHARE,
    STAND_SPAN,
    THIN_SPAN,
    THINNED,
    TOWN_BLEND,
    TOWN_EARTH,
    TOWN_EARTH_COLOUR,
    TOWN_FRAY,
    TOWN_FROM,
    TOWN_HOLES,
    TOWN_JOINT,
    TOWN_JOINTS,
    TOWN_REACH,
    TOWN_SETTS,
    TOWN_SOFT,
    TOWN_STONE,
    TREE_SHADE,
    TRUNK_BARK,
    TRUNK_BELOW,
    TRUNK_HALF,
    TRUNK_LENGTH,
    TUFT,
    TUFT_SHARE,
    UP_LEFT,
    WARP,
    WARP_BROAD,
    WARP_FINE,
    WATER,
    WATER_FURTHER,
    WAVE_LIGHT,
    WAVE_RANGE,
    WAVE_STEP,
    WET_GAIN,
    WET_REACH,
    WHEAT,
    WHEAT_DARK,
    WHEAT_EARS,
    WHEAT_LIGHT,
    WOODS,
    WOODS_FLOOR,
    WOODS_SIDE,
    YARD_COLOURS,
    YOUNG,
    Grove,
)
from railroads.ui.pixel.lattice import BoolArray, Lattice
from railroads.ui.pixel.noise import Floats, noise, ridged, table_hash
from railroads.ui.pixel.projection import CHUNK, RGB, PixelProjection
from railroads.ui.pixel.relief import (
    LIGHT,
    SHADOW_TINT,
    Blend,
    Elevation,
    Raised,
    mountain_cover,
    sample,
    smooth,
)
from railroads.ui.pixel.streams import (
    BANK_WIDTH,
    HALF_WIDTH,
    STREAM_BANK,
    STREAM_WATER,
    stream_distance,
    stream_links,
)


class Ground:
    """The ground of one map, drawn a block of art pixels at a time, and what it is drawn from:
    each cell's biome, the slopes' light, and the fields and groves."""

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
        self._hills = land & (tile_map.relief == Relief.HILLS)
        self._relief_light = self._light_slopes(angle_deg)
        self._wet, self._foot = self._surroundings()
        # Each cell's links along the minor rivers through it (railroads.ui.pixel.streams).
        self._streams = stream_links(tile_map)
        # Where trees stand in the picture, which shade the grass round them, and where buildings
        # stand, what ground lies round each and how far it reaches, both set by the world.
        self.trees: tuple[FloatArray, FloatArray] = (np.zeros(0), np.zeros(0))
        self.buildings: tuple[FloatArray, FloatArray, npt.NDArray[np.str_], FloatArray] = (
            np.zeros(0),
            np.zeros(0),
            np.zeros(0, dtype=str),
            np.zeros(0),
        )
        # Where logging camps stand in the picture and how far the clearing round each reaches.
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
        # Cells with another biome or mountains within two rings of cells round them, whose
        # pixels may take them: those unlike a cell round them, and those beside such a cell.
        kind = self._biomes * 2 + (self.tile_map.relief == Relief.MOUNTAINS)
        unlike = self._off_round | (kind.ravel()[self._round] != kind).any(axis=0)
        self._mixed = unlike | unlike.ravel()[self._round].any(axis=0)
        built = np.zeros(self.tile_map.grid.shape, dtype=bool)
        woodland = np.zeros(self.tile_map.grid.shape, dtype=bool)
        in_woods: list[Cell] = []
        elsewhere: list[Cell] = []
        for construction in self.constructions:
            if isinstance(construction, Building) and construction.kind in IN_THE_WOODS:
                in_woods.append(construction.cell)
            else:
                elsewhere.append(construction.cell)
        for mask, cells in ((woodland, in_woods), (built, elsewhere)):
            if cells:
                rows, cols = np.array(cells, dtype=np.int64).T
                mask[rows, cols] = True
        # Each cell's say in whether woods cover a point near it: 1 for woods and logging camps,
        # 0 for open land, and CLEARING where anything else is built, so the woods give it room.
        woods = np.isin(self._biomes, WOODS) | woodland
        self._woods = np.where(built, CLEARING, woods).astype(np.float32)
        town = np.zeros(self.tile_map.grid.shape, dtype=bool)
        for construction in self.constructions:
            if isinstance(construction, Building) and construction.kind not in ON_THE_LAND:
                town[construction.cell] = True
        self._town = (town & self._land).astype(np.float32)
        rows, cols = np.nonzero(self._town)
        px, py = self.projection.to_pixels(*self.tile_map.grid.center(rows, cols))
        self._town_pixels = (np.atleast_1d(px), np.atleast_1d(py))
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

        Each pixel is coloured by the one kind of ground it shows: water, one of the GRASSES
        (meadow, savanna, tundra or desert) or the plain ramp of its biome, each worked out only
        at its own pixels. Where raised ground
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
        x, y = points.plane(left, top, CHUNK, rows)
        wr, wc = self.winding(x, y, r, c)
        biomes = self._biomes[wr, wc]
        rocky = self.tile_map.relief[wr, wc] == Relief.MOUNTAINS
        seen = np.ones(x.shape, dtype=bool)
        if raised is not None:
            seen = mountain_cover(raised.mountains[:rows], rocky) < 1
        dry = ~wet
        grasses = {kind: _GRASSES[kind][biomes] & dry & seen for kind in GRASSES}
        grassy = np.logical_or.reduce(list(grasses.values()))
        plain = dry & seen & ~grassy
        lit = grassy | plain
        light = np.zeros(x.shape, dtype=np.float64)
        light[lit] = sample(
            self._relief_light, x[lit] * RELIEF_STEPS_PER_CELL, y[lit] * RELIEF_STEPS_PER_CELL
        )
        rgb = np.zeros((*x.shape, 3), dtype=np.float32)
        shade = np.zeros(x.shape, dtype=np.int64)
        down, across = np.nonzero(wet & seen)
        if len(down):
            open_water = (down, across)
            rgb[open_water] = self._water(x[open_water], y[open_water], left + across, top + down)
        down, across = np.nonzero(plain)
        if len(down):
            ix, iy = left + across, top + down
            fx, fy = x[plain].astype(np.float32), y[plain].astype(np.float32)
            value = 1.45 + 1.6 * (noise(fx, fy, 2.4, 41) - 0.5)
            value += 0.8 * (noise(fx, fy, 0.75, 42) - 0.5)
            value += 0.45 * (noise(fx, fy, 0.22, 43) - 0.5) + 0.45 * BAYER[iy % 4, ix % 4]
            value += light[plain] + HILL_LIFT * self._hills[r[plain], c[plain]]
            shade[plain] = np.clip(np.rint(value), 0, 3).astype(np.int64)
        rgb[plain] = self.ramps[biomes[plain], shade[plain]]
        desert = grasses["desert"]
        if desert.any():
            light[desert] += _dunes(x[desert], y[desert])
        if grassy.any():
            shaded = self._tree_shade(left, top, x.shape)
            for kind, grass in grasses.items():
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
        land = lit & dry
        if land.any():
            rgb[land] *= self._natural(x[land], y[land])
        self._draw_streams(rgb, x, y, r, c, land, left, top)
        self._shores(rgb, water, rows, inside)
        if self.tile_map.deposits is not None:
            _sign_deposits(rgb, self.tile_map.deposits[r, c], left, top, wet)
        alpha = inside.astype(np.float32)
        if raised is not None:
            self.elevation.colour(rgb, raised.top(rows), x, y, rocky)
        return np.clip(rgb, 0, 255) * alpha[..., None], alpha

    def _draw_streams(
        self, rgb: FloatArray, x: FloatArray, y: FloatArray, r: IntArray, c: IntArray,
        land: BoolArray, left: int, top: int,
    ) -> None:  # fmt: skip
        """Draws the minor rivers over the land of a block: water along their lines and darker
        banks beside it."""
        grid = self.tile_map.grid
        if not isinstance(grid, HexGrid) or not (land & (self._streams[r, c] > 0)).any():
            return
        gap = stream_distance(self._streams, grid, x, y, r, c)
        rgb[land & (gap < BANK_WIDTH)] *= STREAM_BANK
        down, across = np.nonzero(land & (gap < HALF_WIDTH))
        if len(down):
            at = (down, across)
            water = self._water(x[at], y[at], left + across, top + down)
            rgb[at] = (water + np.array(STREAM_WATER, dtype=np.float32)) / 2

    def _water(self, x: FloatArray, y: FloatArray, ix: IntArray, iy: IntArray) -> FloatArray:
        """The colour of water at plane points `x`, `y`, art pixels `ix`, `iy`: shallow or deep
        by how much water lies round it, lit on its waves, with a few glints."""
        at = Blend.at(self._wet.shape, x * RELIEF_STEPS_PER_CELL, y * RELIEF_STEPS_PER_CELL)
        depth = np.clip((at.read(self._wet) - DEPTH_FROM) / DEPTH_SPAN, 0, 1)[:, None]
        colour = np.where(
            depth < 0.5,
            SHALLOW + (OPEN_WATER - SHALLOW) * (depth * 2),
            OPEN_WATER + (DEEP_WATER - OPEN_WATER) * (depth * 2 - 1),
        )
        fx, fy = x.astype(np.float32), y.astype(np.float32)
        e = np.float32(WAVE_STEP)
        here = _waves(fx, fy)
        gx = (_waves(fx + e, fy) - here) / e
        gy = (_waves(fx, fy + e) - here) / e
        facing = -(gx * np.float32(LIGHT[0]) + gy * np.float32(LIGHT[1])) * np.float32(WAVE_LIGHT)
        colour = colour * np.clip(1 + facing, *WAVE_RANGE)[:, None]
        glint = (facing > SPARKLE_FACING) & (table_hash(ix, iy, 403) > 1 - SPARKLE_SHARE)
        colour[glint] = colour[glint] * (1 - SPARKLE_MIX) + SPARKLE * SPARKLE_MIX
        return colour.astype(np.float32)

    def _shores(self, rgb: FloatArray, water: BoolArray, rows: int, inside: BoolArray) -> None:
        """Softens the shores of a block of `rows` by CHUNK pixels: sand fading into the land
        beside water, light shallows fading into the water beside land and the bank's shade on
        the water below it; `water` holds where water is over the APRON round the block."""
        a = APRON
        wet = water[a : a + rows, a : a + CHUNK]
        land = ~water
        sand = np.array(SAND, dtype=np.float32)
        beside = _near(water, a, rows, BESIDE)
        beach = ~wet & ~beside & _near(water, a, rows, FURTHER) & inside
        rgb[beach] = rgb[beach] * (1 - BEACH[1]) + sand * BEACH[1]
        beach = ~wet & beside & inside
        rgb[beach] = rgb[beach] * (1 - BEACH[0]) + sand * BEACH[0]
        beside = _near(land, a, rows, BESIDE) & wet & inside
        rgb[beside] = rgb[beside] * (1 - SHORE_MIX[0]) + SHORE_WATER * SHORE_MIX[0]
        further = _near(land, a, rows, WATER_FURTHER) & wet & inside & ~beside
        rgb[further] = rgb[further] * (1 - SHORE_MIX[1]) + SHORE_WATER * SHORE_MIX[1]
        shaded = _near(land, a, rows, UP_LEFT) & wet & inside & ~beside
        rgb[shaded] *= BANK_SHADE

    def winding(
        self, x: FloatArray, y: FloatArray, r: IntArray, c: IntArray
    ) -> tuple[IntArray, IntArray]:
        """The cells pixels at plane points `x`, `y` in cells `r`, `c` take their biome and
        mountains from, so biomes and mountains meet along a winding line: their own cells, but
        for land near another biome or mountains, the cell under a point pushed by noise (WARP),
        where that cell is not water."""
        near = self._mixed[r, c] & ~_WATER[self._biomes[r, c]]
        if not near.any():
            return r, c
        fx, fy = x[near].astype(np.float32), y[near].astype(np.float32)
        pr, pc = self.tile_map.grid.cell_on_map(*_wound(fx, fy))
        keep = _WATER[self._biomes[pr, pc]]
        wr, wc = r.copy(), c.copy()
        wr[near] = np.where(keep, r[near], pr)
        wc[near] = np.where(keep, c[near], pc)
        return wr, wc

    def _natural(self, x: FloatArray, y: FloatArray) -> FloatArray:
        """How much each colour of plain land at these points is multiplied by for the light on
        its swells, the water and the mountains near it."""
        fx, fy = x.astype(np.float32), y.astype(np.float32)
        here = _rolling(fx, fy)
        e = np.float32(ROLL_STEP)
        gx = (_rolling(fx + e, fy) - here) / e
        gy = (_rolling(fx, fy + e) - here) / e
        facing = -(gx * np.float32(LIGHT[0]) + gy * np.float32(LIGHT[1]))
        deep = ROLL_LEAST + (ROLL_MOST - ROLL_LEAST) * noise(fx, fy, ROLL_SPAN, 306)
        roll = np.clip(1 + ROLL_LIGHT * deep * facing, *ROLL_RANGE)
        region = noise(fx, fy, PLAINS_SPAN, 305) * 2 - 1
        plains = np.where((region < 0)[:, None], GREENER_PLAINS, DRIER_PLAINS)
        at = Blend.at(self._wet.shape, x * RELIEF_STEPS_PER_CELL, y * RELIEF_STEPS_PER_CELL)
        lush = np.clip(at.read(self._wet) * WET_GAIN, 0, 1) * LUSH
        foot = np.clip(at.read(self._foot) * FOOT_GAIN, 0, 1) * FOOT_SHADE
        tint = 1 + (plains - 1) * (np.abs(region) * PLAINS_TINT)[:, None]
        tint *= 1 + (LUSH_TINT - 1) * lush[:, None]
        tint *= 1 + (FOOT_TINT - 1) * foot[:, None]
        return (tint * roll[:, None]).astype(np.float32)

    def _surroundings(self) -> tuple[FloatArray, FloatArray]:
        """How much water and how many mountains lie round the ground, from 0 to 1, over
        WET_REACH and FOOT_REACH, on the raster of the slopes' light."""
        grid = self.tile_map.grid
        steps = RELIEF_STEPS_PER_CELL
        rows, cols = grid.raster_cells(steps)
        water = (self.tile_map.terrain[rows, cols] != Terrain.LAND).astype(np.float64)
        mountains = (self.tile_map.relief[rows, cols] == Relief.MOUNTAINS).astype(np.float64)
        wet = smooth(water, round(WET_REACH * steps)).astype(np.float32)
        foot = smooth(mountains, round(FOOT_REACH * steps)).astype(np.float32)
        return wet, foot

    @property
    def shading(self) -> FloatArray:
        """The slopes' light, how much water and how many mountains lie round the ground
        (`_surroundings`) and nothing, four to each step of their raster, for the graphics card."""
        light = self._relief_light.astype(np.float32)
        return np.stack([light, self._wet, self._foot, np.zeros_like(light)], axis=-1)

    @property
    def biomes(self) -> npt.NDArray[np.integer]:
        """Each cell's biome."""
        return self._biomes

    @property
    def hills(self) -> BoolArray:
        """Where land is hilly."""
        return self._hills

    @property
    def woodland(self) -> FloatArray:
        """Each cell's say in whether woods cover a point near it, 1 where it is land and 1 where
        it lays town ground (ON_THE_LAND), as (rows, cols, 3) floats, for the graphics card."""
        return np.stack([self._woods, self._land.astype(np.float32), self._town], axis=-1)

    @property
    def town_pixels(self) -> tuple[FloatArray, FloatArray]:
        """Where the middle of each cell laying town ground lies in the picture."""
        return self._town_pixels

    @property
    def plot_table(self) -> FloatArray:
        """Every field and grove, a row each, for the graphics card: where the middle of its
        cell lies in the picture, its cell's row and column, that middle on the plane, which way
        the plot faces, how wide it spreads, and its kind as its place in FIELD_KINDS."""
        grid = self.tile_map.grid
        parts = []
        for code, kind in enumerate(FIELD_KINDS):
            plots = self._plots[kind]
            if not plots:
                continue
            px, py = self._plot_pixels[kind]
            cells = np.array([plot.cell for plot in plots], dtype=np.int64).reshape(-1, 2)
            cx, cy = grid.center(cells[:, 0], cells[:, 1])
            facing = np.array([plot.facing for plot in plots], dtype=np.float64).reshape(-1, 2)
            spread = np.array([plot.spread for plot in plots], dtype=np.float64)
            parts.append(
                np.column_stack(
                    [px, py, cells, cx, cy, facing, spread, np.full(len(plots), float(code))]
                ).astype(np.float32)
            )
        if not parts:
            return np.zeros((0, 10), dtype=np.float32)
        return np.concatenate(parts)

    @property
    def yard_table(self) -> FloatArray:
        """Every building's yard, a row each, for the graphics card: where the building stands
        in the picture, how far its yard reaches, and its kind as its place in YARD_COLOURS."""
        xs, ys, kinds, reaches = self.buildings
        codes = [list(YARD_COLOURS).index(kind) for kind in kinds.tolist()]
        return np.stack([xs, ys, reaches, np.array(codes, dtype=np.float32)], axis=-1).astype(
            np.float32
        )

    @property
    def streams(self) -> npt.NDArray[np.uint8]:
        """Each cell's links along the minor rivers through it (streams.stream_links)."""
        return self._streams

    @property
    def mountains(self) -> BoolArray:
        """Where cells are mountains, which take the mountains' colours as they rise."""
        return self.tile_map.relief == Relief.MOUNTAINS

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
        grass = GRASS_BIOMES[self._biomes[r, c]] & inside
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
        painted = np.zeros(paint.shape[:2], dtype=np.float32)
        # Each patch works out the plane points and cells of its own pixels, unless patches
        # overlap so much that working them out once over the box round all of them is less.
        boxes = {(p[0].start, p[0].stop, p[1].start, p[1].stop) for _, p, _, _ in jobs}
        shared = None
        if sum((b[1] - b[0]) * (b[3] - b[2]) for b in boxes) > paint.shape[0] * paint.shape[1]:
            shared = self._patch_points((down, bottom, across, right), left - a, top - a, k, points)
        worked: dict[tuple[int, int, int, int], tuple] = {}
        # Where each patch's pixels lie off the minor rivers and their banks, which fields leave
        # showing.
        dry: dict[tuple[int, int, int, int], FloatArray | float] = {}
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
                cover = self._sow(*args, plot, crop, k)
            else:
                cover = self._plant(*args, plot, grove, k)
            if box not in dry:
                dry[box] = self._off_streams(*worked[box][:3])
            np.maximum(painted[place], cover * dry[box], out=painted[place])
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
        cover = painted[inside]
        laid = (cover > 0) & (under[..., 3] > 0 if shown is None else shown[region])
        if not laid.any():
            return
        colour = inner[laid]
        cover = cover[laid][:, None]
        if lit is not None:
            art_rows, art_cols = np.nonzero(laid)
            art_rows, art_cols = (
                (art_rows + region[0].start) // k,
                (art_cols + region[1].start) // k,
            )
            colour = colour * lit[art_rows, art_cols][:, None]
        under[laid, :3] = under[laid, :3] * (1 - cover) + np.clip(colour, 0, 255) / 255 * cover
        under[laid, 3:] = under[laid, 3:] * (1 - cover) + cover

    def _off_streams(self, x: FloatArray, y: FloatArray, flat: IntArray) -> FloatArray | float:
        """1 where pixels at plane points `x`, `y` in flat cells `flat` lie off the minor rivers
        and their banks, 0 on them."""
        grid = self.tile_map.grid
        if not isinstance(grid, HexGrid) or not (self._streams.ravel()[flat] > 0).any():
            return 1.0
        r, c = np.divmod(flat, grid.cols)
        gap = stream_distance(self._streams, grid, x, y, r, c)
        return (gap >= BANK_WIDTH).astype(np.float32)

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
            r, c = grid.cell_on_map(x, y)
        ix = np.broadcast_to(left * k + fx, shape)
        iy = np.broadcast_to(top * k + fy, shape)
        return x, y, r * grid.cols + c, ix, iy

    def yards(
        self, canvas: FloatArray, left: int, top: int, k: int, shown: BoolArray | None = None
    ) -> None:
        """Paints the ground round buildings over a picture of the ground from (`left`, `top`),
        `k` pixels to an art pixel, RGB from 0 to 1 multiplied by its alpha (YARD_REACH), where
        the picture shows the map, as `fields` says."""
        self._pave_town(canvas, left, top, k, shown)
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

    def _pave_town(
        self, canvas: FloatArray, left: int, top: int, k: int, shown: BoolArray | None
    ) -> None:
        """Lays town ground over the cells that lay it (ON_THE_LAND) in a picture of the ground
        from (`left`, `top`), `k` pixels to an art pixel, off the minor rivers, given out into
        the grass at its frayed edge."""
        if not self._town.any():
            return
        rows, cols = canvas.shape[0] // k, canvas.shape[1] // k
        grid = self.tile_map.grid
        px, py = self._town_pixels
        reach = TOWN_REACH
        near = (px > left - reach) & (px < left + cols + reach)
        near &= (py > top - reach) & (py < top + rows + reach)
        if not near.any():
            return
        first_col = max(int((px[near].min() - reach - left) * k), 0)
        last_col = min(int((px[near].max() + reach - left) * k) + 1, cols * k)
        first_row = max(int((py[near].min() - reach - top) * k), 0)
        last_row = min(int((py[near].max() + reach - top) * k) + 1, rows * k)
        if first_col >= last_col or first_row >= last_row:
            return
        box = (first_row, last_row, first_col, last_col)
        place = (slice(first_row, last_row), slice(first_col, last_col))
        x, y, flat, ix, iy = self._patch_points(box, left, top, k, None)
        x, y = x.astype(np.float32), y.astype(np.float32)
        land = self._land.ravel()[flat]
        share = np.zeros(x.shape, dtype=np.float32)
        for dx, dy in AROUND:
            sr, sc = grid.cell_on_map(x + dx * TOWN_BLEND, y + dy * TOWN_BLEND)
            share += self._town[sr, sc]
        share /= len(AROUND)
        edge = noise(x, y, 0.55, 63) * 0.55 + noise(x, y, 0.16, 64) * 0.3
        cover = np.clip((share + (edge - 0.42) * TOWN_FRAY - TOWN_FROM) * TOWN_SOFT, 0, 1)
        cover *= land
        holes_size, holes_from = TOWN_HOLES
        cover[noise(x, y, holes_size, 66) > holes_from] = 0
        if not (cover > 0).any():
            return
        cover *= _on_map(canvas, shown)[place]
        cover *= self._off_streams(x, y, flat)
        colour = town_ground(x, y, ix, iy, k)
        _over(canvas[place], colour / 255, cover)

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
            left, right, top, bottom = GARDEN
            garden = (u > left) & (u < right) & (v > top) & (v < bottom)
            colour[garden] = SOIL * 0.95
            rows = garden & (np.floor((v - top) * GARDEN_ROWS) % 2 == 0)
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
        art pixel, RGB from 0 to 1 multiplied by its alpha: a roof of lit crowns, the dark side
        of the trees along its lower edge and their shadow on the ground beside them, worked out
        at every pixel of the picture like the mountains, so they are as smooth as the picture is
        fine, where the picture shows the map, as `fields` says. Finer than an art pixel to a
        pixel, crowns show their clumps of leaves and reach past the woods' edge, and the front
        row its trunks. `points` may hold the plane points and cells of the art pixels already."""
        rows, cols = canvas.shape[0] // k, canvas.shape[1] // k
        fine = k > 1
        reach = CANOPY_REACH + (OVERHANG if fine else 0)
        after = (reach, reach) if fine else (1, 2)
        block = (left - reach, top - reach, cols + reach + after[0], rows + reach + after[1])
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

        def shifted(grid: BoolArray, down: int, across: int) -> BoolArray:
            return grid[reach + down : reach + down + rows, reach + across : reach + across + cols]

        near = _spread(woods, OVERHANG, OVERHANG + 3) if fine else woods
        shadow = shifted(near, -SHADOW_REACH[1], -SHADOW_REACH[0])
        side = shifted(woods, 0, 0) & ~(shifted(woods, 1, 0) & shifted(woods, 2, 0))
        # Inside the woods the floor shows between crowns; at their edge, the ground does.
        inner = (
            shifted(woods, -1, 0)
            & shifted(woods, 1, 0)
            & shifted(woods, 0, -1)
            & shifted(woods, 0, 1)
        )
        over = shifted(near, 0, 0)
        if k > 1:
            over, shadow, side = (np.repeat(np.repeat(a, k, 0), k, 1) for a in (over, shadow, side))
            inner = _rounded(woods[reach - 2 : reach + rows + 2, reach - 2 : reach + cols + 2], k)
        shown = _on_map(canvas, shown)
        painted = np.zeros(over.shape, dtype=bool)
        down, across = np.nonzero(over & shown)
        if len(down):
            edges = (side, inner, woods)
            self._crowns(canvas, painted, down, across, (left, top, k), points, block, edges)
        self._cast_shadow(canvas, shadow & shown & ~painted, (left, top, k), block, woods)

    def _crowns(
        self,
        canvas: FloatArray,
        painted: BoolArray,
        down: IntArray,
        across: IntArray,
        place: tuple[int, int, int],
        points: Lattice,
        block: tuple[int, int, int, int],
        edges: tuple[BoolArray, BoolArray, BoolArray],
    ) -> None:
        """Paints the crowns over the woods' pixels `down`, `across` of a picture from
        (`left`, `top`), `k` pixels to an art pixel, and the floor between them inside the woods,
        marking the pixels painted. `edges` holds which pixels show the dark side of the trees
        along the woods' lower edge, which lie inside the woods rather than at their edge, and
        which art pixels of the block the woods cover: at their edge only crowns standing in the
        woods show, and the ground between them."""
        side, inner, rooted = edges
        left, top, k = place
        fine = k > 1
        reach = left - block[0]
        px, py = left + (across + 0.5) / k, top + (down + 0.5) / k
        height, dx, dy, radius, gx, gy = _tallest_crown(px, py, CROWN_GAP)
        foot_x = np.floor(px - dx).astype(np.int64) - block[0]
        foot_y = np.floor(py - dy).astype(np.int64) - block[1]
        foot_x = np.clip(foot_x, 0, rooted.shape[1] - 1)
        foot_y = np.clip(foot_y, 0, rooted.shape[0] - 1)
        # Crowns rise by their lumps up to LUMPS / 2 and their clumps up to CLUMP_EDGE * 0.7.
        lowest = -(LUMPS / 2 + CLUMP_EDGE) if fine else -LUMPS / 2
        inside = inner[down, across]
        keep = inside | ((height > lowest) & rooted[foot_y, foot_x])
        art_down, art_across = down // k + reach, across // k + reach
        if fine:
            hidden = ~keep & _below(rooted, TRUNK_BELOW)[art_down, art_across]
            trunk, bark = _trunks(px[hidden], py[hidden], rooted, block)
            hidden = np.flatnonzero(hidden)[trunk]
            canvas[down[hidden], across[hidden], :3] = bark[trunk] / 255
            canvas[down[hidden], across[hidden], 3] = 1.0
            painted[down[hidden], across[hidden]] = True
        down, across, px, py, inside = down[keep], across[keep], px[keep], py[keep], inside[keep]
        height, dx, dy, radius, gx, gy = (a[keep] for a in (height, dx, dy, radius, gx, gy))
        foot_x, foot_y = foot_x[keep], foot_y[keep]
        art_down, art_across = art_down[keep], art_across[keep]
        r, c, _ = points.cells(*block)
        if k == 1:
            x, y = points.plane(*block)
            x, y = x[art_down, art_across], y[art_down, art_across]
        else:
            x, y = self.projection.to_plane(px, py)
        fx, fy = x.astype(np.float32), y.astype(np.float32)
        biomes = self._biomes[r[art_down, art_across], c[art_down, art_across]]
        grain = noise(fx, fy, LEAF_GRAIN, 68) - 0.5
        height += LUMPS * grain
        conifers = np.where(biomes == Biome.TAIGA, 0.9, 0.0)
        broadleaved = (biomes != Biome.TAIGA) & (biomes != Biome.JUNGLE)
        stands = noise(fx[broadleaved], fy[broadleaved], CONIFER_STANDS, 64)
        conifers[broadleaved] = 0.05 + 0.85 * np.clip((stands - 0.5) * 4 + 0.5, 0, 1)
        is_conifer = table_hash(gx, gy, 63) < conifers
        conifer = is_conifer[:, None]
        tone = table_hash(gx, gy, 66)[:, None] * CROWN_TONES
        first = np.where(conifer, CONIFER_CROWNS[0], LEAFY_CROWNS[0])
        crowns = first + (np.where(conifer, CONIFER_CROWNS[1], LEAFY_CROWNS[1]) - first) * tone
        crowns *= (0.9 + 0.2 * noise(fx, fy, 0.9, 65))[:, None]
        region = noise(fx, fy, REGION_SPAN, 72)[:, None] * 2 - 1
        crowns *= 1 + (np.where(region < 0, DARK_GREENS, GOLDEN_GREENS) - 1) * np.abs(region)
        towards = (dx * CROWN_LIGHT[0] + dy * CROWN_LIGHT[1]) / radius * CROWN_ROUND
        up = np.sqrt(np.clip(1 - (1 - height) * CROWN_ROUND**2, 0, 1))
        facing = towards + up * CROWN_LIGHT[2]
        solid = height > 0
        if fine:
            detail = _crown_detail(px, py, dx, dy, radius, height, is_conifer, gx, gy)
            solid, facing, hollow, branch, depth, height = detail
        facing += self._swell(fx, fy) + CROWN_SLOPES * sample(
            self._relief_light, x * RELIEF_STEPS_PER_CELL, y * RELIEF_STEPS_PER_CELL
        )
        s = np.clip(facing, 0, 1).astype(np.float32)
        lit = np.clip(1 + 1.3 * (s - np.float32(CROWN_LIGHT[2])), 0.5, 1.35)
        dark = np.clip((0.45 - s) * 2, 0, 1)
        deep = CROWN_DEPTH + (1 - CROWN_DEPTH) * np.sqrt(np.clip(height * 2, 0, 1))
        rim = np.where(height < CROWN_RIM[0], CROWN_RIM[1], 1.0)
        shine = 1 + np.clip(s - CROWN_SHINE[0], 0, 1) * CROWN_SHINE[1] * 5
        if fine:
            rim = np.where(height < CROWN_RIM[0], (1 + CROWN_RIM[1]) / 2, 1.0) * depth
            behind = np.clip(
                foot_y + np.ceil(radius * 0.7).astype(np.int64), 0, rooted.shape[0] - 1
            )
            front = ~rooted[behind, foot_x]
            edge = np.where(front, np.clip(dy / radius * 1.5 + 0.3, 0, 1), np.float32(0))
        else:
            edge = side[down, across].astype(np.float32)
        cool = np.maximum(dark, edge)
        light = lit * (1 - dark * 0.35) * deep * rim * shine * (1 - (1 - WOODS_SIDE) * edge)
        leaves = crowns * np.maximum(light, CROWN_DARKEST)[:, None]
        leaves *= 1 - (cool * 0.6)[:, None] * (1 - SHADOW_TINT)
        if fine:
            leaves[hollow] *= HOLLOW_DARK
            bark = BRANCH * (0.8 + 0.3 * np.clip(s[branch], 0, 1))[:, None]
            leaves[branch] = bark * (1 - 0.5 * (1 - SHADOW_TINT))
        gap = ~solid
        undergrowth = 0.8 + 0.4 * noise(fx[gap], fy[gap], LEAF_GRAIN * 1.5, 73)
        leaves[gap] = WOODS_FLOOR * undergrowth[:, None]
        shows = inside | solid
        down, across = down[shows], across[shows]
        canvas[down, across, :3] = leaves[shows] / 255
        canvas[down, across, 3] = 1.0
        painted[down, across] = True

    def _cast_shadow(
        self,
        canvas: FloatArray,
        under: BoolArray,
        place: tuple[int, int, int],
        block: tuple[int, int, int, int],
        rooted: BoolArray,
    ) -> None:
        """Lays the woods' shadow over the pixels `under` woods SHADOW_REACH art pixels up and
        to the left of them, where a crown stands there, so it takes the crowns' round shapes;
        finer than an art pixel to a pixel, only of crowns standing in the woods, fading out at
        its edge."""
        left, top, k = place
        down, across = np.nonzero(under)
        if not len(down):
            return
        px = left + (across + 0.5) / k - SHADOW_REACH[0]
        py = top + (down + 0.5) / k - SHADOW_REACH[1]
        height, dx, dy, *_ = _tallest_crown(px, py, CROWN_GAP)
        if k > 1:
            foot_x = np.clip(np.floor(px - dx).astype(np.int64) - block[0], 0, rooted.shape[1] - 1)
            foot_y = np.clip(np.floor(py - dy).astype(np.int64) - block[1], 0, rooted.shape[0] - 1)
            strength = np.clip(height * SHADOW_SOFT, 0, 1) * rooted[foot_y, foot_x]
        else:
            strength = (height > 0).astype(np.float32)
        cast = strength > 0
        down, across, strength = down[cast], across[cast], strength[cast] * EDGE_SHADOW
        shaded = canvas[down, across]
        s = strength[:, None]
        shaded[:, :3] = shaded[:, :3] * (1 - s) + SHADOW_COLOUR / 255 * s
        shaded[:, 3] = shaded[:, 3] * (1 - strength) + strength
        canvas[down, across] = shaded

    def _swell(self, x: Floats, y: Floats) -> Floats:
        """How much more the woods' roof faces CROWN_LIGHT for the broad swells it rolls in."""
        e = np.float32(0.05)
        here = noise(x, y, CANOPY_SWELL_SIZE, 67)
        gx = (noise(x + e, y, CANOPY_SWELL_SIZE, 67) - here) / e
        gy = (noise(x, y + e, CANOPY_SWELL_SIZE, 67) - here) / e
        return -(gx * CROWN_LIGHT[0] + gy * CROWN_LIGHT[1]) * CANOPY_SWELL * CANOPY_SWELL_SIZE

    def _clear_round_mills(self, woods: BoolArray, block: tuple[int, int, int, int]) -> None:
        """Takes the woods out of the clearing round each logging camp in the block, an oval
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
                sr, sc = grid.cell_on_map(x + dx, y + dy)
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
        steps = RELIEF_STEPS_PER_CELL
        rows, cols = grid.raster_cells(steps)
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
    ) -> FloatArray:
        """Lays a field of crops in rows over the half of a cell a farm or crop takes up, `k`
        pixels to an art pixel, ploughed into the ground rather than laid on it: a farm grows
        wheat, other crops grow in rows, lit on the side towards the light, with darker soil in
        the furrows, the whole lit and tinted like the plains round it (_natural) and fading
        into the grass at a frayed edge. Returns how fully each pixel of the block the field
        covers."""
        grid = self.tile_map.grid
        row, col = plot.cell
        fx, fy = grid.center(row, col)
        field = (cells == row * grid.cols + col) & plot.covers(x - fx, y - fy)
        cover = np.zeros(field.shape, dtype=np.float32)
        if not field.any():
            return cover
        pick = row * 7 + col * 13
        across, along = (x - fx, y - fy) if pick % 2 else (y - fy, x - fx)
        u, w = across[field] * FIELD_STRIPES, along[field] * FIELD_STRIPES
        px, py = ix[field], iy[field]
        grain = (table_hash(px, py, 71) - 0.5)[:, None]
        gx, gy = x[field].astype(np.float32), y[field].astype(np.float32)
        line = np.floor(u).astype(np.int64)
        fu = u - line
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
        grey = out.mean(axis=1, keepdims=True)
        out = grey + (out - grey) * FIELD_SATURATION
        out *= self._natural(x[field], y[field])
        rgb[field] = out
        cover[field] = self._feathered(field, gx, gy, k)
        return cover

    def _feathered(self, field: BoolArray, x: Floats, y: Floats, k: int) -> FloatArray:
        """How fully each pixel of a field covers the ground, from 0 at its frayed edge up to
        1 FIELD_FEATHER art pixels in, in the order of the field's pixels."""
        depth = _depths(field, FIELD_FEATHER * k).astype(np.float32)
        fray = (noise(x, y, FIELD_FRAY_SIZE, 74) - 0.5) * FIELD_FRAY * k
        return np.clip((depth + 0.5 + fray) / (FIELD_FEATHER * k), 0, 1)

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
        k: int = 1,
    ) -> FloatArray:
        """Lays a grove of trees in rows over the half of a cell an orchard, date palms or
        plantation takes up: each crown lit on the side towards the light, with its shadow on
        the ground beside it, and some bearing fruit, over ground of the grove's own colour
        laid thinly over the grass, the whole lit and tinted like the plains round it. Returns
        how fully each pixel of the block the grove covers."""
        grid = self.tile_map.grid
        row, col = plot.cell
        fx, fy = grid.center(row, col)
        ground = (cells == row * grid.cols + col) & plot.covers(x - fx, y - fy)
        cover = np.zeros(ground.shape, dtype=np.float32)
        if not ground.any():
            return cover
        px, py = ix[ground], iy[ground]
        grain = 1 + 0.12 * (table_hash(px, py, 23) - 0.5)
        floor = np.array(grove.ground, dtype=np.float32)
        colors = (floor * grain[:, None]).astype(np.float32)
        gx, gy = x[ground].astype(np.float32), y[ground].astype(np.float32)
        fade = self._feathered(ground, gx, gy, k)
        here = fade * GROVE_FLOOR
        if grove.herd:
            part, lit = herd_parts((px + 0.5) / k, (py + 0.5) / k, grove)
            trodden = part == 4
            colors[trodden] = np.array(grove.trodden, dtype=np.float32) * grain[trodden, None]
            here[trodden] = np.minimum(fade[trodden] * 2, 1)
            colors[part == 1] = SHADOW_COLOUR
            here[part == 1] = fade[part == 1] * GROVE_SHADOW
            body = part == 2
            colors[body] = np.array(grove.tree, dtype=np.float32) * lit[body, None]
            colors[part == 3] = np.array(grove.fruit[0], dtype=np.float32) * 0.95
            roof = part == 5
            colors[roof] = np.array(grove.roof, dtype=np.float32) * lit[roof, None]
            solid = (part == 2) | (part == 3) | roof
            here[solid] = np.minimum(fade[solid] * 3, 1)
            colors *= self._natural(x[ground], y[ground])
            rgb[ground] = colors
            cover[ground] = here
            return cover
        u = (x[ground] - fx) * grove.rows + 0.5
        v = (y[ground] - fy) * grove.rows + 0.5
        du, dv = u - np.round(u), v - np.round(v)
        near = np.hypot(du, dv)
        shadow = np.hypot(du - 0.12, dv - 0.1) < grove.crown
        colors[shadow] = SHADOW_COLOUR
        here[shadow] = fade[shadow] * GROVE_SHADOW
        tree = near < grove.crown
        leaves = np.array(grove.tree, dtype=np.float32)
        towards = -(du + dv) / max(grove.crown, 1e-6)
        lit = np.clip(0.82 + 0.3 * towards, 0.62, 1.22)
        colors[tree] = np.clip(leaves * (lit[tree] * grain[tree])[:, None], 0, 255)
        here[tree] = np.minimum(fade[tree] * 3, 1)
        fruit = tree & (table_hash(px, py, 21) > 0.82)
        pick = (table_hash(px, py, 22) * len(grove.fruit)).astype(np.int64)
        for index, color in enumerate(grove.fruit):
            colors[fruit & (pick == index)] = color
        colors *= self._natural(x[ground], y[ground])
        rgb[ground] = colors
        cover[ground] = here
        return cover


def herd_parts(qx: FloatArray, qy: FloatArray, grove: Grove) -> tuple[IntArray, FloatArray]:
    """What of a herd shows at art points (qx, qy): 0 its ground, 1 a shadow, 2 an animal's body
    or 3 its head, 4 the trodden patch its group stands on or 5 the group's shelter, with how lit
    a body or shelter is there, lighter on top and a shelter rounded like an ark."""
    flock = grove.flock * ART_METRE
    reach = HERD_PATCH * flock / 2
    fx, fy = np.floor(qx / flock).astype(np.int64), np.floor(qy / flock).astype(np.int64)
    centre_x, centre_y = _flock_centre(fx, fy, flock)
    across, along = (qx - centre_x) / reach, (qy - centre_y) / (reach * 0.7)
    wobble = 1 + 0.15 * np.sin(3 * np.arctan2(along, across) + 6.283 * table_hash(fx, fy, 96))
    part = np.zeros(np.shape(qx), dtype=np.int64)
    part[across * across + along * along < wobble * wobble] = 4
    long = grove.shelter * ART_METRE / 2
    sx, sy = (qx - centre_x) / long, (qy - centre_y + 0.45 * reach * 0.7) / long
    part[(np.abs(sx - 0.25) < 1) & (np.abs(sy - 0.3) < 0.55)] = 1
    gap = grove.gap * ART_METRE
    half = grove.herd * ART_METRE / 2
    gx, gy = np.floor(qx / gap).astype(np.int64), np.floor(qy / gap).astype(np.int64)
    middle_x, middle_y = (gx + 0.5) * gap, (gy + 0.5) * gap
    home_x, home_y = _flock_centre(
        np.floor(middle_x / flock).astype(np.int64),
        np.floor(middle_y / flock).astype(np.int64),
        flock,
    )
    inside = ((middle_x - home_x) / reach) ** 2 + ((middle_y - home_y) / (reach * 0.7)) ** 2 < 0.7
    there = inside & (table_hash(gx, gy, 91) < HERD_SHARE)
    cx = middle_x + (table_hash(gx, gy, 92) - 0.5) * 0.5 * gap
    cy = middle_y + (table_hash(gx, gy, 93) - 0.5) * 0.5 * gap
    facing = np.where(table_hash(gx, gy, 94) > 0.5, 1.0, -1.0)
    dx, dy = (qx - cx) / half, (qy - cy) / half
    part[((dx - 0.2) ** 2 + ((dy - 0.45) / 0.35) ** 2 < 1) & there] = 1
    part[(dx * dx + (dy / 0.5) ** 2 < 1) & there] = 2
    part[((dx - facing * 0.95) ** 2 + (dy + 0.3) ** 2 < 0.38**2) & there] = 3
    roofed = (np.abs(sx) < 1) & (np.abs(sy) < 0.5)
    part[roofed] = 5
    arch = 0.7 + 0.5 * np.sqrt(np.clip(1 - ((sy + 0.1) / 0.6) ** 2, 0, 1))
    lit = np.where(roofed, arch, np.clip(0.85 - 0.4 * dy, 0.6, 1.1))
    return part, lit


def _flock_centre(fx: IntArray, fy: IntArray, flock: float) -> tuple[FloatArray, FloatArray]:
    """Where in its square `flock` art pixels a side the group of a herd gathers."""
    return (
        (fx + 0.5 + (table_hash(fx, fy, 95) - 0.5) * 0.3) * flock,
        (fy + 0.5 + (table_hash(fx, fy, 97) - 0.5) * 0.3) * flock,
    )


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


def town_ground(x: FloatArray, y: FloatArray, ix: IntArray, iy: IntArray, k: int) -> FloatArray:
    """The colour of town ground at plane points `x`, `y`, pixels `ix`, `iy` of a picture `k`
    to an art pixel (ON_THE_LAND), RGB from 0 to 255."""
    u = (ix + 0.5) / k
    v = (iy + 0.5) / k
    width, height = TOWN_STONE
    row = np.floor(v / height)
    along = u / width + 0.5 * (row % 2)
    stone = np.floor(along)
    tone = 0.88 + 0.16 * table_hash(stone.astype(np.int64), row.astype(np.int64), 91)
    colour = TOWN_SETTS * tone[..., None]
    joint = ((along - stone) * width < TOWN_JOINT) | ((v / height - row) * height < TOWN_JOINT)
    colour[joint] = TOWN_JOINTS
    size, earth_from = TOWN_EARTH
    earth = np.clip((noise(x, y, size, 65) - earth_from) * 6, 0, 1)[..., None]
    colour = colour * (1 - earth) + TOWN_EARTH_COLOUR * earth
    return colour * (0.94 + 0.1 * table_hash(ix, iy, 92))[..., None]


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


def crown_radii(gx: IntArray, gy: IntArray, jx: FloatArray, jy: FloatArray) -> Floats:
    """How wide the crowns of the squares `gx`, `gy` standing at picture points `jx`, `jy`
    are: CROWN_SIZE, grown from YOUNG to OLD with the age of their stand, and none at all where
    a thin patch leaves them out."""
    size = CROWN_SIZE[0] + CROWN_SIZE[1] * table_hash(gx, gy, 53)
    age = YOUNG + (OLD - YOUNG) * noise(jx, jy, STAND_SPAN, 70)
    thin = np.clip((noise(jx, jy, THIN_SPAN, 71) - 0.62) * 2.2, 0, THINNED)
    gone = table_hash(gx, gy, 54) < thin
    return np.where(gone, np.float32(0.01), size * age).astype(np.float32)


def _rounded(woods: BoolArray, k: int) -> BoolArray:
    """Which pixels, `k` to an art pixel, lie well inside the woods of these art pixels, by a
    rounded outline instead of the art pixels' steps; the art pixels hold two more each way."""
    rows, cols = woods.shape
    filled = np.pad(woods.astype(np.float32), 1, mode="edge")
    blurred = sum(
        filled[1 + dy : 1 + dy + rows, 1 + dx : 1 + dx + cols]
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
    ) / np.float32(9)

    def between(count: int) -> tuple[IntArray, IntArray, FloatArray]:
        at = (np.arange(2 * k, (count - 2) * k) + 0.5) / k - 0.5
        first = np.clip(np.floor(at).astype(np.int64), 0, count - 2)
        return first, first + 1, (at - first).astype(np.float32)

    r0, r1, tr = between(rows)
    c0, c1, tc = between(cols)
    upper = blurred[r0][:, c0] * (1 - tc) + blurred[r0][:, c1] * tc
    lower = blurred[r1][:, c0] * (1 - tc) + blurred[r1][:, c1] * tc
    return upper * (1 - tr[:, None]) + lower * tr[:, None] > FLOOR_FROM


def _spread(woods: BoolArray, around: int, below: int) -> BoolArray:
    """The art pixels within `around` art pixels of the woods, and up to `below` below them."""
    near = woods.copy()
    for _ in range(around):
        grown = near.copy()
        grown[1:] |= near[:-1]
        grown[:-1] |= near[1:]
        grown[:, 1:] |= near[:, :-1]
        grown[:, :-1] |= near[:, 1:]
        near = grown
    spread = near.copy()
    for step in range(1, below - around + 1):
        spread[step:] |= near[:-step]
    return spread


def _below(woods: BoolArray, rows: int) -> BoolArray:
    """The art pixels outside the woods with woods up to `rows` rows above them."""
    above = np.zeros_like(woods)
    for step in range(1, rows + 1):
        above[step:] |= woods[:-step]
    return above & ~woods


def _crown_detail(
    px: FloatArray,
    py: FloatArray,
    dx: Floats,
    dy: Floats,
    radius: Floats,
    height: Floats,
    conifer: BoolArray,
    gx: IntArray,
    gy: IntArray,
) -> tuple[BoolArray, Floats, BoolArray, BoolArray, Floats, Floats]:
    """What a crown shows at picture points finer than art pixels, given the crown standing
    highest over them: where it is solid, how much it faces CROWN_LIGHT, where its hollows and
    the branches in them show, how deep in its shade it lies and how high it stands. Broadleaved
    crowns are heaped from clumps of leaves, each lit on its own; conifers are stars of drooping
    points in tiers."""
    light = CROWN_LIGHT
    solid = np.zeros(px.shape, dtype=bool)
    facing = np.zeros(px.shape, dtype=np.float32)
    hollow = np.zeros(px.shape, dtype=bool)
    branch = np.zeros(px.shape, dtype=bool)
    depth = np.ones(px.shape, dtype=np.float32)
    leafy = np.flatnonzero(~conifer & (height > -CLUMP_EDGE * 0.7))
    height = np.maximum(height, np.float32(0.01)).astype(np.float32)
    grain = table_hash(np.floor(px * 4).astype(np.int64), np.floor(py * 4).astype(np.int64), 59)
    if leafy.size:
        lx, ly, lh, lr = dx[leafy], dy[leafy], height[leafy], radius[leafy]
        clump, cdx, cdy, cr, *_ = _tallest_dome(px[leafy], py[leafy], CLUMP_GAP, 151, clump_radii)
        bump = np.sqrt(np.clip(clump, 0, 1))
        crown_up = np.sqrt(np.clip(1 - (1 - lh) * CROWN_ROUND**2, 0, 1))
        crown_facing = (lx * light[0] + ly * light[1]) / lr * CROWN_ROUND + crown_up * light[2]
        clump_facing = (cdx * light[0] + cdy * light[1]) / cr + bump * light[2]
        facing[leafy] = (
            (1 - CLUMP_LIGHT) * crown_facing
            + CLUMP_LIGHT * clump_facing
            + 0.05 * (grain[leafy] - 0.5)
        )
        whole = lh + CLUMP_EDGE * (np.clip(clump, -0.6, 1) - 0.3) > 0
        solid[leafy] = whole
        depth[leafy] = 0.82 + 0.18 * bump
        holes = whole & (clump < (HOLLOWS - lh) * 0.9)
        hollow[leafy] = holes
        at = leafy[holes]
        if at.size:
            hx, hy = dx[at], dy[at]
            twist = table_hash(gx[at], gy[at], 57)
            turn = np.arctan2(hy, hx) / np.float32(2 * math.pi)
            ray = (np.round(turn * BRANCHES + twist) - twist) * np.float32(2 * math.pi / BRANCHES)
            branch[at] = np.abs(hx * np.sin(ray) - hy * np.cos(ray)) < BRANCH_WIDTH
    needles = np.flatnonzero(conifer)
    if needles.size:
        nx, ny = dx[needles], dy[needles]
        d = np.hypot(nx, ny) + np.float32(1e-6)
        points = np.arctan2(ny, nx) / np.float32(2 * math.pi) * CONIFER_POINTS
        points += table_hash(gx[needles], gy[needles], 58)
        notch = np.abs(points - np.round(points)) * 2
        cone = d / (radius[needles] * CONIFER_SLIM * (1 - 0.25 * notch))
        tier = (cone * CONIFER_TIERS) % 1
        facing[needles] = (
            (nx * light[0] + ny * light[1]) / d * 0.75
            + 0.66 * light[2]
            + 0.3 * (0.4 - tier)
            + 0.16 * (grain[needles] - 0.5)
        )
        solid[needles] = cone < 1
        depth[needles] = 0.92 - 0.12 * tier
        height[needles] = 1 - cone
    return solid, facing, hollow, branch, depth, height


def _trunks(
    px: FloatArray, py: FloatArray, rooted: BoolArray, block: tuple[int, int, int, int]
) -> tuple[BoolArray, FloatArray]:
    """Which of these picture points show the trunk of a tree in the front row of the woods,
    below its crown, and the colour of its bark there, lit on its left; `rooted` says which art
    pixels of the block the woods cover."""
    on = np.zeros(px.shape, dtype=bool)
    bark = np.zeros((*px.shape, 3), dtype=np.float32)
    if not px.size:
        return on, bark
    rows, cols = rooted.shape
    cx, cy = np.floor(px / CROWN_GAP).astype(np.int64), np.floor(py / CROWN_GAP).astype(np.int64)
    first_x, first_y = int(cx.min()) - 1, int(cy.min()) - 2
    gx = np.arange(first_x, int(cx.max()) + 2)[None, :]
    gy = np.arange(first_y, int(cy.max()) + 1)[:, None]
    jx, jy = crown_spot(gx, gy, CROWN_GAP)
    radius = crown_radii(gx, gy, jx, jy)
    fx = np.clip(np.floor(jx).astype(np.int64) - block[0], 0, cols - 1)
    fy = np.clip(np.floor(jy).astype(np.int64) - block[1], 0, rows - 1)
    behind = np.clip(fy + np.ceil(radius * 0.7).astype(np.int64), 0, rows - 1)
    front = (rooted[fy, fx] & ~rooted[behind, fx] & (radius > 0.5)).ravel()
    length = (TRUNK_LENGTH[0] + TRUNK_LENGTH[1] * table_hash(gx, gy, 59)).ravel()
    jx, jy, radius = jx.ravel(), jy.ravel(), radius.ravel()
    width = gx.shape[1]
    home = (cy - first_y) * width + (cx - first_x)
    for ox in (-1, 0, 1):
        for oy in (-2, -1, 0):
            square = home + (oy * width + ox)
            off = px - jx[square]
            reach = py - jy[square]
            here = front[square] & (np.abs(off) < TRUNK_HALF) & (reach > radius[square] * 0.55)
            here &= reach < radius[square] + length[square]
            if not here.any():
                continue
            out = np.clip((reach[here] - radius[square][here]) / length[square][here], 0, 1)
            shade = np.where(off[here] < 0, 1.15, 0.8) * (0.55 + 0.45 * out)
            bark[here] = TRUNK_BARK * shade[:, None]
            on |= here
    return on, bark


def _tallest_crown(
    px: FloatArray, py: FloatArray, gap: float
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, IntArray, IntArray]:
    """The crown standing highest over each picture point, crowns standing one to a square `gap`
    pixels a side at a random spot in it, each a dome as wide as crown_radii says: how high it stands
    there as a share of its top (below 0 where no crown covers the point), which way the point
    lies from its middle, how wide it is and which square it stands in. Where the crown of every
    square round the points stands is worked out once a square; ties go to the square met
    first, going down each column of the three by three round a point's own."""
    return _tallest_dome(px, py, gap, 51, crown_radii)


def crown_spot(gx: IntArray, gy: IntArray, gap: float, seed: int = 51) -> tuple[Floats, Floats]:
    """Where in the squares `gx`, `gy`, `gap` pixels a side, a crown (or a clump of leaves, by
    its own `seed`) stands."""
    jx = (gx + 0.15 + 0.7 * table_hash(gx, gy, seed)) * gap
    jy = (gy + 0.15 + 0.7 * table_hash(gx, gy, seed + 1)) * gap
    return jx, jy


def clump_radii(gx: IntArray, gy: IntArray, jx: FloatArray, jy: FloatArray) -> Floats:
    """How wide the clumps of leaves of the squares `gx`, `gy` are."""
    size = CLUMP_SIZE[0] + CLUMP_SIZE[1] * table_hash(gx, gy, 155)
    return (CLUMP_GAP * size).astype(np.float32)


def _tallest_dome(
    px: FloatArray,
    py: FloatArray,
    gap: float,
    seed: int,
    radii_of: Callable[[IntArray, IntArray, FloatArray, FloatArray], Floats],
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, IntArray, IntArray]:
    cx, cy = np.floor(px / gap).astype(np.int64), np.floor(py / gap).astype(np.int64)
    first_x, first_y = int(cx.min()) - 1, int(cy.min()) - 1
    gx = np.arange(first_x, int(cx.max()) + 2)[None, :]
    gy = np.arange(first_y, int(cy.max()) + 2)[:, None]
    jx, jy = crown_spot(gx, gy, gap, seed)
    radii = radii_of(gx, gy, jx, jy).ravel()
    jx, jy = jx.ravel(), jy.ravel()
    width = gx.shape[1]
    home = (cy - first_y) * width + (cx - first_x)
    best = np.full(px.shape, -np.inf, dtype=np.float32)
    tallest = home.copy()
    for ox in (-1, 0, 1):
        for oy in (-1, 0, 1):
            square = home + (oy * width + ox)
            dx, dy = (px - jx[square]).astype(np.float32), (py - jy[square]).astype(np.float32)
            radius = radii[square]
            height = 1 - (dx * dx + dy * dy) / (radius * radius)
            higher = height > best
            np.copyto(best, height, where=higher)
            np.copyto(tallest, square, where=higher)
    dx, dy = (px - jx[tallest]).astype(np.float32), (py - jy[tallest]).astype(np.float32)
    return best, dx, dy, radii[tallest], tallest % width + first_x, tallest // width + first_y


def _dunes(x: FloatArray, y: FloatArray) -> Floats:
    """How much more desert at these points faces the light for the dunes it lies in."""
    fx, fy = np.asarray(x, np.float32), np.asarray(y, np.float32)
    e = np.float32(DUNE_STEP)
    here = _dune_height(fx, fy)
    gx = (_dune_height(fx + e, fy) - here) / e
    gy = (_dune_height(fx, fy + e) - here) / e
    return -(gx * np.float32(LIGHT[0]) + gy * np.float32(LIGHT[1])) * np.float32(DUNE_LIGHT)


def _dune_height(x: Floats, y: Floats) -> Floats:
    """How high the desert's dunes rise at these points: crests winding across it."""
    bent = x + np.float32(0.3) * noise(x, y, 0.8, 311)
    return ridged(bent, y, 0.45, 312) + np.float32(0.3) * noise(x, y, 0.15, 313)


def _waves(x: Floats, y: Floats) -> Floats:
    """How high the low waves on water rise at these points."""
    return noise(x, y, 0.1, 401) + np.float32(0.5) * noise(x, y, 0.045, 402)


def _rolling(x: Floats, y: Floats) -> Floats:
    """How high the low swells of the plains rise at these points."""
    return noise(x, y, 1.6, 301) + 0.45 * noise(x, y, 0.55, 302) + 0.12 * noise(x, y, 0.18, 303)


def _wound(x: Floats, y: Floats) -> tuple[Floats, Floats]:
    """Points pushed by noise up to WARP, WARP_FINE and WARP_BROAD cells, for biomes' winding
    borders."""
    wx = x + WARP * 2 * (noise(x, y, 0.7, 211) - 0.5)
    wx += WARP_FINE * 2 * (noise(x, y, 0.2, 213) - 0.5)
    wx += WARP_BROAD * 2 * (noise(x, y, 1.8, 215) - 0.5)
    wy = y + WARP * 2 * (noise(x, y, 0.7, 212) - 0.5)
    wy += WARP_FINE * 2 * (noise(x, y, 0.2, 214) - 0.5)
    wy += WARP_BROAD * 2 * (noise(x, y, 1.8, 216) - 0.5)
    return wx, wy


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
_GRASSES = {
    "meadow": _MEADOW,
    "savanna": _biome_table((Biome.SAVANNA,)),
    "tundra": _biome_table((Biome.TUNDRA,)),
    "desert": _biome_table((Biome.DESERT,)),
}
GRASS_BIOMES = np.logical_or.reduce(list(_GRASSES.values()))
