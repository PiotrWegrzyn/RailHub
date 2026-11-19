"""Everything standing on the pixel map's ground, back to front: trees, rocks, and every building,
station and farmstead, each with its sprite, lifted with the ground under it where hills and
mountains rise (railroads.ui.pixel.relief)."""

import math
import weakref
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Self

import numpy as np
from PIL import Image, ImageDraw

from railroads.sim.constructions import GREW_FROM, BuildingKind, Constructions, Track, TrackWidth
from railroads.sim.grids import BoolGrid, Cell, FloatArray, Grid, IntArray
from railroads.sim.tilemap import Relief, Terrain, TileMap
from railroads.sim.track_rules import TrackBed
from railroads.ui.biomes import Biome, biome_grid
from railroads.ui.building_spots import BuildingSpots
from railroads.ui.iso_camera import PIXEL_CELL_WIDTH
from railroads.ui.pixel.ground import FIELD_KINDS
from railroads.ui.pixel.projection import PixelProjection, hash01
from railroads.ui.pixel.relief import Elevation
from railroads.ui.pixel.sheet import SpriteSheet
from railroads.ui.pixel.track_art import (
    BANK,
    BRIDGE_WIDTH,
    DOUBLE_SPACING,
    PLATFORM_WIDTH,
    centre_line,
    station_house,
    station_line,
)

# What grows on each biome as single trees: how many per cell (a fraction is a chance of one)
# and which kinds, by sprite name prefix. Woods are drawn into the ground as a canopy instead
# (railroads.ui.pixel.ground).
TREES: dict[Biome, tuple[float, tuple[str, ...]]] = {
    Biome.GRASSLAND: (0.06, ("leafy",)),
    Biome.SAVANNA: (0.04, ("leafy",)),
    Biome.TUNDRA: (0.03, ("pine",)),
}


ROCKS_ON = (Biome.MOUNTAIN, Biome.SNOW)


ROCK_CHANCE = 0.4


TREE_SPREAD = 0.4


# How far a building's foot may reach past the edge of its cell, in pixels, and how far along
# the way from where it stands to the cell's middle it is moved in turn until it fits.
CELL_SLACK = 1.0
FIT_STEPS = np.linspace(0.0, 1.0, 6)
# How opaque a sprite's pixel must be to count as part of what it shows.
FOOT_ALPHA = 0.1
# How far above its anchor, in pixels, a sprite's column of pixels may end and still stand on the
# ground.
GROUND_REACH = 3


# Buildings near track stand where they hide as little of it as they can, so it shows: each art
# pixel of track a building would cover where it stands counts HIDE_COST against standing there,
# each pixel its foot reaches out of its cell OUT_COST, each pixel of another building in its cell
# it covers or is covered by CROWD_COST and each pixel it is moved from where it would stand
# MOVE_COST. It stands no further than SHORE_SHARE of the way from the cell's middle to an edge
# water lies beyond. Track counts TRACK_MARGIN pixels wider either side than its bank, and a
# sprite's pixel covers what is under it from HIDING_ALPHA.
HIDE_COST = 10.0
OUT_COST = 4.0
CROWD_COST = 0.5
MOVE_COST = 0.3
SHORE_SHARE = 0.6
TRACK_MARGIN = 1.0
HIDING_ALPHA = 0.5


# Buildings drawn in one of several looks, picked by sprite name prefix; others by one name.
VARIED_SPRITES = {
    BuildingKind.HOUSE: "house",
    BuildingKind.TENEMENT: "tenement",
    BuildingKind.FLATS: "flats",
}


BUILDING_SPRITES = {
    BuildingKind.FARM: "farm",
    BuildingKind.MINE: "mine",
    BuildingKind.FACTORY: "factory",
    BuildingKind.BIG_FACTORY: "big_factory",
    BuildingKind.MEGA_FACTORY: "mega_factory",
    BuildingKind.SKYSCRAPER: "skyscraper",
    BuildingKind.ORCHARD: "orchard",
    BuildingKind.IRON_MINE: "iron_mine",
    BuildingKind.COPPER_MINE: "copper_mine",
    BuildingKind.GOLD_MINE: "gold_mine",
    BuildingKind.DIAMOND_MINE: "diamond_mine",
    BuildingKind.OIL_WELL: "oil_well",
    BuildingKind.SALT_WORKS: "salt_works",
    BuildingKind.FISHERY: "fishery",
    BuildingKind.SHEEP_FARM: "sheep_farm",
    BuildingKind.COTTON_FIELD: "cotton_field",
    BuildingKind.VINEYARD: "vineyard",
    BuildingKind.COFFEE_PLANTATION: "coffee_plantation",
    BuildingKind.RUBBER_PLANTATION: "rubber_plantation",
    BuildingKind.TEXTILE_MILL: "textile_mill",
    BuildingKind.FURNITURE_MAKER: "furniture_maker",
    BuildingKind.TOOL_WORKS: "tool_works",
    BuildingKind.WINERY: "winery",
    BuildingKind.REFINERY: "refinery",
    BuildingKind.CHEMICAL_WORKS: "chemical_works",
    BuildingKind.ELECTRONICS_PLANT: "electronics_plant",
    BuildingKind.CAR_FACTORY: "car_factory",
    BuildingKind.CAR_DEALER: "car_dealer",
    BuildingKind.JEWELLER: "jeweller",
    BuildingKind.ORANGE_FARM: "orange_farm",
    BuildingKind.OUTPOST: "outpost",
    BuildingKind.BASE: "base",
    BuildingKind.HEADQUARTERS: "headquarters",
    BuildingKind.GUN_FACTORY: "gun_factory",
    BuildingKind.TANK_FACTORY: "tank_factory",
    BuildingKind.PLANE_FACTORY: "plane_factory",
    BuildingKind.AIRPORT: "airport",
    BuildingKind.BARREL_MAKER: "barrel_maker",
    BuildingKind.BEEKEEPER: "beekeeper",
    BuildingKind.INCENSE_GROVE: "incense_grove",
    BuildingKind.CHURCH: "church",
    BuildingKind.BASILICA: "basilica",
    BuildingKind.CATHEDRAL: "cathedral",
    BuildingKind.ST_PAULS: "st_pauls",
    BuildingKind.DATE_PALMS: "date_palms",
    BuildingKind.BANANAS: "bananas",
    BuildingKind.SUGAR: "sugar",
    BuildingKind.FOOD_PLANT: "food_plant",
    BuildingKind.BAKERY: "bakery",
    BuildingKind.GASWORKS: "gasworks",
    BuildingKind.COKE_PLANT: "coke_plant",
    BuildingKind.POTTER: "potter",
    BuildingKind.TILE_WORKS: "tile_works",
    BuildingKind.FOUNDRY: "foundry",
    BuildingKind.DAIRY: "dairy",
    BuildingKind.SAUSAGE_MAKER: "sausage_maker",
    BuildingKind.CARPET_WEAVER: "carpet_weaver",
    BuildingKind.CLOCK_MAKER: "clock_maker",
    BuildingKind.MIRROR_MAKER: "mirror_maker",
    BuildingKind.MANOR: "manor",
    BuildingKind.THEATRE: "theatre",
    BuildingKind.HOSPITAL: "hospital",
    BuildingKind.RAILWAY_WORKS: "railway_works",
    BuildingKind.STEEL_MILL: "steel_mill",
    BuildingKind.SHIPYARD: "shipyard",
    BuildingKind.LOGGING_CAMP: "logging_camp",
    BuildingKind.SAWMILL: "sawmill",
    BuildingKind.PORT: "port",
    BuildingKind.CLAY_PIT: "clay_pit",
    BuildingKind.LIMESTONE_QUARRY: "limestone_quarry",
    BuildingKind.SAND_PIT: "sand_pit",
    BuildingKind.POTATO_FARM: "potato_farm",
    BuildingKind.RICE_PADDY: "rice_paddy",
    BuildingKind.SOY_FIELD: "soy_field",
    BuildingKind.TEA_PLANTATION: "tea_plantation",
    BuildingKind.SOY_SAUCE_BREWERY: "soy_sauce_brewery",
    BuildingKind.TOFU_MAKER: "tofu_maker",
    BuildingKind.MISO_MAKER: "miso_maker",
    BuildingKind.FISH_DRYER: "fish_dryer",
    BuildingKind.CATTLE_RANCH: "cattle_ranch",
    BuildingKind.PIG_FARM: "pig_farm",
    BuildingKind.CHICKEN_FARM: "chicken_farm",
    BuildingKind.FLAX_FIELD: "flax_field",
    BuildingKind.HOP_GARDEN: "hop_garden",
    BuildingKind.MACHINE_WORKS: "machine_works",
    BuildingKind.BRICKWORKS: "brickworks",
    BuildingKind.CEMENT_WORKS: "cement_works",
    BuildingKind.GLASSWORKS: "glassworks",
    BuildingKind.BREWERY: "brewery",
    BuildingKind.DISTILLERY: "distillery",
    BuildingKind.SAKE_BREWERY: "sake_brewery",
    BuildingKind.TANNERY: "tannery",
    BuildingKind.SHOE_FACTORY: "shoe_factory",
    BuildingKind.PLASTICS_WORKS: "plastics_works",
    BuildingKind.APPLIANCE_FACTORY: "appliance_factory",
    BuildingKind.PAPER_MILL: "paper_mill",
    BuildingKind.PRINTING_WORKS: "printing_works",
    BuildingKind.SUGAR_REFINERY: "sugar_refinery",
    BuildingKind.PORCELAIN_WORKS: "porcelain_works",
    BuildingKind.MEAT_FACTORY: "meat_factory",
    BuildingKind.POWER_PLANT: "power_plant",
    BuildingKind.DIESEL_POWER_PLANT: "diesel_power_plant",
    BuildingKind.GAS_POWER_PLANT: "gas_power_plant",
    BuildingKind.SPACE_CENTER: "space_center",
    BuildingKind.PEAR_HQ: "pear_hq",
    BuildingKind.ROYAL_PALACE: "royal_palace",
    BuildingKind.SALT_CATHEDRAL: "salt_cathedral",
    BuildingKind.SPA_RESORT: "spa_resort",
    BuildingKind.WORLDS_FAIR: "worlds_fair",
    BuildingKind.FILM_STUDIO: "film_studio",
    BuildingKind.AMUSEMENT_PARK: "amusement_park",
    BuildingKind.OLYMPIC_STADIUM: "olympic_stadium",
    BuildingKind.TAVERN: "tavern",
    BuildingKind.CLUB: "tavern",
    BuildingKind.SUSHI_HUT: "sushi_hut",
    BuildingKind.FILLING_STATION: "car_dealer",
    BuildingKind.DEPARTMENT_STORE: "department_store",
    BuildingKind.HOTEL: "hotel",
    BuildingKind.JUICE_WORKS: "winery",
}

# Big and mega works have sprites of their own (railroads.ui.pixel.art.GROWN_SCALE).
for _grown, _kind in GREW_FROM.items():
    BUILDING_SPRITES[_grown] = f"{_grown.name.split('_')[0].lower()}_{BUILDING_SPRITES[_kind]}"


def platform_sprite(line: str | None = None) -> str:
    """The sprite of a platform added to a station, turned like station_sprite."""
    return "platform" if line is None else f"platform_{line}"


def station_sprite(level: int, line: str | None = None) -> str:
    """The sprite of a station raised to a level, turned along track running `line` on the
    screen (railroads.ui.pixel.track_art.station_line) if it is given."""
    name = "station" if level <= 1 else f"station_{level}"
    return name if line is None else f"{name}_{line}"


@dataclass(frozen=True)
class Scenery:
    """Everything standing on the ground, sorted back to front: cell by cell, by how far the
    middle of the cell each stands in lies from the viewer, and within a cell by where it stands,
    so what stands in a cell nearer the viewer, such as a station house wider than its cell, is
    drawn over what stands in the cells behind. For each: where it stands in art pixels, which
    sprite it is, how many art pixels the ground under it is lifted, and whether it is drawn
    under the track, as a building that cannot stand clear of the track is, so the track shows
    over it."""

    x: IntArray
    y: IntArray
    sprite: IntArray
    lift: IntArray
    under_track: BoolGrid

    @classmethod
    def of(
        cls,
        tile_map: TileMap,
        constructions: Constructions,
        projection: PixelProjection,
        sheet: SpriteSheet,
        station_levels: Mapping[Cell, int] | None = None,
        elevation: Elevation | None = None,
    ) -> Self:
        grid = tile_map.grid
        biomes = biome_grid(tile_map)
        built = np.zeros(grid.shape, dtype=bool)
        for construction in constructions:
            built[construction.cell] = True
        xs: list[FloatArray] = []
        ys: list[FloatArray] = []
        sprites: list[IntArray] = []
        for biome, (per_cell, kinds) in TREES.items():
            rows, cols = np.nonzero((biomes == biome) & ~built)
            choices = np.concatenate([sheet.variants(kind) for kind in kinds])
            weights = np.concatenate([np.full(len(sheet.variants(k)), 1.0) for k in kinds])
            weights /= weights.sum()
            for k in range(math.ceil(per_cell)):
                chance = min(1.0, per_cell - k)
                keep = hash01(rows, cols, k, 1) < chance
                r, c = rows[keep], cols[keep]
                x, y = _jitter(grid, r, c, k)
                pick = np.searchsorted(np.cumsum(weights), hash01(r, c, k, 2), side="right")
                xs.append(x)
                ys.append(y)
                sprites.append(choices[np.minimum(pick, len(choices) - 1)])
        rocky = np.isin(biomes, ROCKS_ON) & (tile_map.relief != Relief.MOUNTAINS)
        rows, cols = np.nonzero(rocky & ~built)
        keep = hash01(rows, cols, 3) < ROCK_CHANCE
        x, y = _jitter(grid, rows[keep], cols[keep], 9)
        xs.append(x)
        ys.append(y)
        sprites.append(np.full(len(x), sheet.index("rocks")))
        stations = constructions.every_station_cell()
        station_from = sum(len(x) for x in xs)
        if stations:
            x, y = _station_houses(constructions, projection, stations)
            xs.append(x)
            ys.append(y)
            levels = station_levels or {}
            turned = grid.kind == "hex"
            sprites.append(
                np.array(
                    [
                        sheet.index(
                            station_sprite(levels.get(cell, 1), line)
                            if constructions.main_station(cell) == cell
                            else platform_sprite(line)
                        )
                        for cell in stations
                        for line in [station_line(constructions, cell) if turned else None]
                    ]
                )
            )
        spots = BuildingSpots.of(constructions, frozenset(FIELD_KINDS))
        looks = {kind: sheet.variants(name) for kind, name in VARIED_SPRITES.items()}
        built_looks = np.array(
            [
                looks[kind][variant % len(looks[kind])]
                if kind in looks
                else sheet.index(BUILDING_SPRITES[kind])
                for kind, variant in zip(spots.kinds, spots.variants, strict=True)
            ],
            dtype=np.int64,
        )
        before = sum(len(x) for x in xs)
        xs.append(np.asarray(spots.x, dtype=np.float64))
        ys.append(np.asarray(spots.y, dtype=np.float64))
        sprites.append(built_looks)
        mx, my = np.concatenate(xs), np.concatenate(ys)
        px, py = projection.to_pixels(mx, my)
        in_rows, in_cols = grid.cell_at(mx, my)
        in_rows, in_cols = np.array(in_rows, dtype=np.int64), np.array(in_cols, dtype=np.int64)
        if stations:
            own = np.array(stations, dtype=np.int64)
            in_rows[station_from : station_from + len(own)] = own[:, 0]
            in_cols[station_from : station_from + len(own)] = own[:, 1]
        depth = projection.to_pixels(*grid.center(in_rows, in_cols))[1]
        fitted = np.flatnonzero(~np.isin(spots.kinds, list(FIELD_KINDS)))
        if fitted.size:
            px[before + fitted], py[before + fitted] = _inside_cells(
                grid,
                projection,
                sheet,
                spots.x[fitted],
                spots.y[fitted],
                built_looks[fitted],
            )
        px, py = np.rint(px).astype(np.int64), np.rint(py).astype(np.int64)
        sprite = np.concatenate(sprites).astype(np.int64)
        rows, cols = grid.cell_at(spots.x, spots.y)
        cells = list(zip(np.asarray(rows).tolist(), np.asarray(cols).tolist(), strict=True))
        under = _off_track(constructions, projection, sheet, (px, py), sprite, before, cells)
        lift = np.zeros(px.shape, dtype=np.int64)
        if elevation is not None and px.size:
            x, y = projection.to_plane(px, py)
            lift = np.rint(elevation.lift(x, y)).astype(np.int64)
        order = np.lexsort((px, py, np.round(depth, 6)))
        return cls(px[order], py[order], sprite[order], lift[order], under[order])

    def __len__(self) -> int:
        return len(self.x)


def _inside_cells(
    grid: Grid,
    projection: PixelProjection,
    sheet: SpriteSheet,
    x: FloatArray,
    y: FloatArray,
    sprite: IntArray,
) -> tuple[FloatArray, FloatArray]:
    """Where in the picture buildings standing at (x, y) on the map are drawn so the foot of each
    stays inside its cell (_feet): moved no further than it must towards the middle of its cell,
    and all the way if even there it does not fit."""
    rows, cols = grid.cell_at(x, y)
    px, py = projection.to_pixels(x, y)
    cx, cy = projection.to_pixels(*grid.center(rows, cols))
    outward, reach = _cell_edges(grid, projection)
    reach = reach + CELL_SLACK
    unique, which = np.unique(sprite, return_inverse=True)
    feet = _feet(sheet, unique)[which]
    away = np.stack([px - cx, py - cy], axis=1)
    kept = np.zeros(len(px))
    for share in FIT_STEPS:
        points = feet + (away * share)[:, None, :]
        fits = (np.einsum("npj,ej->npe", points, outward) <= reach).all(axis=(1, 2))
        kept[fits] = share
    return cx + away[:, 0] * kept, cy + away[:, 1] * kept


def _cell_edges(grid: Grid, projection: PixelProjection) -> tuple[FloatArray, FloatArray]:
    """The edges of a cell in the picture: a unit vector out across each, and how far out from
    the cell's middle each lies along it, in pixels."""
    middle = np.array(projection.to_pixels(*grid.center(0, 0)))
    corners = np.array([projection.to_pixels(*c) for c in grid.corners(0, 0)]) - middle
    outward = np.roll(corners, -1, axis=0) - corners
    outward = np.stack([outward[:, 1], -outward[:, 0]], axis=1)
    outward /= np.linalg.norm(outward, axis=1, keepdims=True)
    outward *= np.sign(np.einsum("ij,ij->i", outward, corners))[:, None]
    return outward, np.einsum("ij,ij->i", outward, corners)


# A building standing in a cell: its sprite and where it stands, in pixels from the pixel nearest
# the cell's middle.
Standing = tuple[int, int, int]


def _off_track(
    constructions: Constructions,
    projection: PixelProjection,
    sheet: SpriteSheet,
    at: tuple[IntArray, IntArray],
    sprite: IntArray,
    first: int,
    cells: list[Cell],
) -> BoolGrid:
    """Moves each building in or next to a cell with track to where it hides as little of the
    track as it can (_placed_off_track), changing where things stand in `at`; returns which still
    hide some, to be drawn under the track. The buildings are everything from `first` on, and
    `cells` holds the cell of each. Stations fill their cells, so no building shares one."""
    px, py = at
    under = np.zeros(len(px), dtype=bool)
    near = _track_near(constructions)
    if not near:
        return under
    grid = constructions.tile_map.grid
    here: dict[Cell, list[int]] = {}
    for i, cell in enumerate(cells, start=first):
        if cell in near:
            here.setdefault(cell, []).append(i)
    known = _KNOWN.setdefault(constructions, {})
    for cell, found in here.items():
        cx, cy = projection.to_pixels(*grid.center(*cell))
        origin = (int(np.rint(cx)), int(np.rint(cy)))
        wanted = tuple(
            (int(sprite[i]), int(px[i]) - origin[0], int(py[i]) - origin[1]) for i in found
        )
        key = (id(sheet), projection.matrix.tobytes(), near[cell], wanted)
        entry = known.get(cell)
        if entry is None or entry[0] != key:
            placed = _placed_off_track(
                constructions.tile_map, projection, sheet, cell, near[cell], wanted
            )
            entry = known[cell] = (key, placed)
        for i, (dx, dy, hiding) in zip(found, entry[1], strict=True):
            px[i], py[i] = origin[0] + dx, origin[1] + dy
            under[i] = hiding
    return under


# What _off_track worked out for each cell, for each Constructions, with what it was worked out
# from.
_KNOWN: weakref.WeakKeyDictionary[Constructions, dict[Cell, tuple]] = weakref.WeakKeyDictionary()


# The track near each cell (_track_near) for each Constructions, with the railway version it was
# found for.
_NEAR: weakref.WeakKeyDictionary[Constructions, tuple[int, dict[Cell, tuple]]] = (
    weakref.WeakKeyDictionary()
)


def _track_near(constructions: Constructions) -> dict[Cell, tuple[tuple[Track, bool], ...]]:
    """The track drawn in or next to each cell, each piece with whether it runs through a
    station, with its platform beside it. Track in tunnels is out of sight and left out. Found
    again only once track or stations changed."""
    version = constructions.railway_version
    kept = _NEAR.get(constructions)
    if kept is not None and kept[0] == version:
        return kept[1]
    grid = constructions.tile_map.grid
    pieces = [(piece, False) for piece in constructions.track() if piece.bed != TrackBed.TUNNEL]
    pieces += [
        (Track(cell, constructions.station_ends(cell)), True)
        for cell in constructions.every_station_cell()
    ]
    near: dict[Cell, list[tuple[Track, bool]]] = {}
    for piece, station in pieces:
        for cell in (piece.cell, *grid.neighbours(*piece.cell)):
            near.setdefault(cell, []).append((piece, station))
    found_near = {cell: tuple(found) for cell, found in near.items()}
    _NEAR[constructions] = (version, found_near)
    return found_near


def _placed_off_track(
    tile_map: TileMap,
    projection: PixelProjection,
    sheet: SpriteSheet,
    cell: Cell,
    pieces: tuple[tuple[Track, bool], ...],
    wanted: tuple[Standing, ...],
) -> list[tuple[int, int, bool]]:
    """Where each building wanted where it is in a cell near track stands, in pixels from the
    pixel nearest the cell's middle, and whether it still hides some of the track. One clear of
    the track where it is stays there; the others, biggest first, stand where hiding the track,
    reaching out of the cell, crowding the others in it and moving cost least (HIDE_COST and
    the rest), with their anchor inside the cell and further in from the shore (_room)."""
    grid = tile_map.grid
    cx, cy = projection.to_pixels(*grid.center(*cell))
    middle = np.array([cx - np.rint(cx), cy - np.rint(cy)])
    outward, reach = _room(tile_map, projection, cell)
    corners = np.array([projection.to_pixels(*c) for c in grid.corners(0, 0)])
    sx, sy = np.ceil(np.ptp(corners, axis=0) / 2).astype(int).tolist()
    half = 2 * max(sx, sy) + int(sheet.sizes.max()) + 1
    track = _track_mask(grid, projection, cell, pieces, half, TRACK_MARGIN)
    bare = _track_mask(grid, projection, cell, pieces, half, 0.0)
    crowded = np.zeros_like(track)
    masks = {
        index: (sheet.pixels[index][..., 3] > HIDING_ALPHA).astype(np.float32)
        for index, _, _ in wanted
    }
    offsets = np.stack(np.meshgrid(np.arange(-sx, sx + 1), np.arange(-sy, sy + 1)), axis=-1)
    inside = ((offsets - middle) @ outward.T <= reach).all(axis=-1)
    placed: list[tuple[int, int, bool] | None] = [None] * len(wanted)
    order = []
    for n, (index, dx, dy) in enumerate(wanted):
        if _covered(track, masks[index], sheet.anchors[index], half, dx, dy) == 0:
            placed[n] = (dx, dy, False)
            _stamp(crowded, masks[index], sheet.anchors[index], half, dx, dy)
        else:
            order.append(n)
    order.sort(key=lambda n: -masks[wanted[n][0]].sum())
    feet = _feet(sheet, np.array([wanted[n][0] for n in order], dtype=np.int64))
    for n, foot in zip(order, feet, strict=True):
        index, dx, dy = wanted[n]
        mask, anchor = masks[index], sheet.anchors[index]
        hidden = _covering(track, mask, anchor, half, (sx, sy))
        crowding = _covering(crowded, mask, anchor, half, (sx, sy))
        reached = (offsets[:, :, None, :] - middle + foot) @ outward.T - reach - CELL_SLACK
        out = np.clip(reached, 0, None).sum(axis=(2, 3))
        moved = np.hypot(offsets[..., 0] - dx, offsets[..., 1] - dy)
        cost = HIDE_COST * hidden + OUT_COST * out + CROWD_COST * crowding + MOVE_COST * moved
        best = np.unravel_index(np.argmin(np.where(inside, cost, np.inf)), cost.shape)
        x, y = (int(v) for v in offsets[best])
        placed[n] = (x, y, _covered(bare, mask, anchor, half, x, y) > 0)
        _stamp(crowded, mask, anchor, half, x, y)
    return [spot for spot in placed if spot is not None]


def _room(
    tile_map: TileMap, projection: PixelProjection, cell: Cell
) -> tuple[FloatArray, FloatArray]:
    """The edges of a cell in the picture as _cell_edges finds them, with how far out towards
    each a building may stand: to the edge, or SHORE_SHARE of the way where water lies beyond
    it, as the shore winds into the cell."""
    grid = tile_map.grid
    outward, reach = _cell_edges(grid, projection)
    cx, cy = projection.to_pixels(*grid.center(*cell))
    for row, col in grid.neighbours(*cell):
        if tile_map.terrain[row, col] != Terrain.LAND:
            nx, ny = projection.to_pixels(*grid.center(row, col))
            reach[np.argmax(outward @ [nx - cx, ny - cy])] *= SHORE_SHARE
    return outward, reach


def _track_mask(
    grid: Grid,
    projection: PixelProjection,
    cell: Cell,
    pieces: tuple[tuple[Track, bool], ...],
    half: int,
    margin: float,
) -> FloatArray:
    """Where the track lies round a cell in the picture, as 1 where it does and 0 elsewhere,
    over 2 * `half` pixels each way with the pixel nearest the cell's middle at (`half`,
    `half`): as wide as its bank, both tracks of double track, a bridge's deck or a station's
    platform reach, and `margin` pixels more either side."""
    cx, cy = projection.to_pixels(*grid.center(*cell))
    left, top = np.rint(cx) - half + 0.5, np.rint(cy) - half + 0.5
    steps = grid.track_steps()
    image = Image.new("L", (2 * half, 2 * half))
    paint = ImageDraw.Draw(image)
    for piece, station in pieces:
        side = (BRIDGE_WIDTH if piece.bed == TrackBed.BRIDGE else BANK) / 2
        side += DOUBLE_SPACING / 2 if piece.width == TrackWidth.DOUBLE else 0.0
        side += PLATFORM_WIDTH if station else 0.0
        line = centre_line(grid, steps, projection.matrix, piece, side * PIXEL_CELL_WIDTH)
        x, y = projection.to_pixels(*np.array(line).T)
        width = round(2 * (side * PIXEL_CELL_WIDTH + margin))
        points = list(zip((x - left).tolist(), (y - top).tolist(), strict=True))
        paint.line(points, fill=255, width=width, joint="curve")
    return (np.asarray(image) > 0).astype(np.float32)


def _place(
    area: FloatArray, mask: FloatArray, anchor: IntArray, half: int, dx: int, dy: int
) -> tuple[slice, slice] | None:
    """The part of `area` a sprite's mask lies over with its anchor `dx`, `dy` from the middle;
    None if it reaches past the area."""
    top, left = half + dy - int(anchor[1]), half + dx - int(anchor[0])
    bottom, right = top + mask.shape[0], left + mask.shape[1]
    if top < 0 or left < 0 or bottom > area.shape[0] or right > area.shape[1]:
        return None
    return slice(top, bottom), slice(left, right)


def _covered(
    area: FloatArray, mask: FloatArray, anchor: IntArray, half: int, dx: int, dy: int
) -> float:
    """How much of `area` a sprite's mask covers with its anchor `dx`, `dy` from the middle; all
    of it, as far as anything goes, if the sprite reaches past it."""
    part = _place(area, mask, anchor, half, dx, dy)
    return math.inf if part is None else float((area[part] * mask).sum())


def _stamp(
    area: FloatArray, mask: FloatArray, anchor: IntArray, half: int, dx: int, dy: int
) -> None:
    part = _place(area, mask, anchor, half, dx, dy)
    if part is not None:
        area[part] += mask


def _covering(
    area: FloatArray, mask: FloatArray, anchor: IntArray, half: int, span: tuple[int, int]
) -> FloatArray:
    """How much of `area` a sprite's mask covers with its anchor at each pixel up to `span`
    across and down from the middle, as a (2 * down + 1, 2 * across + 1) array."""
    (sx, sy), (h, w) = span, mask.shape
    top, left = half - sy - int(anchor[1]), half - sx - int(anchor[0])
    region = area[top : top + 2 * sy + h, left : left + 2 * sx + w]
    windows = np.lib.stride_tricks.sliding_window_view(region, (h, w))
    return np.einsum("ijkl,kl->ij", windows, mask)


def _feet(sheet: SpriteSheet, sprites: IntArray) -> FloatArray:
    """Where each sprite meets the ground at its left and right ends and at its lowest, in
    pixels from its anchor: of the columns of pixels reaching down to within GROUND_REACH of the
    anchor, so smoke and chimneys do not count."""
    feet = np.zeros((len(sprites), 3, 2))
    for i, index in enumerate(sprites.tolist()):
        painted = sheet.pixels[index][..., 3] > FOOT_ALPHA
        lowest = painted.shape[0] - 1 - np.argmax(painted[::-1], axis=0)
        reaching = painted.any(axis=0) & (lowest >= sheet.anchors[index][1] - GROUND_REACH)
        columns = np.flatnonzero(reaching)
        if not columns.size:
            continue
        bottom = columns[np.argmax(lowest[columns])]
        feet[i] = [
            (column, lowest[column]) for column in (columns[0], columns[-1], bottom)
        ] - sheet.anchors[index]
    return feet


def _station_houses(
    constructions: Constructions, projection: PixelProjection, stations: list[Cell]
) -> tuple[FloatArray, FloatArray]:
    """Where each station's house stands: set back from the track through it, away from the
    viewer, so the track runs past in front of it (railroads.ui.pixel.track_art)."""
    grid = constructions.tile_map.grid
    xs, ys = [], []
    for cell in stations:
        back = np.linalg.solve(projection.matrix, station_house(constructions, projection, cell))
        cx, cy = grid.center(*cell)
        xs.append(float(cx) + float(back[0]))
        ys.append(float(cy) + float(back[1]))
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)


def _jitter(
    grid: Grid, rows: IntArray, cols: IntArray, k: int, spread: float = TREE_SPREAD
) -> tuple[FloatArray, FloatArray]:
    cx, cy = grid.center(rows, cols)
    dx = (hash01(rows, cols, k, 4) - 0.5) * 2 * spread
    dy = (hash01(rows, cols, k, 5) - 0.5) * 2 * spread
    return np.asarray(cx + dx, dtype=np.float64), np.asarray(cy + dy, dtype=np.float64)
