"""Big and mega works of fire, each drawn as a building of its own: foundries glowing round their cupolas, tile works, potters
and porcelain works among clusters of bottle kilns and stacked ware, brickworks of Hoffmann kilns
under tall chimneys, cement works of glowing rotary kilns under dusty silos, glassworks of brick
cones lit at their mouths and gasworks of great gasholders in iron guide frames."""

import itertools
import math

import numpy as np

from railroads.ui.pixel.art.commerce import (
    _lamp_post,
)
from railroads.ui.pixel.art.crafts import (
    BEATEN_EARTH,
    BLACK_SAND,
    CAST_IRON,
    CELADON,
    CLAY,
    COBALT,
    CREAM_CLAY,
    CREAM_GLAZE,
    FOUNDRY_YARD,
    FURNACE_GLOW,
    GILT,
    GRAVEL,
    MOLTEN,
    MOULDING_SAND,
    OFFICE_WALL,
    PANTILE,
    PLANK,
    POTTER_WALL,
    ROOF_TILE,
    SHOP_GLOW,
    SLATE,
    SOOTED_BRICK,
    STOVE_GREEN,
    TILE_WHITE,
    WORKS_YARD,
    _column,
    _craft_arch,
    _craft_band,
    _craft_beam,
    _craft_bell,
    _craft_bottle_kiln,
    _craft_casement,
    _craft_cogwheel,
    _craft_cupola,
    _craft_drying_board,
    _craft_flasks,
    _craft_hack,
    _craft_hoop,
    _craft_kiln_block,
    _craft_log_pile,
    _craft_monitor,
    _craft_mound,
    _craft_open_shed,
    _craft_pantiles,
    _craft_pipes,
    _craft_pot,
    _craft_pour,
    _craft_pug_mill,
    _craft_smoke,
    _craft_tile_panel,
    _craft_tile_stack,
    _craft_ware_stall,
    _craft_wheelset,
    _hip,
)
from railroads.ui.pixel.art.homes import (
    _outward,
)
from railroads.ui.pixel.art.industry import (
    BRICK_CLAY,
    BRICK_DUST,
    BRICK_FIRED,
    BRICK_GLOW,
    BRICK_KILN,
    BRICK_ORANGE,
    BRICK_PANTILE,
    DRESSING,
    GLASS_AMBER,
    GLASS_AQUA,
    GLASS_COBALT,
    GLASS_GREEN,
    GLASS_SEA,
    PUDDLE,
    SMOKE,
    SOIL,
    _bed,
    _brick_hack,
    _brick_pallet,
    _brick_tub,
    _crate,
    _door,
    _footing,
    _girders,
    _glass_baskets,
    _glass_carboys,
    _glass_cone,
    _glass_crates,
    _glass_cullet,
    _glass_house,
    _glass_lehr,
    _glass_packing,
    _glass_panes,
    _glass_yard,
    _mill_window,
    _on_roof,
    _ring,
    _roof_rows,
    _round_head,
    _sack,
    _skylight,
    _smoke,
    _smoking_chimney,
    _spill,
    _stains,
    _tall_chimney,
    _track,
    _window,
)
from railroads.ui.pixel.art.painter import (
    BRICK,
    CHIMNEY_SHARE,
    DOORWAY,
    RGB,
    TIMBER,
    WINDOW,
    Painter,
    Point,
    Sprite,
    Top,
    shade,
)
from railroads.ui.pixel.art.parts import (
    DOOR,
    METRE,
    _band,
    _bar,
    _gable_across,
    _heap,
    _iso,
    _on_wall,
    _patch,
    _plinth,
    _ridge,
    _tiles,
    _turned,
    _vault,
    _windows,
)

STEAM: RGB = (238, 238, 242)

WHITE_HOT: RGB = (255, 226, 120)

BALLAST: RGB = (126, 118, 106)

RAIL: RGB = (92, 92, 100)

WAGON: RGB = (104, 74, 54)

OVEN_ROOF: RGB = (70, 66, 64)

OVEN_DOOR: RGB = (40, 36, 36)

QUENCH: RGB = (146, 138, 126)

COKE: RGB = (70, 70, 76)

COAL: RGB = (46, 44, 50)

CONCRETE: RGB = (176, 172, 162)

TANK: RGB = (110, 118, 112)


def _mid(a: Point, b: Point) -> Point:
    return (a[0] + b[0]) / 2, (a[1] + b[1]) / 2


def _out(point: Point, reach: float) -> Point:
    """The point `reach` out over the ground from a lit wall's foot at `point`, towards the eye."""
    return point[0] - reach, point[1] + reach / 2


def _rails(p: Painter, a: Point, b: Point) -> None:
    """A railway siding on a bed of ballast along the ground from `a` to `b`, which run the way
    lit walls do, with its sleepers across and two rails along it."""
    p.polygon([a, b, _out(b, 3), _out(a, 3)], BALLAST)
    steps = max(2, round(abs(b[0] - a[0]) / 1.6))
    for k in range(steps + 1):
        t = k / steps
        x, y = a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t
        p.line([_out((x, y), 0.4), _out((x, y), 2.6)], shade(TIMBER, 0.6))
    for reach in (0.9, 2.1):
        p.line([_out(a, reach), _out(b, reach)], RAIL)


def _wagon(p: Painter, x: float, y: float, load: RGB, glowing: bool = False) -> None:
    """An open railway wagon standing on rails at (x, y), heaped with `load`, glowing if hot."""
    top = p.hall(x, y, 6, 2.6, 2.2, WAGON, None)
    left, front, right, back = top
    _band(p, left, front, 1.1, shade(WAGON, 0.7))
    p.polygon([left, front, right, back], load, shade(load, 0.7))
    p.blob(*_mid(left, back), *_mid(front, right), shade(load, 1.15))
    if glowing:
        for t in (0.3, 0.6):
            p.point(*_on_wall(left, front, t, -0.6), WHITE_HOT)
    for t in (0.2, 0.8):
        wx, wy = _on_wall(left, front, t, 2.2)
        p.blob(wx - 0.8, wy - 0.6, wx + 0.8, wy + 0.8, OVEN_DOOR)


def _coke_heap(p: Painter, x: float, y: float, half: float) -> None:
    _spill(p, x, y, half, COKE)
    _heap(p, x, y, half, half * 0.9, COKE)
    for dx, dy in ((-1, -1.5), (0.6, -2.4)):
        p.point(x + dx, y + dy, shade(COKE, 1.5))


def big_foundry() -> Sprite:
    """An iron foundry grown large: a long casting hall of soot-stained red brick with a louvred
    monitor along its slate ridge and arched windows glowing, two great doors open on ladles
    pouring white-hot iron into the moulds, their light spilling over the yard; two riveted
    cupola furnaces side by side on one charging floor, tapping into ladles and throwing sparks;
    heaps of tan and black moulding sand by stacked moulding boxes; and a crane yard where an
    iron travelling crane on rails carries a new bell over castings set out to be fetched: bells
    in their frames, railway wheels, a cogwheel, girders and pipes."""
    m = METRE
    p = Painter(56, 44, (28, 30))

    def at(u: float, v: float) -> Point:
        return _iso(0, 0, (u - 20) * m, (v - 12.5) * m)

    p.polygon(
        [at(-1, -1), at(41, -1), at(41, 26), at(-1, 26)], FOUNDRY_YARD, shade(FOUNDRY_YARD, 0.85)
    )
    _fire_casting_hall(p, *at(14, 19.5), 28 * m, 10 * m, 8 * m, (0.28, 0.6))
    _craft_cupola(p, *at(32, 23))
    _craft_cupola(p, *at(32, 18))
    _craft_mound(p, *at(3, 9.5), 2.4 * m, 2.0 * m, MOULDING_SAND)
    _craft_mound(p, *at(6.5, 7.5), 1.8 * m, 1.4 * m, BLACK_SAND)
    _craft_mound(p, *at(2.5, 5), 1.6 * m, 1.2 * m, MOULDING_SAND)
    _craft_flasks(p, *at(10, 9.5), 3)
    _craft_flasks(p, *at(11.5, 7.8), 2)
    _craft_flasks(p, *at(8, 4), 2)
    for v in (10.5, 1.5):
        _fire_crane_rail(p, at(15, v), at(40, v))
    hook = _fire_gantry(p, at(31, 1.5), at(31, 10.5), 7.5 * m, 0.55)
    _craft_bell(p, *at(19, 7))
    _craft_cogwheel(p, *at(23, 8.5), 1.4 * m)
    _girders(p, *at(36.5, 8), 6, 3)
    _craft_wheelset(p, *at(25.5, 4))
    _craft_wheelset(p, *at(27.5, 3))
    _craft_pipes(p, *at(36, 4), 4)
    _craft_bell(p, *at(19.5, 2.5))
    _fire_hung_bell(p, *hook, 2.0 * m)
    _fire_gantry_leg(p, *at(31, 1.5), 7.5 * m)
    return p.sprite()


def mega_foundry() -> Sprite:
    """A great iron works of founders: two casting halls of soot-stained red brick one behind the
    other under slate roofs with louvred monitors, arched windows glowing and great doors open on
    ladles pouring; three riveted cupola furnaces in a row on one long charging floor, tapping and
    throwing sparks; heaps of moulding sand by stacked moulding boxes; and a crane yard spanned by
    two iron travelling cranes on rails, one tipping a great ladle to pour a white-hot stream
    into a casting pit sunk in the yard, glowing and throwing sparks, the other carrying a new
    bell, over a yard of castings: a flywheel stood on edge, bells, railway wheels, girders and
    pipes."""
    m = METRE
    p = Painter(66, 50, (33, 34))

    def at(u: float, v: float) -> Point:
        return _iso(0, 0, (u - 23.5) * m, (v - 15) * m)

    p.polygon(
        [at(-1, -1.5), at(47.5, -1.5), at(47.5, 31), at(-1, 31)],
        FOUNDRY_YARD,
        shade(FOUNDRY_YARD, 0.85),
    )
    _fire_casting_hall(p, *at(15, 26.5), 30 * m, 9 * m, 9.5 * m)
    _craft_cupola(p, *at(35, 27.5))
    _craft_cupola(p, *at(35, 23))
    _fire_casting_hall(p, *at(13, 16), 26 * m, 10 * m, 8 * m, (0.26, 0.62))
    _craft_cupola(p, *at(35, 18.5))
    for u, v, half, tall, colour in (
        (2.5, 8, 2.4, 2.0, MOULDING_SAND),
        (6, 6, 1.8, 1.4, BLACK_SAND),
        (2.5, 3.5, 1.8, 1.4, MOULDING_SAND),
        (6.5, 2, 1.4, 1.1, BLACK_SAND),
    ):
        _craft_mound(p, *at(u, v), half * m, tall * m, colour)
    for u, v, layers in ((10, 9, 3), (11.5, 7.3, 2), (10, 4.5, 3), (11.5, 2.8, 2)):
        _craft_flasks(p, *at(u, v), layers)
    for v in (9.5, 0):
        _fire_crane_rail(p, at(15, v), at(47, v))
    hook = _fire_gantry(p, at(24, 0), at(24, 9.5), 8.5 * m, 0.5)
    lip = _fire_great_ladle(p, *hook, 1.3)
    _fire_casting_pit(p, *at(22.5, 4.0), lip)
    _fire_gantry_leg(p, *at(24, 0), 8.5 * m)
    hook = _fire_gantry(p, at(38, 0), at(38, 9.5), 8.5 * m, 0.6)
    _craft_cogwheel(p, *_iso(*at(31, 7), 0, -2.4 * m), 2.2 * m, 22)
    _girders(p, *at(43.5, 7.5), 6, 4)
    _craft_bell(p, *at(33, 3))
    _craft_wheelset(p, *at(29, 3.5))
    _craft_wheelset(p, *at(30.5, 2.2))
    _craft_pipes(p, *at(44, 3.5), 4)
    _fire_hung_bell(p, *hook, 2.4 * m)
    _fire_gantry_leg(p, *at(38, 0), 8.5 * m)
    for u in (16, 18):
        _lamp_post(p, *at(u, -0.8), 4.2)
    return p.sprite()


RETORT_ROOF: RGB = (90, 76, 70)

GIRDER: RGB = (120, 122, 130)


def _arc(x: float, y: float, r: float, start: float, end: float, steps: int) -> list[Point]:
    """Points round the ground ring of radius `r` about (x, y) from angle `start` to `end`."""
    return [
        (
            x + r * math.cos(start + (end - start) * k / steps),
            y + r / 2 * math.sin(start + (end - start) * k / steps),
        )
        for k in range(steps + 1)
    ]


def _gasholder(
    p: Painter, x: float, y: float, r: float, frame: float, bell: float, lifts: int = 1
) -> None:
    """A round gasholder standing on (x, y): its grey-green bell of `lifts` telescoping lifts
    riding `bell` high out of a brick water tank, inside a guide frame of iron columns `frame`
    high tied by two rings of girders and lattice bracing at the front."""
    ring = r + 1.2
    columns = [math.pi / 8 + math.pi / 4 * k for k in range(8)]
    feet = [(x + ring * math.cos(a), y + ring / 2 * math.sin(a)) for a in columns]
    ties = (frame, frame * 0.55)
    for (cx, cy), a in zip(feet, columns, strict=True):
        if math.sin(a) < 0:
            p.line([(cx, cy), (cx, cy - frame)], shade(GIRDER, 0.6))
    back = _arc(x, y, ring, math.pi, 2 * math.pi, 10)
    for h in ties:
        p.line([(px, py - h) for px, py in back], shade(GIRDER, 0.6))
    _turned(p, x, y, [(0, r + 0.7), (2.4, r + 0.7)], shade(BRICK, 0.8))
    lift = bell / lifts
    for i in range(lifts):
        half = r - 0.8 * i
        foot = y - 2 + 0.6 - lift * i
        profile = [(0, half), (lift, half)]
        cap = None
        if i == lifts - 1:
            profile.append((lift + 1.4, half * 0.72))
            cap = shade(TANK, 1.18)
        _turned(p, x, foot, profile, TANK, cap)
        _ring(p, x, foot - lift + 0.6, half, shade(TANK, 0.7))
    front = _arc(x, y, ring, 0, math.pi, 10)
    for h in ties:
        p.line([(px, py - h) for px, py in front], GIRDER)
    lower, upper = ties[1], ties[0]
    near = [(cx, cy) for (cx, cy), a in zip(feet, columns, strict=True) if math.sin(a) >= 0]
    for (ax, ay), (bx, by) in itertools.pairwise(near):
        p.line([(ax, ay - lower), (bx, by - upper)], shade(GIRDER, 0.8))
        p.line([(ax, ay - upper), (bx, by - lower)], shade(GIRDER, 0.8))
    for cx, cy in near:
        p.line([(cx, cy), (cx, cy - frame)], GIRDER)
        p.point(cx, cy - frame - 1, shade(GIRDER, 1.3))


def _retort_house(
    p: Painter, x: float, y: float, length: float, depth: float, height: float, mouths: int
) -> Top:
    """A brick retort house under a dark roof with a louvred vent along its ridge, a row of tall
    arched windows and the retort mouths glowing low along its front."""
    top = p.hall(x, y, length, depth, height, BRICK, None)
    left, front, right, _ = top
    _plinth(p, top, height, BRICK)
    spans = tuple((k + 0.5) / mouths for k in range(mouths))
    _windows(p, left, front, spans, (1.6,), 3, shade(WINDOW, 1.05), shade(BRICK, 1.35))
    for t in spans:
        _patch(
            p,
            left,
            front,
            (t - 0.25 / mouths, t + 0.25 / mouths),
            (height - 2.2, height - 0.9),
            MOLTEN,
        )
    _door(p, front, right, (0.3, 0.7), 2.5, height, DOORWAY, shade(BRICK, 1.3))
    p.gable(top, 3.5, RETORT_ROOF, BRICK)
    rb, rf = _mid(top[0], top[3]), _mid(top[1], top[2])
    vent = p.hall(*_mid(rb, rf), length * 0.6, 2, 1.2, shade(RETORT_ROOF, 0.6), None)
    p.gable(vent, 1.2, RETORT_ROOF, shade(RETORT_ROOF, 0.6))
    return top


def big_gasworks() -> Sprite:
    """A gasworks with a great gasholder riding high in its iron guide frame and a smaller one
    beside it, a long brick retort house with its retort mouths glowing and a louvred vent on its
    ridge, a tall chimney smoking, a coal wagon on the siding and a heap of coal."""
    p = Painter(54, 56, (27, 44))
    _smoking_chimney(p, -18, -8, -32, shade(BRICK, 0.9), close=True)
    p.shadow(22, 5)
    _gasholder(p, 8, -5, 8.5, 20, 12)
    _gasholder(p, 19, 3.5, 5, 12, 5)
    foot = _footing(-12, 2, 16, 7)
    _bed(p, foot, SOIL)
    _retort_house(p, -12, 2, 16, 7, 7, 5)
    a = _out(_on_wall(foot[0], foot[1], 0.25, 0), 2)
    b = _out(_on_wall(foot[0], foot[1], 1.05, 0), 2)
    _rails(p, a, b)
    x, y = _out(_on_wall(foot[0], foot[1], 0.6, 0), 3.5)
    _wagon(p, x, y, COAL)
    _spill(p, 9, 8, 3, COAL)
    _heap(p, 9, 8, 3.5, 3, COAL)
    return p.sprite()


def mega_gasworks() -> Sprite:
    """A great gasworks: two huge gasholders of telescoping lifts in tall guide frames braced with
    lattice girders and a third before them, a long brick retort house glowing along its front
    under a louvred ridge, two tall chimneys smoking, coal wagons on the siding and heaps of coal
    and coke."""
    p = Painter(64, 74, (31, 54))
    for x, foot, tip in ((-25, -4, -38), (-17, -8, -42)):
        _smoking_chimney(p, x, foot, tip, shade(BRICK, 0.9), close=True)
    p.shadow(28, 6)
    _gasholder(p, -4, -12, 9, 26, 18, 2)
    _gasholder(p, 15, -5, 10, 26, 10, 2)
    _gasholder(p, 23, 10, 6, 14, 7)
    foot = _footing(-12, 6, 20, 8)
    _bed(p, foot, SOIL)
    _retort_house(p, -12, 6, 20, 8, 8, 6)
    a = _out(_on_wall(foot[0], foot[1], 0.12, 0), 1.5)
    b = _out(_on_wall(foot[0], foot[1], 1.2, 0), 1.5)
    _rails(p, a, b)
    for t in (0.25, 0.75):
        x, y = _out(_on_wall(foot[0], foot[1], t, 0), 3)
        _wagon(p, x, y, COAL)
    _spill(p, 8, 14, 3, COAL)
    _heap(p, 8, 14, 3.5, 3, COAL)
    _coke_heap(p, 14, 17, 2.5)
    return p.sprite()


GLOW: RGB = (240, 132, 40)

BRIGHT: RGB = (255, 214, 120)


def big_glassworks() -> Sprite:
    """A big glassworks round two English glass cones of brick, 26 and 22 m high, hooped with
    iron and sooted at their smoking tops, their furnace mouths glowing with gathers of molten
    glass drawn out on blowpipes; a long brick glasshouse joining them, its arched windows
    glowing under a roof glazed in aqua with a glass lantern; a smaller glazed glasshouse
    behind; a long lehr shed with amber windows and an aqua skylight; a packing house with a
    show window of coloured glass; and a yard of big racks of window panes glinting aqua and sea
    green, crates and baskets of bottles in green, amber and cobalt, carboys in wicker, and
    heaps of coloured cullet."""
    m = METRE
    p = Painter(58, 56, (28, 40))

    def at(u: float, v: float) -> Point:
        return _iso(0, 0, (u - 22) * m, (v - 8) * m)

    _glass_yard(
        p, [(at(18, 6), 16), (at(6, -2), 8), (at(34, 2), 10), (at(38, 12), 8), (at(27, -5), 8)]
    )
    for u, v, length, depth in ((22, 8, 16, 9), (40, 8, 14, 5), (37, 17, 10, 6), (40, -2, 8, 5)):
        _bed(p, _footing(*at(u, v), length * m, depth * m), margin=0.8)
    _glass_cone(p, *at(26, 18), 6.5 * m, 22 * m)
    _glass_house(p, *at(37, 17), (10 * m, 6 * m, 6 * m))
    _glass_cone(p, *at(8, 9), 7.5 * m, 26 * m)
    _glass_house(p, *at(22, 8), (16 * m, 9 * m, 7 * m))
    _glass_lehr(p, *at(40, 8), 14 * m, 5 * m)
    _glass_packing(p, *at(40, -2), 8 * m, 5 * m)
    _glass_cullet(p, *at(-2, 2), 3.2 * m, 2.6 * m, (GLASS_AQUA, GLASS_SEA, GLASS_GREEN))
    _glass_cullet(p, *at(0.5, -2.5), 2.4 * m, 1.9 * m, (GLASS_COBALT, GLASS_AMBER, GLASS_AQUA))
    for u, v in ((9.5, -3.6), (15, -4.4), (20.5, -5.2)):
        _glass_panes(p, *at(u, v), 4.5, 2.8)
    _glass_crates(p, *at(26, -2.8), (GLASS_GREEN, GLASS_AMBER, GLASS_COBALT, GLASS_GREEN))
    _glass_crates(p, *at(26.6, -4.4), (GLASS_COBALT, GLASS_GREEN, GLASS_AMBER))
    _glass_carboys(
        p, *at(23.5, -6.2), (GLASS_GREEN, GLASS_AMBER, GLASS_GREEN, GLASS_COBALT, GLASS_AMBER)
    )
    _glass_baskets(p, *at(30, -6), (GLASS_AMBER, GLASS_GREEN, GLASS_COBALT))
    return p.sprite()


def mega_glassworks() -> Sprite:
    """A great glassworks round three English glass cones of brick, the middle one 28 m high,
    each hooped with iron, sooted at its smoking top and glowing at its furnace mouth with gathers
    of molten glass drawn out on blowpipes; a long brick glasshouse joining them under a roof
    glazed in aqua with a glass lantern along its ridge, its arched windows glowing; a smaller
    glazed glasshouse behind; two long lehr sheds with amber windows and aqua skylights; a
    packing house with a show window of coloured glass and a blue glass globe lamp; and a yard
    of big racks of window panes glinting aqua and sea green, crates and baskets of bottles in
    green, amber and cobalt, rows of carboys in wicker, and heaps of coloured cullet."""
    m = METRE
    p = Painter(64, 62, (31, 45))

    def at(u: float, v: float) -> Point:
        return _iso(0, 0, (u - 25) * m, (v - 9) * m)

    _glass_yard(
        p,
        [
            (at(22, 7), 18),
            (at(6, -2), 9),
            (at(42, 3), 11),
            (at(44, 15), 9),
            (at(16, 21), 9),
            (at(30, -5), 9),
        ],
    )
    for u, v, length, depth in (
        (26, 8, 22, 9),
        (45, 6, 16, 5),
        (45, 13, 16, 5),
        (40, -3, 9, 5),
        (12, 21, 10, 6),
    ):
        _bed(p, _footing(*at(u, v), length * m, depth * m), margin=0.8)
    _glass_house(p, *at(12, 21), (10 * m, 6 * m, 6 * m))
    _glass_cone(p, *at(26, 20), 8 * m, 28 * m)
    _glass_cone(p, *at(41, 23.5), 6.5 * m, 22 * m)
    _glass_lehr(p, *at(45, 13), 16 * m, 5 * m)
    _glass_cone(p, *at(7, 9), 7 * m, 24 * m)
    _glass_house(p, *at(26, 8), (22 * m, 9 * m, 7.5 * m))
    _glass_lehr(p, *at(45, 6), 16 * m, 5 * m)
    _glass_packing(p, *at(40, -3), 9 * m, 5 * m)
    _glass_cullet(p, *at(-3, 2), 3.4 * m, 2.8 * m, (GLASS_AQUA, GLASS_SEA, GLASS_GREEN))
    _glass_cullet(p, *at(0, -2.5), 2.6 * m, 2 * m, (GLASS_COBALT, GLASS_AMBER, GLASS_AQUA))
    for u, v in ((9.5, -3.6), (15, -4.4), (20.5, -5.2), (26, -6)):
        _glass_panes(p, *at(u, v), 4.5, 2.8)
    _glass_crates(
        p, *at(29, -2.6), (GLASS_GREEN, GLASS_AMBER, GLASS_COBALT, GLASS_GREEN, GLASS_AMBER)
    )
    _glass_crates(p, *at(29.6, -4.2), (GLASS_COBALT, GLASS_GREEN, GLASS_AMBER, GLASS_COBALT))
    _glass_carboys(
        p,
        *at(29.5, -6.6),
        (GLASS_GREEN, GLASS_AMBER, GLASS_GREEN, GLASS_COBALT, GLASS_AMBER, GLASS_GREEN),
    )
    _glass_baskets(p, *at(36.5, -6.6), (GLASS_AMBER, GLASS_GREEN, GLASS_COBALT))
    return p.sprite()


def _bottle_oven(
    p: Painter,
    x: float,
    y: float,
    s: float,
    colour: RGB,
    glowing: bool = False,
    smoking: bool = True,
    hoop: RGB | None = None,
) -> None:
    """A brick bottle oven `s` times the size of the small one standing on (x, y), its neck sooted,
    bound with iron hoops if `hoop` is given, its firing mouth glowing if `glowing`."""
    profile = [(0, 4.5 * s), (4 * s, 5 * s), (8 * s, 4 * s), (11 * s, 2 * s), (13 * s, 1.8 * s)]
    _turned(p, x, y, profile, colour, (60, 46, 42))
    if hoop is not None:
        for h, half in ((3, 4.9), (6.5, 4.6)):
            _ring(p, x, y - h * s, half * s, hoop)
    else:
        _ring(p, x, y - 4 * s, 5 * s, shade(colour, 0.75))
    p.line([(x - 1.5 * s, y - 12.5 * s), (x + 1.5 * s, y - 12.5 * s)], shade(colour, 0.5))
    mouth = (60, 40, 34)
    p.rect(x - 2.2 * s, y - 3.2 * s, x - 0.4 * s, y + 0.4, mouth)
    if glowing:
        p.blob(x - 2 * s, y - 2.6 * s, x - 0.6 * s, y, GLOW)
        p.point(x - 1.4 * s, y - 1.2 * s, BRIGHT)
    if smoking:
        _smoke(p, x + 0.5, y - 13 * s - 2.5, SMOKE, 3)


def big_tile_works() -> Sprite:
    """A roof-tile and stove-tile works grown large: two long drying sheds open to the wind
    under roofs of the works' own red tiles with racks of raw tiles on their shelves, two brick
    kilns under ribbed vaults glowing at their firing mouths and a square brick stack between
    them, a glazing shop with its doors open on tubs of blue, green and white glaze and racks of
    glazed tiles drying before it, a heap of cream clay by a horse-turned pug mill, hacks and
    pallets of red roof tiles and green stove tiles, and an office of two storeys faced all over
    in blue-and-white tiles under a green glazed roof laid with amber diamonds."""
    m = METRE
    p = Painter(56, 44, (28, 30))

    def at(u: float, v: float) -> Point:
        return _iso(0, 0, (u - 19) * m, (v - 13) * m)

    p.polygon([at(-1, -2), at(39, -2), at(39, 28), at(-1, 28)], WORKS_YARD, shade(WORKS_YARD, 0.85))
    _craft_open_shed(p, *at(10, 24.5), 20 * m, 6 * m, 3.2 * m, 2.2 * m)
    _fire_square_stack(p, *at(35.5, 19.5), 30 * m)
    _craft_kiln_block(p, *at(30, 23.5))
    _craft_open_shed(p, *at(10, 16.5), 20 * m, 6 * m, 3.2 * m, 2.2 * m)
    _craft_kiln_block(p, *at(30, 15.5))
    _craft_pug_mill(p, *at(26.5, 9.5))
    _craft_mound(p, *at(36.5, 9.8), 2.4 * m, 2.0 * m, CREAM_CLAY)
    for v in (11.2, 9.4):
        _craft_hack(p, *at(5.5, v), 9 * m)
    _fire_glazing_shop(p, *at(6, 3.5), 12 * m, 5.5 * m)
    _fire_tile_rack(p, *at(3.5, -0.5), 5 * m, (COBALT, TILE_WHITE, STOVE_GREEN, GLAZE_AMBER))
    _fire_tile_rack(p, *at(9.5, -0.5), 5 * m, (STOVE_GREEN, TILE_WHITE, COBALT, TILE_WHITE))
    office = _fire_tiled_house(p, *at(19.5, 5), 10 * m, 7 * m, 2, False)
    _fire_glazed_roof(p, office, 3.4 * m, STOVE_GREEN, GLAZE_AMBER)
    for u, v, colour, high in (
        (28, 4.2, ROOF_TILE, 1.2),
        (30.5, 4.2, ROOF_TILE, 1.5),
        (33, 4.2, STOVE_GREEN, 1.0),
        (35.5, 4.2, ROOF_TILE, 1.1),
        (28.5, 1.6, STOVE_GREEN, 1.3),
        (31, 1.6, ROOF_TILE, 0.9),
        (33.5, 1.6, ROOF_TILE, 1.4),
        (36, 1.6, STOVE_GREEN, 1.0),
    ):
        _craft_tile_stack(p, *at(u, v), colour, high * m)
    return p.sprite()


def mega_tile_works() -> Sprite:
    """A great tile works: its house on the street a palace of tiles three storeys high, faced all
    over in blue and white with a great tiled picture of a ship in full sail across its front,
    under a green glazed roof laid with amber diamonds and a corner tower capped by a green
    glazed dome and a gilt finial; behind it three long drying sheds open to the wind with racks
    of raw tiles, three vaulted brick kilns glowing at their mouths beside two square brick
    stacks, a glazing shop with tubs of glaze and racks of blue, green, white and amber tiles
    before it, hacks, a heap of cream clay by a horse-turned pug mill, and a stockyard of red
    roof tiles and green stove tiles."""
    m = METRE
    p = Painter(66, 52, (33, 36))

    def at(u: float, v: float) -> Point:
        return _iso(0, 0, (u - 22) * m, (v - 15) * m)

    def up(u: float, v: float, z: float) -> Point:
        x, y = at(u, v)
        return x, y - z * m

    p.polygon([at(-1, -3), at(45, -3), at(45, 32), at(-1, 32)], WORKS_YARD, shade(WORKS_YARD, 0.85))
    for v in (28.5, 21):
        _craft_open_shed(p, *at(10, v), 20 * m, 6 * m, 3.2 * m, 2.2 * m)
    _fire_square_stack(p, *at(41.5, 28.5), 34 * m)
    _craft_kiln_block(p, *at(28, 27.5))
    _craft_kiln_block(p, *at(37, 27.5))
    _fire_square_stack(p, *at(41.5, 19), 30 * m)
    _craft_kiln_block(p, *at(36, 18.5))
    _craft_open_shed(p, *at(10, 13.5), 20 * m, 6 * m, 3.2 * m, 2.2 * m)
    _craft_mound(p, *at(27, 17), 2.8 * m, 2.4 * m, CREAM_CLAY)
    _craft_pug_mill(p, *at(26, 11.5))
    _fire_glazing_shop(p, *at(6, 5), 12 * m, 5.5 * m)
    _fire_tile_rack(p, *at(3.5, 0.8), 5 * m, (COBALT, TILE_WHITE, STOVE_GREEN, GLAZE_AMBER))
    _fire_tile_rack(p, *at(9.5, 0.8), 5 * m, (STOVE_GREEN, TILE_WHITE, COBALT, TILE_WHITE))
    for v in (9.6, 7.8):
        _craft_hack(p, *at(17, v), 7 * m)
    for u, v, colour, high in (
        (36, 10.5, ROOF_TILE, 1.2),
        (38.5, 10.5, ROOF_TILE, 1.5),
        (41, 10.5, STOVE_GREEN, 1.0),
        (36.5, 8, STOVE_GREEN, 1.3),
        (39, 8, ROOF_TILE, 0.9),
        (41.5, 8, ROOF_TILE, 1.4),
        (37, 5.5, ROOF_TILE, 1.1),
        (39.5, 5.5, STOVE_GREEN, 1.2),
        (42, 5.5, ROOF_TILE, 1.3),
    ):
        _craft_tile_stack(p, *at(u, v), colour, high * m)
    house = _fire_tiled_house(p, *at(24, 3.5), 16 * m, 8 * m, 3, True)
    _fire_glazed_roof(p, house, 4 * m, STOVE_GREEN, GLAZE_AMBER)
    tower = p.hall(*at(31, 0.5), 4 * m, 4 * m, 13.6 * m, TILE_WHITE, None)
    _fire_tile_facade(p, tower[0], tower[1], 4 * m, (0, 13.6 * m))
    _fire_tile_facade(p, tower[1], tower[2], 4 * m, (0, 13.6 * m))
    _plinth(p, tower, 13.6 * m, COBALT, 0.6 * m)
    _craft_band(p, tower, 0.35 * m, 0.5 * m, STOVE_GREEN)
    for f in range(3):
        head = f * 3.6 * m + 1.4 * m
        for a, b, glass in ((tower[0], tower[1], WINDOW), (tower[1], tower[2], shade(WINDOW, 0.8))):
            _craft_casement(p, a, b, 4 * m, 0.5, 1.1 * m, (head, head + 1.8 * m), glass, COBALT)
    dx, dy = up(33, 2.5, 13.6)
    dome = [
        (0, 2.4 * m),
        (0.4 * m, 2.4 * m),
        (1.4 * m, 2.2 * m),
        (2.4 * m, 1.6 * m),
        (3.2 * m, 0.8 * m),
        (3.6 * m, 0),
    ]
    _turned(p, dx, dy + 1.0 * m, [(0, 2.5 * m), (1.0 * m, 2.5 * m)], TILE_WHITE)
    _craft_hoop(p, dx, dy + 0.5 * m, 2.5 * m, 0.4 * m, COBALT)
    _turned(p, dx, dy, dome, STOVE_GREEN)
    for h, half in ((1.0 * m, 2.25 * m), (2.0 * m, 1.85 * m)):
        _craft_hoop(p, dx, dy - h, half, 0.15 * m, GLAZE_AMBER)
    _bar(p, (dx, dy - 3.5 * m), (dx, dy - 5.0 * m), 0.18 * m, GILT)
    p.blob(dx - 0.35 * m, dy - 5.3 * m, dx + 0.35 * m, dy - 4.6 * m, GILT)
    for u in (17, 30.5):
        _lamp_post(p, *at(u, -2), 4)
    return p.sprite()


def big_potter() -> Sprite:
    """A pottery works round its yard, a potbank: a long throwing shed washed cream under
    terracotta pantiles with skylights for the throwers, three brick bottle kilns of different
    sizes bound with iron hoops, glowing at their mouths and smoking, bungs of fireclay saggars
    and a heap of coal by them, rows of boards of unfired pots drying in the yard by a heap of
    clay, and a two-storey ware warehouse on the street with blue doors, taking-in doors under a
    hoist, crates of packed ware and a stall of finished ware glazed cobalt, green and cream."""
    m = METRE
    p = Painter(56, 44, (28, 30))

    def at(u: float, v: float) -> Point:
        return _iso(0, 0, (u - 19.5) * m, (v - 11.5) * m)

    p.polygon(
        [at(-1, -1.5), at(40, -1.5), at(40, 25), at(-1, 25)],
        BEATEN_EARTH,
        shade(BEATEN_EARTH, 0.85),
    )
    _fire_pot_range(p, *at(10.5, 19.5), 21 * m, 7 * m, 4.6 * m, 3.2 * m, (0.3, 0.7), True)
    _craft_bottle_kiln(p, *at(27, 21), 1.3)
    _craft_mound(p, *at(37.5, 22), 2.0 * m, 1.5 * m, KILN_COAL)
    _craft_bottle_kiln(p, *at(35, 19.5), 1.0)
    for u, v, tall in ((24.5, 13.8, 5), (25.7, 13.1, 4), (23.6, 12.7, 3)):
        _fire_saggars(p, *at(u, v), tall)
    _craft_bottle_kiln(p, *at(31, 14), 1.15)
    _craft_log_pile(p, *at(37, 12), 3 * m, 1.4 * m)
    _craft_mound(p, *at(5, 12), 2.0 * m, 1.6 * m, CLAY)
    _craft_mound(p, *at(9, 12.8), 1.4 * m, 1.1 * m, shade(CLAY, 0.9))
    for v in (8.6, 6.8, 5.0, 3.2, 1.4):
        _craft_drying_board(p, *at(6.5, v), 10 * m, 15)
    high = 7.6 * m
    store = _fire_pot_range(p, *at(25, 4.5), 18 * m, 7 * m, high, 3.6 * m, (0.2,))
    _fire_hoist(p, store[0], store[1], 0.72, high, 2)
    _fire_crates(p, *at(14.4, -0.4), (3, 2, 1))
    _craft_ware_stall(p, *at(26.5, -0.6))
    for u, v, tall, colour in (
        (31.5, 0.2, 1.3, COBALT),
        (32.6, -0.8, 1.0, CELADON),
        (33.6, 0.3, 1.2, CREAM_GLAZE),
        (35, -0.6, 1.1, COBALT),
    ):
        _craft_pot(p, *at(u, v), tall * m, colour, True)
    return p.sprite()


def mega_potter() -> Sprite:
    """A famous pottery: on the street its showroom of blue ware, 22 by 8 m and 9 m high in cream
    stucco banded with cobalt tiles, six tall round-headed windows in gilt frames shelved with
    cobalt plates and vases, a portico of four white columns under a pediment crowned by a great
    blue-and-white urn with gilt handles, and lamps before it; behind it a potbank of two
    throwing sheds lit by skylights, five brick bottle kilns smoking over the roofs with saggars
    and coal heaped among them, boards of unfired pots drying in the yard by a heap of clay, and a
    ware warehouse with a hoist and crates of packed ware."""
    m = METRE
    p = Painter(66, 50, (33, 33))

    def at(u: float, v: float) -> Point:
        return _iso(0, 0, (u - 24) * m, (v - 14) * m)

    def up(u: float, v: float, z: float) -> Point:
        x, y = at(u, v)
        return x, y - z * m

    p.polygon(
        [
            at(0, 2),
            at(11, 2),
            at(11, -3.5),
            at(37, -3.5),
            at(37, 0),
            at(48.5, 0),
            at(48.5, 29.5),
            at(0, 29.5),
        ],
        BEATEN_EARTH,
        shade(BEATEN_EARTH, 0.85),
    )
    p.polygon([at(11, -3.5), at(37, -3.5), at(37, 0.5), at(11, 0.5)], GRAVEL)
    _fire_pot_range(p, *at(10, 25), 20 * m, 6 * m, 4.6 * m, 3.2 * m, (0.25, 0.75), True)
    _craft_bottle_kiln(p, *at(26, 27), 1.45)
    _craft_bottle_kiln(p, *at(34, 28), 1.2)
    _fire_pot_range(p, *at(5, 15), 8 * m, 10 * m, 4.6 * m, 3.2 * m, (0.5,), True)
    _craft_bottle_kiln(p, *at(42, 25), 1.35)
    _craft_mound(p, *at(47, 21), 2.2 * m, 1.6 * m, KILN_COAL)
    for u, v, tall in ((24.5, 18.5, 5), (25.7, 17.8, 4), (23.6, 17.4, 3), (36, 19, 5)):
        _fire_saggars(p, *at(u, v), tall)
    _craft_bottle_kiln(p, *at(30.5, 20.5), 1.25)
    _craft_bottle_kiln(p, *at(41, 16.5), 1.1)
    _craft_mound(p, *at(15, 16), 2.0 * m, 1.6 * m, CLAY)
    for v in (11.5, 9.7, 7.9, 6.1, 4.3):
        _craft_drying_board(p, *at(5.5, v), 9 * m, 13)
    high = 7.6 * m
    store = _fire_pot_range(p, *at(43, 7), 11 * m, 7 * m, high, 3.6 * m, (0.3,))
    _fire_hoist(p, store[0], store[1], 0.75, high, 2)
    _fire_crates(p, *at(44, 1.5), (3, 2, 1))
    high, length, deep = 9 * m, 22 * m, 8 * m
    top = p.hall(*at(24, 5), length, deep, high, SHOWROOM_WALL, None)
    left, front, right, _ = top
    _plinth(p, top, high, shade(SHOWROOM_WALL, 0.8), 0.8 * m)
    _craft_band(p, top, 0.5 * m, 0.6 * m, COBALT)
    _craft_band(p, top, 0.9 * m, 0.12 * m, GILT)
    _craft_band(p, top, high - 0.95 * m, 0.25 * m, COBALT)
    for k in range(1, 30):
        if k % 2:
            _craft_beam(
                p, left, front, (k / 30, 0.25 * m), (k / 30, 0.75 * m), 0.28 * m, CREAM_GLAZE
            )
    for t in (0.1, 0.27, 0.73, 0.9):
        _fire_blue_window(p, left, front, length, t, (1.8 * m, high - 1.0 * m))
    for t in (0.25, 0.75):
        _fire_blue_window(p, front, right, deep, t, (1.8 * m, high - 1.0 * m))
    _patch(p, left, front, (0.4, 0.6), (1.4 * m, high), shade(SHOWROOM_WALL, 0.8))
    _patch(p, left, front, (0.46, 0.54), (4.2 * m, high), DOOR)
    p.polygon(_round_head(left, front, length, 0.5, 1.8 * m, 3.9 * m, high), GILT)
    p.polygon(_round_head(left, front, length, 0.5, 1.5 * m, 4.1 * m, high), (70, 52, 40))
    _patch(p, left, front, (0.48, 0.52), (4.6 * m, 6.6 * m), SHOP_GLOW)
    _hip(p, top, 3.2 * m, 0.25, SHOWROOM_ROOF, (0.35, 0.7))
    step = shade(SHOWROOM_WALL, 0.9)
    p.hall(*at(24, -1.6), 8.4 * m, 3 * m, 0.6 * m, step, shade(step, 1.06))
    for v in (-0.6, -2.6):
        for u in (20.6, 22.9, 25.1, 27.4) if v < -1 else (20.6, 27.4):
            _column(p, *up(u, v, 0.6), 6.6 * m)
    ent = p.hall(*up(24, -1.6, 7.2), 8.4 * m, 3 * m, 1.3 * m, SHOWROOM_WALL, None)
    e_left, e_front, _, _ = ent
    _craft_band(p, ent, 0.45 * m, 0.35 * m, COBALT)
    _gable_across(p, ent, 2.4 * m, SHOWROOM_ROOF, SHOWROOM_WALL)
    ox, oy = _on_wall(e_left, e_front, 0.5, -0.8 * m)
    p.ellipse(ox - 0.5 * m, oy - 0.5 * m, ox + 0.5 * m, oy + 0.5 * m, COBALT)
    _fire_urn(p, *_on_wall(e_left, e_front, 0.5, -2.3 * m), 4.6 * m)
    for u in (15.5, 32.5):
        _lamp_post(p, *at(u, -2.6), 4)
    for u, v, tall, colour in ((13, -1.6, 1.3, COBALT), (34.8, -1.6, 1.3, COBALT)):
        _craft_pot(p, *at(u, v), tall * m, colour, True)
    return p.sprite()


OVEN: RGB = (172, 116, 86)

HOOP: RGB = (70, 60, 58)

CREAM: RGB = (230, 218, 192)

CHINA_ROOF: RGB = (122, 88, 70)

CHINA: RGB = (246, 246, 250)

BLUE: RGB = (54, 86, 176)

FRAME: RGB = (120, 88, 58)

SHARD: RGB = (220, 220, 226)

SAGGAR: RGB = (198, 168, 132)


def _ware_racks(p: Painter, x: float, y: float, length: float, rows: int = 2) -> None:
    """Open racks running the lit way from (x, y), their shelves set with blue-and-white ware."""
    a, b = (x, y), (x + length, y + length / 2)
    for t in (0, 1):
        px, py = _on_wall(a, b, t, 0)
        p.line([(px, py), (px, py - 3 * rows - 1)], FRAME)
    for r in range(rows):
        d = -1 - 3 * r
        _band(p, a, b, d, FRAME)
        items = max(3, round(length / 2))
        for k in range(items):
            px, py = _on_wall(a, b, (k + 0.5) / items, d - 0.4)
            p.line([(px, py), (px, py - 1)], CHINA if k % 2 else BLUE)


def _saggars(p: Painter, x: float, y: float, tall: int) -> None:
    """A column of round fireclay saggars stacked `tall` high on (x, y)."""
    for k in range(tall):
        cy = y - 1.3 * k
        p.ellipse(x - 1.6, cy - 1.6, x + 1.6, cy, SAGGAR)


def _shards(p: Painter, x: float, y: float, half: float) -> None:
    """A heap of broken china, flecked blue."""
    _spill(p, x, y, half, SHARD)
    _heap(p, x, y, half, half * 0.75, SHARD)
    for dx, dy in ((-1, -0.6), (1, 0.4), (-half, 0.6)):
        p.point(x + dx, y + dy, BLUE)


def _vase(p: Painter, x: float, y: float, s: float = 1) -> None:
    """A tall blue-and-white vase standing on (x, y)."""
    _turned(
        p,
        x,
        y,
        [(0, 0.9 * s), (2 * s, 1.8 * s), (4 * s, 1.2 * s), (5 * s, 0.7 * s), (5.6 * s, 1 * s)],
        CHINA,
        shade(BLUE, 0.7),
    )
    _ring(p, x, y - 2 * s, 1.8 * s, BLUE)
    _ring(p, x, y - 3.2 * s, 1.5 * s, BLUE)


def _china_crates(p: Painter, x: float, y: float, count: int) -> None:
    for k in range(count):
        cx, cy = x + 3 * k, y - 1.5 * k
        _crate(p, cx, cy)
        p.point(cx - 1, cy - 2.5, (226, 196, 120))


def _porcelain_range(
    p: Painter, x: float, y: float, length: float, depth: float, height: float, storeys: int
) -> Top:
    """A cream potbank workshop range under a brown roof with a blue name board under its eaves
    and rows of windows."""
    top = p.hall(x, y, length, depth, height, CREAM, None)
    left, front, right, _ = top
    _plinth(p, top, height, CREAM)
    _stains(p, front, right, height, (0.15, 0.85), shade(CREAM, 0.6), 1.2)
    spans = tuple((k + 0.5) / 7 for k in range(7))
    for k in range(storeys):
        _windows(p, left, front, spans, (2.6 + 3.4 * k,), 2, WINDOW, shade(CREAM, 1.1))
    _patch(p, left, front, (0.15, 0.85), (0.6, 1.8), BLUE)
    for t in (0.3, 0.5, 0.7):
        p.point(*_on_wall(left, front, t, 1), CHINA)
    _window(p, front, right, (0.3, 0.7), 2.6, shade(WINDOW, 0.85))
    p.gable(top, 4, CHINA_ROOF, CREAM)
    _roof_rows(p, top, 4, CHINA_ROOF)
    return top


def big_porcelain_works() -> Sprite:
    """A cluster of three brick bottle ovens bound with iron hoops, one glowing at its firing mouth,
    behind a two-storey cream workshop with a blue name board, blue-and-white ware set out on
    racks, stacks of saggars, a heap of broken shards and crates packed for shipping."""
    p = Painter(50, 54, (24, 42))
    for x, y, s, glow in ((-12, -6, 1.1, False), (-3, -11, 1.3, False), (6, -9, 1.05, False)):
        p.blob(x - 6 * s, y - 3 * s, x + 6 * s, y + 3 * s, SOIL)
        _bottle_oven(p, x, y, s, OVEN, glow, True, HOOP)
    p.shadow(20, 5)
    foot = _footing(9, 1, 14, 7)
    _bed(p, foot, SOIL, 1)
    _track(p, foot[1], foot[2], (0.35, 0.7), 3)
    _porcelain_range(p, 9, 1, 14, 7, 8, 2)
    _bottle_oven(p, -16, 2, 0.9, OVEN, True, False, HOOP)
    _ware_racks(p, -12, 4, 9)
    for x, y in ((-20, 7), (-18, 8.5)):
        _saggars(p, x, y, 3)
    _shards(p, 2, 9, 2)
    _china_crates(p, 14, 10, 2)
    return p.sprite()


def mega_porcelain_works() -> Sprite:
    """A great potbank: six brick bottle ovens bound with iron hoops clustered round a long cream
    workshop range, two glowing at their mouths and the rest smoking, an arched gateway under a
    pediment crowned with a great blue-and-white vase, racks of blue-and-white ware, columns of
    stacked saggars, heaps of shards and crates packed for shipping."""
    p = Painter(64, 66, (32, 50))
    for x, y, s in ((-14, -13, 1.25), (-1, -17, 1.5), (12, -14, 1.3), (22, -6, 1.05)):
        p.blob(x - 6 * s, y - 3 * s, x + 6 * s, y + 3 * s, SOIL)
        _bottle_oven(p, x, y, s, OVEN, x == 12, True, HOOP)
    p.shadow(27, 6)
    foot = _footing(1, 2, 24, 8)
    _bed(p, foot, SOIL, 1)
    top = _porcelain_range(p, 1, 2, 24, 8, 9, 2)
    left, front, _, _ = top
    _patch(p, left, front, (0.44, 0.56), (4.2, 9), DOORWAY)
    _patch(p, left, front, (0.42, 0.58), (3, 4.2), shade(CREAM, 1.08))
    a, b = _on_wall(left, front, 0.4, 0), _on_wall(left, front, 0.6, 0)
    peak = _on_wall(left, front, 0.5, -4)
    p.polygon([a, b, peak], CREAM, shade(CREAM, 0.6))
    p.point(*_on_wall(left, front, 0.5, -1.5), BLUE)
    _vase(p, peak[0], peak[1] + 0.5, 0.9)
    _bottle_oven(p, -22, 0, 1.1, OVEN, True, True, HOOP)
    _ware_racks(p, -24, 6, 10)
    _ware_racks(p, -14, 10.5, 8, 1)
    for x, y, tall in ((-27, 8, 4), (-25, 9.5, 3), (-28.5, 10, 2)):
        _saggars(p, x, y, tall)
    _shards(p, 6, 12, 2.2)
    _china_crates(p, 16, 12, 3)
    return p.sprite()


def big_brickworks() -> Sprite:
    """A brickworks grown large round a Hoffmann ring kiln 30 m long: its sooty oval pierced by
    arched wickets, one glowing, its feeding floor under a pantiled roof on posts and its tall
    chimney smoking from the middle; beside it a second kiln, a round downdraught kiln banded in
    iron under its dome, smoking; a clay pit dug in benches with a puddle on its floor and tipping
    tubs on a narrow-gauge tramway at its rim; two long open sheds over hacks of cream green
    bricks; pallets of fired red bricks; and a railway siding where open wagons stand loaded
    with bricks."""
    m = METRE
    p = Painter(60, 52, (30, 36))

    def at(u: float, v: float) -> Point:
        return _iso(0, 0, (u - 20) * m, (v - 12) * m)

    p.polygon([at(-1, -3), at(41, -3), at(41, 26), at(-1, 26)], BRICK_DUST, shade(BRICK_DUST, 0.85))
    _fire_clay_pit(p, *at(6, 19.5), 12, 12)
    _fire_ring_kiln(p, *at(27, 19.5), 28, 44 * m)
    _fire_narrow_gauge(p, at(1, 12.2), at(16, 12.2))
    _brick_tub(p, *at(9.5, 12.2))
    _brick_tub(p, *at(11.6, 12.2))
    _fire_dome_kiln(p, *at(35, 7))
    for v in (9, 4.5):
        _fire_hack_shed(p, *at(9.5, v), 18)
    pallets = [
        (u, v, BRICK_FIRED if (u + v) % 3 else BRICK_ORANGE) for u in (21, 23, 25) for v in (3, 5.5)
    ]
    for u, v, colour in sorted(pallets, key=lambda e: e[0] - e[1]):
        _brick_pallet(p, *at(u, v), colour)
    _fire_siding(p, at(-1, -1), at(41, -1))
    for u in (24, 31, 38):
        _fire_brick_wagon(p, *at(u, -1))
    return p.sprite()


def mega_brickworks() -> Sprite:
    """A great brickworks of two Hoffmann ring kilns side by side, each 30 m long, their sooty
    ovals pierced by arched wickets glowing here and there, their feeding floors under pantiled
    roofs on posts and their two tall chimneys smoking; a deep clay pit dug in benches where a
    steam navvy bites into the clay face and tipping tubs wait on a narrow-gauge tramway; long
    open sheds over hacks of cream green bricks; pallets of fired red and orange bricks; and a
    railway siding along the front where a train of open wagons is loaded with bricks."""
    m = METRE
    p = Painter(66, 58, (33, 40))

    def at(u: float, v: float) -> Point:
        return _iso(0, 0, (u - 23.5) * m, (v - 14.5) * m)

    p.polygon(
        [at(0, -3), at(47, -3), at(47, 30.5), at(0, 30.5)], BRICK_DUST, shade(BRICK_DUST, 0.85)
    )
    _fire_clay_pit(p, *at(7.5, 21.5), 14, 17)
    _fire_ring_kiln(p, *at(30.5, 24.5), 30, 46 * m)
    _fire_navvy(p, *at(10, 19.5))
    _fire_narrow_gauge(p, at(1, 12.5), at(18, 12.5))
    for u in (8, 10.1, 12.2):
        _brick_tub(p, *at(u, 12.5))
    _fire_ring_kiln(p, *at(30.5, 12), 30, 46 * m)
    for v in (8.5, 4):
        _fire_hack_shed(p, *at(9, v), 17)
    pallets = [
        (u, v, BRICK_FIRED if (u + v) % 3 else BRICK_ORANGE)
        for u in (22, 24, 26, 28)
        for v in (2.5, 5)
    ]
    for u, v, colour in sorted(pallets, key=lambda e: e[0] - e[1]):
        _brick_pallet(p, *at(u, v), colour)
    for v in (2.5, 5):
        _brick_hack(p, *at(39, v), 12, 1.4, 1.3)
    _fire_siding(p, at(0, -1.2), at(47, -1.2))
    for u in (20, 27, 34, 41):
        _fire_brick_wagon(p, *at(u, -1.2))
    return p.sprite()


CEMENT: RGB = (190, 188, 182)

CEMENT_DUST: RGB = (214, 212, 206)

CEMENT_GROUND: RGB = (176, 174, 164)

TUBE: RGB = (134, 128, 120)

LIMESTONE: RGB = (222, 216, 198)


def _cement_silo(p: Painter, x: float, y: float, tall: float) -> None:
    """A tall round grey silo banded at its foot and twice up its height."""
    _turned(p, x, y, [(0, 2.5), (tall, 2.5)], CEMENT, shade(CEMENT, 1.1))
    for h in (tall * 0.3, tall * 0.7):
        _ring(p, x, y - h, 2.5, shade(CEMENT, 0.75))
    _ring(p, x, y - 0.8, 2.5, shade(CEMENT, 0.62))


def _preheater(p: Painter, x: float, y: float, half: float, tall: float) -> Top:
    """A windowed concrete preheater tower in storeys, a door at its foot and a stack on top
    throwing a plume of dust."""
    grey = shade(CEMENT, 0.9)
    tower = p.box(x, y, half, tall, grey, shade(CEMENT, 0.75))
    left, front, right, _ = tower
    storeys = round(tall / 5)
    for k in range(1, storeys + 1):
        drop = tall * k / (storeys + 0.4)
        _band(p, left, front, drop, shade(CEMENT, 0.6))
        p.point(*_on_wall(left, front, 0.5, drop - 2.5), WINDOW)
        p.point(*_on_wall(front, right, 0.5, drop - 2.5), shade(WINDOW, 0.8))
    _plinth(p, tower, tall, grey)
    _patch(p, front, right, (0.3, 0.7), (tall - 2.5, tall), DOORWAY)
    cx, cy = _mid(left, right)
    p.rect(cx - 1, cy - 4, cx + 1, cy, shade(CEMENT, 0.7))
    _smoke(p, cx, cy - 7, CEMENT_DUST, 3)
    return tower


def _rotary_kiln(p: Painter, a: Point, b: Point, lift: float) -> None:
    """A long rotary kiln tube on piers sloping from its firing end at `a` up to `b`, `lift` off
    the ground, banded with its tyres and glowing at its firing end."""
    for t in (0.12, 0.45, 0.78):
        px, py = _on_wall(a, b, t, 0)
        p.blob(px - 2, py - 0.5, px + 2, py + 1.5, shade(CEMENT_GROUND, 0.8))
        p.rect(px - 1, py - lift + 0.5, px + 1, py + 1, shade(CEMENT, 0.8))
    p.polygon(
        [
            (a[0], a[1] - lift - 3),
            (b[0], b[1] - lift - 5),
            (b[0], b[1] - lift),
            (a[0], a[1] - lift),
        ],
        TUBE,
        shade(TUBE, 0.55),
    )
    p.line([(a[0], a[1] - lift - 3), (b[0], b[1] - lift - 5)], shade(TUBE, 1.3))
    for t in (0.12, 0.45, 0.78):
        px, py = _on_wall(a, b, t, -lift)
        p.line([(px, py - 3 - 2 * t), (px, py)], shade(TUBE, 0.6))
    p.ellipse(a[0] - 2, a[1] - lift - 3.5, a[0] + 1, a[1] - lift + 0.5, shade(TUBE, 0.8))
    p.blob(a[0] - 1.5, a[1] - lift - 2.5, a[0] + 0.5, a[1] - lift - 0.5, GLOW)
    p.point(a[0] - 1, a[1] - lift - 1.8, BRIGHT)


def _clinker_store(p: Painter, x: float, y: float, length: float, depth: float) -> None:
    """A long grey clinker store under a ribbed barrel-vault roof, its end open."""
    top = p.hall(x, y, length, depth, 4, shade(CEMENT, 0.92), None)
    left, front, right, _ = top
    _plinth(p, top, 4, shade(CEMENT, 0.92))
    _windows(p, left, front, (0.2, 0.4, 0.6, 0.8), (1.2,), 1, shade(WINDOW, 1.1))
    _vault(p, top, 3, shade(CEMENT, 0.8), shade(CEMENT, 0.92))
    _patch(p, front, right, (0.25, 0.75), (1.5, 4), DOORWAY)


def _dust(p: Painter, spots: tuple[tuple[float, float, float], ...]) -> None:
    for x, y, r in spots:
        p.blob(x - r, y - r, x + r, y + r, CEMENT_DUST)


def big_cement_works() -> Sprite:
    """A cement works on ground whitened by its dust: four banded grey silos joined by a conveyor
    to a tall windowed preheater tower throwing a plume of dust, a long rotary kiln tube on piers
    glowing at its firing end, a barrel-vaulted clinker store, a heap of limestone and bags of
    cement."""
    p = Painter(54, 67, (27, 52))
    p.polygon([(-25, 2), (-8, -14), (22, -12), (25, 4), (4, 13), (-18, 11)], CEMENT_GROUND)
    p.shadow(22, 6)
    _track(p, (4, 8), (16, 2), (0.25, 0.7), 4, shade(CEMENT_GROUND, 0.84))
    _preheater(p, 13, -8, 4, 28)
    for k in range(4):
        _cement_silo(p, -4 - 4 * k, -10 + 2 * k, 19)
    p.line([(-16, -29), (9, -32)], shade(CEMENT, 0.55))
    p.line([(-16, -28), (9, -31)], shade(CEMENT, 0.8))
    _rotary_kiln(p, (-16, 8), (8, -4), 4)
    _clinker_store(p, 15, 5, 10, 5)
    _heap(p, -19, -2, 3, 2.5, LIMESTONE)
    _dust(p, ((-18, 7, 2), (-14, 10, 1.5), (-20, 4, 1.5)))
    for x, y in ((2, 10), (5, 9), (3.5, 7.8)):
        _sack(p, x, y, (200, 198, 190))
    return p.sprite()


def mega_cement_works() -> Sprite:
    """A great cement works on ground whitened by its dust: a row of six banded grey silos, two
    tall windowed preheater towers throwing dust, a tall concrete stack, two long rotary kilns on
    piers glowing at their firing ends, a barrel-vaulted clinker store, heaps of limestone and a
    train of cement wagons on the siding."""
    p = Painter(64, 80, (31, 60))
    p.polygon([(-30, 2), (-10, -20), (26, -16), (31, 6), (8, 18.5), (-22, 15)], CEMENT_GROUND)
    p.shadow(27, 7)
    _smoking_chimney(p, 25, -8, -46, shade(CEMENT, 0.85), smoke=CEMENT_DUST, close=True)
    _preheater(p, 18, -10, 4, 27)
    _preheater(p, 10, -13, 4, 31)
    for k in range(6):
        _cement_silo(p, -2 - 4 * k, -16 + 2 * k, 22)
    p.line([(-22, -35), (6, -41)], shade(CEMENT, 0.55))
    p.line([(-22, -34), (6, -40)], shade(CEMENT, 0.8))
    _rotary_kiln(p, (-22, 2), (4, -11), 4)
    _rotary_kiln(p, (-10, 9), (16, -4), 4)
    _clinker_store(p, 19, 7, 11, 5)
    _heap(p, -26, -1, 3.5, 3, LIMESTONE)
    _dust(p, ((-22, 5, 2), (-14, 11, 2), (-18, 8, 1.5)))
    a, b = (-10, 8.5), (6, 16.5)
    _rails(p, a, b)
    for t in (0.3, 0.7):
        x, y = _out(_on_wall(a, b, t, 0), 1.5)
        _wagon(p, x, y, CEMENT)
    return p.sprite()


GANTRY: RGB = (84, 92, 102)

CRANE_BED: RGB = (104, 96, 86)


def _fire_casting_hall(
    p: Painter,
    x: float,
    y: float,
    long: float,
    deep: float,
    high: float,
    doors: tuple[float, ...] = (),
) -> Top:
    """A foundry casting hall `long` along the lit walls and `deep`, `high` to its eaves, of
    soot-stained red brick on (x, y): pilasters between tall arched windows glowing dimly from
    within, a cream string course, a slate roof with a louvred monitor along its ridge, and at
    each of `doors` along its lit wall a great open arched door through which a ladle pours into
    the moulds, its light spilling out over the ground. Returns its top."""
    m = METRE
    top = p.hall(x, y, long, deep, high, SOOTED_BRICK, None)
    left, front, right, _ = top
    _plinth(p, top, high, SOOTED_BRICK, 0.6 * m)
    soot = shade(SOOTED_BRICK, 0.58)
    for a, b, tone in ((left, front, 1.0), (front, right, 0.78)):
        _patch(p, a, b, (0, 1), (0, 1.2 * m), shade(soot, tone))
        _patch(p, a, b, (0, 1), (1.2 * m, 1.8 * m), shade(soot, tone * 1.25))
        _patch(p, a, b, (0, 1), (2.6 * m, 2.9 * m), shade(CREAM_GLAZE, tone * 0.85))
    bays = max(3, round(long / (3 * m)))
    for k in range(bays + 1):
        _craft_beam(
            p,
            left,
            front,
            (k / bays, 1.2 * m),
            (k / bays, high),
            0.45 * m,
            shade(SOOTED_BRICK, 0.85),
        )
    for k in range(bays):
        t = (k + 0.5) / bays
        if any(abs(t - d) < 3.4 * m / long for d in doors):
            continue
        _mill_window(p, left, front, long, t, 1.4 * m, (3.4 * m, high - 1.2 * m), FURNACE_GLOW)
        if k % 3 == 1:
            _craft_beam(p, left, front, (t, 1.8 * m), (t, 3.2 * m), 0.25 * m, shade(soot, 1.3))
    sides = max(1, round(deep / (3.2 * m)))
    for k in range(sides):
        _mill_window(
            p,
            front,
            right,
            deep,
            (k + 0.5) / sides,
            1.4 * m,
            (3.4 * m, high - 1.2 * m),
            shade(FURNACE_GLOW, 0.7),
        )
    for t in doors:
        _patch(
            p,
            left,
            front,
            (t - 3.1 * m / long, t + 3.1 * m / long),
            (1.2 * m, 1.6 * m),
            shade(soot, 0.9),
        )
        p.polygon(_round_head(left, front, long, t, 6.0 * m, high - 5.7 * m, high), DRESSING)
        p.polygon(_round_head(left, front, long, t, 5.0 * m, high - 5.1 * m, high), (40, 32, 30))
        _craft_pour(p, left, front, long, t, high)
    p.gable(top, 3.4 * m, SLATE, SOOTED_BRICK)
    _tiles(p, top, 3.4 * m, SLATE, (0.3, 0.6))
    _craft_monitor(p, top, 3.4 * m, shade(SOOTED_BRICK, 0.8), SLATE)
    for t in doors:
        gx, gy = _on_wall(left, front, t, high)
        p.soft(
            [(gx - 3.8 * m, gy - 0.4 * m, gx + 2.2 * m, gy + 2.6 * m)], (255, 166, 76), 0.42, 0.8
        )
    return top


def _fire_crane_rail(p: Painter, a: Point, b: Point) -> None:
    """The track of a travelling crane on the ground from `a` to `b`: a rail on a bed of
    sleepers and cinders."""
    m = METRE
    _bar(p, a, b, 0.9 * m, CRANE_BED)
    for k in range(1, 24):
        sx, sy = _on_wall(a, b, k / 24, 0)
        p.blob(sx - 0.45 * m, sy - 0.12 * m, sx + 0.45 * m, sy + 0.12 * m, shade(TIMBER, 0.6))
    _bar(p, a, b, 0.16 * m, (150, 152, 160))


def _fire_gantry_leg(p: Painter, x: float, y: float, high: float) -> None:
    """An A-framed leg of a travelling crane standing on its rail at (x, y), `high` to the bridge:
    two iron legs splayed along the rail and braced, on a wheeled bogie."""
    m = METRE
    a, b = _iso(x, y - 0.6 * m, -1.8 * m, 0), _iso(x, y - 0.6 * m, 1.8 * m, 0)
    head = (x, y - high)
    p.hall(x, y, 4.4 * m, 0.9 * m, 0.6 * m, (54, 56, 62), (70, 72, 78))
    for wx, wy in (_iso(x, y, -1.6 * m, -0.45 * m), _iso(x, y, 1.6 * m, -0.45 * m)):
        p.ellipse(wx - 0.35 * m, wy - 0.55 * m, wx + 0.35 * m, wy + 0.05 * m, (40, 40, 44))
    _bar(p, a, head, 0.4 * m, GANTRY)
    _bar(p, b, head, 0.4 * m, shade(GANTRY, 0.78))
    for f in (0.35, 0.65):
        _bar(p, _on_wall(a, head, f, 0), _on_wall(b, head, f, 0), 0.18 * m, shade(GANTRY, 0.9))
    _bar(p, _on_wall(a, head, 0.35, 0), _on_wall(b, head, 0.65, 0), 0.12 * m, shade(GANTRY, 0.9))


def _fire_gantry(p: Painter, near: Point, far: Point, high: float, crab: float) -> Point:
    """The far leg, the bridge and the crab of a travelling crane whose legs stand on the ground
    at `far` and `near`, `high` to its bridge, a lattice girder with the crab `crab` of the way
    along it from the far leg; returns where its hook hangs. Its near leg is drawn last, with
    _fire_gantry_leg, over what hangs from the hook."""
    m = METRE
    _fire_gantry_leg(p, *far, high)
    t_far, t_near = (far[0], far[1] - high), (near[0], near[1] - high)
    deep = 1.2 * m
    p.polygon(
        [t_far, t_near, (t_near[0], t_near[1] + deep), (t_far[0], t_far[1] + deep)],
        shade(GANTRY, 0.8),
    )
    for k in range(12):
        f0, f1 = k / 12, (k + 1) / 12
        a = _on_wall(t_far, t_near, f0, deep if k % 2 else 0)
        b = _on_wall(t_far, t_near, f1, 0 if k % 2 else deep)
        _bar(p, a, b, 0.12 * m, shade(GANTRY, 0.55))
    _bar(p, t_far, t_near, 0.25 * m, shade(GANTRY, 1.2))
    _bar(p, (t_far[0], t_far[1] + deep), (t_near[0], t_near[1] + deep), 0.2 * m, shade(GANTRY, 0.6))
    cx, cy = _on_wall(t_far, t_near, crab, 0)
    p.hall(cx, cy, 1.6 * m, 2.2 * m, 1.0 * m, (196, 150, 52), shade((196, 150, 52), 1.15))
    hook = (cx, cy + 3.0 * m)
    _bar(p, (cx - 0.2 * m, cy + 0.3 * m), hook, 0.08 * m, (36, 34, 34))
    _bar(p, (cx + 0.2 * m, cy + 0.3 * m), hook, 0.08 * m, (36, 34, 34))
    p.blob(
        hook[0] - 0.25 * m, hook[1] - 0.2 * m, hook[0] + 0.25 * m, hook[1] + 0.3 * m, (40, 38, 40)
    )
    return hook


def _fire_hung_bell(p: Painter, x: float, y: float, tall: float) -> None:
    """A newly cast bell `tall` high hung by its crown from a hook at (x, y), grey iron lit on
    its left, with a sling round its crown."""
    t = tall
    _bar(p, (x, y), (x, y + 0.15 * t), 0.06 * t, (36, 34, 34))
    top = y + 0.15 * t
    outline = [
        (x - 0.2 * t, top),
        (x + 0.2 * t, top),
        (x + 0.36 * t, top + 0.25 * t),
        (x + 0.4 * t, top + 0.7 * t),
        (x + 0.6 * t, top + t),
        (x - 0.6 * t, top + t),
        (x - 0.4 * t, top + 0.7 * t),
        (x - 0.36 * t, top + 0.25 * t),
    ]
    p.polygon(outline, shade(CAST_IRON, 0.85))
    p.polygon([outline[0], (x, top), (x, top + t), outline[5], outline[6], outline[7]], CAST_IRON)
    _bar(
        p,
        (x - 0.6 * t, top + 0.92 * t),
        (x + 0.6 * t, top + 0.92 * t),
        0.08 * t,
        shade(CAST_IRON, 1.3),
    )
    _bar(
        p,
        (x - 0.38 * t, top + 0.5 * t),
        (x + 0.38 * t, top + 0.5 * t),
        0.04 * t,
        shade(CAST_IRON, 0.7),
    )


def _fire_great_ladle(p: Painter, x: float, y: float, size: float) -> Point:
    """A great ladle hung from a crane hook at (x, y) by its bail, `size` across, tipped to pour:
    a riveted iron bucket brimming with molten iron, white hot at its lip. Returns its lip."""
    m = METRE * size
    cx, cy = x, y + 2.4 * m
    c, s = math.cos(-0.45), math.sin(-0.45)

    def tipped(u: float, v: float) -> Point:
        return cx + (u * c - v * s) * m, cy + (u * s + v * c) * m

    for side in (-0.95, 0.95):
        _bar(p, (x, y), tipped(side, -0.4), 0.12 * m, (36, 34, 34))
    p.polygon(
        [tipped(-0.9, -0.75), tipped(0.9, -0.75), tipped(0.65, 0.8), tipped(-0.65, 0.8)],
        (54, 50, 52),
    )
    p.polygon(
        [tipped(0, -0.75), tipped(0.9, -0.75), tipped(0.65, 0.8), tipped(0, 0.8)], (42, 40, 42)
    )
    for v in (-0.2, 0.35):
        _bar(p, tipped(-0.82, v), tipped(0.78, v), 0.08 * m, (80, 76, 78))
    p.polygon(
        [tipped(-0.82, -0.72), tipped(0.82, -0.72), tipped(0.78, -0.48), tipped(-0.78, -0.48)],
        MOLTEN,
    )
    p.polygon(
        [tipped(-0.82, -0.72), tipped(-0.2, -0.72), tipped(-0.22, -0.55), tipped(-0.78, -0.55)],
        (255, 238, 176),
    )
    return tipped(-0.95, -0.75)


def _fire_casting_pit(p: Painter, x: float, y: float, lip: Point) -> None:
    """A casting pit dug into the yard at (x, y), its sand mould filling from a stream of
    white-hot iron falling from `lip`, glowing and throwing sparks, its light on the ground."""
    m = METRE
    p.soft([(x - 4.5 * m, y - 2.2 * m, x + 4.5 * m, y + 2.2 * m)], (255, 160, 70), 0.45, 0.8)
    pit = [_iso(x, y, du * 2 * m, dv * 1.6 * m) for du, dv in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    p.polygon(pit, (58, 50, 46))
    p.polygon(
        [pit[0], pit[3], _iso(*pit[3], 0.4 * m, -0.3 * m), _iso(*pit[0], 0.4 * m, 0.3 * m)],
        (40, 34, 32),
    )
    inner = [
        _iso(x, y, du * 1.3 * m, dv * 1.0 * m) for du, dv in ((-1, -1), (1, -1), (1, 1), (-1, 1))
    ]
    p.polygon(inner, MOULDING_SAND)
    p.blob(x - 0.7 * m, y - 0.35 * m, x + 0.7 * m, y + 0.35 * m, MOLTEN)
    p.blob(x - 0.35 * m, y - 0.18 * m, x + 0.35 * m, y + 0.18 * m, (255, 238, 176))
    _bar(p, lip, (lip[0] - 0.1 * m, lip[1] + 0.4 * m), 0.35 * m, (255, 238, 176))
    _bar(p, (lip[0] - 0.1 * m, lip[1] + 0.4 * m), (x, y), 0.2 * m, (255, 238, 176))
    for dx, dy in ((-1.2, -0.8), (1.0, -1.2), (-0.6, -1.6), (1.6, -0.5), (0.3, -2.0), (-1.8, -0.3)):
        q = 0.12 * m
        p.blob(x + dx * m - q, y + dy * m - q, x + dx * m + q, y + dy * m + q, (255, 210, 110))


GLAZE_AMBER: RGB = (226, 170, 56)

GLAZE_TUB: RGB = (132, 96, 62)

STACK_BRICK: RGB = (156, 80, 58)


def _fire_square_stack(
    p: Painter, x: float, y: float, tall: float, colour: RGB = STACK_BRICK
) -> None:
    """A square brick kiln stack drawn for `tall` units and lowered (CHIMNEY_SHARE), standing on
    (x, y): a plinth with a cream string course, a shaft tapering to a corbelled cap with a cream
    band under it, and a wisp of smoke."""
    m = METRE
    tall *= CHIMNEY_SHARE
    base = p.hall(x, y, 3 * m, 3 * m, 2.5 * m, colour, shade(colour, 1.1))
    _craft_band(p, base, 0.3 * m, 0.3 * m, CREAM_GLAZE)
    foot, head = y - 2.5 * m, y - tall
    w0, w1 = 1.2 * m, 0.8 * m

    def face(sign: int, tone: float) -> None:
        p.polygon(
            [
                (x, foot + w0 / 2),
                (x + sign * w0, foot),
                (x + sign * w1, head),
                (x, head + w1 / 2),
            ],
            shade(colour, tone),
        )

    face(-1, 1.0)
    face(1, 0.76)
    cap = p.hall(x, head + 0.5 * m, 2.4 * m, 2.4 * m, 0.9 * m, shade(colour, 0.8), (40, 34, 34))
    _craft_band(p, cap, 0.45 * m, 0.25 * m, CREAM_GLAZE)
    for k in range(1, 4):
        hy = foot + (head - foot) * k / 4
        hw = w0 + (w1 - w0) * k / 4
        _bar(p, (x - hw, hy), (x, hy + hw / 2), 0.15 * m, shade(colour, 0.6))
        _bar(p, (x, hy + hw / 2), (x + hw, hy), 0.15 * m, shade(colour, 0.5))
    _craft_smoke(p, x + 0.8 * m, head - 2.4 * m, 2.4 * m, 2)


def _fire_glaze_tub(p: Painter, x: float, y: float, glaze: RGB) -> None:
    """A half barrel of liquid glaze 1.2 m across on (x, y), bound with iron hoops, the glaze
    showing at its brim."""
    m = METRE
    _turned(p, x, y, [(0, 0.55 * m), (0.8 * m, 0.6 * m)], GLAZE_TUB, glaze)
    _craft_hoop(p, x, y - 0.3 * m, 0.57 * m, 0.08 * m, (60, 58, 60))


def _fire_tile_rack(p: Painter, x: float, y: float, long: float, colours: tuple[RGB, ...]) -> None:
    """An open rack `long` along the lit walls on (x, y), 2 m high, its four shelves laid with
    freshly glazed tiles in rows of `colours`, drying before the second firing."""
    m = METRE
    high = 2 * m
    left, front, _, _ = p.hall(x, y, long, 0.8 * m, high, (98, 76, 58), None)
    shelves = 4
    for k in range(shelves):
        d0 = high * k / shelves + 0.1 * m
        d1 = d0 + high / shelves - 0.2 * m
        colour = colours[k % len(colours)]
        _patch(p, left, front, (0.02, 0.98), (d0, d1), colour)
        for t in np.linspace(0.04, 0.96, round(long / (0.4 * m))):
            _craft_beam(p, left, front, (t, d0), (t, d1), 0.05 * m, shade(colour, 0.7))
    for t in np.linspace(0, 1, 4):
        _craft_beam(p, left, front, (t, 0), (t, high), 0.14 * m, (84, 64, 50))


def _fire_glazing_shop(p: Painter, x: float, y: float, long: float, deep: float) -> Top:
    """A glazing shop `long` along the lit walls and `deep` of cream render under red tiles on
    (x, y): wide doors standing open on the dipping vats, a row of windows and a louvred vent
    on its ridge. Returns its top."""
    m = METRE
    high = 4.4 * m
    top = p.hall(x, y, long, deep, high, OFFICE_WALL, None)
    left, front, right, _ = top
    _plinth(p, top, high, (176, 168, 150), 0.5 * m)
    _craft_band(p, top, 0.2 * m, 0.35 * m, shade(OFFICE_WALL, 0.84))
    count = max(2, round(long / (3 * m)))
    for k in range(count):
        t = (k + 0.5) / count
        if abs(t - 0.5) < 0.7 / count:
            continue
        _craft_casement(p, left, front, long, t, 1.2 * m, (1.0 * m, 2.6 * m), WINDOW)
    _craft_casement(p, front, right, deep, 0.5, 1.2 * m, (1.0 * m, 2.6 * m), shade(WINDOW, 0.8))
    half = 1.5 * m / long
    _patch(p, left, front, (0.5 - half, 0.5 + half), (1.2 * m, high), (64, 52, 46))
    _patch(p, left, front, (0.5 - half, 0.5 + half), (2.8 * m, high), (110, 92, 74))
    for k, glaze in enumerate((COBALT, STOVE_GREEN, TILE_WHITE)):
        gx, gy = _on_wall(left, front, 0.5 - half * 0.6 + half * 0.6 * k, high - 0.2 * m)
        _fire_glaze_tub(p, gx, gy, glaze)
    p.gable(top, 2.8 * m, ROOF_TILE, OFFICE_WALL)
    _tiles(p, top, 2.8 * m, ROOF_TILE, (0.3, 0.6))
    ridge_back, ridge_front = _ridge(top, 2.8 * m)
    vx, vy = _on_wall(ridge_back, ridge_front, 0.5, 0)
    vent = p.hall(vx, vy + 0.4 * m, 2.4 * m, 1.2 * m, 0.9 * m, shade(OFFICE_WALL, 0.9), None)
    for d in (0.3 * m, 0.6 * m):
        _craft_beam(p, vent[0], vent[1], (0.05, d), (0.95, d), 0.1 * m, (60, 52, 48))
    p.gable(vent, 0.5 * m, ROOF_TILE, OFFICE_WALL)
    return top


def _fire_glazed_roof(p: Painter, top: Top, rise: float, roof: RGB, pattern: RGB) -> None:
    """A gable roof of glazed tiles in `roof` on a block's top, laid with a band of diamonds in
    `pattern` along its lit side and a glint of light on its glaze."""
    m = METRE
    p.gable(top, rise, roof, OFFICE_WALL)
    _tiles(p, top, rise, roof, (0.2, 0.4, 0.6, 0.8))
    count = max(3, round(abs(top[1][0] - top[0][0]) / (0.9 * m)))
    for k in range(count):
        a0, a1 = (k + 0.15) / count, (k + 0.85) / count
        mid = (a0 + a1) / 2
        p.polygon(
            [
                _on_roof(top, rise, a0, 0.5),
                _on_roof(top, rise, mid, 0.4),
                _on_roof(top, rise, a1, 0.5),
                _on_roof(top, rise, mid, 0.6),
            ],
            pattern,
        )
    _bar(
        p, _on_roof(top, rise, 0.05, 0.9), _on_roof(top, rise, 0.3, 0.9), 0.12 * m, shade(roof, 1.5)
    )


def _fire_tile_facade(
    p: Painter, a: Point, b: Point, length: float, drop: tuple[float, float]
) -> None:
    """The wall under the edge from `a` to `b`, which is `length` units long, faced from
    `drop[0]` down it to `drop[1]` in white glazed tiles 0.6 m square, every other one painted
    with a small blue flower, between cobalt pilasters at its corners."""
    m = METRE
    d0, d1 = drop
    _patch(p, a, b, (0, 1), drop, TILE_WHITE)
    cols, rows = max(2, round(length / (0.6 * m))), max(2, round((d1 - d0) / (0.6 * m)))
    q = 0.09 * m
    for i in range(cols):
        for j in range(rows):
            if (i + j) % 2:
                continue
            x, y = _on_wall(a, b, (i + 0.5) / cols, d0 + (d1 - d0) * (j + 0.5) / rows)
            p.blob(x - q, y - q, x + q, y + q, COBALT)
    rim = 0.3 * m / length
    for t in (rim, 1 - rim):
        _craft_beam(p, a, b, (t, d0), (t, d1), 0.55 * m, COBALT)


def _fire_tiled_house(
    p: Painter, x: float, y: float, long: float, deep: float, floors: int, mural: bool
) -> Top:
    """The works' office `long` along the lit walls and `deep`, `floors` storeys of 3.6 m, its
    walls faced all over in the works' own blue-and-white tiles between cobalt bands: tall
    windows in cobalt frames, a green door under a cobalt hood and, if `mural`, a great tiled
    picture of a ship in blue across its middle, or else two framed panels of patterned tiles
    either side of its door. Returns its top."""
    m = METRE
    high = floors * 3.6 * m
    top = p.hall(x, y, long, deep, high, TILE_WHITE, None)
    left, front, right, _ = top
    _fire_tile_facade(p, left, front, long, (0, high))
    _fire_tile_facade(p, front, right, deep, (0, high))
    _plinth(p, top, high, COBALT, 0.6 * m)
    _craft_band(p, top, 0.35 * m, 0.5 * m, STOVE_GREEN)
    for f in range(1, floors):
        _craft_band(p, top, f * 3.6 * m, 0.3 * m, COBALT)
    if not mural:
        for t in (0.3, 0.7):
            _craft_tile_panel(
                p, left, front, long, (t - 0.09, t + 0.09), (high - 3.0 * m, high - 0.9 * m)
            )
    count = max(2, round(long / (3 * m)))
    for f in range(floors):
        head = f * 3.6 * m + 1.0 * m
        for k in range(count):
            t = (k + 0.5) / count
            if mural and f < floors - 1 and abs(t - 0.5) < 0.3:
                continue
            if f == floors - 1 and (
                abs(t - 0.5) < 0.6 / count or not mural and abs(abs(t - 0.5) - 0.2) < 0.6 / count
            ):
                continue
            _craft_casement(
                p, left, front, long, t, 1.2 * m, (head, head + 1.9 * m), WINDOW, COBALT
            )
        for t in (0.3, 0.7):
            _craft_casement(
                p,
                front,
                right,
                deep,
                t,
                1.2 * m,
                (head, head + 1.9 * m),
                shade(WINDOW, 0.8),
                COBALT,
            )
    if mural:
        d0, d1 = 0.8 * m, (floors - 1) * 3.6 * m - 0.4 * m
        _patch(p, left, front, (0.22, 0.78), (d0, d1), COBALT)
        _patch(p, left, front, (0.235, 0.765), (d0 + 0.25 * m, d1 - 0.25 * m), (220, 232, 246))
        sea = (d0 + d1) / 2 + 0.9 * m
        _patch(p, left, front, (0.235, 0.765), (sea, d1 - 0.25 * m), (70, 110, 190))
        for k in range(4):
            wave = sea + 0.3 * m + k * 0.35 * m
            _craft_beam(p, left, front, (0.26, wave), (0.74, wave), 0.08 * m, (200, 216, 240))
        hx, hy = _on_wall(left, front, 0.5, sea)
        p.polygon(
            [
                (hx - 2.2 * m, hy - 0.2 * m),
                (hx + 2.4 * m, hy - 0.2 * m),
                (hx + 1.6 * m, hy + 0.7 * m),
                (hx - 1.6 * m, hy + 0.7 * m),
            ],
            COBALT,
        )
        for dx, sail in ((-1.0, 2.6), (0.6, 3.2)):
            mx = hx + dx * m
            _bar(p, (mx, hy - 0.2 * m), (mx, hy - (sail + 0.4) * m), 0.1 * m, COBALT)
            p.polygon(
                [
                    (mx + 0.1 * m, hy - (sail + 0.2) * m),
                    (mx + 1.3 * m, hy - 0.9 * m),
                    (mx + 0.1 * m, hy - 0.7 * m),
                ],
                (40, 70, 160),
            )
    half = 0.9 * m / long
    _patch(p, left, front, (0.5 - half, 0.5 + half), (high - 2.8 * m, high), STOVE_GREEN)
    _craft_beam(
        p, left, front, (0.5, high - 2.8 * m), (0.5, high), 0.06 * m, shade(STOVE_GREEN, 0.6)
    )
    _patch(
        p,
        left,
        front,
        (0.5 - half * 1.5, 0.5 + half * 1.5),
        (high - 3.3 * m, high - 2.9 * m),
        COBALT,
    )
    return top


KILN_COAL: RGB = (52, 50, 56)

CRATE_WOOD: RGB = (196, 160, 104)

STRAW: RGB = (230, 206, 120)

SHOWROOM_WALL: RGB = (240, 230, 206)

SHOWROOM_ROOF: RGB = (64, 74, 96)


def _fire_pot_range(
    p: Painter,
    x: float,
    y: float,
    long: float,
    deep: float,
    high: float,
    rise: float,
    doors: tuple[float, ...] = (),
    lights: bool = False,
) -> Top:
    """A pottery range `long` along the lit walls and `deep`, `high` to its eaves, washed warm
    cream under a roof of terracotta pantiles: a casement window every 3 m on each floor, blue
    doors at each of `doors` along its lit wall and, if `lights`, skylights along its roof for
    the throwers and turners at their wheels. Returns its top."""
    m = METRE
    top = p.hall(x, y, long, deep, high, POTTER_WALL, None)
    left, front, right, _ = top
    _plinth(p, top, high, (180, 156, 120), 0.5 * m)
    _craft_band(p, top, 0.2 * m, 0.4 * m, shade(POTTER_WALL, 0.84))
    floors = max(1, round(high / (3.8 * m)))
    storey = high / floors
    for a, b, length, glass in (
        (left, front, long, WINDOW),
        (front, right, deep, shade(WINDOW, 0.8)),
    ):
        count = max(1, round(length / (3 * m)))
        for f in range(floors):
            head = f * storey + 0.8 * m
            for k in range(count):
                t = (k + 0.5) / count
                if a is left and any(abs(t - d) < 0.7 / count for d in doors):
                    continue
                _craft_casement(p, a, b, length, t, 1.1 * m, (head, head + 1.5 * m), glass)
    for t in doors:
        half, rim = 1.0 * m / long, 0.15 * m / long
        _patch(p, left, front, (t - half - rim, t + half + rim), (high - 2.9 * m, high), PLANK)
        _patch(p, left, front, (t - half, t + half), (high - 2.75 * m, high), COBALT)
        _craft_beam(p, left, front, (t, high - 2.75 * m), (t, high), 0.06 * m, shade(COBALT, 0.6))
    p.gable(top, rise, PANTILE, POTTER_WALL)
    _craft_pantiles(p, top, rise, PANTILE)
    if lights:
        for t0 in np.arange(0.06, 0.9, 0.2):
            _skylight(p, top, rise, (t0, t0 + 0.12))
    return top


def _fire_hoist(p: Painter, a: Point, b: Point, t: float, high: float, floors: int) -> None:
    """A column of taking-in doors one over another about `t` along the wall under the edge from
    `a` to `b`, one to each upper floor of a building `high` to its eaves, under a hoist beam
    with its rope and hook."""
    m = METRE
    storey = high / floors
    span = abs(b[0] - a[0])
    half = 0.8 * m / span
    for f in range(floors - 1):
        head = f * storey + 0.6 * m
        _patch(
            p,
            a,
            b,
            (t - half - 0.1 * m / span, t + half + 0.1 * m / span),
            (head - 0.1 * m, head + 2.3 * m),
            PLANK,
        )
        _patch(p, a, b, (t - half, t + half), (head, head + 2.2 * m), COBALT)
        _craft_beam(p, a, b, (t, head), (t, head + 2.2 * m), 0.06 * m, shade(COBALT, 0.6))
    bx, by = _on_wall(a, b, t, 0.2 * m)
    out = _outward(a, b, 1.4 * m)
    end = (bx + out[0], by + out[1])
    _bar(p, (bx, by), end, 0.25 * m, (70, 60, 52))
    _bar(p, end, (end[0], end[1] + 2.6 * m), 0.05 * m, (60, 56, 52))
    p.blob(end[0] - 0.15 * m, end[1] + 2.5 * m, end[0] + 0.15 * m, end[1] + 2.8 * m, (60, 56, 52))


def _fire_saggars(p: Painter, x: float, y: float, tall: int) -> None:
    """A bung of `tall` round fireclay saggars stacked one on another on (x, y), the boxes the
    ware is fired in, each rim showing as a dark ring."""
    m = METRE
    r, step = 0.45 * m, 0.38 * m
    _turned(p, x, y, [(0, r), (tall * step, r)], SAGGAR, shade(SAGGAR, 0.8))
    for k in range(1, tall):
        _craft_hoop(p, x, y - k * step, r, 0.05 * m, shade(SAGGAR, 0.62))


def _fire_crate(p: Painter, x: float, y: float) -> None:
    """A slatted crate of packed ware 1.3 by 1 m and 1 m high on (x, y), with straw sticking up
    between the slats of its lid."""
    m = METRE
    top = p.hall(x, y, 1.3 * m, 1.0 * m, 1.0 * m, CRATE_WOOD, shade(CRATE_WOOD, 1.1))
    left, front, right, _ = top
    for a, b, tone in ((left, front, 0.7), (front, right, 0.58)):
        for d in (0.33, 0.66):
            _craft_beam(p, a, b, (0, d * m), (1, d * m), 0.06 * m, shade(CRATE_WOOD, tone))
        for t in (0.06, 0.94):
            _craft_beam(p, a, b, (t, 0), (t, 1 * m), 0.1 * m, shade(CRATE_WOOD, tone))
    cx, cy = (left[0] + right[0]) / 2, (left[1] + right[1]) / 2
    for dx, dy in ((-0.3, 0), (0.2, -0.1), (0.1, 0.15)):
        p.blob(
            cx + (dx - 0.18) * m,
            cy + (dy - 0.1) * m,
            cx + (dx + 0.18) * m,
            cy + (dy + 0.1) * m,
            STRAW,
        )


def _fire_crates(p: Painter, x: float, y: float, rows: tuple[int, ...]) -> None:
    """Crates of packed ware stacked on (x, y): a row of `rows[0]` along the lit walls, `rows[1]`
    on top of those and so on."""
    m = METRE
    for level, count in enumerate(rows):
        for k in range(count):
            cx, cy = _iso(x, y, (k + level * 0.5) * 1.4 * m, 0)
            _fire_crate(p, cx, cy - level * 1.0 * m)


def _fire_blue_window(
    p: Painter, a: Point, b: Point, length: float, t: float, span: tuple[float, float]
) -> None:
    """A tall round-headed display window 2.4 m wide in a gilt frame about `t` along the wall under
    the edge from `a` to `b`, which is `length` units long, from `span[0]` down the wall to
    `span[1]`: lit warm inside, with shelves of cobalt-blue ware, plates on edge and vases."""
    m = METRE
    head, foot = span
    p.polygon(_round_head(a, b, length, t, 2.6 * m, head - 0.15 * m, foot + 0.15 * m), GILT)
    p.polygon(_round_head(a, b, length, t, 2.3 * m, head, foot), SHOP_GLOW)
    half = 1.1 * m / length
    shelves = np.linspace(head + 2.0 * m, foot - 0.1 * m, 3)
    for row, shelf in enumerate(shelves):
        _craft_beam(p, a, b, (t - half, shelf), (t + half, shelf), 0.1 * m, (120, 90, 60))
        for k in range(3):
            x, y = _on_wall(a, b, t - half + 2 * half * (k + 0.5) / 3, shelf - 0.05 * m)
            if (k + row) % 2:
                p.ellipse(x - 0.32 * m, y - 0.66 * m, x + 0.32 * m, y, COBALT)
                p.blob(x - 0.15 * m, y - 0.5 * m, x + 0.15 * m, y - 0.18 * m, CREAM_GLAZE)
            else:
                _craft_pot(p, x, y, 0.75 * m, COBALT, True)
    _craft_beam(p, a, b, (t, head), (t, foot), 0.08 * m, GILT)


def _fire_urn(p: Painter, x: float, y: float, tall: float) -> None:
    """A great urn glazed cobalt blue and white standing on (x, y), `tall` high: a swelling body
    on a gilt foot, white bands painted round it, a flaring gilt-rimmed neck, two gilt scroll
    handles and a glint of light on its glaze."""
    t = tall
    _turned(p, x, y, [(0, 0.2 * t), (0.08 * t, 0.16 * t)], GILT)
    body = [
        (0, 0.13 * t),
        (0.18 * t, 0.3 * t),
        (0.42 * t, 0.36 * t),
        (0.66 * t, 0.26 * t),
        (0.8 * t, 0.15 * t),
        (0.92 * t, 0.2 * t),
    ]
    _turned(p, x, y - 0.08 * t, body, COBALT, GILT)
    for h, half in ((0.3 * t, 0.34 * t), (0.58 * t, 0.31 * t)):
        _craft_hoop(p, x, y - h, half, 0.05 * t, CREAM_GLAZE)
    for side in (-1, 1):
        _bar(
            p,
            (x + side * 0.16 * t, y - 0.85 * t),
            (x + side * 0.38 * t, y - 0.78 * t),
            0.04 * t,
            GILT,
        )
        _bar(
            p,
            (x + side * 0.38 * t, y - 0.78 * t),
            (x + side * 0.33 * t, y - 0.6 * t),
            0.04 * t,
            GILT,
        )
    p.blob(x - 0.24 * t, y - 0.56 * t, x - 0.12 * t, y - 0.4 * t, (180, 200, 250))


WAGON_WOOD: RGB = (118, 92, 70)

NAVVY: RGB = (66, 80, 70)


def _fire_wicket(p: Painter, x: float, y: float, high: float, kind: int) -> None:
    """An arched wicket into a ring kiln's chamber with its foot at (x, y) in a wall `high` tall:
    bricked up for the firing if `kind` is 0, open and dark if 1, glowing if 2, under a sooty
    streak."""
    m = METRE

    def arch(half: float, crown: float) -> list[Point]:
        turns = np.linspace(0, math.pi, 7)
        return [
            (x - half, y),
            (x + half, y),
            *[(x + half * math.cos(a), y - crown + half - half * math.sin(a)) for a in turns],
        ]

    _bar(p, (x, y - 2.0 * m), (x + 0.2 * m, y - high + 0.1 * m), 0.5 * m, (92, 54, 44))
    p.polygon(arch(0.78 * m, 2.0 * m), shade(BRICK_KILN, 1.12))
    fill = (shade(BRICK_KILN, 0.96), DOORWAY, BRICK_GLOW)[kind]
    p.polygon(arch(0.58 * m, 1.8 * m), fill)
    if kind == 0:
        for drop in (0.35, 0.75, 1.15):
            _bar(
                p,
                (x - 0.55 * m, y - drop * m),
                (x + 0.55 * m, y - drop * m),
                0.05 * m,
                shade(fill, 0.75),
            )
    if kind == 2:
        p.blob(x - 0.32 * m, y - 1.3 * m, x + 0.32 * m, y - 0.1 * m, (255, 214, 120))


def _fire_ring_kiln(p: Painter, x: float, y: float, long: float, chimney: float) -> None:
    """A Hoffmann ring kiln centred on (x, y), `long` metres end to end along the lit walls and
    11 m across: a long low oval of sooty brick, its battered walls pierced all round by arched
    wickets, some bricked up for the firing and one glowing, the feeding floor on its crown under
    a pantiled hipped roof on open posts, and its tall round chimney rising from the middle,
    drawn for `chimney` units."""
    m = METRE
    r = 5.5
    half = long / 2 - r

    def at(u: float, v: float) -> Point:
        return _iso(x, y, u * m, v * m)

    def up(u: float, v: float, z: float) -> Point:
        px, py = at(u, v)
        return px, py - z * m

    _bed(p, _footing(x, y, long * m, 11 * m))
    high = 3.6 * m
    f = r * math.sqrt(2) * m
    ends = (at(-half, 0), at(half, 0))
    batter = [(0, f + 0.35 * m), (0.9 * m, f), (high, f)]
    _turned(p, *ends[0], batter, BRICK_KILN)
    b_left, b_front, _, _ = p.hall(x, y, 2 * half * m, 11 * m, high, BRICK_KILN, None)
    _turned(p, *ends[1], batter, BRICK_KILN)
    _patch(p, b_left, b_front, (0, 1), (high - 0.9 * m, high), shade(BRICK_KILN, 0.84))
    lid = (110, 92, 84)
    for ex, ey in ends:
        p.blob(ex - f, ey - high - f / 2, ex + f, ey - high + f / 2, lid)
    p.polygon([up(-half, -r, 3.6), up(half, -r, 3.6), up(half, r, 3.6), up(-half, r, 3.6)], lid)
    count = max(3, round(2 * half / 2.6))
    for k in range(count):
        kind = (1, 0, 1, 2, 0, 1, 0, 0, 1, 0)[k % 10]
        _fire_wicket(p, *_on_wall(b_left, b_front, (k + 0.5) / count, high), high, kind)
    for (ex, ey), turns in ((ends[0], (2.75, 2.45)), (ends[1], (1.95, 1.2, 0.5))):
        for kind, a in zip((0, 1, 0), turns, strict=False):
            _fire_wicket(p, ex + f * math.cos(a), ey + f / 2 * math.sin(a), high, kind)
    store = p.hall(*up(0, 0, 3.6), (2 * half + 4) * m, 7.6 * m, 1.3 * m, (50, 40, 36), None)
    posts = max(4, round((2 * half + 4) / 2.2))
    for a, b, n in ((store[0], store[1], posts), (store[1], store[2], 3)):
        for k in range(n + 1):
            px, py = _on_wall(a, b, k / n, 1.3 * m)
            _bar(p, (px, py), (px, py - 1.3 * m), 0.22 * m, shade(TIMBER, 0.8))
    tile = BRICK_PANTILE
    ru0, ru1, rv0, rv1, eave, rise = -half - 4.6, half + 4.6, -4.2, 4.2, 4.9, 2.8
    run = (rv1 - rv0) / 2
    r_left, r_front = up(ru0, rv0, eave), up(ru1, rv0, eave)
    r_right, r_back = up(ru1, rv1, eave), up(ru0, rv1, eave)
    ridge_back = up(ru0 + run, rv0 + run, eave + rise)
    ridge_front = up(ru1 - run, rv0 + run, eave + rise)
    p.polygon([r_back, r_right, ridge_front, ridge_back], shade(tile, 0.68), shade(tile, 0.45))
    p.polygon([r_left, r_back, ridge_back], shade(tile, 0.9), shade(tile, 0.55))
    p.polygon([r_front, r_right, ridge_front], shade(tile, 0.74), shade(tile, 0.5))
    p.polygon([r_left, r_front, ridge_front, ridge_back], tile, shade(tile, 0.6))
    for u in np.linspace(ru0 + 0.5, ru1 - 0.5, round((ru1 - ru0) * 1.6)):
        reach = min(u - ru0, ru1 - u, run)
        _bar(
            p,
            up(u, rv0, eave),
            up(u, rv0 + reach, eave + rise * reach / run),
            0.06 * m,
            shade(tile, 0.86),
        )
    _bar(p, r_left, r_front, 0.14 * m, shade(tile, 0.7))
    _bar(p, ridge_back, ridge_front, 0.22 * m, shade(tile, 1.2))
    _tall_chimney(p, *up(0, 0, eave + rise - 1.0), chimney)


def _fire_hack_shed(p: Painter, x: float, y: float, long: float) -> None:
    """An open drying shed `long` metres along the lit walls and 4 m across centred on (x, y): a
    long hack of cream green bricks stacked in an open lattice under a pantiled roof on posts."""
    m = METRE
    shed = _footing(x, y - 2.8 * m, long * m, 4 * m)
    s_left, s_front, s_right, s_back = shed
    posts = max(3, round(long / 3))
    for k in range(posts + 1):
        px, py = _on_wall(s_back, s_right, k / posts, 2.8 * m)
        _bar(p, (px, py), (px, py - 2.8 * m), 0.2 * m, shade(TIMBER, 0.6))
    _brick_hack(p, x, y, long - 2, 1.6, 1.5)
    for k in range(posts + 1):
        px, py = _on_wall(s_left, s_front, k / posts, 2.8 * m)
        _bar(p, (px, py), (px, py - 2.8 * m), 0.2 * m, shade(TIMBER, 0.75))
    p.gable(shed, 1.2 * m, shade(BRICK_PANTILE, 0.8), TIMBER)
    _roof_rows(p, shed, 1.2 * m, shade(BRICK_PANTILE, 0.8))


def _fire_clay_pit(p: Painter, x: float, y: float, long: float, wide: float) -> None:
    """A clay pit dug into the ground with its near corner's foot at the middle of (x, y): `long`
    metres along the lit walls and `wide` across, stepped down its far sides in benches of raw
    clay to a floor with a puddle in it, grass at its rim."""
    m = METRE

    def at(u: float, v: float) -> Point:
        return _iso(x, y, u * m, v * m)

    u0, u1, v0, v1, d = -long / 2, long / 2, -wide / 2, wide / 2, 2.4
    p.polygon([at(u0, v0), at(u0, v1), at(u0 + d, v1 - d), at(u0 + d, v0)], shade(BRICK_CLAY, 0.62))
    p.polygon([at(u0, v1), at(u1, v1), at(u1, v1 - d), at(u0 + d, v1 - d)], shade(BRICK_CLAY, 0.92))
    for q, tone in ((0.3, 0.8), (0.55, 1.04), (0.8, 0.78)):
        _bar(p, at(u0 + q * d, v1 - q * d), at(u1, v1 - q * d), 0.12 * m, shade(BRICK_CLAY, tone))
        _bar(
            p,
            at(u0 + q * d, v0),
            at(u0 + q * d, v1 - q * d),
            0.12 * m,
            shade(BRICK_CLAY, tone * 0.8),
        )
    _bar(p, at(u0, v1), at(u1, v1), 0.25 * m, (88, 104, 56))
    _bar(p, at(u0, v0), at(u0, v1), 0.25 * m, (88, 104, 56))
    p.polygon(
        [at(u0 + d, v0), at(u1, v0), at(u1, v1 - d), at(u0 + d, v1 - d)], shade(BRICK_CLAY, 0.8)
    )
    px, py = at(u0 + d + (u1 - u0 - d) * 0.4, (v0 + v1 - d) / 2)
    p.blob(px - 2.4 * m, py - 0.9 * m, px + 2.4 * m, py + 0.9 * m, PUDDLE)
    p.blob(px - 1.5 * m, py - 0.55 * m, px - 0.2 * m, py - 0.1 * m, shade(PUDDLE, 1.3))


def _fire_narrow_gauge(p: Painter, a: Point, b: Point) -> None:
    """A narrow-gauge tramway on the ground from `a` to `b` along the lit walls: two light rails
    on sleepers, 0.6 m apart."""
    m = METRE
    for k in range(1, 30):
        sx, sy = _on_wall(a, b, k / 30, 0)
        p.blob(sx - 0.35 * m, sy - 0.18 * m, sx + 0.35 * m, sy + 0.18 * m, shade(TIMBER, 0.6))
    for w in (-0.3, 0.3):
        _bar(p, _iso(*a, 0, w * m), _iso(*b, 0, w * m), 0.08 * m, (70, 66, 64))


def _fire_siding(p: Painter, a: Point, b: Point) -> None:
    """A standard-gauge railway siding on the ground from `a` to `b` along the lit walls: two
    rails on timber sleepers in a bed of ballast."""
    m = METRE
    _bar(p, a, b, 3.0 * m, (126, 118, 106))
    for k in range(1, 60):
        sx, sy = _on_wall(a, b, k / 60, 0)
        lo, hi = _iso(sx, sy, 0, -1.3 * m), _iso(sx, sy, 0, 1.3 * m)
        _bar(p, lo, hi, 0.26 * m, (92, 72, 56))
    for w in (-0.72, 0.72):
        _bar(p, _iso(*a, 0, w * m), _iso(*b, 0, w * m), 0.12 * m, (150, 152, 160))


def _fire_brick_wagon(p: Painter, x: float, y: float) -> None:
    """A four-wheeled open wagon 6 m long along the lit walls standing on (x, y) on its siding:
    planked sides on a dark frame with buffers at its ends, heaped with fired red bricks."""
    m = METRE
    p.hall(x, y - 0.5 * m, 6.4 * m, 2.0 * m, 0.4 * m, (44, 42, 44), None)
    body = p.hall(x, y - 0.9 * m, 6 * m, 2.6 * m, 1.3 * m, WAGON_WOOD, None)
    left, front, right, back = body
    for a, b, tone in ((left, front, 0.75), (front, right, 0.62)):
        for d in (0.43, 0.87):
            _craft_beam(p, a, b, (0, d * m), (1, d * m), 0.05 * m, shade(WAGON_WOOD, tone))
        for t in (0.04, 0.5, 0.96):
            _craft_beam(p, a, b, (t, 0), (t, 1.3 * m), 0.12 * m, (54, 50, 50))
    load = [left, front, right, back]
    p.polygon(load, BRICK_FIRED)
    p.polygon(
        [_on_wall(left, back, 0.5, -0.4 * m), _on_wall(front, right, 0.5, -0.4 * m), right, back],
        shade(BRICK_FIRED, 0.85),
    )
    p.polygon(
        [left, front, _on_wall(front, right, 0.5, -0.4 * m), _on_wall(left, back, 0.5, -0.4 * m)],
        shade(BRICK_FIRED, 1.08),
    )
    for k in range(1, 8):
        _bar(
            p,
            _on_wall(left, front, k / 8, 0),
            _on_wall(
                _on_wall(left, back, 0.5, -0.4 * m), _on_wall(front, right, 0.5, -0.4 * m), k / 8, 0
            ),
            0.05 * m,
            shade(BRICK_FIRED, 0.8),
        )
    for t in (0.22, 0.78):
        wx, wy = _on_wall(left, front, t, 1.3 * m + 0.9 * m)
        p.ellipse(wx - 0.5 * m, wy - 0.5 * m, wx + 0.5 * m, wy + 0.5 * m, (46, 44, 46))
        p.blob(wx - 0.15 * m, wy - 0.15 * m, wx + 0.15 * m, wy + 0.15 * m, (110, 110, 116))
    for end in (left, front):
        bx, by = end[0], end[1] + 1.6 * m
        p.blob(bx - 0.2 * m, by - 0.2 * m, bx + 0.2 * m, by + 0.2 * m, (40, 38, 40))


def _fire_navvy(p: Painter, x: float, y: float) -> None:
    """A steam navvy digging clay on (x, y): a riveted iron truck on wheels carrying a house with
    an upright boiler and its chimney steaming, and a jib reaching out to the left with the
    dipper arm and toothed bucket biting into the clay face."""
    m = METRE
    p.hall(x, y, 5 * m, 3 * m, 0.9 * m, (50, 50, 54), (60, 60, 64))
    for t in (-1.6, 1.6):
        wx, wy = _iso(x, y, t * m, -1.5 * m)
        p.ellipse(wx - 0.5 * m, wy - 0.95 * m, wx + 0.5 * m, wy + 0.05 * m, (40, 40, 44))
    house = p.hall(x + 0.3 * m, y - 0.9 * m, 3.6 * m, 2.6 * m, 2.4 * m, NAVVY, None)
    h_left, h_front, h_right, _ = house
    _patch(p, h_left, h_front, (0.55, 0.85), (0.4 * m, 1.6 * m), (40, 38, 36))
    _patch(p, h_front, h_right, (0.3, 0.7), (0.5 * m, 1.3 * m), shade(WINDOW, 0.8))
    p.gable(house, 0.6 * m, shade(NAVVY, 0.7), NAVVY)
    cx, cy = _on_wall(h_left, h_right, 0.75, 0)
    _bar(p, (cx, cy), (cx, cy - 2.4 * m), 0.4 * m, (36, 36, 40))
    _craft_smoke(p, cx + 0.4 * m, cy - 3.6 * m, 1.6 * m, 2)
    foot = _iso(x, y - 1.6 * m, -2.3 * m, 0)
    tip = _iso(x, y - 6.0 * m, -6.5 * m, 0)
    _bar(p, foot, tip, 0.35 * m, shade(NAVVY, 0.85))
    _bar(p, (tip[0], tip[1]), (tip[0] - 0.2 * m, tip[1] + 4.2 * m), 0.06 * m, (36, 34, 34))
    pivot = _on_wall(foot, tip, 0.45, 0)
    bucket = (tip[0] - 0.9 * m, tip[1] + 4.6 * m)
    _bar(p, pivot, bucket, 0.28 * m, shade(NAVVY, 0.7))
    p.polygon(
        [
            (bucket[0] - 0.8 * m, bucket[1] - 0.5 * m),
            (bucket[0] + 0.5 * m, bucket[1] - 0.8 * m),
            (bucket[0] + 0.6 * m, bucket[1] + 0.5 * m),
            (bucket[0] - 0.6 * m, bucket[1] + 0.6 * m),
        ],
        (46, 46, 50),
    )
    p.blob(
        bucket[0] - 0.7 * m,
        bucket[1] - 0.9 * m,
        bucket[0] + 0.4 * m,
        bucket[1] - 0.4 * m,
        BRICK_CLAY,
    )


def _fire_dome_kiln(p: Painter, x: float, y: float) -> None:
    """A round downdraught kiln 7 m across standing on (x, y): a squat drum of brick bound with
    iron bands under a shallow dome, with firing mouths round its foot, one of them glowing."""
    m = METRE
    r = 3.5 * m
    _turned(p, x, y, [(0, r), (3.2 * m, r)], BRICK_KILN)
    for h in (1.0 * m, 2.4 * m):
        _craft_hoop(p, x, y - h, r, 0.15 * m, (64, 60, 62))
    _turned(
        p,
        x,
        y - 3.2 * m,
        [(0, r), (0.8 * m, r * 0.85), (1.6 * m, r * 0.5), (2.0 * m, 0)],
        shade(BRICK_KILN, 0.82),
    )
    for a, glow in ((2.6, False), (1.9, True), (1.1, False)):
        fx, fy = x + r * math.cos(a), y + r / 2 * math.sin(a)
        p.polygon(_craft_arch(fx, fy, 0.9 * m, 1.3 * m), shade(BRICK_KILN, 0.6))
        p.polygon(_craft_arch(fx, fy, 0.6 * m, 1.0 * m), BRICK_GLOW if glow else DOORWAY)
    _craft_smoke(p, x + 0.4 * m, y - 6.4 * m, 2.0 * m, 2)
