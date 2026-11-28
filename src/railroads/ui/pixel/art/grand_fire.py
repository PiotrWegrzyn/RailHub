"""Big and mega works of fire, each drawn as a building of its own: foundries glowing round their cupolas, tile works, potters
and porcelain works among clusters of bottle kilns and stacked ware, brickworks of Hoffmann kilns
under tall chimneys, cement works of glowing rotary kilns under dusty silos, glassworks of brick
cones lit at their mouths and gasworks of great gasholders in iron guide frames."""

import itertools
import math

from railroads.ui.pixel.art.commerce import _glass_rack
from railroads.ui.pixel.art.crafts import MOLTEN
from railroads.ui.pixel.art.industry import (
    GLASS_AMBER,
    GLASS_GREEN,
    GLASS_SAND,
    GLASS_SODA,
    SMOKE,
    SOIL,
    _bed,
    _crate,
    _door,
    _footing,
    _glass_cone,
    _glass_crates,
    _glass_heap,
    _glass_house,
    _glass_lehr,
    _glass_packing,
    _glass_yard,
    _on_roof,
    _ring,
    _roof_rows,
    _sack,
    _smoke,
    _smoking_chimney,
    _spill,
    _stains,
    _track,
    _window,
)
from railroads.ui.pixel.art.painter import (
    BRICK,
    DOORWAY,
    RGB,
    TIMBER,
    WHITEWASH,
    WINDOW,
    Painter,
    Point,
    Sprite,
    Top,
    shade,
)
from railroads.ui.pixel.art.parts import (
    METRE,
    _band,
    _bands,
    _heap,
    _iso,
    _on_wall,
    _patch,
    _plinth,
    _turned,
    _vault,
    _windows,
)

STEAM: RGB = (238, 238, 242)
WHITE_HOT: RGB = (255, 226, 120)
IRON: RGB = (66, 62, 64)
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


def _sparks(p: Painter, points: tuple[Point, ...]) -> None:
    for k, (x, y) in enumerate(points):
        p.point(x, y, WHITE_HOT if k % 2 else MOLTEN)


def _coke_heap(p: Painter, x: float, y: float, half: float) -> None:
    _spill(p, x, y, half, COKE)
    _heap(p, x, y, half, half * 0.9, COKE)
    for dx, dy in ((-1, -1.5), (0.6, -2.4)):
        p.point(x + dx, y + dy, shade(COKE, 1.5))


def _column(p: Painter, x: float, y: float, half: float, tall: float, colour: RGB) -> None:
    """A tall round banded column or tank standing on (x, y) under a domed cap."""
    _turned(
        p,
        x,
        y,
        [(0, half), (tall, half), (tall + half * 0.5, half * 0.6)],
        colour,
        shade(colour, 1.1),
    )
    for f in (0.3, 0.65):
        _ring(p, x, y - tall * f, half, shade(colour, 0.7))


FOUNDRY_WALL: RGB = (126, 64, 45)
FOUNDRY_ROOF: RGB = (88, 84, 92)
SAND: RGB = (214, 196, 146)
CASTING: RGB = (74, 76, 82)
CUPOLA: RGB = (96, 92, 96)
CRANE: RGB = (200, 160, 56)


def _foundry_hall(
    p: Painter, x: float, y: float, length: float, depth: float, height: float, rise: float
) -> Top:
    """A soot-dark brick foundry hall under a slate roof, a row of windows glowing from the fire
    inside and its great door open on the glow."""
    top = p.hall(x, y, length, depth, height, FOUNDRY_WALL, None)
    left, front, right, _ = top
    _plinth(p, top, height, FOUNDRY_WALL)
    _stains(p, left, front, height, (0.1, 0.35, 0.8), shade(FOUNDRY_WALL, 0.6), 2)
    spans = tuple(t for t in (0.08, 0.2, 0.32, 0.44, 0.76, 0.88) if t < 0.95)
    _windows(p, left, front, spans, (2,), 3, shade(MOLTEN, 0.8), shade(FOUNDRY_WALL, 1.4))
    _door(p, left, front, (0.54, 0.68), 3, height, MOLTEN, shade(FOUNDRY_WALL, 1.4))
    _patch(p, left, front, (0.57, 0.65), (4.5, height), WHITE_HOT)
    _window(p, front, right, (0.35, 0.65), 2, shade(MOLTEN, 0.7))
    p.gable(top, rise, FOUNDRY_ROOF, FOUNDRY_WALL)
    _roof_rows(p, top, rise, FOUNDRY_ROOF)
    return top


def _cupola(p: Painter, x: float, y: float, tall: float) -> None:
    """A tall banded iron cupola furnace standing on (x, y), fire and sparks at its mouth."""
    _turned(p, x, y, [(0, 2.4), (tall, 2.2)], CUPOLA)
    for f in (0.2, 0.45, 0.7):
        _ring(p, x, y - tall * f, 2.3, shade(CUPOLA, 0.6))
    p.ellipse(x - 2.2, y - tall - 1.1, x + 2.2, y - tall + 1.1, MOLTEN)
    p.blob(x - 1.1, y - tall - 0.6, x + 1.1, y - tall + 0.6, WHITE_HOT)
    _sparks(p, ((x - 1, y - tall - 3), (x + 1.5, y - tall - 4.5), (x, y - tall - 6.5)))


def _charging_deck(p: Painter, x: float, y: float, length: float, height: float) -> None:
    """A timber charging deck `height` up on iron legs, running along the lit way from (x, y),
    with a ladder up to it."""
    deck = p.hall(x, y - height, length, 3, 0.8, shade(TIMBER, 0.8), TIMBER)
    left, front, right, _ = deck
    for corner in (left, front, right):
        p.line([(corner[0], corner[1] + 0.8), (corner[0], corner[1] + height + 0.8)], IRON)
    lx, ly = _on_wall(left, front, 0.15, 0.8)
    p.line([(lx, ly), (lx - 2, ly + height)], shade(TIMBER, 0.7))
    p.line([(lx + 1, ly), (lx - 1, ly + height)], shade(TIMBER, 0.7))


def _crucible(p: Painter, x: float, y: float) -> None:
    """An iron ladle standing on (x, y), brimming with molten iron."""
    p.ellipse(x - 2.2, y - 3.6, x + 2.2, y, CASTING)
    p.blob(x - 1.6, y - 3.8, x + 1.6, y - 2.6, MOLTEN)
    p.point(x - 0.4, y - 3.4, WHITE_HOT)


def _casting_bed(p: Painter, x: float, y: float, half: float, moulds: int) -> None:
    """A floor of foundry sand centred at (x, y) with a row of dark moulding boxes, their pouring
    holes glowing."""
    p.blob(x - half, y - half * 0.35, x + half, y + half * 0.4, SAND)
    p.blob(x - half * 0.7, y - half * 0.2, x + half * 0.5, y + half * 0.15, shade(SAND, 1.06))
    for k in range(moulds):
        mx, my = x - half * 0.6 + 2.6 * k, y - 1 + 1.3 * k
        box = p.hall(mx, my, 2.4, 2, 1, shade(SAND, 0.55), shade(SAND, 0.7))
        p.point(*_mid(box[0], box[2]), MOLTEN if k % 2 == 0 else WHITE_HOT)


def _wheel(p: Painter, x: float, y: float, r: float) -> None:
    """A new cast-iron wheel standing upright on the sand at (x, y), its spokes showing."""
    p.ellipse(x - r * 0.8, y - 2 * r, x + r * 0.8, y, shade(CASTING, 1.25))
    p.blob(x - r * 0.8 + 1, y - 2 * r + 1, x + r * 0.8 - 1, y - 1, SAND)
    p.line([(x, y - 2 * r + 0.5), (x, y - 0.5)], CASTING)
    p.line([(x - r * 0.8 + 0.5, y - r), (x + r * 0.8 - 0.5, y - r)], CASTING)
    p.blob(x - 0.6, y - r - 0.6, x + 0.6, y - r + 0.6, shade(CASTING, 0.8))


def _pour(p: Painter, a: Point, b: Point) -> None:
    """A thin stream of molten iron running over the ground from `a` to `b`."""
    p.line([a, b], MOLTEN)
    p.line([(a[0], a[1] - 0.5), (b[0], b[1] - 0.5)], shade(MOLTEN, 1.1))


def big_foundry() -> Sprite:
    """A long soot-dark foundry hall under a slate roof, its windows and great door glowing, two
    tall iron cupolas throwing sparks beside a charging deck, molten iron running from their tap
    into a ladle, glowing moulds in a bed of sand, castings laid out and a chimney smoking."""
    p = Painter(52, 58, (25, 46))
    _smoking_chimney(p, -11, -12, -35, shade(BRICK, 0.7), close=True)
    p.shadow(20, 5)
    foot = _footing(-3, -1, 20, 8)
    _bed(p, foot, SOIL)
    _charging_deck(p, 17.5, -4.5, 8, 11)
    _cupola(p, 15, -6, 19)
    _cupola(p, 20, -3.5, 17)
    _foundry_hall(p, -3, -1, 20, 8, 7, 4)
    _pour(p, (18, -3), (15, 3))
    _crucible(p, 15, 5.5)
    _casting_bed(p, -14, 7, 8, 3)
    p.blob(2, 6, 12, 10.5, SAND)
    for x, y in ((5, 9.5), (9, 8.5)):
        _wheel(p, x, y, 2)
    _sparks(p, ((12, -30), (17, -32), (21, -29), (14, -34)))
    return p.sprite()


def _gantry(p: Painter, a: Point, b: Point, height: float, along: float) -> Point:
    """A travelling crane on two A-frame legs standing at `a` and `b`, its girder `height` up,
    with its hook hanging `along` of the way over; returns where the hook hangs from."""
    for x, y in (a, b):
        p.polygon([(x - 2.4, y), (x - 1.4, y), (x + 0.5, y - height), (x - 0.5, y - height)], CRANE)
        p.polygon(
            [(x + 1.4, y), (x + 2.4, y), (x + 0.5, y - height), (x - 0.5, y - height)],
            shade(CRANE, 0.75),
        )
        p.line([(x - 1.4, y - height * 0.4), (x + 1.4, y - height * 0.4)], shade(CRANE, 0.6))
    top_a, top_b = (a[0], a[1] - height), (b[0], b[1] - height)
    p.polygon(
        [(top_a[0], top_a[1] - 2), (top_b[0], top_b[1] - 2), top_b, top_a],
        CRANE,
        shade(CRANE, 0.55),
    )
    hx, hy = _on_wall(top_a, top_b, along, 0)
    p.rect(hx - 1, hy - 1.6, hx + 1, hy, (60, 58, 60))
    return hx, hy


def mega_foundry() -> Sprite:
    """A great iron foundry: a long casting hall under a slate roof with a lantern along its ridge,
    every window glowing, three tall cupolas on a charging deck throwing sparks, a yellow
    travelling crane pouring a ladle of molten iron into glowing moulds on the sand, a great
    flywheel and pig iron laid out, two chimneys smoking and a wagon of castings on the siding."""
    p = Painter(64, 72, (31, 56))
    for x, foot, tip in ((-15, -17, -45), (-3, -14, -42)):
        _smoking_chimney(p, x, foot, tip, shade(BRICK, 0.7), close=True)
    p.shadow(27, 6)
    foot = _footing(-4, 0, 28, 10)
    _bed(p, foot, SOIL)
    _charging_deck(p, 22, -8, 12, 13)
    for x, y, tall in ((17.5, -10.5, 24), (21.5, -8.5, 22), (25.5, -6.5, 20)):
        _cupola(p, x, y, tall)
    top = _foundry_hall(p, -4, 0, 28, 10, 9, 5)
    rx, ry = _mid(_mid(top[0], top[3]), _mid(top[1], top[2]))
    lantern = p.hall(rx, ry - 3.5, 18, 3, 2.4, FOUNDRY_WALL, None)
    _windows(p, lantern[0], lantern[1], (0.2, 0.35, 0.5, 0.65, 0.8), (0.8,), 2, MOLTEN)
    p.gable(lantern, 1.6, FOUNDRY_ROOF, FOUNDRY_WALL)
    _pour(p, (24, -6.5), (19, 2))
    _crucible(p, 18.5, 4)
    _casting_bed(p, -16, 10, 10, 4)
    hx, hy = _gantry(p, (-26, 5), (-11, 12.5), 15, 0.5)
    p.line([(hx, hy), (hx, hy + 4)], IRON)
    p.ellipse(hx - 2.2, hy + 4, hx + 2.2, hy + 8, CASTING)
    p.blob(hx - 1.4, hy + 3.8, hx + 1.4, hy + 5, MOLTEN)
    p.polygon(
        [(hx + 1.6, hy + 5), (hx + 2.6, hy + 5), (hx + 2.2, hy + 13), (hx + 1.6, hy + 13)], MOLTEN
    )
    p.line([(hx + 2, hy + 6), (hx + 2, hy + 12)], WHITE_HOT)
    p.blob(0, 6, 16, 13, SAND)
    _wheel(p, 5, 11.5, 3.5)
    _wheel(p, 10.5, 11.5, 2)
    for x, y in ((12.5, 8), (14.5, 9)):
        p.rect(x - 2.5, y - 1, x + 2.5, y, shade(CASTING, 1.3))
    _sparks(
        p, ((13, -40), (19, -42), (25, -38), (16, -45), (23, -46), (28, -33), (-14, 6), (-12, 3))
    )
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
    iron and sooted at their smoking tops, their furnace mouths glowing in deep arches; a long
    brick glasshouse joining them whose arched windows and ridge vent glow with the furnaces; a
    long lehr shed with amber windows and flues; a brick warehouse; a boarded packing house; and
    a yard of crates of bottles in green and amber glass, racks of window panes glinting aqua,
    and heaps of sand and soda."""
    m = METRE
    p = Painter(58, 56, (28, 40))

    def at(u: float, v: float) -> Point:
        return _iso(0, 0, (u - 22) * m, (v - 8) * m)

    _glass_yard(p, [(at(18, 6), 16), (at(6, -2), 8), (at(34, 2), 10), (at(38, 12), 8)])
    for u, v, length, depth in ((22, 8, 16, 9), (40, 8, 14, 5), (37, 17, 10, 6), (40, -2, 8, 5)):
        _bed(p, _footing(*at(u, v), length * m, depth * m), margin=0.8)
    _glass_cone(p, *at(26, 18), 6.5 * m, 22 * m)
    _glass_house(p, *at(37, 17), (10 * m, 6 * m, 6 * m))
    _glass_cone(p, *at(8, 9), 7.5 * m, 26 * m)
    _glass_house(p, *at(22, 8), (16 * m, 9 * m, 7 * m))
    _glass_lehr(p, *at(40, 8), 14 * m, 5 * m)
    _glass_packing(p, *at(40, -2), 8 * m, 5 * m)
    _glass_heap(p, *at(-2, 2), 3.2 * m, 2.8 * m, GLASS_SAND)
    _glass_heap(p, *at(0.5, -2), 2.2 * m, 1.9 * m, GLASS_SODA)
    for u, v in ((11, -4), (15.5, -4.5), (20, -5)):
        _glass_rack(p, *at(u, v))
    _glass_crates(p, *at(26, -3), (GLASS_GREEN, GLASS_AMBER, GLASS_GREEN, GLASS_AMBER))
    _glass_crates(p, *at(26.6, -4.6), (GLASS_AMBER, GLASS_GREEN, GLASS_AMBER))
    return p.sprite()


def mega_glassworks() -> Sprite:
    """A great glassworks round three English glass cones of brick, the middle one 28 m high,
    each hooped with iron, sooted at its smoking top and glowing at its furnace mouth; a long
    brick glasshouse under a roof glazed in aqua panes joining them, its arched windows and
    ridge vent glowing; two long lehr sheds with amber windows and flues; a boarded packing house
    and a brick warehouse; and a yard of crates of bottles in green and amber glass, racks of
    window panes glinting aqua, and heaps of sand and soda."""
    m = METRE
    p = Painter(64, 62, (31, 45))

    def at(u: float, v: float) -> Point:
        return _iso(0, 0, (u - 25) * m, (v - 9) * m)

    _glass_yard(
        p, [(at(22, 7), 18), (at(6, -2), 9), (at(42, 3), 11), (at(44, 15), 9), (at(16, 21), 9)]
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
    _glass_house(p, *at(26, 8), (22 * m, 9 * m, 7.5 * m), glazed=True)
    _glass_lehr(p, *at(45, 6), 16 * m, 5 * m)
    _glass_packing(p, *at(40, -3), 9 * m, 5 * m)
    _glass_heap(p, *at(-3, 2), 3.4 * m, 3 * m, GLASS_SAND)
    _glass_heap(p, *at(0, -2.5), 2.4 * m, 2 * m, GLASS_SODA)
    for u, v in ((10, -4), (14.5, -4.5), (19, -5), (23.5, -5.5)):
        _glass_rack(p, *at(u, v))
    _glass_crates(p, *at(28, -3), (GLASS_GREEN, GLASS_AMBER, GLASS_GREEN, GLASS_AMBER, GLASS_GREEN))
    _glass_crates(p, *at(28.6, -4.6), (GLASS_AMBER, GLASS_GREEN, GLASS_AMBER, GLASS_GREEN))
    return p.sprite()


KILN_BRICK: RGB = (143, 73, 51)
TILE: RGB = (70, 120, 170)
GLAZE: RGB = (236, 236, 226)
TILE_ROOF: RGB = (88, 84, 92)


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


def _tile_stack(p: Painter, x: float, y: float, rows: int = 3) -> None:
    """A stack of glazed blue stove tiles on (x, y), white glaze showing between the rows."""
    top = p.hall(x, y, 3, 2.4, 0.9 * rows, TILE, shade(TILE, 1.25))
    left, front, right, _ = top
    for row in range(1, rows):
        _bands(p, left, front, right, 0.9 * row, GLAZE, shade(GLAZE, 0.8))


def _checker(
    p: Painter, a: Point, b: Point, span: tuple[float, float], drop: tuple[float, float], cols: int
) -> None:
    """A field of blue and white tiles in a chequer on the wall under the edge from `a` to `b`."""
    (t0, t1), (d0, d1) = span, drop
    rows = max(1, round((d1 - d0) / 1.2))
    for i in range(cols):
        for j in range(rows):
            colour = TILE if (i + j) % 2 else GLAZE
            _patch(
                p,
                a,
                b,
                (t0 + (t1 - t0) * i / cols, t0 + (t1 - t0) * (i + 1) / cols),
                (d0 + (d1 - d0) * j / rows, d0 + (d1 - d0) * (j + 1) / rows),
                colour,
            )


def _tile_hall(
    p: Painter, x: float, y: float, length: float, depth: float, height: float, rise: float
) -> Top:
    """A brick tile works hall under a slate roof, sooted at its end, with a door there."""
    top = p.hall(x, y, length, depth, height, BRICK, None)
    _, front, right, _ = top
    _plinth(p, top, height, BRICK)
    _stains(p, front, right, height, (0.2, 0.8), shade(BRICK, 0.6), 1.4)
    _door(p, front, right, (0.3, 0.7), height - 4.5, height, DOORWAY, shade(BRICK, 1.3))
    p.gable(top, rise, TILE_ROOF, BRICK)
    _roof_rows(p, top, rise, TILE_ROOF)
    return top


def _open_shed(
    p: Painter, x: float, y: float, length: float, depth: float, height: float, rack: RGB
) -> None:
    """An open-sided timber drying shed under a low roof, racks of `rack` drying on its shelves."""
    left, front, right, back = _footing(x, y, length, depth)
    post = shade(TIMBER, 0.75)
    for corner in (back, left, right):
        p.line([corner, (corner[0], corner[1] - height)], post)
    for d in (height * 0.35, height * 0.7):
        p.line([_on_wall(left, front, 0.05, -d + 0.4), _on_wall(left, front, 0.95, -d + 0.4)], rack)
        p.line(
            [_on_wall(left, front, 0.05, -d), _on_wall(left, front, 0.95, -d)], shade(TIMBER, 0.9)
        )
    for t in (0.5, 1.0):
        px, py = _on_wall(left, front, t, 0)
        p.line([(px, py), (px, py - height)], post)
    top = tuple((cx, cy - height) for cx, cy in (left, front, right, back))
    p.gable(top, 2, (120, 96, 72), TIMBER)


def big_tile_works() -> Sprite:
    """A long brick tile works under a slate roof with a chequered board of blue and white tiles
    over its windows, two brick bottle kilns behind it, one glowing at its mouth, both smoking,
    and stacks of glazed blue stove tiles laid out before it."""
    p = Painter(50, 52, (24, 40))
    p.shadow(20, 5)
    _bottle_oven(p, 9, -10, 1.15, KILN_BRICK, True)
    _bottle_oven(p, 16, -5, 1.0, KILN_BRICK)
    foot = _footing(-4, 0, 18, 8)
    _bed(p, foot, SOIL)
    _track(p, foot[1], foot[2], (0.3, 0.7), 4)
    top = _tile_hall(p, -4, 0, 18, 8, 7, 4)
    left, front, _, _ = top
    _checker(p, left, front, (0.1, 0.9), (0.8, 2.6), 12)
    _windows(p, left, front, (0.15, 0.3, 0.45, 0.6, 0.75, 0.9), (3.6,), 2, shade(WINDOW, 1.05))
    for x, y in ((-20, 4), (-16.5, 6), (-13, 8), (-17, 9.5)):
        _tile_stack(p, x, y, 3)
    _tile_stack(p, 14, 7, 2)
    return p.sprite()


def mega_tile_works() -> Sprite:
    """A great tile works: a long brick hall whose whole front is clad in a chequer of blue and
    white glazed tiles, three tall bottle kilns behind it glowing and smoking, an open drying
    shed of tiles on racks, and rows of stacked blue stove tiles on the yard."""
    p = Painter(62, 64, (30, 48))
    p.shadow(26, 6)
    for x, y, s, glow in ((-10, -14, 1.3, True), (4, -15, 1.45, False), (17, -9, 1.25, True)):
        _bottle_oven(p, x, y, s, KILN_BRICK, glow)
    foot = _footing(-6, 0, 26, 9)
    _bed(p, foot, SOIL)
    _track(p, foot[1], foot[2], (0.3, 0.7), 4)
    top = _tile_hall(p, -6, 0, 26, 9, 9, 5)
    left, front, _, _ = top
    _checker(p, left, front, (0.03, 0.97), (0.8, 7.6), 22)
    spans = (0.12, 0.26, 0.4, 0.6, 0.74, 0.88)
    _windows(p, left, front, spans, (2.2,), 2, shade(WINDOW, 1.05))
    _windows(p, left, front, spans, (5.4,), 2, shade(WINDOW, 1.05))
    _door(p, left, front, (0.46, 0.54), 4, 9, DOORWAY)
    _open_shed(p, 19, 5, 10, 5, 5, TILE)
    for x, y in ((-26, 4), (-22.5, 6), (-19, 8), (-15.5, 10), (-23, 9), (-19.5, 11)):
        _tile_stack(p, x, y, 3)
    for x, y in ((8, 12), (11.5, 13.5)):
        _tile_stack(p, x, y, 2)
    return p.sprite()


POT: RGB = (176, 104, 70)
POTTER_ROOF: RGB = (150, 70, 52)
POT_KILN: RGB = (151, 77, 54)
CLAY: RGB = (156, 128, 100)


def _pots(p: Painter, x: float, y: float, count: int, step: float = 2.4) -> None:
    """A row of terracotta pots set out to dry from (x, y)."""
    for k in range(count):
        px, py = x + step * k, y + step * k * 0.3
        p.ellipse(px - 1, py - 2.5, px + 1, py, POT)
        p.line([(px - 0.8, py - 2.3), (px + 0.8, py - 2.3)], shade(POT, 0.6))


def _urn(p: Painter, x: float, y: float, s: float = 1) -> None:
    """A big terracotta storage jar standing on (x, y)."""
    _turned(
        p,
        x,
        y,
        [
            (0, 0.8 * s),
            (1.4 * s, 1.6 * s),
            (3 * s, 1.4 * s),
            (3.8 * s, 0.7 * s),
            (4.3 * s, 0.9 * s),
        ],
        POT,
        shade(POT, 0.55),
    )


def _plank(p: Painter, x: float, y: float, length: float, pots: int) -> None:
    """A drying plank on two trestles running the lit way from (x, y), pots set out along it."""
    a, b = (x, y), (x + length, y + length / 2)
    for t in (0.1, 0.9):
        px, py = _on_wall(a, b, t, 0)
        p.line([(px - 0.6, py), (px, py - 2)], shade(TIMBER, 0.7))
        p.line([(px + 0.6, py), (px, py - 2)], shade(TIMBER, 0.7))
    _band(p, a, b, -2, TIMBER)
    for k in range(pots):
        px, py = _on_wall(a, b, (k + 0.5) / pots, -2.2)
        p.ellipse(px - 0.9, py - 2, px + 0.9, py, POT)


def _beehive(p: Painter, x: float, y: float, r: float, glowing: bool = True) -> None:
    """A round domed brick kiln standing on (x, y) with a stubby stack on its crown smoking and
    its firing mouth glowing."""
    _turned(p, x, y, [(0, r), (r * 0.5, r), (r * 0.95, r * 0.7), (r * 1.2, r * 0.3)], POT_KILN)
    _ring(p, x, y - r * 0.5, r, shade(POT_KILN, 0.7))
    p.rect(x - 1.2, y - r * 1.2 - 3, x + 1.2, y - r * 1.2 + 0.5, shade(POT_KILN, 0.8))
    _smoke(p, x + 0.5, y - r * 1.2 - 5.5, SMOKE, 2)
    p.ellipse(x - r * 0.6, y - r * 0.6, x - r * 0.05, y + 0.6, (60, 40, 34))
    if glowing:
        p.blob(x - r * 0.5, y - r * 0.45, x - r * 0.15, y + 0.4, GLOW)


def _potter_house(
    p: Painter, x: float, y: float, length: float, depth: float, height: float, storeys: int
) -> Top:
    """A whitewashed potter's workshop under a red tiled roof, with rows of windows, a door and a
    sign of a pot."""
    top = p.hall(x, y, length, depth, height, WHITEWASH, None)
    left, front, right, _ = top
    _plinth(p, top, height, WHITEWASH)
    spans = (0.12, 0.3, 0.7, 0.88)
    for k in range(storeys):
        for t in spans:
            _window(p, left, front, (t - 0.05, t + 0.05), 1.6 + 3.4 * k, shade(WINDOW, 1.05))
    _door(p, left, front, (0.44, 0.56), height - 4, height, (120, 70, 44))
    _patch(p, left, front, (0.42, 0.58), (height - 6, height - 4.6), POT)
    _window(p, front, right, (0.3, 0.7), 2, shade(WINDOW, 0.85))
    p.gable(top, 4, POTTER_ROOF, WHITEWASH)
    _roof_rows(p, top, 4, POTTER_ROOF)
    return top


def big_potter() -> Sprite:
    """A long whitewashed potter's workshop under a red roof with a pot for a sign, a brick bottle
    kiln smoking behind it and a domed kiln glowing beside it, planks of pots drying on trestles,
    big storage jars by the door and a heap of clay."""
    p = Painter(46, 48, (22, 36))
    p.shadow(18, 4)
    _bottle_oven(p, 9, -9, 1.0, POT_KILN)
    foot = _footing(-4, -1, 14, 7)
    _bed(p, foot, SOIL, 1)
    _track(p, foot[0], foot[1], (0.4, 0.6), 3, (190, 172, 136), ruts=False)
    _potter_house(p, -4, -1, 14, 7, 6, 1)
    _beehive(p, 14, 2, 5)
    _heap(p, -17, 0, 3, 2.5, CLAY)
    _plank(p, -20, 4, 8, 3)
    _urn(p, 4, 8)
    _urn(p, 7, 9.5, 0.8)
    _pots(p, -10, 9, 3)
    return p.sprite()


def mega_potter() -> Sprite:
    """A great pottery: a long two-storey whitewashed workshop range under a red roof with a pot for
    a sign, two brick bottle kilns and a domed kiln behind and beside it glowing and smoking, an
    open drying shed of pots on racks, rows of big storage jars, planks of pots drying, a cart of
    pots and a heap of clay."""
    p = Painter(60, 58, (29, 42))
    p.shadow(24, 5)
    _bottle_oven(p, -8, -14, 1.25, POT_KILN, True)
    _bottle_oven(p, 6, -13, 1.1, POT_KILN)
    _heap(p, -22, -4, 3.5, 3, CLAY)
    foot = _footing(-4, -1, 22, 8)
    _bed(p, foot, SOIL, 1)
    _track(p, foot[0], foot[1], (0.44, 0.56), 4, (190, 172, 136), ruts=False)
    _potter_house(p, -4, -1, 22, 8, 9, 2)
    _beehive(p, 17, -1, 6)
    _open_shed(p, 20, 9, 9, 4, 5, POT)
    _plank(p, -26, 3, 9, 4)
    _plank(p, -20, 7, 9, 4)
    for x, y, s in ((2, 9, 1.0), (5, 10.5, 1.0), (8, 12, 0.9), (-1, 11, 0.8)):
        _urn(p, x, y, s)
    cx, cy = -6, 13
    p.rect(cx - 4, cy - 2, cx + 3, cy, (120, 84, 56))
    _pots(p, cx - 3, cy - 1.8, 3, 2.2)
    for wx in (cx - 3, cx + 2):
        p.ellipse(wx - 1, cy - 0.5, wx + 1, cy + 1.5, (70, 54, 40))
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


HOFFMANN: RGB = (152, 98, 76)
KILN_TOP: RGB = (96, 86, 86)
BRICK_CLAY: RGB = (176, 122, 86)
BRICK_DUST: RGB = (146, 104, 82)


def _brick_stack(p: Painter, x: float, y: float, length: float = 4) -> None:
    """A stack of new red bricks on (x, y), their courses and joints showing."""
    stack = p.hall(x, y, length, 3, 3, BRICK, shade(BRICK, 1.25))
    for a, b in ((stack[0], stack[1]), (stack[1], stack[2])):
        _band(p, a, b, 1.5, shade(BRICK, 1.3))
        for t in (0.25, 0.5, 0.75):
            d = (0, 1.5) if t == 0.5 else (1.5, 3)
            p.line([_on_wall(a, b, t, d[0]), _on_wall(a, b, t, d[1])], shade(BRICK, 0.7))


def _firing_doors(p: Painter, a: Point, b: Point, spans: tuple[float, ...], height: float) -> None:
    """Dark arched firing doors along the wall under the edge from `a` to `b`, sooted above."""
    for t in spans:
        _patch(p, a, b, (t - 0.03, t + 0.03), (height - 3.6, height), (52, 36, 32))
        p.point(*_on_wall(a, b, t, height - 4.4), shade(HOFFMANN, 1.35))
        p.line(
            [_on_wall(a, b, t, height - 5), _on_wall(a, b, t, height - 5.6)], shade(HOFFMANN, 0.6)
        )


def _long_kiln(p: Painter, x: float, y: float, length: float, depth: float, chimney: float) -> Top:
    """A long Hoffmann kiln: a low brick block with a row of arched firing doors sooted above along
    its sides, feed holes along its low roof and a tall banded chimney rising from its middle to
    `chimney` above the ground, smoking."""
    height = 6
    top = p.hall(x, y, length, depth, height, HOFFMANN, None)
    left, front, right, _ = top
    _bands(p, left, front, right, 0.8, shade(HOFFMANN, 0.7), shade(HOFFMANN, 0.55))
    steps = max(3, round(length / 4))
    _firing_doors(p, left, front, tuple((k + 0.5) / steps for k in range(steps)), height)
    _firing_doors(p, front, right, (0.35, 0.65), height)
    p.gable(top, 2.5, KILN_TOP, HOFFMANN)
    for along in (0.15, 0.35, 0.65, 0.85):
        p.point(*_on_roof(top, 2.5, along, 0.5), (60, 54, 54))
    cx, cy = _mid(_mid(top[0], top[3]), _mid(top[1], top[2]))
    _smoking_chimney(p, cx, cy - 2, y - chimney, shade(BRICK, 0.9))
    return top


def _round_kiln(p: Painter, x: float, y: float, r: float, chimney: float) -> None:
    """A round Hoffmann kiln ringed with arched firing doors sooted above, round its tall banded
    chimney rising to `chimney` above the ground, feed holes on its roof."""
    _turned(p, x, y, [(0, r), (6, r)], HOFFMANN)
    _ring(p, x, y - 0.8, r, shade(HOFFMANN, 0.7))
    for k in range(7):
        a = 0.3 + k * 0.42
        ax, ay = x + r * math.cos(a), y + r / 2 * math.sin(a)
        p.rect(ax - 1, ay - 4, ax + 1, ay, (52, 36, 32))
        p.point(ax - 0.5, ay - 5, shade(HOFFMANN, 1.35))
        p.line([(ax, ay - 5.5), (ax, ay - 6)], shade(HOFFMANN, 0.62))
    _turned(p, x, y - 6, [(0, r + 1), (3, r - 4)], KILN_TOP, (118, 106, 104))
    for k in range(6):
        a = 0.4 + k * 0.5
        p.point(x + (r - 1) * math.cos(a), y - 7.5 + (r - 1) / 2 * math.sin(a), (60, 54, 54))
    _smoking_chimney(p, x, y - 8, y - chimney, shade(BRICK, 0.9))


def _clay_heap(p: Painter, x: float, y: float, half: float) -> None:
    _spill(p, x, y, half, BRICK_CLAY)
    _heap(p, x, y, half, half * 0.85, BRICK_CLAY)


def _tub(p: Painter, x: float, y: float) -> None:
    """A little tipping tub of clay on narrow-gauge wheels at (x, y)."""
    p.polygon([(x - 2, y - 2.6), (x + 2, y - 2.6), (x + 1.4, y - 0.8), (x - 1.4, y - 0.8)], IRON)
    p.blob(x - 1.8, y - 3.4, x + 1.8, y - 2.2, BRICK_CLAY)
    for dx in (-1, 1):
        p.blob(x + dx - 0.6, y - 1, x + dx + 0.6, y + 0.2, OVEN_DOOR)


def big_brickworks() -> Sprite:
    """A long Hoffmann kiln of arched firing doors sooted above, its tall banded chimney smoking
    from the middle of its roof, open drying sheds of bricks on racks behind, a tub of clay on
    its narrow-gauge line from a heap of clay, and stacks of new red bricks on ground red with
    brick dust."""
    p = Painter(54, 70, (26, 57))
    p.blob(-24, -10, 22, 10, SOIL)
    p.blob(2, 2, 22, 11, BRICK_DUST)
    _open_shed(p, 15, -11, 12, 4, 4, BRICK)
    p.shadow(20, 5)
    foot = _footing(-3, -2, 24, 9)
    _track(p, foot[0], foot[1], (0.45, 0.62), 4)
    _long_kiln(p, -3, -2, 24, 9, 40)
    _rails_narrow(p, (-22, 3), (-8, 10))
    _clay_heap(p, -19, 4, 3.5)
    _tub(p, -13, 9)
    for x, y in ((14, 3), (18.5, 5.5), (9.5, 7), (14, 9.5)):
        _brick_stack(p, x, y)
    return p.sprite()


def _rails_narrow(p: Painter, a: Point, b: Point) -> None:
    """A narrow-gauge line of two thin rails over the ground from `a` to `b`."""
    for reach in (0, 1.4):
        p.line([_out(a, reach), _out(b, reach)], RAIL)


def mega_brickworks() -> Sprite:
    """A great brickworks: a round Hoffmann kiln and a long one behind it, each ringed with arched
    firing doors and smoking from its own tall banded chimney, an open drying shed of bricks on
    racks, a great heap of clay with tubs on the narrow-gauge line and rows of new red bricks
    stacked high."""
    p = Painter(64, 80, (31, 66))
    p.blob(-30, -16, 30, 10, SOIL)
    p.blob(-4, 2, 26, 12, BRICK_DUST)
    _open_shed(p, 20, -14, 12, 4, 4, BRICK)
    p.shadow(27, 6)
    _long_kiln(p, 10, -9, 22, 9, 44)
    _round_kiln(p, -14, -2, 11, 42)
    _rails_narrow(p, (-27, 4), (-12, 11.5))
    _clay_heap(p, -23, 5, 4.5)
    _tub(p, -17, 9.8)
    _tub(p, -13.5, 11.5)
    for x, y in ((3, 5), (7.5, 7.5), (12, 10), (8, 3), (12.5, 5.5), (17, 8), (21.5, 6), (17, 3.5)):
        _brick_stack(p, x, y)
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
