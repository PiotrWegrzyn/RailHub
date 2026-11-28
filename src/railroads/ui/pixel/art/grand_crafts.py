"""Big and mega works of the crafts grown into industry, each drawn as a building of its own:
yellow-brick cotton mills of Łódź with dyed cloth drying, carpet halls hung with rugs, tanneries
of drying lofts and tan pits, shoe factories under giant boots, furniture works in lumber yards,
paper mills turning waterwheels, newspaper press houses under big signs, glowing forges of tool
works and great engine halls of machine works under gantry cranes."""

import math

from railroads.ui.pixel.art.crafts import (
    BUCKLE,
    HARNESS,
    SADDLE,
    SADDLE_TAN,
    SADDLER_SIGN,
    _collar,
    _saddle,
)
from railroads.ui.pixel.art.industry import (
    BUNDLE,
    FRAME,
    GLAZING,
    JOINER_WALL,
    SMOKE,
    SOIL,
    _anvil,
    _bed,
    _bundle,
    _chair,
    _chimney,
    _crate,
    _door,
    _drying_cloth,
    _footing,
    _forge_stack,
    _gear,
    _hide_frame,
    _lengths,
    _locomotive,
    _oil,
    _on_roof,
    _plank_walls,
    _puddle,
    _ring,
    _roof_rows,
    _sawtooth,
    _skylight,
    _smoke,
    _smoking_chimney,
    _spill,
    _spool,
    _stack,
    _stains,
    _track,
    _window,
)
from railroads.ui.pixel.art.painter import (
    BRICK,
    GLASS,
    RGB,
    VERDIGRIS,
    WINDOW,
    Painter,
    Point,
    Sprite,
    Top,
    shade,
)
from railroads.ui.pixel.art.parts import (
    _band,
    _bands,
    _barrel,
    _flag,
    _heap,
    _on_wall,
    _patch,
    _plinth,
    _turned,
    _windows,
)

MILL_WALL: RGB = (206, 164, 92)
MILL_ROOF: RGB = (80, 76, 84)
MILL_DOOR: RGB = (60, 44, 36)
CLOTH_BALE: RGB = (232, 220, 196)

CARPET_WALL: RGB = (214, 200, 176)
CARPET_SIGN: RGB = (150, 40, 40)
CARPET_ROOF: RGB = (88, 84, 92)
RUGS: list[tuple[RGB, RGB]] = [
    ((176, 40, 44), (232, 190, 80)),
    ((50, 80, 160), (220, 70, 60)),
    ((40, 110, 90), (236, 214, 150)),
    ((120, 40, 90), (232, 190, 80)),
]
GOLD: RGB = (232, 190, 80)

TAN_WALL: RGB = (184, 160, 124)
TAN_ROOF: RGB = (96, 82, 74)
HIDES: list[RGB] = [(198, 150, 100), (162, 108, 68), (214, 180, 130), (140, 92, 58)]
BARK: RGB = (140, 92, 58)
SODDEN: RGB = (92, 84, 56)
PIT_RIM: RGB = (138, 116, 86)
LIQUORS: list[RGB] = [(96, 66, 40), (70, 50, 34), (214, 210, 190), (120, 84, 48)]

SHOE_WALL: RGB = (204, 170, 116)
SHOE_ROOF: RGB = (124, 66, 52)
BOOT: RGB = (140, 54, 40)
SHOE_BOX: RGB = (196, 160, 112)
PAVING: RGB = (176, 170, 158)

BOARD: RGB = (226, 190, 134)
JOINER_ROOF: RGB = (98, 74, 58)
SAWDUST: RGB = (222, 196, 140)
LEG: RGB = (110, 80, 56)

PAPER_WALL: RGB = (192, 168, 136)
PAPER_ROOF: RGB = (100, 86, 80)
WATER: RGB = (84, 140, 196)
BANK: RGB = (96, 88, 62)
WHEEL: RGB = (98, 72, 50)
FOAM: RGB = (226, 238, 244)
ROLL: RGB = (240, 240, 234)
LOG: RGB = (196, 160, 112)
STEAM: RGB = (232, 232, 236)

PRESS_WALL: RGB = (220, 204, 172)
PRESS_ROOF: RGB = (110, 106, 116)
PAGE: RGB = (242, 240, 230)
INK: RGB = (40, 40, 48)
PRINT: RGB = (150, 150, 160)
MASTHEAD: RGB = (170, 40, 40)

FORGE_WALL: RGB = (92, 84, 82)
FORGE_ROOF: RGB = (60, 60, 68)
FORGE_GROUND: RGB = (62, 58, 56)
GLOW: RGB = (238, 132, 40)
HOT: RGB = (255, 208, 112)
COAL: RGB = (48, 46, 50)
SIGNBOARD: RGB = (214, 202, 176)
BAR: RGB = (120, 122, 130)
HANDLE: RGB = (150, 112, 74)

ENGINE_ROOF: RGB = (82, 84, 96)
BRASS: RGB = (222, 180, 74)
CRANE: RGB = (214, 168, 40)
GRAVEL: RGB = (140, 134, 120)
CASTING: RGB = (88, 90, 98)
RAIL: RGB = (96, 92, 90)
SLEEPER: RGB = (110, 84, 60)
DOORWAY_PRESS: RGB = (54, 46, 48)
METAL_GREEN: RGB = (60, 100, 80)


def _rails(p: Painter, a: Point, b: Point, ties: int = 6) -> None:
    """A short rail siding on sleepers across the ground from `a` to `b`."""
    for k in range(ties + 1):
        x, y = _on_wall(a, b, k / ties, 0)
        p.line([(x - 1, y - 0.5), (x + 1, y + 0.5)], SLEEPER)
    for off in (-0.6, 0.6):
        p.line([(a[0] + off, a[1] + off / 2), (b[0] + off, b[1] + off / 2)], RAIL)


def _wagon(p: Painter, x: float, y: float, colour: RGB) -> None:
    """A low open goods wagon on (x, y), its side lit."""
    p.rect(x - 3.5, y - 3, x + 3.5, y - 1, colour)
    p.line([(x - 3, y - 2.2), (x + 3, y - 2.2)], shade(colour, 1.25))
    for wx in (x - 2, x + 2):
        p.ellipse(wx - 0.9, y - 1.5, wx + 0.9, y + 0.3, (40, 38, 40))


def _cart(p: Painter, x: float, y: float) -> None:
    """A two-wheeled wooden handcart standing on (x, y), its shafts down on the ground."""
    p.rect(x - 3, y - 3, x + 3, y - 1.5, (130, 92, 60))
    p.line([(x + 3, y - 2), (x + 6, y)], (110, 80, 56))
    p.ellipse(x - 1.6, y - 2.4, x + 0.6, y + 0.2, (70, 54, 40))


def _mill_block(
    p: Painter, x: float, y: float, length: float, depth: float, height: float, storeys: int
) -> Top:
    """A yellow-brick mill of `storeys` floors of small arched windows over a dark plinth stained
    by the damp, with a cornice and string courses, its flat roof behind a parapet."""
    top = p.hall(x, y, length, depth, height, MILL_WALL, MILL_ROOF)
    left, front, right, _ = top
    lintel = shade(MILL_WALL, 1.3)
    _plinth(p, top, height, MILL_WALL)
    _stains(p, left, front, height, (0.04, 0.3, 0.56, 0.82), shade(MILL_WALL, 0.6), 2)
    _stains(p, front, right, height, (0.2, 0.8), shade(MILL_WALL, 0.5), 2)
    _bands(p, left, front, right, 1, shade(MILL_WALL, 1.2), shade(MILL_WALL, 0.95))
    p.line([left, front, right], shade(MILL_ROOF, 1.3))
    floor = (height - 2.5) / storeys
    drops = tuple(2.8 + floor * f for f in range(storeys))
    n = max(3, round(length / 3.2))
    spans = tuple((k + 0.5) / n for k in range(n))
    _windows(p, left, front, spans, drops, 2, arch=lintel)
    m = max(2, round(depth / 3.4))
    side = tuple((k + 0.5) / m for k in range(m))
    _windows(p, front, right, side, drops, 2, shade(WINDOW, 0.75), lintel)
    for f in range(1, storeys):
        _bands(p, left, front, right, drops[f] - 1.3, shade(MILL_WALL, 0.88), shade(MILL_WALL, 0.7))
    return top


def _mill_tower(p: Painter, x: float, y: float, half: float, tall: float, dome: bool) -> None:
    """A square yellow-brick stair tower standing on (x, y) with narrow arched windows, a
    battlemented parapet and either a green copper dome or a dark pointed cap."""
    top = p.box(x, y, half, tall, MILL_WALL, MILL_ROOF)
    left, front, right, _ = top
    lintel = shade(MILL_WALL, 1.3)
    _plinth(p, top, tall, MILL_WALL)
    drops = tuple(3 + 4.2 * k for k in range(int((tall - 4) / 4.2)))
    _windows(p, left, front, (0.5,), drops, 2, arch=lintel)
    _windows(p, front, right, (0.5,), drops, 2, shade(WINDOW, 0.75), lintel)
    _bands(p, left, front, right, 1.2, shade(MILL_WALL, 0.8), shade(MILL_WALL, 0.65))
    for a, b in ((left, front), (front, right)):
        for t in (0.0, 0.5, 1.0):
            mx, my = _on_wall(a, b, t, 0)
            p.rect(mx - 0.6, my - 1.4, mx + 0.6, my, shade(MILL_WALL, 0.95))
    cx, cy = x, y - tall
    if dome:
        _turned(
            p,
            cx,
            cy,
            [(0, half * 0.8), (half * 0.6, half * 0.75), (half * 1.1, half * 0.4), (half * 1.4, 0)],
            VERDIGRIS,
        )
        p.line([(cx, cy - half * 1.4), (cx, cy - half * 1.4 - 2.5)], GOLD)
    else:
        apex = (cx, cy - half * 1.6)
        p.polygon([left, front, apex], MILL_ROOF, shade(MILL_ROOF, 0.6))
        p.polygon([front, right, apex], shade(MILL_ROOF, 0.75), shade(MILL_ROOF, 0.5))
        p.line([apex, (apex[0], apex[1] - 2)], (70, 66, 62))


def _weaving_shed(p: Painter, x: float, y: float, length: float, depth: float, teeth: int) -> Top:
    """A low yellow-brick weaving shed under a sawtooth roof glazed along its teeth."""
    top = p.hall(x, y, length, depth, 5, MILL_WALL, None)
    left, front, right, _ = top
    _plinth(p, top, 5, MILL_WALL)
    _stains(p, front, right, 5, (0.15, 0.9), shade(MILL_WALL, 0.5), 1.2)
    n = max(3, round(length / 3))
    _windows(
        p,
        left,
        front,
        tuple((k + 0.5) / n for k in range(n)),
        (2.5,),
        1,
        arch=shade(MILL_WALL, 1.3),
    )
    _sawtooth(p, top, teeth, 4, (104, 100, 108), GLAZING, MILL_WALL)
    return top


def _bales(p: Painter, x: float, y: float, count: int) -> None:
    """Bales of grey cloth piled on the ground from (x, y)."""
    for k in range(count):
        bx, by = x + 2.6 * (k % 3), y + 1.3 * (k % 3) - 2 * (k // 3)
        p.box(bx, by, 1.6, 1.8, CLOTH_BALE, shade(CLOTH_BALE, 1.06))
        p.line([(bx - 1.6, by - 0.9), (bx, by - 0.1)], shade(CLOTH_BALE, 0.7))


def big_textile_mill() -> Sprite:
    """A bigger yellow-brick cotton mill of Łódź: a long block of five storeys of small arched
    windows behind a stair tower with a dark pointed cap, a long sawtooth weaving shed in front of
    it, a tall banded chimney smoking, two racks of cloth in every dye hung out to dry and bales of
    cloth by the door."""
    p = Painter(52, 74, (25, 56))
    mill_foot, shed_foot = _footing(-4, -3, 22, 9), _footing(11, 6, 12, 9)
    _bed(p, mill_foot)
    _bed(p, shed_foot)
    p.shadow(20, 5, reach=4)
    _track(p, shed_foot[1], shed_foot[2], (0.35, 0.7), 3)
    cap = _chimney(p, 15, -3, -42, shade(MILL_WALL, 0.92))
    _ring(p, 15, -3 + (cap + 3) * 0.45, 1.5, shade(MILL_WALL, 0.7))
    _smoke(p, 16, cap - 3, puffs=3)
    top = _mill_block(p, -4, -3, 22, 9, 20, 5)
    left, front, _, _ = top
    p.line([_on_wall(left, front, 0.66, 1.5), _on_wall(left, front, 0.66, 19.6)], (70, 64, 66))
    tx, ty = _on_wall(mill_foot[0], mill_foot[1], 0.42, 1.6)
    _mill_tower(p, tx, ty, 3.2, 25, dome=False)
    _patch(p, mill_foot[0], mill_foot[1], (0.36, 0.48), (-3, 0), MILL_DOOR)
    _weaving_shed(p, 11, 6, 12, 9, 4)
    _spool(p, *_on_wall(left, front, 0.16, 4))
    _drying_cloth(p, (-24, 2), (-14, 7), 8)
    _drying_cloth(p, (-17, 8.5), (-6, 14), 7)
    _bales(p, 2, 12, 3)
    return p.sprite()


def mega_textile_mill() -> Sprite:
    """A Łódź mill palace: a great yellow-brick mill of six storeys of small arched windows with
    battlemented corner towers under green copper domes and a taller gate tower before its middle,
    long sawtooth weaving sheds beside it, two tall banded chimneys smoking, racks of cloth in
    every dye hung out to dry in rows before it and bales of cloth stacked by the door."""
    p = Painter(64, 82, (31, 62))
    mill_foot, shed_foot = _footing(-6, -5, 28, 10), _footing(18, 5, 12, 11)
    _bed(p, mill_foot)
    _bed(p, shed_foot)
    p.shadow(25, 6, reach=4)
    _track(p, shed_foot[1], shed_foot[2], (0.3, 0.6), 3)
    for x, foot, tall in ((14, -10, 38), (22, -6, 40)):
        cap = _chimney(p, x, foot, foot - tall, shade(MILL_WALL, 0.92))
        _ring(p, x, foot - (foot - cap) * 0.4, 1.5, shade(MILL_WALL, 0.7))
        _smoke(p, x + 1, cap - 3, puffs=3)
    left_corner = mill_foot[0]
    _mill_tower(p, left_corner[0] + 1, left_corner[1] + 0.2, 3.4, 29, dome=True)
    top = _mill_block(p, -6, -5, 28, 10, 24, 6)
    left, front, _, _ = top
    for t in (0.3, 0.72):
        p.line([_on_wall(left, front, t, 1.5), _on_wall(left, front, t, 23.6)], (70, 64, 66))
    _mill_tower(p, mill_foot[1][0], mill_foot[1][1] - 0.2, 3.4, 29, dome=True)
    gx, gy = _on_wall(mill_foot[0], mill_foot[1], 0.5, 2)
    _mill_tower(p, gx, gy, 4, 33, dome=False)
    _patch(p, mill_foot[0], mill_foot[1], (0.44, 0.56), (-2, 2), MILL_DOOR)
    _weaving_shed(p, 18, 5, 12, 11, 4)
    _spool(p, *_on_wall(left, front, 0.22, 5))
    _drying_cloth(p, (-30, -2), (-20, 3), 8)
    _drying_cloth(p, (-26, 5), (-14, 11), 8)
    _drying_cloth(p, (-18, 12), (-6, 18), 7)
    _bales(p, 5, 14, 5)
    return p.sprite()


def _rug(p: Painter, x: float, y: float, half: float, tall: float, field: RGB, border: RGB) -> None:
    """A patterned rug hung from a line, its foot centred on (x, y), with a border, a medallion and
    a fringe."""

    def at(dx: float, dy: float) -> Point:
        return x + dx, y + dy + dx / 2

    p.polygon([at(-half, -tall), at(half, -tall), at(half, 0), at(-half, 0)], border)
    inset = min(0.9, half * 0.35)
    p.polygon(
        [
            at(-half + inset, -tall + inset),
            at(half - inset, -tall + inset),
            at(half - inset, -inset),
            at(-half + inset, -inset),
        ],
        field,
    )
    p.point(*at(-0.5, -tall / 2 - 0.5), border)
    if tall > 6:
        p.point(*at(-0.5, -tall / 2 - 2), shade(border, 0.8))
        p.point(*at(-0.5, -tall / 2 + 1), shade(border, 0.8))
    p.line([at(-half, 0.6), at(half, 0.6)], (240, 232, 210))


def _rug_line(p: Painter, a: Point, b: Point, height: float, rugs: int, first: int = 0) -> None:
    """A line on two poles from `a` to `b` on the ground, `height` up, hung with `rugs` rugs."""
    pole = (110, 84, 60)
    for x, y in (a, b):
        p.line([(x, y), (x, y - height)], pole)
    p.line([(a[0], a[1] - height), (b[0], b[1] - height)], shade(pole, 1.2))
    span = (b[0] - a[0]) / rugs
    for k in range(rugs):
        field, border = RUGS[(first + k) % len(RUGS)]
        x0, y0 = _on_wall(a, b, (k + 0.5) / rugs, 0)
        _rug(
            p,
            x0,
            y0 - 1.2 - 0.6 * (k % 2),
            span * 0.42,
            height - 1.6 - 0.6 * (k % 2),
            field,
            border,
        )


def _wall_rug(
    p: Painter, a: Point, b: Point, span: tuple[float, float], drop: tuple[float, float], k: int
) -> None:
    """A rug hung flat on the wall under the edge from `a` to `b`, with its border and medallion."""
    field, border = RUGS[k % len(RUGS)]
    t0, t1 = span
    d0, d1 = drop
    _patch(p, a, b, span, drop, border)
    w = (t1 - t0) * 0.22
    _patch(p, a, b, (t0 + w, t1 - w), (d0 + 0.8, d1 - 0.8), field)
    p.point(*_on_wall(a, b, (t0 + t1) / 2, (d0 + d1) / 2), border)
    p.line([_on_wall(a, b, t0, d1 + 0.5), _on_wall(a, b, t1, d1 + 0.5)], (240, 232, 210))


def _rolled(p: Painter, x: float, y: float, colour: RGB) -> None:
    """A rolled carpet lying on (x, y), its end showing the spiral."""
    p.polygon(
        [(x - 4, y - 3), (x + 1, y - 0.5), (x + 1, y + 1), (x - 4, y - 1.5)],
        colour,
        shade(colour, 0.6),
    )
    p.ellipse(x - 0.2, y - 1, x + 2, y + 1.4, shade(colour, 1.2))
    p.point(x + 0.8, y + 0.1, shade(colour, 0.6))


def _carpet_hall(p: Painter, x: float, y: float, length: float, depth: float, height: float) -> Top:
    """A cream carpet house with a red sign along under its eaves and rows of windows."""
    top = p.hall(x, y, length, depth, height, CARPET_WALL, None)
    left, front, right, _ = top
    _plinth(p, top, height, CARPET_WALL)
    _stains(p, front, right, height, (0.2, 0.8), shade(CARPET_WALL, 0.55), 1.4)
    _patch(p, left, front, (0, 1), (0.4, 1.8), CARPET_SIGN)
    _patch(p, front, right, (0, 1), (0.4, 1.8), shade(CARPET_SIGN, 0.78))
    for t in (0.2, 0.4, 0.6, 0.8):
        p.point(*_on_wall(left, front, t, 1.1), GOLD)
    return top


def big_carpet_weaver() -> Sprite:
    """A long two-storey carpet house with a red sign under its eaves and a shop window of carpets,
    rugs hung out of its upper windows, a weaving shed behind it, two lines of patterned rugs out
    in the yard and rolled carpets piled by the door."""
    p = Painter(50, 44, (24, 32))
    foot = _footing(4, -1, 18, 8)
    _bed(p, foot, SOIL, 1)
    p.shadow(14, 4)
    shed = p.hall(12, -8, 10, 6, 5, shade(CARPET_WALL, 0.92), None)
    _sawtooth(p, shed, 3, 3, CARPET_ROOF, GLAZING, shade(CARPET_WALL, 0.92))
    _track(p, foot[0], foot[1], (0.8, 0.94), 3, (190, 172, 136), ruts=False)
    top = _carpet_hall(p, 4, -1, 18, 8, 11)
    left, front, right, _ = top
    for k, t in enumerate((0.1, 0.3, 0.5, 0.7)):
        _wall_rug(p, left, front, (t - 0.06, t + 0.06), (2.6, 6.2), k)
    _patch(p, left, front, (0.05, 0.7), (7.4, 9.6), shade(WINDOW, 0.9))
    for k, t in enumerate((0.15, 0.3, 0.45, 0.6)):
        p.point(*_on_wall(left, front, t, 8.6), RUGS[k % 4][0])
    _door(p, left, front, (0.8, 0.94), 6.5, 11, CARPET_SIGN)
    _window(p, front, right, (0.3, 0.7), 3, shade(WINDOW, 0.8))
    _wall_rug(p, front, right, (0.3, 0.7), (6.8, 10), 1)
    p.gable(top, 5, CARPET_ROOF, CARPET_WALL)
    _roof_rows(p, top, 5, CARPET_ROOF)
    x, y = _on_roof(top, 5, 0.3, 0.9)
    p.rect(x - 1, y - 3, x, y, BRICK)
    _rug_line(p, (-23, -4), (-12, 1.5), 9, 3)
    _rug_line(p, (-18, 4), (-7, 9.5), 8, 3, 2)
    for k, (x, y) in enumerate(((10, 9), (12, 8), (11, 6.8))):
        _rolled(p, x, y, RUGS[k][0])
    return p.sprite()


def mega_carpet_weaver() -> Sprite:
    """A great carpet hall of three storeys under a slate roof, a red and gold sign along its
    eaves and great patterned rugs hung down its whole front, a long weaving shed behind, lines of
    rugs in every colour filling the yard, a carpet laid out on the ground and a cart loaded with
    rolled carpets."""
    p = Painter(64, 58, (32, 40))
    foot = _footing(4, -3, 26, 10)
    _bed(p, foot, SOIL, 1)
    p.shadow(20, 5)
    shed = p.hall(16, -12, 14, 8, 6, shade(CARPET_WALL, 0.92), None)
    _sawtooth(p, shed, 4, 3, CARPET_ROOF, GLAZING, shade(CARPET_WALL, 0.92))
    _track(p, foot[0], foot[1], (0.86, 0.96), 4, (190, 172, 136), ruts=False)
    top = _carpet_hall(p, 4, -3, 26, 10, 16)
    left, front, right, _ = top
    for k in range(6):
        t = 0.06 + k * 0.14
        _wall_rug(p, left, front, (t, t + 0.1), (2.6, 11.8), k)
    _windows(p, left, front, tuple(0.13 + k * 0.14 for k in range(5)), (4.5, 9.5), 2)
    _door(p, left, front, (0.88, 0.97), 10.5, 16, CARPET_SIGN)
    _patch(p, left, front, (0.04, 0.84), (13, 15.2), shade(WINDOW, 0.9))
    for k in range(9):
        p.point(*_on_wall(left, front, 0.08 + k * 0.09, 14.2), RUGS[k % 4][k % 2])
    for k, t in enumerate((0.25, 0.75)):
        _wall_rug(p, front, right, (t - 0.16, t + 0.16), (2.6, 10), k + 2)
    p.gable(top, 6, CARPET_ROOF, CARPET_WALL)
    _roof_rows(p, top, 6, CARPET_ROOF, (0.25, 0.5, 0.75))
    for along in (0.25, 0.7):
        x, y = _on_roof(top, 6, along, 0.9)
        p.rect(x - 1, y - 3, x, y, BRICK)
    rx, ry = _on_roof(top, 6, 0.5, 1)
    p.rect(rx - 6, ry - 6, rx + 6, ry - 2, CARPET_SIGN)
    p.rect(rx - 5, ry - 5, rx + 5, ry - 3, RUGS[0][1])
    p.rect(rx - 3, ry - 4.6, rx + 3, ry - 3.4, CARPET_SIGN)
    _rug_line(p, (-30, -6), (-19, -0.5), 10, 3)
    _rug_line(p, (-27, 2), (-15, 8), 9, 3, 1)
    _rug_line(p, (-21, 9.5), (-9, 15.5), 8, 3, 2)
    p.polygon([(-6, 13), (2, 9), (8, 12), (0, 16)], RUGS[1][1])
    p.polygon([(-4, 13), (2, 10), (6, 12), (0, 15)], RUGS[1][0])
    p.point(1, 12.5, RUGS[1][1])
    cx, cy = 14, 13
    p.rect(cx - 5, cy - 3, cx + 5, cy - 1.5, (130, 92, 60))
    for k in range(3):
        _rolled(p, cx - 1 + 2.6 * k - 1.5, cy - 2.2 - (k % 2), RUGS[k + 1][0])
    for wx in (cx - 3, cx + 3):
        p.ellipse(wx - 1.2, cy - 1.6, wx + 1.2, cy + 0.8, (70, 54, 40))
    return p.sprite()


def _tan_shed(
    p: Painter, x: float, y: float, length: float, depth: float, height: float, loft: bool
) -> Top:
    """A tanning shed with a door, windows and a dark plinth, under a gable roof, with a slatted
    drying loft along its ridge if `loft`."""
    top = p.hall(x, y, length, depth, height, TAN_WALL, None)
    left, front, right, _ = top
    _plinth(p, top, height, TAN_WALL, 0.7)
    _stains(p, front, right, height, (0.2, 0.8), shade(TAN_WALL, 0.52), 1)
    n = max(2, round(length / 5))
    for k in range(n):
        t = (k + 0.5) / n
        if k == n - 1:
            _patch(p, left, front, (t - 0.12, t + 0.12), (height - 3, height), (70, 56, 44))
            p.line(
                [_on_wall(left, front, t, height - 3), _on_wall(left, front, t, height)],
                (54, 42, 34),
            )
        else:
            _window(p, left, front, (t - 0.06, t + 0.06), height - 3, tall=1.2)
    if height > 6:
        slats = tuple((k + 0.5) / (n * 2) for k in range(n * 2))
        for t in slats:
            p.line(
                [_on_wall(left, front, t, 1), _on_wall(left, front, t, height - 4.5)], (110, 92, 70)
            )
            p.line(
                [_on_wall(front, right, t, 1), _on_wall(front, right, t, height - 4.5)],
                (96, 80, 60),
            )
        _band(p, left, front, height - 4, shade(TAN_WALL, 0.8))
    _window(p, front, right, (0.35, 0.65), height - 2.8, shade(WINDOW, 0.8))
    rise = 3 if length < 14 else 4
    p.gable(top, rise, TAN_ROOF, TAN_WALL)
    _roof_rows(p, top, rise, TAN_ROOF, (0.5,))
    if loft:
        slat = (150, 124, 92)
        vent = p.hall(x, y - height - rise + 0.6, length * 0.7, 2, 2.4, slat, None)
        for a, b, tone in ((vent[0], vent[1], 0.6), (vent[1], vent[2], 0.5)):
            for t in (0.1, 0.25, 0.4, 0.55, 0.7, 0.85):
                p.line([_on_wall(a, b, t, 0.3), _on_wall(a, b, t, 2.2)], shade(slat, tone))
        p.gable(vent, 1.4, TAN_ROOF, slat)
    return top


def _pit(p: Painter, x: float, y: float, liquor: RGB) -> None:
    """A round pit of tan liquor sunk in the ground at (x, y), with a timber rim."""
    p.ellipse(x - 3, y - 1.5, x + 3, y + 1.5, PIT_RIM)
    p.blob(x - 2.2, y - 1, x + 2.2, y + 1, liquor)
    p.point(x - 1, y - 0.5, shade(liquor, 1.4))


def big_tannery() -> Sprite:
    """A bigger tannery: a long tanning shed under a slatted drying loft beside a lower shed, a
    boiler chimney smoking, a row of three pits of tan liquor in sodden ground, five hides
    stretched on frames to dry, a heap of oak bark spreading over the ground and a barrow."""
    p = Painter(52, 58, (26, 42))
    near = _footing(-6, -2, 16, 7)
    _bed(p, _footing(6, -9, 10, 5))
    _bed(p, near)
    p.blob(-18, 3, -8, 8, SODDEN)
    p.blob(-12, 5, -3, 10, SODDEN)
    _spill(p, -20, 0, 3, BARK)
    p.shadow(16, 4)
    _smoking_chimney(p, 13, -9, -30, shade(BRICK, 0.8))
    _tan_shed(p, 6, -9, 10, 5, 4, False)
    _track(p, near[0], near[1], (0.75, 0.92), 3)
    _tan_shed(p, -6, -2, 16, 7, 9, True)
    for k, x in enumerate((-15, -10, -5)):
        _pit(p, x, 5 + 2.5 * (k % 2) - 0.5 * k, LIQUORS[k])
    _heap(p, -20, 0, 3, 3, BARK)
    for k in range(5):
        x, y = 1 + 4 * k, 10 - 2 * k
        _hide_frame(p, x, y, HIDES[k % len(HIDES)])
    _cart(p, -12, 12)
    p.ellipse(-15, 8.5, -11, 10, BARK)
    return p.sprite()


def mega_tannery() -> Sprite:
    """A great tannery: a long hall under a tall slatted drying loft and two sheds beside it, a
    tall banded chimney smoking, a yard of six pits of tan liquor and lime in sodden ground, two
    rows of hides stretched on frames to dry, hides hung over a rail, a great heap of oak bark and
    a cart piled with hides."""
    p = Painter(64, 74, (32, 54))
    hall_foot = _footing(-6, -6, 22, 9)
    _bed(p, _footing(10, -13, 12, 6))
    _bed(p, _footing(-22, -6, 8, 5))
    _bed(p, hall_foot)
    p.blob(-25, 3, -13, 10, SODDEN)
    p.blob(-18, 6, -4, 13, SODDEN)
    p.blob(-22, 9, -12, 14, SODDEN)
    _spill(p, 24, 2, 4, BARK)
    p.shadow(22, 5)
    cap = _chimney(p, 18, -14, -40, shade(BRICK, 0.8))
    _ring(p, 18, -14 + (cap + 14) * 0.46, 1.5, shade(BRICK, 0.6))
    _smoke(p, 19, cap - 3, SMOKE, 3)
    _tan_shed(p, 10, -13, 12, 6, 5, False)
    _tan_shed(p, -22, -6, 8, 5, 4, False)
    _track(p, hall_foot[0], hall_foot[1], (0.8, 0.94), 3.5)
    _tan_shed(p, -6, -6, 22, 9, 12, True)
    for k in range(6):
        x, y = -21 + 6 * (k % 3) + 2 * (k // 3), 4 + 3 * (k % 3) * 0.5 + 4.5 * (k // 3)
        _pit(p, x, y, LIQUORS[k % len(LIQUORS)])
    _heap(p, 24, 2, 4, 4, BARK)
    for row, (x0, y0) in enumerate(((10, 6), (4, 13))):
        for k in range(4):
            _hide_frame(p, x0 + 4 * k, y0 - 2 * k, HIDES[(k + row) % len(HIDES)])
    p.line([(-27, -4), (-27, 2)], FRAME)
    p.line([(-17, 1), (-17, 7)], FRAME)
    p.line([(-27, -4), (-17, 1)], shade(FRAME, 1.3))
    for k in range(3):
        x0 = -26 + 3.2 * k
        p.polygon(
            [
                (x0, -3.5 + 1.6 * k),
                (x0 + 2.6, -2.2 + 1.6 * k),
                (x0 + 2.6, 1.6 * k + 1),
                (x0, 1.6 * k - 0.6),
            ],
            HIDES[k],
            shade(HIDES[k], 0.65),
        )
    cx, cy = -8, 17
    p.rect(cx - 4, cy - 2.6, cx + 4, cy - 1, (130, 92, 60))
    for k in range(3):
        p.blob(cx - 4 + 2.6 * k, cy - 4.6 + 0.5 * (k % 2), cx - 1 + 2.6 * k, cy - 2.2, HIDES[k])
    p.ellipse(cx - 2, cy - 1.8, cx + 0.4, cy + 0.6, (70, 54, 40))
    return p.sprite()


def _boot(p: Painter, x: float, y: float, size: float) -> None:
    """A big red boot sign, its shaft's top left corner at (x, y), `size` times the small one."""
    s = size
    pts = [(2, 0), (6, 0), (6, 8), (0, 8), (0, 6), (1, 5), (2, 5)]
    p.polygon([(x + dx * s, y + dy * s) for dx, dy in pts], BOOT, shade(BOOT, 0.5))
    p.line([(x + 3 * s, y + 1 * s), (x + 3 * s, y + 5 * s)], shade(BOOT, 1.4))
    p.line([(x, y + 8 * s), (x + 6 * s, y + 8 * s)], (40, 32, 30))
    p.line([(x + 2 * s, y + 0.5), (x + 6 * s, y + 0.5)], shade(BOOT, 0.7))


def _shoe_block(p: Painter, x: float, y: float, length: float, depth: float, floors: int) -> Top:
    """A workshop of `floors` storeys with cornices between its floors, windows with sills and a
    long shop window of glass over a red fascia on its ground floor."""
    h = 4 * floors
    top = p.hall(x, y, length, depth, h, SHOE_WALL, None)
    left, front, right, _ = top
    _plinth(p, top, h, SHOE_WALL)
    _stains(p, front, right, h, (0.15, 0.5, 0.85), shade(SHOE_WALL, 0.58), 1.5)
    for f in range(1, floors):
        _bands(p, left, front, right, 4 * f - 0.5, shade(SHOE_WALL, 1.15), shade(SHOE_WALL, 0.85))
    n = max(2, round(length / 4.6))
    m = max(2, round(depth / 4))
    for f in range(floors - 1):
        for k in range(n):
            t = (k + 0.5) / n
            _window(p, left, front, (t - 0.3 / n, t + 0.3 / n), 1.2 + 4 * f, tall=1.6)
    for f in range(floors):
        for k in range(m):
            t = (k + 0.5) / m
            _window(
                p, front, right, (t - 0.3 / m, t + 0.3 / m), 1.2 + 4 * f, shade(WINDOW, 0.75), 1.6
            )
    _patch(p, left, front, (0.04, 0.96), (h - 3.8, h - 3.2), SHOE_ROOF)
    _patch(p, left, front, (0.06, 0.72), (h - 3.2, h - 1.2), GLASS)
    for k in range(4):
        p.line(
            [
                _on_wall(left, front, 0.1 + 0.16 * k, h - 2.6),
                _on_wall(left, front, 0.14 + 0.16 * k, h - 2.8),
            ],
            (220, 236, 244),
        )
        p.point(*_on_wall(left, front, 0.15 + 0.16 * k, h - 1.6), BOOT)
    _door(p, left, front, (0.78, 0.92), h - 3.2, h, (84, 60, 46))
    p.point(*_on_wall(left, front, 0.9, h - 1.6), (226, 196, 120))
    return top


def _pavement(p: Painter, foot: Top, joints: int) -> None:
    _bed(p, foot, PAVING, 2.4)
    _bed(p, foot, SOIL, 0.8)
    for k in range(1, joints):
        t = k / joints
        p.line(
            [_on_wall(foot[0], foot[1], t, 0.8), _on_wall(foot[0], foot[1], t, 2.2)],
            shade(PAVING, 0.8),
        )


def _shoe_boxes(p: Painter, x: float, y: float, count: int) -> None:
    for k in range(count):
        bx, by = x + 2.4 * (k % 3), y - 1.2 * (k % 3) - 1.6 * (k // 3)
        _crate(p, bx, by, SHOE_BOX)
        p.point(bx - 1, by - 1.6, BOOT)


def big_shoe_factory() -> Sprite:
    """A five-storey shoe factory with cornices between its floors and rows of windows, a long
    shop window over a red fascia on its ground floor, a square chimney smoking behind, a big boot
    hung from a bracket as a sign, a pavement of flags along its front, boxes of shoes piled up
    and a handcart."""
    p = Painter(46, 60, (22, 48))
    foot = _footing(2, -1, 14, 7)
    _pavement(p, foot, 4)
    p.shadow(13, 3.5)
    _track(p, foot[0], foot[1], (0.78, 0.92), 2.6, shade(PAVING, 0.9), ruts=False)
    cap = _stack(p, 9, -10, -36, shade(BRICK, 0.85))
    _smoke(p, 10.5, cap - 2, SMOKE, 3)
    top = _shoe_block(p, 2, -1, 14, 7, 5)
    p.gable(top, 4, SHOE_ROOF, SHOE_WALL)
    _roof_rows(p, top, 4, SHOE_ROOF)
    x, y = _on_roof(top, 4, 0.3, 0.9)
    p.rect(x - 1, y - 2.5, x, y, BRICK)
    left, front, _, _ = top
    bx, by = _on_wall(left, front, 0.02, 7)
    p.line([(bx, by), (bx - 3, by)], (60, 56, 54))
    _boot(p, bx - 7.5, by + 0.5, 1.3)
    _shoe_boxes(p, 9, 8.5, 5)
    _cart(p, -14, 8)
    for k in range(2):
        _crate(p, -15 + 2 * k, 5 - 0.8 * k, SHOE_BOX)
    return p.sprite()


def mega_shoe_factory() -> Sprite:
    """A great shoe factory: a long block of six storeys with cornices and rows of windows, a shop
    front of glass over a red fascia all along its ground floor, a giant red boot standing on its
    roof as a sign, a corner turret, a tall square chimney smoking, a pavement of flags along its
    front, and a delivery cart loaded with boxes of shoes among more boxes stacked up."""
    p = Painter(60, 78, (29, 62))
    foot = _footing(0, -2, 22, 9)
    _pavement(p, foot, 6)
    p.shadow(18, 5)
    _track(p, foot[0], foot[1], (0.78, 0.92), 3, shade(PAVING, 0.9), ruts=False)
    cap = _stack(p, 13, -14, -46, shade(BRICK, 0.85))
    _smoke(p, 14.5, cap - 2, SMOKE, 3)
    top = _shoe_block(p, 0, -2, 22, 9, 6)
    left, front, _, _ = top
    p.gable(top, 5, SHOE_ROOF, SHOE_WALL)
    _roof_rows(p, top, 5, SHOE_ROOF, (0.25, 0.5, 0.75))
    tx, ty = front
    turret = p.box(tx, ty - 24, 2.6, 4, SHOE_WALL, None)
    _windows(p, turret[0], turret[1], (0.5,), (1.2,), 2)
    apex = (tx, ty - 24 - 4 - 6)
    p.polygon([turret[0], turret[1], apex], SHOE_ROOF, shade(SHOE_ROOF, 0.6))
    p.polygon([turret[1], turret[2], apex], shade(SHOE_ROOF, 0.75), shade(SHOE_ROOF, 0.5))
    _flag(p, apex[0], apex[1], 4, BOOT)
    rx, ry = _on_roof(top, 5, 0.45, 1)
    for dx in (-3, 3):
        p.line([(rx + dx, ry + 1), (rx + dx, ry - 3)], (70, 66, 62))
    _boot(p, rx - 6, ry - 16, 2)
    bx, by = _on_wall(left, front, 0.02, 9)
    p.line([(bx, by), (bx - 3, by)], (60, 56, 54))
    _boot(p, bx - 7.5, by + 0.5, 1.3)
    _shoe_boxes(p, 14, 13, 6)
    cx, cy = -16, 12
    p.rect(cx - 5, cy - 3, cx + 5, cy - 1.5, (130, 92, 60))
    for k in range(4):
        _crate(p, cx - 3.5 + 2.4 * k, cy - 3, SHOE_BOX)
    _crate(p, cx - 1, cy - 5.6, SHOE_BOX)
    for wx in (cx - 3, cx + 3):
        p.ellipse(wx - 1.2, cy - 1.6, wx + 1.2, cy + 0.8, (70, 54, 40))
    p.line([(cx - 5, cy - 2), (cx - 8, cy)], (110, 80, 56))
    return p.sprite()


def _boards(p: Painter, x: float, y: float, length: float, depth: float, h: float) -> None:
    """A stack of sawn boards on bearers, the gaps between them showing."""
    for t in (0.15, 0.85):
        bx, by = _on_wall(*_footing(x, y, length, depth)[:2], t, 0)
        p.rect(bx - 0.5, by - 1, bx + 0.5, by + 0.5, LEG)
    stack = p.hall(x, y - 0.8, length, depth, h, BOARD, shade(BOARD, 1.12))
    gap = shade(BOARD, 0.5)
    for drop in range(2, int(h), 2):
        for a, b in ((stack[0], stack[1]), (stack[1], stack[2])):
            _band(p, a, b, drop, gap)


def _furniture(p: Painter, x: float, y: float) -> None:
    """A table, a chair and a tall wardrobe stood out on the ground from (x, y)."""
    wardrobe = (130, 86, 54)
    p.rect(x - 2, y - 7, x + 1, y, wardrobe)
    p.line([(x - 0.5, y - 6.5), (x - 0.5, y - 0.5)], shade(wardrobe, 0.6))
    p.point(x - 1, y - 3.5, GOLD)
    p.point(x, y - 3.5, GOLD)
    p.rect(x + 2.5, y - 2.6, x + 7, y - 2, BOARD)
    for lx in (x + 3, x + 6.5):
        p.line([(lx, y - 2), (lx, y)], LEG)
    _chair(p, x + 8, y + 0.5, 0.6)


def big_furniture_maker() -> Sprite:
    """A bigger joiner's works: a two-storey plank-walled workshop with a loft door under a hoist
    beam and a big red chair on its ridge, an open saw shed beside it, a yard of sawn boards
    stacked on bearers, heaps of sawdust and a table, chair and wardrobe stood out by the door."""
    p = Painter(50, 48, (23, 34))
    foot = _footing(-4, -2, 16, 8)
    _bed(p, foot)
    p.blob(2, -6, 24, 4, SOIL)
    p.shadow(18, 4)
    _track(p, foot[0], foot[1], (0.62, 0.86), 3)
    for x, y, h in ((14, -6, 5), (17, -1, 4), (10, 2, 3)):
        _boards(p, x, y, 8, 3, h)
    top = p.hall(-4, -2, 16, 8, 11, JOINER_WALL, None)
    left, front, right, _ = top
    _plank_walls(p, top, 11)
    _windows(p, left, front, (0.12, 0.3, 0.48), (2.5,), 2)
    _window(p, left, front, (0.08, 0.22), 7)
    _patch(p, left, front, (0.6, 0.76), (1.6, 4.6), (56, 40, 30))
    _door(p, left, front, (0.62, 0.86), 6.6, 11, (56, 40, 30), shade(JOINER_WALL, 1.3))
    p.line([_on_wall(left, front, 0.74, 6.8), _on_wall(left, front, 0.74, 11)], (88, 64, 44))
    _window(p, front, right, (0.3, 0.6), 2, shade(WINDOW, 0.8))
    _window(p, front, right, (0.3, 0.6), 6.5, shade(WINDOW, 0.8))
    p.gable(top, 5, JOINER_ROOF, JOINER_WALL)
    _roof_rows(p, top, 5, JOINER_ROOF)
    hx, hy = _on_wall(left, front, 0.68, 0)
    p.line([(hx, hy - 1), (hx - 3, hy - 0.5)], LEG)
    p.line([(hx - 2.6, hy - 0.5), (hx - 2.6, hy + 3)], (70, 66, 62))
    rx, ry = _on_roof(top, 5, 0.35, 1)
    _chair(p, rx - 3, ry + 0.5, 1.3)
    p.blob(-18, 6, -10, 9, SAWDUST)
    p.blob(-13, 9, -7, 11, shade(SAWDUST, 0.92))
    for x in (-16, -12):
        p.line([(x - 1, 8), (x, 5), (x + 1, 8)], LEG)
    p.line([(-17, 4), (-10, 6.5)], BOARD)
    p.line([(-17, 5), (-10, 7.5)], shade(BOARD, 0.7))
    _furniture(p, 4, 11)
    return p.sprite()


def mega_furniture_maker() -> Sprite:
    """A great furniture factory: a long plank-walled hall under a glazed sawtooth roof with a giant
    red chair raised on its roof, a timber drying kiln tower with a louvred top, a banded steam
    stack smoking, a lumber yard of sawn boards stacked high, heaps of sawdust and a row of
    finished tables, chairs and wardrobes out front by a cart."""
    p = Painter(64, 76, (32, 60))
    foot = _footing(-6, -4, 24, 10)
    _bed(p, foot)
    p.blob(4, -12, 30, 6, SOIL)
    p.shadow(24, 5)
    _track(p, foot[0], foot[1], (0.78, 0.94), 3.5)
    _smoking_chimney(p, 2, -14, -44, shade(BRICK, 0.85), smoke=STEAM)
    kiln = p.box(16, -10, 4, 16, shade(JOINER_WALL, 0.9), None)
    _plank_walls(p, kiln, 16)
    for a, b in ((kiln[0], kiln[1]), (kiln[1], kiln[2])):
        _patch(p, a, b, (0.15, 0.85), (1.2, 3.6), (70, 54, 40))
        for t in (0.3, 0.5, 0.7):
            p.line(
                [_on_wall(a, b, t - 0.08, 1.4), _on_wall(a, b, t + 0.08, 3.4)], shade(BOARD, 0.8)
            )
    p.gable(kiln, 3, JOINER_ROOF, JOINER_WALL)
    _smoke(p, 17, -32, STEAM, 2)
    for x, y, h in ((22, -6, 7), (20, 1, 5), (25, 2, 4), (13, 5, 3)):
        _boards(p, x, y, 8, 3, h)
    top = p.hall(-6, -4, 24, 10, 10, JOINER_WALL, None)
    left, front, right, _ = top
    _plank_walls(p, top, 10)
    _windows(p, left, front, tuple(0.06 + 0.12 * k for k in range(6)), (2.4, 5.4), 2)
    _door(p, left, front, (0.78, 0.94), 4, 10, (56, 40, 30), shade(JOINER_WALL, 1.3))
    p.line([_on_wall(left, front, 0.86, 4.2), _on_wall(left, front, 0.86, 10)], (88, 64, 44))
    _windows(p, front, right, (0.25, 0.5, 0.75), (2.4, 5.4), 2, shade(WINDOW, 0.8))
    _sawtooth(p, top, 5, 4, JOINER_ROOF, GLAZING, JOINER_WALL)
    rx, ry = _on_wall(top[0], top[2], 0.45, -3)
    for dx in (-2, 4):
        p.line([(rx + dx, ry + 2), (rx + dx, ry - 1)], (70, 66, 62))
    _chair(p, rx - 4, ry, 2.2)
    p.blob(3, 9, 15, 14, SAWDUST)
    _heap(p, 9, 12, 3.5, 3, SAWDUST)
    for x, y in ((-22, 1.5), (-13, 6)):
        _furniture(p, x, y)
    _cart(p, 22, 12)
    for k in range(2):
        p.line([(20 + 2 * k, 9), (24 + 2 * k, 10.5)], BOARD)
    return p.sprite()


def _river(p: Painter, a: Point, b: Point, width: float) -> None:
    """A stream between muddy banks flowing over the ground from `a` to `b`, `width` across."""
    (ax, ay), (bx, by) = a, b
    p.polygon(
        [
            (ax - 1, ay - width / 2 - 1),
            (bx + 1, by - width / 2),
            (bx - 1, by + width / 2 + 1.5),
            (ax - 1, ay + width / 2 + 1),
        ],
        BANK,
    )
    p.polygon(
        [
            (ax, ay - width / 2),
            (bx, by - width / 2 + 0.6),
            (bx - 1, by + width / 2),
            (ax, ay + width / 2),
        ],
        WATER,
    )
    for t in (0.2, 0.55):
        x, y = _on_wall(a, b, t, -0.5)
        p.line([(x, y), (x + 3, y + 1.5)], (150, 196, 226))


def _waterwheel(p: Painter, cx: float, cy: float, r: float) -> None:
    """A wooden waterwheel turning in the stream at (cx, cy), churning foam below it."""
    p.ellipse(cx - r * 0.6, cy - r, cx + r * 0.6, cy + r * 0.8, WHEEL)
    p.ellipse(cx - r * 0.4, cy - r * 0.8, cx + r * 0.4, cy + r * 0.6, shade(WHEEL, 1.3))
    p.line([(cx, cy - r * 0.8), (cx, cy + r * 0.6)], WHEEL)
    p.line([(cx - r * 0.4, cy - 0.1 * r), (cx + r * 0.4, cy - 0.1 * r)], WHEEL)
    for a in range(8):
        ang = a * math.pi / 4
        p.point(
            cx + r * 0.6 * math.cos(ang), cy - 0.1 * r + r * 0.9 * math.sin(ang), shade(WHEEL, 0.6)
        )
    for dx, dy in ((-1.5, r * 0.8), (1, r * 0.9 + 0.5), (2.5, r * 0.8 + 1)):
        p.point(cx + dx, cy + dy, FOAM)


def _rolls(p: Painter, x: float, y: float, rows: int) -> None:
    """A pyramid of paper rolls standing on end from (x, y), `rows` deep."""
    for row in range(rows):
        for k in range(rows - row):
            rx = x + 3 * k + 1.5 * row
            ry = y - 1.5 * k - 3 * row + 0.75 * row
            p.ellipse(rx - 1.5, ry - 3, rx + 1.5, ry, ROLL)
            p.point(rx, ry - 1.5, shade(ROLL, 0.55))


def _logs(p: Painter, x: float, y: float, rows: int) -> None:
    """A pile of pulpwood logs with their pale ends showing, among scattered bark."""
    for dx, dy in ((-1, 1.5), (3, 2), (6, 0.8)):
        p.point(x + dx, y + dy, (150, 112, 74))
    for row in range(rows):
        for k in range(rows + 1 - row):
            lx, ly = x + 2 * k + row, y - 1 * k - 1.8 * row
            p.ellipse(lx - 1, ly - 2, lx + 1, ly, LOG)
            p.point(lx, ly - 1, (150, 112, 74))


def _paper_mill_hall(
    p: Painter, x: float, y: float, length: float, depth: float, height: float, storeys: int
) -> Top:
    top = p.hall(x, y, length, depth, height, PAPER_WALL, None)
    left, front, right, _ = top
    lintel = shade(PAPER_WALL, 1.2)
    _plinth(p, top, height, PAPER_WALL)
    _stains(p, left, front, height, (0.45, 0.65, 0.85), shade(PAPER_WALL, 0.6), 2.2)
    _stains(p, front, right, height, (0.15, 0.85), shade(PAPER_WALL, 0.5), 1.6)
    n = max(3, round(length / 3.4))
    drops = tuple(2 + 3.6 * f for f in range(storeys))
    _windows(p, left, front, tuple((k + 0.5) / n for k in range(n)), drops, 2, arch=lintel)
    _windows(p, front, right, (0.3, 0.75), drops, 2, shade(WINDOW, 0.8), lintel)
    _door(p, front, right, (0.4, 0.65), height - 4, height, (70, 56, 48), lintel)
    return top


def big_paper_mill() -> Sprite:
    """A bigger riverside paper mill: a long two-storey mill turning two waterwheels in its stream,
    a drying loft of louvres along its ridge, a tall banded stack smoking, a pyramid of paper rolls
    in the yard and a big pile of pulpwood logs among scattered bark."""
    p = Painter(54, 68, (27, 54))
    _river(p, (-25, -6), (2, 7.5), 5)
    foot = _footing(1, -6, 18, 9)
    _bed(p, foot)
    p.blob(10, 2, 24, 10, (120, 96, 66))
    p.shadow(16, 4)
    _track(p, foot[1], foot[2], (0.4, 0.65), 4)
    _smoking_chimney(p, 11, -8, -38)
    top = _paper_mill_hall(p, 1, -6, 18, 9, 10, 2)
    left, front, _, _ = top
    p.gable(top, 5, PAPER_ROOF, PAPER_WALL)
    _roof_rows(p, top, 5, PAPER_ROOF, (0.34,))
    for t in (0.2, 0.35, 0.5, 0.65, 0.8):
        a, b = _on_roof(top, 5, t - 0.04, 0.6), _on_roof(top, 5, t + 0.04, 0.95)
        p.line([a, b], shade(PAPER_ROOF, 1.35))
    for t in (0.2, 0.55):
        cx, cy = _on_wall(left, front, t, 8)
        _waterwheel(p, cx, cy, 4)
    _rolls(p, 6, 10, 3)
    _logs(p, 17, 6, 3)
    return p.sprite()


def mega_paper_mill() -> Sprite:
    """A great paper works on its river: a long mill turning three waterwheels, a tall machine hall
    under a glazed sawtooth roof behind, two round pulp digesters banded in iron, two tall stacks
    smoking and steam rising, pyramids of paper rolls in the yard and a great pile of pulpwood logs
    among scattered bark."""
    p = Painter(64, 80, (32, 64))
    _river(p, (-30, -8), (2, 9), 6)
    foot = _footing(-1, -8, 24, 10)
    _bed(p, foot)
    _bed(p, _footing(16, -12, 12, 9))
    p.blob(8, 2, 30, 14, (120, 96, 66))
    p.shadow(22, 5)
    for x, foot_y, tall in ((4, -18, 34), (24, -12, 34)):
        _smoking_chimney(p, x, foot_y, foot_y - tall)
    for x, y in ((20, -14), (25, -11.5)):
        _turned(
            p,
            x,
            y,
            [(0, 2.4), (14, 2.4), (16, 1.4)],
            shade(PAPER_WALL, 0.85),
            shade(PAPER_WALL, 0.7),
        )
        for h in (4, 9):
            _ring(p, x, y - h, 2.4, (90, 84, 82))
    _smoke(p, 21, -32, STEAM, 2)
    mh = p.hall(16, -6, 12, 8, 9, PAPER_WALL, None)
    _plinth(p, mh, 9, PAPER_WALL)
    _windows(p, mh[0], mh[1], (0.2, 0.4, 0.6, 0.8), (2.5,), 4, arch=shade(PAPER_WALL, 1.2))
    _windows(p, mh[1], mh[2], (0.3, 0.7), (2.5,), 4, shade(WINDOW, 0.8))
    _sawtooth(p, mh, 3, 3, PAPER_ROOF, GLAZING, PAPER_WALL)
    _track(p, mh[1], (mh[1][0] + 8, mh[1][1] - 4), (0.3, 0.6), 4)
    top = _paper_mill_hall(p, -1, -8, 24, 10, 13, 3)
    left, front, _, _ = top
    p.gable(top, 6, PAPER_ROOF, PAPER_WALL)
    _roof_rows(p, top, 6, PAPER_ROOF, (0.34,))
    for t in (0.15, 0.3, 0.45, 0.6, 0.75, 0.9):
        a, b = _on_roof(top, 6, t - 0.03, 0.6), _on_roof(top, 6, t + 0.03, 0.95)
        p.line([a, b], shade(PAPER_ROOF, 1.35))
    for t in (0.15, 0.45, 0.75):
        cx, cy = _on_wall(left, front, t, 11)
        _waterwheel(p, cx, cy, 4.5)
    _rolls(p, 4, 14, 3)
    _rolls(p, 13, 10, 2)
    _logs(p, 20, 9, 4)
    return p.sprite()


def _newspaper(p: Painter, x: float, y: float, half: float, tall: float) -> None:
    """A newspaper page held up on posts with its feet at (x, y) as a sign: a red masthead, a
    headline, a picture and columns of print."""
    for dx in (-half * 0.5, half * 0.5):
        p.line([(x + dx, y), (x + dx, y - 3)], (70, 66, 62))
    t, b = y - 3 - tall, y - 3
    p.rect(x - half, t, x + half, b, PAGE)
    p.rect(x - half + 1, t + 1, x + half - 1, t + 2.2, MASTHEAD)
    p.line([(x - half + 1.5, t + 3.6), (x + half - 1.5, t + 3.6)], INK)
    if tall > 9:
        p.line([(x - half + 1.5, t + 4.8), (x + half * 0.4, t + 4.8)], INK)
    p.rect(x - half + 1.5, t + tall * 0.55, x - 1, b - 1.5, PRINT)
    for yy in range(int(t + tall * 0.55), int(b - 1), 2):
        p.line([(x + 1, yy), (x + half - 1.5, yy)], PRINT)


def _bundles(p: Painter, x: float, y: float, count: int) -> None:
    for k in range(count):
        _bundle(p, x + 3.4 * (k % 3) - 1.7 * (k // 3), y - 1.7 * (k % 3) - 1.6 * (k // 3))


def _press_house(
    p: Painter, x: float, y: float, length: float, depth: float, height: float, storeys: int
) -> Top:
    top = p.hall(x, y, length, depth, height, PRESS_WALL, PRESS_ROOF)
    left, front, right, _ = top
    lintel = shade(PRESS_WALL, 1.12)
    _plinth(p, top, height, PRESS_WALL)
    _stains(p, front, right, height, (0.12, 0.38, 0.62, 0.88), shade(PRESS_WALL, 0.56), 1.5)
    _band(p, left, front, 1, lintel)
    p.line([left, front, right], shade(PRESS_ROOF, 1.3))
    n = max(4, round(length / 3))
    spans = tuple((k + 0.5) / n for k in range(n))
    floor = (height - 2.5) / storeys
    for f in range(storeys):
        drop = 2.5 + floor * f
        _windows(
            p, left, front, spans, (drop,), max(2, round(floor) - 2), arch=shade(PRESS_WALL, 0.75)
        )
        _windows(
            p,
            front,
            right,
            (0.25, 0.5, 0.75),
            (drop,),
            max(2, round(floor) - 2),
            shade(WINDOW, 0.75),
        )
        if f:
            _bands(
                p, left, front, right, drop - 1.2, shade(PRESS_WALL, 0.85), shade(PRESS_WALL, 0.68)
            )
    return top


def big_printing_works() -> Sprite:
    """A bigger printing house: three storeys of tall arched windows over a dark plinth with a
    great newspaper sign held up on its roof, a glazed press hall beside it, a pavement of flags
    with a path to the door, a cart loaded with bundles of papers, more bundles piled up and ink
    barrels by a stain of spilt ink."""
    p = Painter(48, 54, (23, 38))
    foot = _footing(-2, -1, 16, 9)
    _pavement(p, foot, 5)
    _bed(p, _footing(14, -2, 8, 7))
    p.shadow(15, 4)
    _track(p, foot[0], foot[1], (0.62, 0.8), 2.6, shade(PAVING, 0.9), ruts=False)
    _oil(p, 10, 8, 1.6)
    annex = p.hall(14, -2, 8, 7, 6, PRESS_WALL, None)
    _plinth(p, annex, 6, PRESS_WALL)
    _patch(p, annex[1], annex[2], (0.2, 0.8), (2, 5), DOORWAY_PRESS)
    p.gable(annex, 3, PRESS_ROOF, PRESS_WALL)
    _skylight(p, annex, 3, (0.1, 0.9))
    top = _press_house(p, -2, -1, 16, 9, 16, 3)
    left, front, _, _ = top
    _door(p, left, front, (0.62, 0.78), 11.5, 16, (70, 56, 48), shade(PRESS_WALL, 0.75))
    rx, ry = _on_wall(top[0], top[2], 0.5, 0)
    _newspaper(p, rx, ry, 9, 10)
    cx, cy = 8, 12
    p.rect(cx - 4, cy - 2.6, cx + 4, cy - 1, (130, 92, 60))
    for k in range(3):
        bundle = p.box(cx - 2.5 + 2.6 * k, cy - 2.4, 1.4, 1.4, BUNDLE, shade(BUNDLE, 1.04))
        p.line([bundle[1], (bundle[1][0], bundle[1][1] + 1.4)], (150, 110, 80))
    for wx in (cx - 2.5, cx + 2.5):
        p.ellipse(wx - 1.1, cy - 1.5, wx + 1.1, cy + 0.7, (70, 54, 40))
    _bundles(p, 15, 7, 4)
    for x, y in ((-16, 6), (-13.5, 7.5)):
        _barrel(p, x, y)
    return p.sprite()


def mega_printing_works() -> Sprite:
    """A great newspaper press house: a long building of four storeys of tall arched windows with a
    clock over its door and a huge newspaper sign raised on a frame on its roof, a glazed press
    hall where the rotary presses roll, a flag flying, rolls of newsprint delivered on a wagon,
    bundles of papers stacked high and carts waiting to take them away."""
    p = Painter(64, 74, (32, 52))
    foot = _footing(-4, -3, 24, 10)
    _pavement(p, foot, 7)
    _bed(p, _footing(18, -4, 12, 9))
    p.shadow(22, 5)
    _track(p, foot[0], foot[1], (0.7, 0.84), 3, shade(PAVING, 0.9), ruts=False)
    _oil(p, 14, 12, 1.8)
    hall = p.hall(18, -4, 12, 9, 8, PRESS_WALL, None)
    _plinth(p, hall, 8, PRESS_WALL)
    _patch(p, hall[0], hall[1], (0.1, 0.9), (1.6, 6), GLASS)
    for t in (0.2, 0.45, 0.7):
        cx, cy = _on_wall(hall[0], hall[1], t + 0.05, 4.4)
        p.ellipse(cx - 1.2, cy - 1.2, cx + 1.2, cy + 1.2, INK)
        p.point(cx, cy, PRINT)
    p.line([_on_wall(hall[0], hall[1], 0.1, 3), _on_wall(hall[0], hall[1], 0.9, 3)], PAGE)
    _door(p, hall[1], hall[2], (0.2, 0.8), 3, 8, DOORWAY_PRESS)
    _sawtooth(p, hall, 4, 3, PRESS_ROOF, GLAZING, PRESS_WALL)
    top = _press_house(p, -4, -3, 24, 10, 22, 4)
    left, front, right, _ = top
    _door(p, left, front, (0.44, 0.56), 17, 22, (70, 56, 48), shade(PRESS_WALL, 0.75))
    cx, cy = _on_wall(left, front, 0.5, 14.4)
    p.ellipse(cx - 1.6, cy - 1.6, cx + 1.6, cy + 1.6, (248, 246, 236))
    p.line([(cx, cy), (cx, cy - 1.1)], INK)
    p.line([(cx, cy), (cx + 1, cy)], INK)
    rx, ry = _on_wall(top[0], top[2], 0.5, 0)
    for dx in (-8, 8):
        p.line([(rx + dx, ry + 1), (rx + dx, ry - 4)], (70, 66, 62))
    p.line([(rx - 8, ry - 2), (rx + 8, ry - 2)], (70, 66, 62))
    _newspaper(p, rx, ry - 2, 13, 14)
    _flag(p, right[0] - 1, right[1], 7, MASTHEAD)
    _rails(p, (15, 16.5), (29, 9.5), 7)
    wx, wy = 22, 13
    p.rect(wx - 5, wy - 2.6, wx + 5, wy - 1, (70, 72, 80))
    for k in range(3):
        rx2 = wx - 3 + 3 * k
        p.ellipse(rx2 - 1.4, wy - 5.4, rx2 + 1.4, wy - 2.4, ROLL)
        p.point(rx2, wy - 3.9, shade(ROLL, 0.55))
    for wxx in (wx - 3, wx + 3):
        p.ellipse(wxx - 1, wy - 1.6, wxx + 1, wy + 0.4, (40, 38, 40))
    _bundles(p, -20, 6, 6)
    _bundles(p, 4, 14, 3)
    for x, y in ((-10, 14), (-3, 17)):
        p.rect(x - 4, y - 2.6, x + 4, y - 1, (130, 92, 60))
        for j in range(2):
            bundle = p.box(x - 1.4 + 2.8 * j, y - 2.4, 1.3, 1.3, BUNDLE, shade(BUNDLE, 1.04))
            p.line([bundle[1], (bundle[1][0], bundle[1][1] + 1.3)], (150, 110, 80))
        p.ellipse(x - 1.2, y - 1.5, x + 1.2, y + 0.7, (70, 54, 40))
    return p.sprite()


def _forge_hall(
    p: Painter,
    x: float,
    y: float,
    length: float,
    depth: float,
    height: float,
    doors: tuple[float, ...],
) -> Top:
    """A dark forge hall with doors open on glowing hearths along its lit wall."""
    top = p.hall(x, y, length, depth, height, FORGE_WALL, None)
    left, front, right, _ = top
    _plinth(p, top, height, FORGE_WALL)
    _stains(p, front, right, height, (0.15, 0.5, 0.85), shade(FORGE_WALL, 0.55), 1.8)
    w = 0.34 * 13 / length
    for t in doors:
        _patch(p, left, front, (t - w / 2, t + w / 2), (height - 4.5, height), GLOW)
        _patch(p, left, front, (t - w / 4, t + w / 4), (height - 2.5, height), HOT)
        p.line(
            [
                _on_wall(left, front, t - w / 2 - 0.02, height - 5.1),
                _on_wall(left, front, t + w / 2 + 0.02, height - 5.1),
            ],
            shade(FORGE_WALL, 1.5),
        )
    _windows(p, front, right, (0.3, 0.7), (3,), 1, (240, 150, 60), shade(FORGE_WALL, 1.5))
    return top


def _tools(p: Painter, x: float, y: float, count: int) -> None:
    """A rack of finished shovels and picks leant on a rail standing on (x, y)."""
    p.line([(x - 0.5, y - 4), (x + count * 1.6, y - 4 + count * 0.8)], LEG)
    for k in range(count):
        tx, ty = x + 1.6 * k, y + 0.8 * k
        p.line([(tx, ty - 5), (tx, ty)], HANDLE)
        if k % 2:
            p.line([(tx - 1, ty - 5), (tx + 1, ty - 4.6)], BAR)
        else:
            p.rect(tx - 0.6, ty - 1, tx + 0.6, ty, BAR)


def _sparks(p: Painter, points: tuple[Point, ...]) -> None:
    for k, (x, y) in enumerate(points):
        p.point(x, y, HOT if k % 2 else GLOW)


def big_tool_works() -> Sprite:
    """A bigger tool works: a long dark forge hall with two doors open on glowing hearths and an
    anvil painted on a board on its wall, a trip-hammer shed glowing beside it, two banded stacks
    smoking and sparking, a heap of coal spilling over the sooty ground, a quenching tub, racks of
    finished shovels and picks and a pile of iron bars."""
    p = Painter(48, 58, (24, 42))
    foot = _footing(-3, -2, 18, 9)
    _bed(p, foot, FORGE_GROUND, 1.6)
    _bed(p, _footing(13, -5, 7, 6), FORGE_GROUND, 1.4)
    _spill(p, -15, 4, 3.5, COAL)
    p.shadow(17, 4)
    _track(p, foot[0], foot[1], (0.62, 0.86), 3, (96, 88, 82))
    shed = p.hall(13, -5, 7, 6, 5, shade(FORGE_WALL, 1.1), None)
    _patch(p, shed[1], shed[2], (0.25, 0.75), (1.5, 5), GLOW)
    _patch(p, shed[1], shed[2], (0.4, 0.6), (3, 5), HOT)
    p.gable(shed, 2.5, FORGE_ROOF, FORGE_WALL)
    top = _forge_hall(p, -3, -2, 18, 9, 8, (0.62, 0.86))
    left, front, _, _ = top
    _patch(p, left, front, (0.04, 0.42), (0.6, 6.4), SIGNBOARD)
    _patch(p, left, front, (0.04, 0.42), (5.6, 6.4), shade(SIGNBOARD, 0.8))
    _anvil(p, *_on_wall(left, front, 0.23, 3.4), 1.1)
    p.gable(top, 4, FORGE_ROOF, FORGE_WALL)
    _roof_rows(p, top, 4, FORGE_ROOF)
    for along in (0.3, 0.75):
        x, y = _on_roof(top, 4, along, 0.9)
        cap = _forge_stack(p, x, y + 1, y - 9)
        _smoke(p, x + 1, cap - 4, (96, 94, 98))
    _sparks(p, ((-6, -22), (-3, -24), (2, -21), (5, -25), (-1, -27)))
    _heap(p, -15, 4, 3.5, 3, COAL)
    _puddle(p, 4, 5.6, 2.2)
    p.ellipse(2, 3.5, 6, 4.5, (70, 96, 120))
    _tools(p, -9, 10, 6)
    _lengths(p, 13, 5, BAR, 3)
    _barrel(p, 7, 10)
    return p.sprite()


def mega_tool_works() -> Sprite:
    """A great tool works: a long dark forge hall under a sawtooth roof glowing from the hearths
    inside, three doors open on the fire, a tall steam-hammer house with its door glowing, three
    banded stacks smoking and throwing sparks, a giant anvil sign on a frame, a great heap of coal,
    a rail siding with a wagon of coal, racks of finished shovels and picks, quenching tubs and
    piles of iron bars."""
    p = Painter(62, 76, (31, 56))
    foot = _footing(-6, -5, 26, 10)
    _bed(p, foot, FORGE_GROUND, 2)
    _bed(p, _footing(14, -10, 8, 8), FORGE_GROUND, 1.4)
    p.blob(-28, 1, -8, 11, FORGE_GROUND)
    p.blob(-14, 7, 6, 16, FORGE_GROUND)
    _spill(p, -22, 5, 4.5, COAL)
    p.shadow(23, 5)
    for k, x in enumerate((-12, -2, 8)):
        _smoking_chimney(p, x, -14 + k, -42 + k, (74, 58, 52), smoke=(96, 94, 98))
    _sparks(p, ((-14, -46), (-10, -44), (-4, -47), (0, -45), (6, -46), (10, -44), (-8, -49)))
    house = p.hall(14, -10, 8, 8, 16, shade(FORGE_WALL, 1.08), None)
    _plinth(p, house, 16, FORGE_WALL)
    _patch(p, house[0], house[1], (0.25, 0.75), (8, 16), GLOW)
    _patch(p, house[0], house[1], (0.38, 0.62), (11, 16), HOT)
    _windows(p, house[1], house[2], (0.3, 0.7), (3, 8), 2, (240, 150, 60))
    p.gable(house, 3, FORGE_ROOF, FORGE_WALL)
    top = _forge_hall(p, -6, -5, 26, 10, 10, (0.5, 0.7, 0.9))
    left, front, _, _ = top
    _windows(p, left, front, (0.06, 0.14, 0.22, 0.3, 0.38), (2.5,), 3, GLOW, shade(FORGE_WALL, 1.6))
    _sawtooth(p, top, 4, 3, FORGE_ROOF, shade(GLOW, 0.9), FORGE_WALL)
    sx, sy = _on_wall(foot[0], foot[1], 0.2, 4)
    for dx in (-5, 5):
        p.line([(sx + dx, sy), (sx + dx, sy - 9)], LEG)
    p.rect(sx - 6, sy - 15, sx + 6, sy - 7, SIGNBOARD)
    _anvil(p, sx, sy - 11, 1.6)
    _heap(p, -22, 5, 4.5, 4, COAL)
    p.point(-23, 2, (104, 104, 116))
    _rails(p, (6, 18), (28, 7), 9)
    _wagon(p, 18, 13, (90, 70, 56))
    p.blob(15, 9, 21, 11.5, COAL)
    _puddle(p, -6, 12, 2.4)
    p.ellipse(-8, 10, -4, 11, (70, 96, 120))
    _tools(p, -2, 13, 6)
    _tools(p, -14, 12, 5)
    _lengths(p, 24, 0, BAR, 3)
    _lengths(p, 18, 3, BAR, 2)
    return p.sprite()


def _engine_hall(p: Painter, x: float, y: float, length: float, depth: float, height: float) -> Top:
    """A brick engine hall over a dark plinth stained by the damp, with tall arched windows."""
    top = p.hall(x, y, length, depth, height, BRICK, None)
    left, front, right, _ = top
    lintel = shade(BRICK, 1.3)
    _plinth(p, top, height, BRICK)
    _stains(p, left, front, height, (0.04, 0.21, 0.39, 0.57, 0.73), shade(BRICK, 0.62), 2)
    _stains(p, front, right, height, (0.2, 0.8), shade(BRICK, 0.5), 2)
    _bands(p, left, front, right, 1, lintel, shade(BRICK, 1.05))
    n = max(4, round(length / 3.6))
    spans = tuple((k + 0.5) / (n + 1) for k in range(n))
    _windows(p, left, front, spans, (2.5,), round(height * 0.5), arch=lintel)
    _windows(p, left, front, spans, (height * 0.5 + 3.2,), 1, shade(BRICK, 1.15))
    _door(p, left, front, (n / (n + 1) + 0.02, 0.97), height * 0.45, height, (58, 48, 44), lintel)
    p.line(
        [
            _on_wall(left, front, (n / (n + 1) + 0.99) / 2, height * 0.45),
            _on_wall(left, front, (n / (n + 1) + 0.99) / 2, height),
        ],
        (44, 36, 34),
    )
    return top


def _gantry(p: Painter, feet: tuple[Point, Point], height: float) -> None:
    """A yellow gantry crane on two legs standing on rails, its beam across their tops."""
    for x, y in feet:
        p.line([(x, y), (x, y - height)], CRANE)
        p.line([(x + 0.6, y), (x + 0.6, y - height)], shade(CRANE, 0.75))
        p.line([(x - 1.5, y), (x, y - 4)], shade(CRANE, 0.7))
        p.rect(x - 1.5, y - 0.5, x + 1.5, y + 0.5, (70, 70, 76))
    (x0, y0), (x1, y1) = feet
    p.line([(x0 - 1, y0 - height), (x1 + 1, y1 - height)], CRANE)
    p.line([(x0 - 1, y0 - height + 1), (x1 + 1, y1 - height + 1)], shade(CRANE, 0.7))
    for k in range(1, 4):
        a = _on_wall((x0, y0 - height), (x1, y1 - height), (k - 0.5) / 3, 0)
        b = _on_wall((x0, y0 - height), (x1, y1 - height), k / 3, 1)
        p.line([a, b], shade(CRANE, 0.85))


def _wheel(p: Painter, cx: float, cy: float, rx: float, ry: float, colour: RGB) -> None:
    """A spoked iron wheel seen a little edge on, centred on (cx, cy), open between its spokes."""
    for inset in (0, 0.7):
        ring = [
            (
                cx + (rx - inset) * math.cos(a * math.pi / 8),
                cy + (ry - inset) * math.sin(a * math.pi / 8),
            )
            for a in range(17)
        ]
        p.line(ring, colour if inset else shade(colour, 0.7))
    for a in range(4):
        ang = a * math.pi / 4
        dx, dy = rx * 0.9 * math.cos(ang), ry * 0.9 * math.sin(ang)
        p.line([(cx - dx, cy - dy), (cx + dx, cy + dy)], colour)
    p.point(cx, cy, BRASS)


def _flywheel(p: Painter, x: float, y: float, r: float) -> None:
    """A stationary steam engine on (x, y): a great spoked flywheel beside its green cylinder on
    an iron bed, a brass rod between them."""
    p.rect(x - 1, y - 2, x + 6, y, CASTING)
    _wheel(p, x - r * 0.15, y - r, r * 0.7, r, shade(CASTING, 0.9))
    p.rect(x + 2, y - 4.5, x + 6, y - 2, METAL_GREEN)
    p.line([(x + 2, y - 3.6), (x + 6, y - 3.6)], shade(METAL_GREEN, 1.4))
    p.line([(x - r * 0.15, y - r), (x + 2, y - 3)], BRASS)


def _boiler(p: Painter, x: float, y: float) -> None:
    """A new locomotive boiler lying on blocks at (x, y), its smokebox end dark."""
    p.rect(x - 1, y - 1, x + 1, y, (110, 84, 60))
    p.rect(x + 5, y - 1, x + 7, y, (110, 84, 60))
    p.polygon(
        [(x - 2, y - 5), (x + 7, y - 0.5), (x + 7, y - 4.5), (x - 2, y - 9)],
        CASTING,
        shade(CASTING, 0.6),
    )
    p.line([(x - 2, y - 7.6), (x + 7, y - 3.1)], shade(CASTING, 1.4))
    for t in (0.3, 0.65):
        p.line(
            [(x - 2 + 9 * t, y - 9 + 4.5 * t), (x - 2 + 9 * t, y - 5 + 4.5 * t)],
            shade(CASTING, 0.75),
        )
    p.ellipse(x + 5.4, y - 4.8, x + 8.4, y - 0.4, (40, 38, 42))


def big_machine_works() -> Sprite:
    """A bigger machine works: a long brick engine hall with tall arched windows, a glazed roof
    light and a great brass gear on its end wall, a lower erecting shop beside it, a tall banded
    chimney smoking, and a gravelled yard where a gantry crane on rails lifts a casting over a
    boiler lying on blocks, with pipes piled up and oil stains."""
    p = Painter(54, 70, (26, 54))
    foot = _footing(-6, -3, 20, 10)
    _bed(p, foot)
    _bed(p, _footing(12, -9, 10, 7))
    p.polygon([(4, 9), (17, 2.5), (26, 7), (13, 13.5)], GRAVEL)
    p.shadow(20, 5)
    _smoking_chimney(p, 2, -14, -40, shade(BRICK, 0.85))
    shop = p.hall(12, -9, 10, 7, 7, shade(BRICK, 0.95), None)
    _plinth(p, shop, 7, BRICK)
    _windows(p, shop[0], shop[1], (0.25, 0.5, 0.75), (2.4,), 3, arch=shade(BRICK, 1.3))
    _door(p, shop[1], shop[2], (0.25, 0.75), 2.6, 7, (58, 48, 44))
    p.gable(shop, 3, ENGINE_ROOF, BRICK)
    _track(p, foot[0], foot[1], (0.82, 0.96), 4)
    for dx, dy in ((0, 0), (7, -3.5)):
        p.line([(6 + dx, 10 + dy), (12 + dx, 13 + dy)], RAIL)
    _oil(p, 17, 8, 1.4)
    top = _engine_hall(p, -6, -3, 20, 10, 12)
    _, front, right, _ = top
    p.gable(top, 6, ENGINE_ROOF, BRICK)
    _skylight(p, top, 6, (0.06, 0.94))
    _gear(p, *_on_wall(front, right, 0.5, 6), 3.2, BRASS)
    _lengths(p, -19, 4, (96, 100, 112), 3)
    _boiler(p, 12, 10)
    _gantry(p, ((8, 11), (19, 5.5)), 18)
    p.rect(12.5, -10, 14.5, -8.5, (60, 64, 74))
    p.line([(13.5, -8.5), (13.5, -4)], (70, 64, 58))
    p.rect(11, -4, 16, -0.5, CASTING)
    p.line([(11.5, -3), (15.5, -3)], shade(CASTING, 1.3))
    return p.sprite()


def mega_machine_works() -> Sprite:
    """A great machine works: a long brick engine hall with tall arched windows under a raised
    glazed clerestory, a giant brass gear on its gable, two tall banded chimneys smoking, a big
    gantry crane spanning the yard on rails lifting a flywheel, a stationary steam engine and a
    new locomotive out front on a rail siding, a boiler lying on blocks and pipes piled up."""
    p = Painter(64, 80, (31, 60))
    foot = _footing(-8, -6, 28, 11)
    _bed(p, foot)
    p.polygon([(-2, 10), (16, 1), (31, 8.5), (13, 17.5)], GRAVEL)
    p.shadow(25, 6)
    for x, foot_y, tall in ((6, -18, 30), (12, -15, 32)):
        cap = _chimney(p, x, foot_y, foot_y - tall, shade(BRICK, 0.85))
        _ring(p, x, foot_y - (foot_y - cap) * 0.4, 1.5, shade(BRICK, 0.6))
        _smoke(p, x + 1, cap - 3, SMOKE, 3)
    _track(p, foot[0], foot[1], (0.86, 0.97), 4)
    _oil(p, 20, 12, 1.6)
    _oil(p, 8, 10, 1.2)
    top = _engine_hall(p, -8, -6, 28, 11, 15)
    _, front, right, _ = top
    p.gable(top, 4, ENGINE_ROOF, BRICK)
    clere = p.hall(-8, -6 - 15 - 3, 24, 4, 3, shade(BRICK, 0.95), None)
    _patch(p, clere[0], clere[1], (0.03, 0.97), (0.6, 2.4), GLAZING)
    for t in (0.2, 0.4, 0.6, 0.8):
        p.line(
            [_on_wall(clere[0], clere[1], t, 0.6), _on_wall(clere[0], clere[1], t, 2.4)],
            shade(BRICK, 0.8),
        )
    p.gable(clere, 2, ENGINE_ROOF, BRICK)
    _gear(p, *_on_wall(front, right, 0.5, 6.5), 4, BRASS)
    _rails(p, (-26, 6), (-2, 18), 10)
    _locomotive(p, -12, 13.5)
    _flywheel(p, 0, 9, 4)
    _boiler(p, 18, 13)
    _lengths(p, -24, -2, (96, 100, 112), 3)
    _gantry(p, ((10, 16), (26, 8)), 22)
    p.rect(16, -13, 18.5, -11.5, (60, 64, 74))
    p.line([(17.2, -11.5), (17.2, -7)], (70, 64, 58))
    _wheel(p, 17.2, -4.1, 2.4, 3, shade(CASTING, 0.9))
    return p.sprite()


SADDLER_WALL: RGB = (222, 206, 178)
SADDLER_ROOF: RGB = (92, 60, 48)
CARRIAGE: RGB = (34, 32, 36)
HORSE: RGB = (112, 72, 44)


def _leather_house(
    p: Painter, x: float, y: float, length: float, depth: float, height: float
) -> Top:
    """A cream saddlery with a saddle-brown sign studded with brass along under its eaves."""
    top = p.hall(x, y, length, depth, height, SADDLER_WALL, None)
    left, front, right, _ = top
    _patch(p, left, front, (0, 1), (0.4, 1.8), SADDLER_SIGN)
    _patch(p, front, right, (0, 1), (0.4, 1.8), shade(SADDLER_SIGN, 0.78))
    for t in (0.2, 0.4, 0.6, 0.8):
        p.point(*_on_wall(left, front, t, 1.1), BUCKLE)
    return top


def big_saddler() -> Sprite:
    """A long two-storey saddlery of cream walls under a brown roof and a saddle-brown sign with
    brass studs, a shop window of saddles and bags, collars hung by its doors, a workshop shed
    behind, saddles on trestles out in the yard and rolled hides by the door."""
    p = Painter(50, 44, (24, 32))
    foot = _footing(4, -1, 18, 8)
    _bed(p, foot, SOIL, 1)
    p.shadow(14, 4)
    shed = p.hall(12, -8, 10, 6, 5, shade(SADDLER_WALL, 0.9), None)
    _sawtooth(p, shed, 3, 3, SADDLER_ROOF, GLAZING, shade(SADDLER_WALL, 0.9))
    _track(p, foot[0], foot[1], (0.8, 0.94), 3, (190, 172, 136), ruts=False)
    top = _leather_house(p, 4, -1, 18, 8, 11)
    left, front, right, _ = top
    for t in (0.12, 0.32, 0.52):
        _window(p, left, front, (t - 0.06, t + 0.06), 3, shade(WINDOW, 0.95), 2.4)
    _patch(p, left, front, (0.05, 0.7), (7.4, 9.6), shade(WINDOW, 0.9))
    for k, t in enumerate((0.15, 0.3, 0.45, 0.6)):
        p.point(*_on_wall(left, front, t, 8.6), (SADDLE, SADDLE_TAN)[k % 2])
    _door(p, left, front, (0.8, 0.94), 6.5, 11, SADDLER_SIGN)
    _window(p, front, right, (0.3, 0.7), 3, shade(WINDOW, 0.8))
    _window(p, front, right, (0.3, 0.7), 7, shade(WINDOW, 0.8))
    p.gable(top, 5, SADDLER_ROOF, SADDLER_WALL)
    _roof_rows(p, top, 5, SADDLER_ROOF)
    x, y = _on_roof(top, 5, 0.3, 0.9)
    p.rect(x - 1, y - 3, x, y, BRICK)
    cx, cy = _on_wall(left, front, 0.74, 7)
    _collar(p, cx, cy)
    for k, (x, y) in enumerate(((-19, 1), (-14.5, 3.5), (-10, 6), (-15, 7.5))):
        _saddle(p, x, y, (SADDLE, SADDLE_TAN)[k % 2])
    for k, (x, y) in enumerate(((10, 9), (12, 8), (11, 6.8))):
        _rolled(p, x, y, (SADDLE, SADDLE_TAN, shade(SADDLE, 0.8))[k])
    return p.sprite()


def mega_saddler() -> Sprite:
    """A great leather goods house of three storeys under a brown roof, a saddle-brown and gilt
    sign along its eaves and a gilt horse's head on its roof, shop windows of saddles and bags
    along its whole front, a long workshop shed behind, rows of saddles on trestles in the yard,
    rolled hides piled by the door and a black carriage with a horse in polished harness waiting
    at the kerb."""
    p = Painter(64, 58, (32, 40))
    foot = _footing(4, -3, 26, 10)
    _bed(p, foot, SOIL, 1)
    p.shadow(20, 5)
    shed = p.hall(16, -12, 14, 8, 6, shade(SADDLER_WALL, 0.9), None)
    _sawtooth(p, shed, 4, 3, SADDLER_ROOF, GLAZING, shade(SADDLER_WALL, 0.9))
    _track(p, foot[0], foot[1], (0.86, 0.96), 4, (190, 172, 136), ruts=False)
    top = _leather_house(p, 4, -3, 26, 10, 16)
    left, front, right, _ = top
    for row in (3, 7.5):
        for k in range(6):
            t = 0.08 + k * 0.13
            _window(p, left, front, (t - 0.04, t + 0.04), row, shade(WINDOW, 0.95), 2.2)
    _patch(p, left, front, (0.04, 0.84), (12.4, 15), shade(WINDOW, 0.9))
    for k in range(9):
        p.point(*_on_wall(left, front, 0.08 + k * 0.09, 13.8), (SADDLE, SADDLE_TAN, BUCKLE)[k % 3])
    _door(p, left, front, (0.88, 0.97), 10.5, 16, SADDLER_SIGN, BUCKLE)
    for row in (3, 7.5, 12):
        _window(p, front, right, (0.2, 0.4), row, shade(WINDOW, 0.8))
        _window(p, front, right, (0.6, 0.8), row, shade(WINDOW, 0.8))
    p.gable(top, 6, SADDLER_ROOF, SADDLER_WALL)
    _roof_rows(p, top, 6, SADDLER_ROOF, (0.25, 0.5, 0.75))
    for along in (0.25, 0.7):
        x, y = _on_roof(top, 6, along, 0.9)
        p.rect(x - 1, y - 3, x, y, BRICK)
    rx, ry = _on_roof(top, 6, 0.5, 1)
    p.polygon(
        [
            (rx - 1.5, ry - 1),
            (rx - 1, ry - 5),
            (rx + 1, ry - 7),
            (rx + 2.6, ry - 6),
            (rx + 1.2, ry - 4.6),
            (rx + 1, ry - 1),
        ],
        BUCKLE,
        shade(BUCKLE, 0.6),
    )
    p.point(rx + 1.2, ry - 6, shade(BUCKLE, 0.5))
    for k, (x, y) in enumerate(((-27, -2), (-23, 0), (-19, 2), (-24, 5), (-20, 7), (-16, 9))):
        _saddle(p, x, y, (SADDLE, SADDLE_TAN)[k % 2])
    for k, (x, y) in enumerate(((13, 11), (15, 10), (14, 8.8), (16.5, 8.5))):
        _rolled(p, x, y, (SADDLE, SADDLE_TAN, shade(SADDLE, 0.8), SADDLE_TAN)[k])
    _carriage(p, -4, 14)
    return p.sprite()


def _carriage(p: Painter, x: float, y: float) -> None:
    """A black carriage with yellow wheels and a brown horse in harness before it, seen side on,
    standing on (x, y)."""
    body = CARRIAGE
    p.blob(x - 4, y - 5, x + 2, y - 1.6, body)
    p.rect(x - 3.6, y - 7, x + 1, y - 5, shade(body, 1.3))
    p.rect(x - 3, y - 6.6, x - 1, y - 5.2, (150, 180, 200))
    for wx, r in ((x - 3, 1.5), (x + 1.4, 1.1)):
        p.ellipse(wx - r, y - 2 * r, wx + r, y, (200, 160, 60))
        p.ellipse(wx - r + 0.5, y - 2 * r + 0.5, wx + r - 0.5, y - 0.5, body)
    p.line([(x + 2, y - 2.6), (x + 6, y - 3)], HARNESS)
    hx = x + 7
    horse = HORSE
    p.blob(hx - 2, y - 4.6, hx + 2, y - 2.4, horse)
    for dx in (-1.6, -0.9, 1.0, 1.7):
        p.line([(hx + dx, y - 2.6), (hx + dx, y)], shade(horse, 0.6))
    p.polygon(
        [(hx + 1.4, y - 4.2), (hx + 2.4, y - 6.6), (hx + 3.6, y - 6), (hx + 2.6, y - 3.8)], horse
    )
    p.line([(hx - 1.5, y - 4.4), (hx + 1.6, y - 4.4)], HARNESS)
    p.point(hx, y - 4.4, BUCKLE)
    p.line([(hx - 2, y - 4.2), (hx - 2.6, y - 2.2)], shade(horse, 0.5))
