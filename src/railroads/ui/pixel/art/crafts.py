"""Town crafts and the town's grand buildings: coke plants, potters, tile works, foundries,
dairies, sausage makers, the makers of luxury goods, manors, theatres and hospitals."""

from railroads.ui.pixel.art.homes import DOOR, _chimney_on, _foot, _tiles, _window
from railroads.ui.pixel.art.painter import (
    BRICK,
    GLASS,
    METAL,
    RGB,
    WHITEWASH,
    WINDOW,
    Painter,
    Sprite,
    shade,
)
from railroads.ui.pixel.art.parts import _heap, _on_wall, _patch

ROOF: RGB = (150, 70, 52)
SLATE: RGB = (88, 84, 92)


def _shop(p: Painter, x: float, y: float, wall: RGB, roof: RGB, sign: RGB) -> None:
    """A small workshop: a door, a window, a sign over the door, a tiled roof and a chimney."""
    top = p.box(x, y, 5, 4, wall, None)
    left, front, right, _ = top
    _patch(p, left, front, (0.45, 0.65), (1.3, 4), DOOR)
    _window(p, front, right, (0.3, 0.6), 1.0, shade(WINDOW, 0.8))
    p.line([_on_wall(left, front, 0.35, 0.8), _on_wall(left, front, 0.75, 0.8)], sign)
    _foot(p, top, 4)
    p.gable(top, 3, roof, wall)
    _tiles(p, top, 3, roof)
    _chimney_on(p, top, 3, 0.3, 1.4)


def _hall(p: Painter, wall: RGB, roof: RGB, length: float = 16) -> tuple:
    top = p.hall(0, 0, length, 8, 6, wall, None)
    left, front, _, _ = top
    for f in (0.15, 0.4, 0.65, 0.88):
        _patch(p, left, front, (f - 0.05, f + 0.05), (2, 4), shade(WINDOW, 0.85))
    _foot(p, top, 6)
    p.gable(top, 3.5, roof, wall)
    return top


def coke_plant() -> Sprite:
    """A low battery of brick coke ovens with their doors in a row, a tall chimney smoking above
    it, a quench tower and a heap of grey coke beside."""
    p = Painter(32, 30, (15, 21))
    p.shadow(13, 3.5)
    top = p.hall(-1, 1, 18, 7, 4, BRICK, (70, 66, 64))
    left, front, _, _ = top
    for f in (0.1, 0.25, 0.4, 0.55, 0.7, 0.85):
        _patch(p, left, front, (f - 0.04, f + 0.04), (1.2, 3.4), (40, 36, 36))
    p.rect(9, -18, 10.6, -2, shade(BRICK, 0.85))
    p.line([(9, -19), (10.6, -19)], shade(BRICK, 0.5))
    for x, y in ((10, -21), (11.5, -23), (10.5, -25)):
        p.point(x, y, (196, 196, 200))
    p.rect(-13, -10, -9, 2, (120, 116, 110))
    _heap(p, 12, 6, 3, 3, (70, 70, 76))
    return p.sprite()


def potter() -> Sprite:
    """A potter's cottage with a round brick kiln beside it and pots set out to dry."""
    p = Painter(26, 22, (12, 16))
    p.shadow(9, 2.5)
    _shop(p, -3, 0, WHITEWASH, ROOF, (176, 104, 70))
    p.ellipse(4, -6, 11, 2, shade(BRICK, 0.9))
    p.rect(6, -9, 9, -5, shade(BRICK, 0.8))
    for x in (-10, -7.5, 9, 11.5):
        p.ellipse(x - 1, 2.5, x + 1, 5, (176, 104, 70))
    return p.sprite()


def tile_works() -> Sprite:
    """A brick works shed with a bottle kiln and stacks of glazed blue stove tiles."""
    p = Painter(30, 26, (14, 19))
    p.shadow(12, 3)
    _hall(p, BRICK, SLATE, 14)
    p.ellipse(8, -12, 13, -2, shade(BRICK, 0.85))
    p.rect(9.5, -16, 11.5, -10, shade(BRICK, 0.75))
    for x in (-11, -8):
        p.rect(x, 2, x + 2.4, 5, (70, 120, 170))
        p.line([(x, 3.5), (x + 2.4, 3.5)], (236, 236, 226))
    return p.sprite()


def foundry() -> Sprite:
    """A soot-dark foundry hall with a cupola furnace glowing at its door and iron castings laid
    out on the sand in front."""
    p = Painter(30, 28, (14, 20))
    p.shadow(12, 3)
    top = _hall(p, shade(BRICK, 0.75), SLATE, 15)
    left, front, _, _ = top
    _patch(p, left, front, (0.45, 0.6), (2, 6), (230, 120, 40))
    p.rect(9, -18, 11, -4, (60, 60, 64))
    p.point(10, -19, (230, 120, 40))
    p.blob(-12, 3, -4, 6, (214, 196, 146))
    for x in (-10, -7):
        p.ellipse(x - 1.2, 3.4, x + 1.2, 5.4, (74, 76, 82))
    return p.sprite()


def dairy() -> Sprite:
    """A whitewashed dairy with a blue door, milk churns lined up on a bench and a cheese wheel
    in the window."""
    p = Painter(24, 20, (11, 15))
    p.shadow(9, 2.5)
    _shop(p, -2, 0, WHITEWASH, SLATE, (90, 130, 190))
    for x in (5, 7, 9):
        p.rect(x - 0.8, 1, x + 0.8, 4, (196, 200, 206))
        p.point(x, 0.6, shade((196, 200, 206), 0.6))
    p.ellipse(-10, 2, -6, 5, (238, 200, 90))
    return p.sprite()


def sausage_maker() -> Sprite:
    """A half-timbered butcher's shop with a smokehouse chimney and sausages hung over its door."""
    p = Painter(24, 22, (11, 16))
    p.shadow(9, 2.5)
    _shop(p, -1, 0, (226, 210, 180), ROOF, (130, 80, 50))
    p.rect(6, -9, 8, 0, shade(BRICK, 0.8))
    for x in (-6, -4.5, -3):
        p.line([(x, -1), (x, 1.5)], (170, 74, 54))
    return p.sprite()


def _luxury_shop(sign: RGB, goods: RGB) -> Sprite:
    p = Painter(22, 24, (10, 18))
    p.shadow(8, 2.5)
    top = p.box(0, 0, 5, 6, (214, 200, 176), None)
    left, front, right, _ = top
    _patch(p, left, front, (0.15, 0.85), (2, 4.6), shade(WINDOW, 0.9))
    for f in (0.3, 0.55):
        p.point(*_on_wall(left, front, f, 3.4), goods)
    _patch(p, left, front, (0.15, 0.85), (5, 5.8), sign)
    _window(p, front, right, (0.3, 0.6), 1.2, shade(WINDOW, 0.8))
    _foot(p, top, 6)
    p.gable(top, 3, SLATE, (214, 200, 176))
    return p.sprite()


def carpet_weaver() -> Sprite:
    """A townhouse workshop with a red sign, carpets in reds and blues hung in its window."""
    return _luxury_shop((150, 40, 40), (60, 90, 170))


def clock_maker() -> Sprite:
    """A townhouse shop with a dark green sign and a clock face showing in its window."""
    return _luxury_shop((40, 90, 60), (232, 190, 70))


def mirror_maker() -> Sprite:
    """A townhouse shop with a gilded sign and mirrors glinting in its window."""
    return _luxury_shop((200, 160, 60), GLASS)


def manor() -> Sprite:
    """A white manor house under a tall slate roof, a pillared porch at its door, and a lawn
    with a gravel drive before it."""
    p = Painter(34, 28, (16, 20))
    p.blob(-14, -1, 14, 7, (120, 168, 84))
    p.shadow(13, 3.5)
    top = p.hall(0, 0, 20, 9, 7, WHITEWASH, None)
    left, front, _, _ = top
    for f in (0.1, 0.25, 0.75, 0.9):
        _patch(p, left, front, (f - 0.04, f + 0.04), (2, 3.6), WINDOW)
        _patch(p, left, front, (f - 0.04, f + 0.04), (4.6, 6.2), WINDOW)
    for f in (0.42, 0.5, 0.58):
        p.line([_on_wall(left, front, f, 0), _on_wall(left, front, f, 5)], (240, 236, 226))
    p.polygon(
        [
            _on_wall(left, front, 0.38, 5),
            _on_wall(left, front, 0.62, 5),
            _on_wall(left, front, 0.5, 7),
        ],
        WHITEWASH,
        shade(WHITEWASH, 0.6),
    )
    _foot(p, top, 7)
    p.gable(top, 5, SLATE, WHITEWASH)
    p.line([(-10, 6), (0, 4)], (214, 200, 170))
    return p.sprite()


def theatre() -> Sprite:
    """A theatre with a columned front under a pediment, a dome on its roof and posters by its
    doors."""
    p = Painter(30, 30, (14, 21))
    p.shadow(12, 3.5)
    wall = (226, 214, 188)
    top = p.box(0, 0, 10, 8, wall, (120, 110, 104))
    left, front, _, _ = top
    for f in (0.2, 0.35, 0.5, 0.65, 0.8):
        p.line([_on_wall(left, front, f, 0), _on_wall(left, front, f, 6)], shade(wall, 1.1))
    _patch(p, left, front, (0.44, 0.56), (1.5, 4), DOOR)
    p.polygon(
        [
            _on_wall(left, front, 0.1, 8),
            _on_wall(left, front, 0.9, 8),
            _on_wall(left, front, 0.5, 10),
        ],
        wall,
        shade(wall, 0.6),
    )
    p.ellipse(-3, -16, 3, -10, (80, 130, 110))
    p.point(0, -17, (232, 190, 70))
    for f in (0.12, 0.88):
        _patch(p, left, front, (f - 0.03, f + 0.03), (1.5, 3.5), (200, 60, 60))
    _foot(p, top, 8)
    return p.sprite()


def hospital() -> Sprite:
    """A long whitewashed hospital of two storeys with rows of windows, a red cross over its
    door and a flag on its roof."""
    p = Painter(32, 28, (15, 20))
    p.shadow(13, 3.5)
    top = p.hall(0, 0, 20, 8, 8, WHITEWASH, None)
    left, front, _, _ = top
    for f in (0.1, 0.22, 0.34, 0.66, 0.78, 0.9):
        for height in ((2, 3.4), (5, 6.4)):
            _patch(p, left, front, (f - 0.03, f + 0.03), height, GLASS)
    _patch(p, left, front, (0.45, 0.55), (1.5, 4.5), DOOR)
    x, y = _on_wall(left, front, 0.5, 6.4)
    p.rect(x - 1.5, y - 0.5, x + 1.5, y + 0.5, (200, 40, 40))
    p.rect(x - 0.5, y - 1.5, x + 0.5, y + 1.5, (200, 40, 40))
    _foot(p, top, 8)
    p.gable(top, 3, SLATE, WHITEWASH)
    p.line([(8, -14), (8, -8)], METAL)
    p.rect(8, -14, 11, -12, (200, 40, 40))
    return p.sprite()
