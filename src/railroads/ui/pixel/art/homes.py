"""Houses, tenements, flats and skyscrapers."""

from railroads.ui.pixel.art.painter import (
    BRICK,
    GLASS,
    METAL,
    RGB,
    WINDOW,
    Painter,
    Point,
    Sprite,
    shade,
)
from railroads.ui.pixel.art.parts import _on_wall, _patch

DOOR: RGB = (104, 66, 44)
SILL: RGB = (240, 236, 226)
AWNINGS: list[RGB] = [(178, 52, 46), (52, 104, 70), (60, 88, 150)]
LIT: RGB = (236, 206, 120)
STEP: RGB = (206, 200, 188)
CONCRETE: RGB = (176, 178, 182)
SOIL: RGB = (74, 64, 46)
TRODDEN: RGB = (146, 130, 92)


def _ridge(top: tuple[Point, Point, Point, Point], rise: float) -> tuple[Point, Point]:
    """The back and front ends of the ridge of a gable `rise` high on a box's top (Painter.gable)."""
    left, front, right, back = top
    return (
        ((left[0] + back[0]) / 2, (left[1] + back[1]) / 2 - rise),
        ((front[0] + right[0]) / 2, (front[1] + right[1]) / 2 - rise),
    )


def _tiles(p: Painter, top: tuple[Point, Point, Point, Point], rise: float, roof: RGB) -> None:
    """Rows of tiles across the lit side of a gable roof."""
    left, front, _, _ = top
    ridge_back, ridge_front = _ridge(top, rise)
    for f in (0.34, 0.68):
        a = (left[0] + (ridge_back[0] - left[0]) * f, left[1] + (ridge_back[1] - left[1]) * f)
        b = (front[0] + (ridge_front[0] - front[0]) * f, front[1] + (ridge_front[1] - front[1]) * f)
        p.line([a, b], shade(roof, 0.8))


def _chimney_on(
    p: Painter, top: tuple[Point, Point, Point, Point], rise: float, along: float, tall: float = 2
) -> None:
    """A brick chimney with a dark cap standing on the ridge, `along` of the way to its front."""
    ridge_back, ridge_front = _ridge(top, rise)
    x = ridge_back[0] + (ridge_front[0] - ridge_back[0]) * along
    y = ridge_back[1] + (ridge_front[1] - ridge_back[1]) * along
    p.rect(x - 1, y - tall, x, y + 1, BRICK)
    p.line([(x - 1, y - tall - 1), (x, y - tall - 1)], shade(BRICK, 0.55))


def _window(
    p: Painter, a: Point, b: Point, span: tuple[float, float], drop: float, fill: RGB
) -> None:
    """A window with a light sill under it on the wall under the edge from `a` to `b`."""
    _patch(p, a, b, span, (drop, drop + 1.2), fill)
    p.line([_on_wall(a, b, span[0], drop + 1.5), _on_wall(a, b, span[1], drop + 1.5)], SILL)


def _foot(p: Painter, top: tuple[Point, Point, Point, Point], height: float) -> None:
    """A dark line of shaded earth where a box's two front walls meet the ground."""
    left, front, right, _ = top
    feet = [_on_wall(left, front, 0, height), _on_wall(front, right, 0, height)]
    feet.append(_on_wall(front, right, 1, height))
    p.line([(x + 0.5, y + 0.5) for x, y in feet], SOIL)


def _outward(a: Point, b: Point, reach: float) -> Point:
    """How far to move over the ground to go `reach` straight out from the wall under the edge
    from `a` to `b`, lit or shaded."""
    across = abs(b[0] - a[0]) or 1
    return -2 * reach * (b[1] - a[1]) / across, reach / 2


def _doorstep(
    p: Painter, a: Point, b: Point, span: tuple[float, float], height: float, fill: RGB = STEP
) -> None:
    """A low step of stone on the ground before a door in the wall under the edge from `a` to
    `b`, lit on top with a darker riser."""
    near, far = _on_wall(a, b, span[0], height), _on_wall(a, b, span[1], height)
    dx, dy = _outward(a, b, 1.2)
    out = [(far[0] + dx, far[1] + dy), (near[0] + dx, near[1] + dy)]
    p.polygon([near, far, *out], fill, shade(fill, 0.6))
    p.line([(x, y + 0.6) for x, y in out], shade(fill, 0.62))


def _worn(
    p: Painter,
    a: Point,
    b: Point,
    span: tuple[float, float],
    height: float,
    reach: float = 3,
    fill: RGB = TRODDEN,
) -> None:
    """A patch of trodden earth or paving spreading `reach` out over the ground before the wall
    under the edge from `a` to `b`, from `span[0]` to `span[1]` of the way along it."""
    near, far = _on_wall(a, b, span[0], height), _on_wall(a, b, span[1], height)
    spread = (far[0] - near[0]) * 0.25, (far[1] - near[1]) * 0.25
    dx, dy = _outward(a, b, reach)
    out = [
        (far[0] + spread[0] + dx, far[1] + spread[1] + dy),
        (near[0] - spread[0] + dx, near[1] - spread[1] + dy),
    ]
    p.polygon([near, far, *out], fill)


def _eaves(p: Painter, top: tuple[Point, Point, Point, Point], wall: RGB) -> None:
    """The shadow the eaves cast along the tops of a box's two front walls."""
    left, front, right, _ = top
    p.line([_on_wall(left, front, 0, 0.6), _on_wall(left, front, 1, 0.6)], shade(wall, 0.78))
    p.line([_on_wall(front, right, 0, 0.6), _on_wall(front, right, 1, 0.6)], shade(wall, 0.6))


def house(wall: RGB, roof: RGB) -> Sprite:
    """A cottage on a stone plinth: a door and windows with sills, a tiled gable roof and a
    chimney."""
    p = Painter(15, 16, (7, 12))
    p.shadow(6, 2)
    top = p.box(0, 0, 6, 4, wall, None)
    left, front, right, _ = top
    _patch(p, left, front, (0, 1), (3.2, 4), shade(wall, 0.7))
    _patch(p, front, right, (0, 1), (3.2, 4), shade(wall, 0.5))
    _eaves(p, top, wall)
    for span in ((0.1, 0.27), (0.73, 0.9)):
        _window(p, left, front, span, 1.1, WINDOW)
    _patch(p, left, front, (0.44, 0.6), (1.4, 4), DOOR)
    p.line([_on_wall(left, front, 0.42, 1.2), _on_wall(left, front, 0.62, 1.2)], shade(wall, 0.6))
    _window(p, front, right, (0.3, 0.5), 1.1, shade(WINDOW, 0.8))
    _foot(p, top, 4)
    _doorstep(p, left, front, (0.4, 0.64), 4)
    p.gable(top, 4, roof, wall)
    _tiles(p, top, 4, roof)
    _, ridge_front = _ridge(top, 4)
    p.point(ridge_front[0] - 0.5, ridge_front[1] + 1.6, shade(WINDOW, 0.8))
    _chimney_on(p, top, 4, 0.35, 1.4)
    return p.sprite()


def tenement(wall: RGB, roof: RGB) -> Sprite:
    """A tenement (kamienica): a shop on the ground floor under an awning, two painted floors of
    windows with sills between cornices, and a steep tiled roof with a dormer and chimneys."""
    p = Painter(17, 26, (8, 20))
    p.shadow(7, 2.5)
    s, height = 6, 10
    top = p.box(0, 0, s, height, wall, None)
    left, front, right, _ = top
    awning = AWNINGS[sum(wall) % len(AWNINGS)]
    _patch(p, left, front, (0, 1), (7, height), shade(wall, 0.78))
    _patch(p, front, right, (0, 1), (7, height), shade(wall, 0.56))
    _patch(p, left, front, (0.08, 0.5), (7.7, 9.6), GLASS)
    _patch(p, left, front, (0.64, 0.82), (7.5, height), DOOR)
    p.line([_on_wall(left, front, 0.29, 7.7), _on_wall(left, front, 0.29, 9.6)], shade(wall, 0.6))
    _patch(p, left, front, (0.05, 0.53), (6.6, 7.5), awning)
    for t in (0.17, 0.37):
        p.point(*_on_wall(left, front, t, 6.7), SILL)
    _patch(p, front, right, (0.25, 0.6), (7.7, 9.4), shade(GLASS, 0.75))
    _foot(p, top, height)
    _doorstep(p, left, front, (0.6, 0.86), height)
    for drop in (1.2, 4.0):
        for span in ((0.12, 0.27), (0.43, 0.58), (0.74, 0.89)):
            _window(p, left, front, span, drop, WINDOW)
        for span in ((0.25, 0.4), (0.62, 0.77)):
            _window(p, front, right, span, drop, shade(WINDOW, 0.75))
    for drop in (0.3, 3.3, 6.2):
        p.line([_on_wall(left, front, 0, drop), _on_wall(left, front, 1, drop)], shade(wall, 1.15))
        p.line(
            [_on_wall(front, right, 0, drop), _on_wall(front, right, 1, drop)], shade(wall, 0.85)
        )
    p.gable(top, 5, roof, wall)
    _tiles(p, top, 5, roof)
    ridge_back, ridge_front = _ridge(top, 5)
    dx = (ridge_back[0] + ridge_front[0]) / 2 - 2
    dy = (ridge_back[1] + ridge_front[1]) / 2 + 2
    p.rect(dx - 1, dy - 1, dx + 1, dy + 1, shade(wall, 0.95))
    p.point(dx, dy, WINDOW)
    _chimney_on(p, top, 5, 0.2, 1.5)
    _chimney_on(p, top, 5, 0.8, 1.5)
    return p.sprite()


def flats(wall: RGB) -> Sprite:
    """A block of flats: five floors under a flat roof with a parapet and a lift house, rows of
    windows with balconies, a few lit, and an entrance under a canopy."""
    p = Painter(19, 32, (9, 26))
    p.shadow(8, 3, reach=5)
    s, floors = 7, 5
    height = 3 * floors + 1
    top = p.box(0, 0, s, height, wall, shade(wall, 0.62))
    left, front, right, _ = top
    for floor in range(floors):
        drop = 1 + 3 * floor
        for k, span in enumerate(((0.12, 0.24), (0.44, 0.56), (0.76, 0.88))):
            lit = (floor * 3 + k) % 7 == 2
            _patch(p, left, front, span, (drop, drop + 1), LIT if lit else WINDOW)
        for span in ((0.25, 0.37), (0.63, 0.75)):
            _patch(p, front, right, span, (drop, drop + 1), shade(WINDOW, 0.75))
        if floor < floors - 1:
            p.line(
                [_on_wall(left, front, 0.36, drop + 2), _on_wall(left, front, 0.64, drop + 2)],
                shade(wall, 0.8),
            )
    _patch(p, left, front, (0, 1), (height - 0.8, height), shade(wall, 0.72))
    _patch(p, front, right, (0, 1), (height - 0.8, height), shade(wall, 0.54))
    _patch(p, left, front, (0.44, 0.58), (height - 2.4, height), DOOR)
    _foot(p, top, height)
    _doorstep(p, left, front, (0.4, 0.62), height, CONCRETE)
    p.line(
        [_on_wall(left, front, 0.38, height - 2.8), _on_wall(left, front, 0.64, height - 2.8)],
        shade(wall, 0.55),
    )
    p.line([left, front, right], shade(wall, 1.15))
    p.rect(-1, -height - 3, 1, -height - 1, shade(wall, 0.7))
    p.point(0, -height - 2, shade(METAL, 0.8))
    return p.sprite()


def skyscraper() -> Sprite:
    """A glass tower with a grid of windows and an aerial on top."""
    p = Painter(17, 60, (8, 54))
    p.shadow(7, 2.5, reach=8)
    s, height = 6, 44
    top = p.box(0, 0, s, height, GLASS, shade(GLASS, 0.8))
    left, front, right, _ = top
    lit, dark = shade(GLASS, 1.35), shade(GLASS, 0.62)
    for floor in range(2, height - 4, 3):
        p.line([(-s, -floor), (0, s / 2 - floor)], lit)
        p.line([(0, s / 2 - floor), (s, -floor)], dark)
    p.line([(-s / 2, s / 4 - 4), (-s / 2, s / 4 - height)], shade(GLASS, 0.85))
    p.line([_on_wall(left, front, 0, 0.6), _on_wall(left, front, 1, 0.6)], shade(GLASS, 1.6))
    p.line([_on_wall(front, right, 0, 0.6), _on_wall(front, right, 1, 0.6)], shade(GLASS, 1.1))
    _patch(p, left, front, (0, 1), (height - 3.6, height), CONCRETE)
    _patch(p, front, right, (0, 1), (height - 3.6, height), shade(CONCRETE, 0.74))
    _patch(p, left, front, (0.12, 0.36), (height - 2.8, height - 1), shade(GLASS, 0.75))
    _patch(p, left, front, (0.46, 0.72), (height - 2.8, height), LIT)
    p.line([_on_wall(left, front, 0.59, height - 2.6), _on_wall(left, front, 0.59, height)], DOOR)
    _patch(p, front, right, (0.3, 0.7), (height - 2.8, height - 1), shade(GLASS, 0.6))
    _foot(p, top, height)
    _doorstep(p, left, front, (0.4, 0.78), height, CONCRETE)
    p.box(1.5, -height, 2, 2, shade(METAL, 1.1), shade(METAL, 0.85))
    p.line([(0, -height - 1), (0, -height - 6)], METAL)
    p.point(0, -height - 7, (230, 60, 50))
    return p.sprite()
