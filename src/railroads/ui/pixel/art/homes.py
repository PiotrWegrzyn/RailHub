"""Houses, tenements, flats and skyscrapers."""

from railroads.ui.pixel.art.painter import GLASS, METAL, RGB, WINDOW, Painter, Sprite, shade


def house(wall: RGB, roof: RGB) -> Sprite:
    p = Painter(15, 16, (7, 12))
    p.shadow(6, 2)
    top = p.box(0, 0, 6, 4, wall, None)
    p.gable(top, 4, roof, wall)
    p.point(-4, -2, WINDOW)
    p.point(-2, -1, WINDOW)
    p.point(3, -2, shade(WINDOW, 0.8))
    return p.sprite()


def tenement(wall: RGB, roof: RGB) -> Sprite:
    """A tenement (kamienica): three floors of painted front under a steep tiled roof."""
    p = Painter(17, 26, (8, 20))
    p.shadow(7, 2.5)
    s, floors = 6, 3
    top = p.box(0, 0, s, 3 * floors + 1, wall, None)
    p.gable(top, 5, roof, wall)
    for floor in range(floors):
        rise = 3 * floor + 2.5
        for t in (0.3, 0.7):
            p.point(-s + t * s, t * s / 2 - rise, WINDOW)
            p.point(t * s, s / 2 - t * s / 2 - rise, shade(WINDOW, 0.75))
    p.line([(-s, -3 * floors - 1), (0, s / 2 - 3 * floors - 1)], shade(wall, 1.15))
    return p.sprite()


def flats(wall: RGB) -> Sprite:
    """A block of flats: five floors under a flat roof, with rows of windows."""
    p = Painter(19, 32, (9, 26))
    p.shadow(8, 3, reach=5)
    s, floors = 7, 5
    p.box(0, 0, s, 3 * floors + 1, wall, shade(wall, 0.62))
    for floor in range(floors):
        rise = 3 * floor + 2.5
        for t in (0.2, 0.45, 0.7):
            p.point(-s + t * s * 1.2, t * s * 0.6 - rise, WINDOW)
            p.point(t * s * 1.2, s / 2 - t * s * 0.6 - rise, shade(WINDOW, 0.75))
    p.rect(-1, -3 * floors - 3, 1, -3 * floors - 1, shade(wall, 0.7))
    return p.sprite()


def skyscraper() -> Sprite:
    """A glass tower with a grid of windows and an aerial on top."""
    p = Painter(17, 60, (8, 54))
    p.shadow(7, 2.5, reach=8)
    s, height = 6, 44
    p.box(0, 0, s, height, GLASS, shade(GLASS, 0.8))
    lit, dark = shade(GLASS, 1.35), shade(GLASS, 0.62)
    for floor in range(2, height - 1, 3):
        p.line([(-s, -floor), (0, s / 2 - floor)], lit)
        p.line([(0, s / 2 - floor), (s, -floor)], dark)
    p.line([(-s / 2, s / 4), (-s / 2, s / 4 - height)], shade(GLASS, 0.85))
    p.line([(0, -height - 1), (0, -height - 6)], METAL)
    return p.sprite()
