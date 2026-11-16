"""Big and mega works drawn from the works they grew from: a big works is two of its buildings
side by side in one yard, a mega works three round a tall chimney with a water tower behind."""

from collections.abc import Callable

from railroads.ui.pixel.art.industry import _chimney, _smoke
from railroads.ui.pixel.art.painter import BRICK, METAL, Painter, Sprite, drawn_at, fineness, shade

# Where each building of a grown works stands, as shares of one building's width across and down
# from the works' anchor, back to front, and how big each is drawn against the works it grew from.
LAYOUTS = {
    "big": ((0.3, -0.15), (-0.2, 0.1)),
    "mega": ((-0.36, -0.18), (0.36, -0.18), (0.0, 0.12)),
}
PART_SCALE = 0.85
TANK = (150, 70, 52)
# How tall a mega works' chimney stands at most, in units.
CHIMNEY_MOST = 26.0


def grown(draw: Callable[[], Sprite], size: str, most_width: float) -> Sprite:
    """A big or mega works of the buildings `draw` draws, laid out as LAYOUTS says, no wider than
    `most_width` units."""
    k = fineness()
    layout = LAYOUTS[size]
    with drawn_at(1):
        whole = draw()
    across = max(dx for dx, _ in layout) - min(dx for dx, _ in layout) + 1
    part_scale = min(PART_SCALE, most_width / (whole.image.width * across))
    w, h = whole.image.width * part_scale, whole.image.height * part_scale
    ax, ay = whole.anchor[0] * part_scale, whole.anchor[1] * part_scale
    spots = [(dx * w, dy * w) for dx, dy in layout]
    chimney = 0.25 * h + min(h * 0.9, CHIMNEY_MOST) + 10
    extra = max(0.0, chimney - ay) if size == "mega" else 0.0
    left = -min(x for x, _ in spots) + ax + 1
    right = max(x for x, _ in spots) + w - ax + 1
    top = -min(y for _, y in spots) + ay + extra + 1
    bottom = max(y for _, y in spots) + h - ay + 1
    p = Painter(left + right, top + bottom, (left, top))
    with drawn_at(k * part_scale):
        part = draw()
    image, (px, py) = part.image, part.anchor
    if size == "mega":
        _back_of_mega(p, w, h, spots)
    for x, y in spots:
        p.image.alpha_composite(
            image, (round(p.anchor[0] + x * k - px), round(p.anchor[1] + y * k - py))
        )
    return p.sprite()


def _back_of_mega(p: Painter, w: float, h: float, spots: list[tuple[float, float]]) -> None:
    """A tall banded chimney smoking behind a mega works' middle and a water tower behind its
    right side."""
    back = min(y for _, y in spots)
    foot = back - h * 0.25
    tall = min(h * 0.9, CHIMNEY_MOST)
    _chimney(p, -0.04 * w, foot, foot - tall, shade(BRICK, 0.9))
    _smoke(p, -0.04 * w + 0.5, foot - tall - 2, (214, 214, 218), 3)
    x, y = 0.12 * w, back - h * 0.1
    legs = h * 0.55
    for dx in (-2, 2):
        p.line([(x + dx, y), (x + dx * 0.6, y - legs)], shade(METAL, 0.6))
    p.line([(x - 1.6, y - legs * 0.5), (x + 1.6, y - legs * 0.5)], shade(METAL, 0.6))
    p.rect(x - 2.6, y - legs - 4, x + 2.6, y - legs, TANK)
    p.polygon([(x - 3, y - legs - 4), (x, y - legs - 6), (x + 3, y - legs - 4)], shade(TANK, 0.7))
