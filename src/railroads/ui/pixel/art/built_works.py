"""The gasworks built as a 3D model (railroads.ui.pixel.art.built): a brick gasholder rotunda,
laid out with the painted sprite's own coordinates."""

import math

from railroads.ui.pixel.art.built import (
    along,
    box_at,
    chimney,
    cone,
    dome,
    footing,
    gable_roof,
    spot,
    tall,
    upright,
    walls,
    wide,
)
from railroads.ui.pixel.art.painter import BRICK, DOORWAY, RGB, shade
from railroads.ui.pixel.models import Face, Model

GASHOLDER_BRICK: RGB = (146, 72, 54)
SOOT: RGB = (78, 50, 44)
GASHOLDER_ROOF: RGB = (88, 84, 86)
ARCH_GLASS: RGB = (40, 42, 50)
BAYS = 12


def _on_drum(
    centre: tuple[float, float], radius: float, angle: float, z: float
) -> tuple[float, float, float]:
    u, v = centre
    return u + radius * math.cos(angle), v + radius * math.sin(angle), z


def _drum_patch(
    m: Model,
    centre: tuple[float, float],
    radius: float,
    angles: tuple[float, float],
    z: tuple[float, float],
    color: RGB,
    arched: bool = False,
) -> None:
    """A patch lying on a drum `radius` round `centre`, between `angles` and heights `z`: a
    pilaster or a window, rounded over at the top if `arched`."""
    (a0, a1), (z0, z1) = angles, z
    middle = (a0 + a1) / 2
    points = [(a0, z0), (a1, z0), (a1, z1)]
    if arched:
        rise = (a1 - a0) / 2 * radius
        points += [
            (
                middle + (a1 - a0) / 2 * math.cos(math.pi * k / 6),
                z1 + rise * math.sin(math.pi * k / 6),
            )
            for k in range(1, 6)
        ]
    points.append((a0, z1))
    m.faces.append(
        Face(
            [_on_drum(centre, radius, a, h) for a, h in points],
            color,
            (math.cos(middle), math.sin(middle), 0.0),
        )
    )


def _rotunda(m: Model, centre: tuple[float, float], radius: float, high: float) -> None:
    """A brick gasholder house: a tall round drum split into bays by pilasters, three tiers of
    round-arched windows with round windows on the lowest, cornice bands between the tiers, a
    low roof and a railing round its rim; streaked with soot."""
    u, v = centre
    upright(m, u, v, radius, (0.0, high), GASHOLDER_BRICK)
    step = 2 * math.pi / BAYS
    tiers = [(0.07, 0.24), (0.4, 0.57), (0.72, 0.88)]
    for k in range(BAYS):
        a = k * step
        _drum_patch(
            m,
            centre,
            radius + 0.0008,
            (a - step * 0.07, a + step * 0.07),
            (0.0, high),
            shade(GASHOLDER_BRICK, 1.12),
        )
        if k % 3 == 1:
            _drum_patch(
                m,
                centre,
                radius + 0.0006,
                (a + step * 0.1, a + step * 0.45),
                (0.0, high * 0.8),
                SOOT,
            )
        for t, (low, top) in enumerate(tiers):
            centre_a = a + step / 2
            _drum_patch(
                m,
                centre,
                radius + 0.0012,
                (centre_a - step * 0.2, centre_a + step * 0.2),
                (high * low, high * top - step * 0.2 * radius),
                ARCH_GLASS,
                arched=True,
            )
            if t == 0:
                with m.fine():
                    ring = high * (tiers[0][1] + 0.05)
                    _drum_patch(
                        m,
                        centre,
                        radius + 0.0012,
                        (centre_a - step * 0.07, centre_a + step * 0.07),
                        (ring, ring + step * 0.14 * radius),
                        ARCH_GLASS,
                    )
    with m.fine():
        for low, _ in tiers[1:]:
            band = high * (low - 0.05)
            upright(m, u, v, radius + 0.004, (band, band + 0.01), shade(GASHOLDER_BRICK, 1.3))
        upright(m, u, v, radius + 0.004, (high - 0.01, high), shade(GASHOLDER_BRICK, 1.2))
    dome(m, u, v, radius, (high, high + radius * 0.18), GASHOLDER_ROOF)
    with m.fine():
        posts = 40
        for k in range(posts):
            a = 2 * math.pi * k / posts
            pu, pv, _ = _on_drum(centre, radius - 0.003, a, 0)
            m.box(
                (pu - 0.0012, pu + 0.0012),
                (pv - 0.0012, pv + 0.0012),
                (high, high + 0.014),
                (60, 62, 66),
            )
        upright(m, u, v, radius - 0.002, (high + 0.013, high + 0.016), (60, 62, 66))


def gasworks() -> Model:
    """A brick gasholder house towering over a small brick boiler house with a tall chimney and a
    heap of coal by its door."""
    m = Model()
    centre = spot(5, 1)
    _rotunda(m, centre, wide(8), tall(18))
    u, v = box_at(-8, 1, 5)
    high = tall(6)
    footing(m, u, v)
    walls(m, u, v, high, BRICK)
    gable_roof(m, u, v, (high, high + tall(3)), (90, 76, 70), BRICK)
    m.panel((along(u, 0.4), along(u, 0.62)), (0.0, tall(4.5)), v[1] + 0.0006, DOORWAY)
    cu, cv = spot(-11.3, -1)
    chimney(m, cu, cv, (0.0, tall(15)), shade(BRICK, 0.9), 0.012)
    hu, hv = spot(-12.5, 6)
    cone(m, hu, hv, wide(2.6), tall(2.6), (54, 52, 56))
    return m
