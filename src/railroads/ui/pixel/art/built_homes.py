"""Houses, tenements, flats and skyscrapers built as 3D models (railroads.ui.pixel.art.built)."""

from collections.abc import Callable

from railroads.ui.pixel.art.built import (
    chimney,
    cornice,
    door,
    end_window,
    flat_roof,
    gable_roof,
    modelled,
    walls,
    window,
)
from railroads.ui.pixel.art.painter import (
    BRICK,
    GLASS,
    METAL,
    RGB,
    STONE,
    WINDOW,
    Sprite,
    shade,
)
from railroads.ui.pixel.models import Model

AWNINGS: list[RGB] = [(178, 52, 46), (52, 104, 70), (60, 88, 150)]
CONCRETE: RGB = (176, 178, 182)
LIT: RGB = (246, 214, 128)
SHOP_GLASS: RGB = (120, 156, 180)
# How much homes are stretched across and up to stand as big and as tall as the painted ones.
ACROSS, UP = 1.32, 1.7


def home(build: Callable[[], Model]) -> Callable[[], Sprite]:
    """The draw function of a home built by `build`."""
    return modelled(build, ACROSS, UP)


def house(wall: RGB, roof: RGB) -> Model:
    """A cottage on a stone plinth: a door and windows with sills on its front, a window in its
    gable end, a tiled gable roof and a brick chimney."""
    m = Model()
    u, v, high = (-0.1, 0.1), (-0.085, 0.085), 0.085
    walls(m, u, v, high, wall, STONE)
    for span in ((-0.08, -0.045), (0.045, 0.08)):
        window(m, span, (0.035, 0.065), 0.085, WINDOW)
    door(m, (-0.012, 0.012), 0.062, 0.085)
    for end in (-0.1, 0.1):
        end_window(m, end, (-0.018, 0.018), (0.035, 0.065), shade(WINDOW, 0.85))
    gable_roof(m, u, v, (high, high + 0.085), roof, wall)
    chimney(m, -0.045, 0.03, (high + 0.05, high + 0.11), BRICK)
    return m


def tenement(wall: RGB, roof: RGB) -> Model:
    """A tenement (kamienica): a shop on the ground floor under an awning, two painted floors of
    windows between cornices, and a steep tiled roof with a dormer and two chimneys."""
    m = Model()
    u, v, high = (-0.1, 0.1), (-0.09, 0.09), 0.2
    walls(m, u, v, high, wall)
    m.box((u[0] - 0.001, u[1] + 0.001), (v[0] - 0.001, v[1] + 0.001), (0.0, 0.06), shade(wall, 0.8))
    awning = AWNINGS[sum(wall) % len(AWNINGS)]
    m.panel((-0.085, -0.005), (0.015, 0.05), 0.0912, SHOP_GLASS)
    door(m, (0.03, 0.06), 0.052, 0.0905)
    m.box((-0.09, 0.0), (0.091, 0.112), (0.056, 0.062), awning)
    m.box((-0.09, 0.0), (-0.112, -0.091), (0.056, 0.062), awning)
    for low in (0.085, 0.145):
        for span in ((-0.08, -0.05), (-0.015, 0.015), (0.05, 0.08)):
            window(m, span, (low, low + 0.04), 0.09, WINDOW)
        for end in (-0.1, 0.1):
            for across in ((-0.06, -0.03), (0.03, 0.06)):
                end_window(m, end, across, (low, low + 0.04), shade(WINDOW, 0.85))
    for height in (0.065, 0.135, high - 0.006):
        cornice(m, u, v, height, shade(wall, 1.12))
    gable_roof(m, u, v, (high, high + 0.11), roof, wall)
    m.box((-0.02, 0.02), (0.02, 0.07), (high + 0.03, high + 0.07), shade(wall, 0.95))
    m.roof((-0.025, 0.025), (0.02, 0.075), (high + 0.07, high + 0.09), roof, shade(wall, 0.95))
    m.panel((-0.01, 0.01), (high + 0.038, high + 0.062), 0.0705, WINDOW)
    for at in (-0.075, 0.075):
        chimney(m, at, 0.0, (high + 0.08, high + 0.14), BRICK)
    return m


def flats(wall: RGB) -> Model:
    """A block of flats: five floors under a flat roof with a parapet and a lift house, rows of
    windows with balconies, a few lit, and an entrance under a canopy."""
    m = Model()
    u, v = (-0.12, 0.12), (-0.11, 0.11)
    floors, storey = 5, 0.068
    high = floors * storey + 0.01
    walls(m, u, v, high, wall)
    for floor in range(floors):
        low = 0.02 + floor * storey
        for k, span in enumerate(((-0.1, -0.07), (-0.045, -0.015), (0.015, 0.045), (0.07, 0.1))):
            lit = (floor * 4 + k) % 7 == 2
            window(m, span, (low, low + 0.034), 0.11, LIT if lit else WINDOW)
            if floor and k in (1, 2):
                for side in (1.0, -1.0):
                    rail = sorted((side * 0.11, side * 0.128))
                    m.box(span, (rail[0], rail[1]), (low - 0.008, low - 0.004), shade(wall, 0.9))
                    m.box(
                        span,
                        (side * 0.126 - 0.002, side * 0.126 + 0.002),
                        (low - 0.004, low + 0.016),
                        shade(METAL, 0.9),
                    )
        for end in (-0.12, 0.12):
            for across in ((-0.07, -0.035), (0.035, 0.07)):
                end_window(m, end, across, (low, low + 0.034), shade(WINDOW, 0.85))
    door(m, (-0.018, 0.018), 0.012 + storey * 0.6, 0.11)
    for side in (1.0, -1.0):
        m.box((-0.03, 0.03), sorted((side * 0.11, side * 0.135)), (0.06, 0.066), CONCRETE)
    flat_roof(m, u, v, high, shade(wall, 0.6), wall)
    m.box((-0.04, 0.02), (-0.03, 0.03), (high, high + 0.05), shade(wall, 0.8))
    return m


def skyscraper() -> Model:
    """A glass tower with floors marked across it, a concrete base with a lit entrance, a plant
    room on top and an aerial."""
    m = Model()
    u, v = (-0.1, 0.1), (-0.1, 0.1)
    high = 0.94
    m.box(u, v, (0.0, 0.07), CONCRETE)
    m.box(u, v, (0.07, high), GLASS)
    m.panel((-0.05, 0.05), (0.0, 0.05), 0.1006, LIT)
    with m.fine():
        for k in range(1, 30):
            z = 0.07 + k * (high - 0.07) / 30
            m.box(
                (u[0] - 0.001, u[1] + 0.001),
                (v[0] - 0.001, v[1] + 0.001),
                (z, z + 0.004),
                shade(GLASS, 1.12),
            )
        for k in range(1, 6):
            at = u[0] + k * 0.2 / 6
            for side in (0.1012, -0.1012):
                m.panel((at - 0.002, at + 0.002), (0.07, high), side, shade(GLASS, 0.75))
    m.box((-0.06, 0.06), (-0.06, 0.06), (high, high + 0.05), shade(CONCRETE, 0.85))
    m.box((-0.003, 0.003), (-0.003, 0.003), (high + 0.05, high + 0.16), METAL)
    m.box((-0.006, 0.006), (-0.006, 0.006), (high + 0.16, high + 0.172), (230, 60, 50))
    return m
