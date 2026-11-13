"""What the waiting screen shows while a new game is set up: a line of make-believe work that
changes every few seconds, opened by two jokes when the map is a random place; and, once the game
is ready, the screen breaking into hexes that fall away to show it."""

import math
from dataclasses import dataclass
from typing import Self

import numpy as np
from imgui_bundle import imgui

RANDOM_JOKES = ("Random bullshit coming!", "Lerooorryyy jennnnnkinssss")
JOKE_SECONDS = 5.0
WORK = (
    "Turning cogs",
    "Steaming the engine",
    "Playing with the hexes",
    "Pouring water",
    "Sprinkling sand",
    "Making people",
    "Making more people",
    "Growing them apples",
    "Shoveling cow shit",
    "Handing out bread",
)
WORK_SECONDS = 2.5

# The falling hexes: their size, when the cracks show and the first falls, how much later the
# outermost falls than the middle one, and how hard they fall (pixels a second squared).
HEX_RADIUS = 40.0
CRACK_SECONDS = 0.35
SPREAD_SECONDS = 0.6
JITTER_SECONDS = 0.15
GRAVITY = 2600.0
CRACK_COLOUR = (0.0, 0.0, 0.0, 0.55)
# The most a frame moves the hexes on, so the slow first frames of a game can't skip their fall.
LONGEST_STEP = 0.1


def waiting_line(seconds: float, random_place: bool) -> str:
    """The line shown `seconds` after a new game began to be set up."""
    if random_place:
        if seconds < JOKE_SECONDS * len(RANDOM_JOKES):
            return RANDOM_JOKES[int(seconds // JOKE_SECONDS)]
        seconds -= JOKE_SECONDS * len(RANDOM_JOKES)
    return WORK[int(seconds // WORK_SECONDS) % len(WORK)]


@dataclass(frozen=True)
class Pieces:
    """The hexes a screen breaks into: each one's centre, when it starts to fall, which way it is
    thrown (pixels a second) and how fast it turns (radians a second)."""

    x: np.ndarray
    y: np.ndarray
    falls_at: np.ndarray
    thrown_x: np.ndarray
    thrown_y: np.ndarray
    spin: np.ndarray

    @classmethod
    def covering(
        cls, width: float, height: float, rng: np.random.Generator, radius: float = HEX_RADIUS
    ) -> Self:
        """Flat-topped hexes covering a screen this size, falling from the middle outwards."""
        step_x, step_y = radius * 1.5, radius * math.sqrt(3)
        columns = np.arange(int(width / step_x) + 2)
        rows = np.arange(int(height / step_y) + 2)
        col, row = (grid.ravel() for grid in np.meshgrid(columns, rows))
        x = col * step_x
        y = row * step_y + (col % 2) * step_y / 2
        reach = math.hypot(width, height) / 2
        out = np.hypot(x - width / 2, y - height / 2) / reach
        count = len(x)
        return cls(
            x=x.astype(float),
            y=y.astype(float),
            falls_at=CRACK_SECONDS + out * SPREAD_SECONDS + rng.uniform(0, JITTER_SECONDS, count),
            thrown_x=(x - width / 2) / reach * 150 + rng.uniform(-60, 60, count),
            thrown_y=rng.uniform(-260, -40, count),
            spin=rng.uniform(-4, 4, count),
        )

    def at(self, seconds: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Each hex's centre and turn `seconds` after the screen broke."""
        falling = np.maximum(seconds - self.falls_at, 0.0)
        x = self.x + self.thrown_x * falling
        y = self.y + self.thrown_y * falling + GRAVITY * falling**2 / 2
        return x, y, self.spin * falling

    def nearest(self, x: float, y: float) -> int:
        return int(np.argmin(np.hypot(self.x - x, self.y - y)))

    def gone(self, seconds: float, height: float, radius: float = HEX_RADIUS) -> bool:
        """Whether every hex has fallen out of sight."""
        _, y, _ = self.at(seconds)
        return bool((y - radius > height).all())


@dataclass(frozen=True)
class Label:
    """Something the waiting screen showed, carried off by the hex under its middle: a line of
    text, or with no text a bar filled with the colour."""

    x: float
    y: float
    width: float
    height: float
    colour: int
    text: str | None = None

    @property
    def middle(self) -> tuple[float, float]:
        return self.x + self.width / 2, self.y + self.height / 2


class Shatter:
    """The waiting screen breaking into hexes over the game it was waiting for, its labels falling
    with them."""

    def __init__(
        self, width: float, height: float, colour: int, labels: list[Label], seed: int = 0
    ) -> None:
        self.height = height
        self.colour = colour
        self.pieces = Pieces.covering(width, height, np.random.default_rng(seed))
        self.labels = labels
        self._carriers = [self.pieces.nearest(*label.middle) for label in labels]
        self.seconds = 0.0

    def draw(self, seconds_passed: float) -> bool:
        """Draws the hexes over everything, `seconds_passed` on from the last frame (LONGEST_STEP
        at most); False once they are all gone."""
        self.seconds += min(seconds_passed, LONGEST_STEP)
        if self.pieces.gone(self.seconds, self.height):
            return False
        draw = imgui.get_foreground_draw_list()
        crack = imgui.get_color_u32(
            imgui.ImVec4(*CRACK_COLOUR[:3], CRACK_COLOUR[3] * min(self.seconds / CRACK_SECONDS, 1))
        )
        x, y, turn = self.pieces.at(self.seconds)
        for cx, cy, a in zip(x.tolist(), y.tolist(), turn.tolist(), strict=True):
            if cy - HEX_RADIUS > self.height:
                continue
            # Filled a pixel bigger than its outline, so no seam shows between hexes in place.
            draw.add_convex_poly_filled(_hexagon(cx, cy, a, HEX_RADIUS + 1), self.colour)
            draw.add_polyline(
                _hexagon(cx, cy, a, HEX_RADIUS),
                crack,
                thickness=1.5,
                flags=imgui.ImDrawFlags_.closed.value,
            )
        for label, carrier in zip(self.labels, self._carriers, strict=True):
            dx, dy = x[carrier] - self.pieces.x[carrier], y[carrier] - self.pieces.y[carrier]
            top_left = imgui.ImVec2(label.x + dx, label.y + dy)
            if label.text is None:
                bottom_right = imgui.ImVec2(top_left.x + label.width, top_left.y + label.height)
                draw.add_rect_filled(top_left, bottom_right, label.colour)
            else:
                draw.add_text(top_left, label.colour, label.text)
        return True


def _hexagon(x: float, y: float, turn: float, radius: float) -> list[imgui.ImVec2]:
    """The corners of a flat-topped hexagon turned by `turn` radians."""
    return [
        imgui.ImVec2(x + radius * math.cos(turn + t), y + radius * math.sin(turn + t))
        for t in (n * math.pi / 3 for n in range(6))
    ]
