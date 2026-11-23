"""What the waiting screen shows while a new game is set up without pictures of its map growing
(railroads.ui.history_pictures): a line of make-believe work that changes every few seconds, opened
by a joke when the map is a random place. And, once the game is ready, the screen breaking into
hexes that fall away to show it, each hex carrying the part of the screen it covered: the newest
picture of the map growing, or the plain background, and the lines shown over it."""

import math
from dataclasses import dataclass
from typing import Self

import numpy as np
from imgui_bundle import imgui

from railroads.sim.goods import FITS, Need
from railroads.sim.ownership import LOT, MOST_PRICED, SHARES, satisfaction_price
from railroads.sim.trade import SCARCE_WORTH

RANDOM_JOKES = ("Random bullshit coming!",)
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

*_FIRST_DRINKS, _LAST_DRINK = (good.value for good in FITS[Need.ALCOHOL])
# How many times a starving building's price a thriving one's shares cost.
_THRIVING_PRICE = satisfaction_price(MOST_PRICED) / satisfaction_price(0.0)
# A tip shown under the waiting screen's lines, one picked for each wait.
TIPS = (
    "Hexagons are bestagons!",
    "Stations hand out to your own buildings first.",
    f"Freight pays up to {SCARCE_WORTH:g} times as much where buildings go short.",
    f"Every {LOT} shares sell you {LOT / SHARES:.0%} of a building's output at the market price.",
    "Buy shares cheap where a building starves, then feed it.",
    "Short trains of full cars pay best: every car costs a trip.",
    "Passengers only ride between stations of different towns.",
    f"Taverns and clubs take any alcohol: {', '.join(_FIRST_DRINKS)} or {_LAST_DRINK}.",
    "Hover a category's icon to see what goods are in it.",
    f"A thriving building's shares are worth up to {_THRIVING_PRICE:.0f} times a starving one's.",
    "A station not joined by track to another gathers and hands out nothing.",
    "What buildings must have is handed out before anything they take as a bonus.",
)

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
# A hexagon's corners (_hexagon) as the two quads either side of its middle.
HALVES = ((0, 1, 2, 3), (3, 4, 5, 0))


def tip_for(rng: np.random.Generator) -> str:
    """A tip for one wait."""
    return TIPS[int(rng.integers(len(TIPS)))]


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
        self,
        width: float,
        height: float,
        colour: int,
        labels: list[Label],
        seed: int = 0,
        pictures: list[imgui.ImTextureRef] | None = None,
    ) -> None:
        self.width = width
        self.height = height
        self.pictures = pictures or []
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
        homes = zip(self.pieces.x.tolist(), self.pieces.y.tolist(), strict=True)
        for cx, cy, a, home in zip(x.tolist(), y.tolist(), turn.tolist(), homes, strict=True):
            if cy - HEX_RADIUS > self.height:
                continue
            # Filled a pixel bigger than its outline, so no seam shows between hexes in place.
            corners = _hexagon(cx, cy, a, HEX_RADIUS + 1)
            if not self.pictures:
                draw.add_convex_poly_filled(corners, self.colour)
            else:
                self._draw_piece(draw, corners, _hexagon(*home, 0.0, HEX_RADIUS + 1))
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

    def _draw_piece(
        self, draw: imgui.ImDrawList, corners: list[imgui.ImVec2], home: list[imgui.ImVec2]
    ) -> None:
        """Draws a hex with the part of the pictures it covered in place, one over the other: as
        two quads, the halves of the hexagon above and below its middle."""
        for half in HALVES:
            uv = [imgui.ImVec2(home[i].x / self.width, home[i].y / self.height) for i in half]
            for picture in self.pictures:
                draw.add_image_quad(picture, *(corners[i] for i in half), *uv)


def _hexagon(x: float, y: float, turn: float, radius: float) -> list[imgui.ImVec2]:
    """The corners of a flat-topped hexagon turned by `turn` radians."""
    return [
        imgui.ImVec2(x + radius * math.cos(turn + t), y + radius * math.sin(turn + t))
        for t in (n * math.pi / 3 for n in range(6))
    ]
