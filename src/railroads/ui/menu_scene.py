"""What runs behind the main menu: a dark evening over a strip of land with three tracks one
behind another, on which trains of every era, drawn as the map draws them, run across the screen,
the nearer the faster and brighter, steam engines trailing smoke. A train that has left the
screen makes way for another, its picture drawn on the frame it is wanted, one at a time."""

import random
import time
from dataclasses import dataclass, field

import numpy as np
from imgui_bundle import hello_imgui, imgui

from railroads.sim.goods import Good
from railroads.sim.trains import ENGINES, Car, Power, Train
from railroads.ui.ornate import DEEP, GLOW
from railroads.ui.pixel.train_layer import train_picture
from railroads.ui.pixel.trains import GrownTrainSprites
from railroads.ui.theme import color_u32

# How the map's projection looks at trains here: straight from the side, tilted as the map is.
MATRIX = np.array([[32.0, 0.0], [0.0, 16.0]])
# Each track: how far down the screen it lies, how finely its trains are drawn, how fast they run,
# in pixels a second, and how brightly they are lit, the far ones darker; the farthest first.
TRACKS = ((0.64, 1, 18.0, 0.55), (0.685, 1, 24.0, 0.75), (0.735, 1, 32.0, 1.0))
GOODS = (
    Good.WHEAT,
    Good.COAL,
    Good.LOGS,
    Good.IRON,
    Good.STEEL,
    Good.CATTLE,
    Good.OIL,
    Good.BRICKS,
    Good.GOODS,
    Good.BARRELS,
)
# Where the land begins, as a share of the screen from the top.
HORIZON = TRACKS[0][0] - 0.03
# Where the strip of land under the tracks ends.
LAND_END = TRACKS[-1][0] + 0.025
FEWEST_CARS = 5
MOST_CARS = 12
# How long a track stands empty, at most, before another train comes along.
MOST_WAIT = 6.0
SKY_LOW = (0.2, 0.11, 0.06, 1.0)
GROUND = (0.09, 0.07, 0.05, 1.0)
RAIL = (0.42, 0.36, 0.3, 1.0)
SLEEPER = (0.2, 0.14, 0.09, 1.0)
SMOKE = (0.72, 0.68, 0.62)
PUFF_EVERY = 0.2
PUFF_LIFE = 3.0


@dataclass
class _Puff:
    x: float
    y: float
    age: float = 0.0


@dataclass
class _Runner:
    """A train running across one track: its picture, where its left edge is, which way it goes
    and whether it trails smoke."""

    texture: hello_imgui.TextureGpu
    size: tuple[float, float]
    x: float
    way: int
    steam: bool
    # How many rows of the picture lie below the wheels, empty.
    foot: float = 0.0
    puffs: list[_Puff] = field(default_factory=list)
    since_puff: float = 0.0


@dataclass
class MenuScene:
    rng: random.Random = field(default_factory=random.Random)
    _sprites: GrownTrainSprites | None = None
    _runners: dict[int, _Runner] = field(default_factory=dict)
    # When each empty track's next train sets off.
    _due: dict[int, float] = field(default_factory=dict)
    _last: float | None = None

    def draw(self, low: imgui.ImVec2, high: imgui.ImVec2) -> None:
        """Draws the evening, the tracks and their trains in the rectangle, moved on by the time
        since the last frame."""
        now = time.monotonic()
        seconds = 0.0 if self._last is None else min(now - self._last, 0.1)
        self._last = now
        draw_list = imgui.get_window_draw_list()
        width, height = high.x - low.x, high.y - low.y
        horizon = low.y + height * HORIZON
        draw_list.add_rect_filled_multi_color(
            low, (high.x, horizon), *(color_u32(c) for c in (DEEP, DEEP, SKY_LOW, SKY_LOW))
        )
        land_end = low.y + height * LAND_END
        draw_list.add_rect_filled((low.x, horizon), (high.x, land_end), color_u32(GROUND))
        draw_list.add_rect_filled_multi_color(
            (low.x, land_end), high, *(color_u32(c) for c in (SKY_LOW, SKY_LOW, DEEP, DEEP))
        )
        edge = color_u32((*GLOW, 0.5))
        draw_list.add_line((low.x, horizon), (high.x, horizon), edge, 1.0)
        draw_list.add_line((low.x, land_end), (high.x, land_end), edge, 1.0)
        started = False
        for track, (down, times, speed, light) in enumerate(TRACKS):
            rail_y = low.y + height * down
            _draw_rails(draw_list, low.x, high.x, rail_y, times)
            runner = self._runners.get(track)
            if runner is None:
                due = self._due.setdefault(track, now + self.rng.uniform(0.0, MOST_WAIT / 2))
                if now >= due and not started:
                    runner = self._runners[track] = self._new_runner(times, low.x, width)
                    started = True
                else:
                    continue
            runner.x += runner.way * speed * seconds
            self._draw_runner(draw_list, runner, rail_y, seconds, light)
            far_right = runner.x > high.x + 40
            gone = far_right if runner.way > 0 else runner.x + runner.size[0] < low.x - 40
            if gone:
                del self._runners[track]
                self._due[track] = now + self.rng.uniform(0.5, MOST_WAIT)
        hello_imgui.request_refresh()

    def _new_runner(self, times: int, left: float, width: float) -> _Runner:
        if self._sprites is None:
            self._sprites = GrownTrainSprites(tuple(float(v) for v in MATRIX.ravel()))
        engine = self.rng.choice(sorted(ENGINES))
        cars = [
            Car(self.rng.choice(GOODS), load=1.0)
            for _ in range(self.rng.randint(FEWEST_CARS, MOST_CARS))
        ]
        way = self.rng.choice((1, -1))
        pixels = train_picture(
            self._sprites, MATRIX, Train("", engine=engine, cars=cars), times, (float(way), 0.0)
        )
        texture = hello_imgui.create_texture_gpu_from_rgba_data(np.ascontiguousarray(pixels))
        size = (float(pixels.shape[1]), float(pixels.shape[0]))
        rows = np.flatnonzero(pixels[:, :, 3].any(axis=1))
        foot = float(pixels.shape[0] - rows[-1] - 1) if rows.size else 0.0
        x = left - size[0] - 20 if way > 0 else left + width + 20
        return _Runner(texture, size, x, way, ENGINES[engine].power == Power.STEAM, foot)

    def _draw_runner(
        self,
        draw_list: imgui.ImDrawList,
        runner: _Runner,
        rail_y: float,
        seconds: float,
        light: float,
    ) -> None:
        width, height = runner.size
        top = rail_y - height + runner.foot + 1
        draw_list.add_image(
            imgui.ImTextureRef(runner.texture.texture_id()),
            (runner.x, top),
            (runner.x + width, top + height),
            (0.0, 0.0),
            (1.0, 1.0),
            color_u32((light, light, light, 1.0)),
        )
        if runner.steam:
            runner.since_puff += seconds
            if runner.since_puff >= PUFF_EVERY:
                runner.since_puff = 0.0
                front = runner.x + width - 6 if runner.way > 0 else runner.x + 6
                chimney = front - runner.way * height * 0.55
                runner.puffs.append(_Puff(chimney, top + height * 0.12))
        kept = []
        for puff in runner.puffs:
            puff.age += seconds
            if puff.age >= PUFF_LIFE:
                continue
            share = puff.age / PUFF_LIFE
            puff.y -= seconds * 14.0
            puff.x -= runner.way * seconds * 10.0
            radius = (3.0 + 14.0 * share) * height / 60.0 + 2.0
            draw_list.add_circle_filled(
                (puff.x, puff.y), radius, color_u32((*SMOKE, 0.28 * (1.0 - share)))
            )
            kept.append(puff)
        runner.puffs = kept


def _draw_rails(
    draw_list: imgui.ImDrawList, left: float, right: float, y: float, times: int
) -> None:
    """A track seen from the side: sleepers under two rails, the far one a little higher."""
    step = 10.0 * times
    x = left - (left % step)
    while x < right:
        draw_list.add_rect_filled((x, y), (x + 3.0 * times, y + 1.5 * times), color_u32(SLEEPER))
        x += step
    draw_list.add_line((left, y - times), (right, y - times), color_u32(RAIL), 1.0)
    draw_list.add_line((left, y + 1), (right, y + 1), color_u32(RAIL), max(1.0, times / 2))
