"""Shows the block signals (railroads.sim.signals) over the pixel map as old semaphores, lit from the
top left like the buildings: a mast striped red and white on a dark footing beside the track, on
the right of the way trains pass it, topped with a white finial, with a red arm striped white at its
end that reaches over towards the track. Its spectacle holds a red glass and a green one in front
of the lamp: at danger the arm lies level and the lamp shows red; once the way past the signal is
set for a train coming to it, the arm rises and the lamp shows green, the arm swinging and the
light fading from one to the other. From far off only the lamps show, as points of light that tell
at a glance where trains wait."""

import math

import numpy as np
from imgui_bundle import imgui

from railroads.sim.railway import Railway
from railroads.sim.signals import blocks_of, cleared_signals
from railroads.ui.iso_camera import PIXEL_CELL_WIDTH, IsoCamera, Vec2
from railroads.ui.pixel.models import HEIGHT_PER_WIDTH
from railroads.ui.pixel.track_art import DOUBLE_SPACING
from railroads.ui.pixel.world import PixelWorld

# How far to the side of the track's middle a signal stands, in cells: clear of the outer rail of
# double track.
SIDE = DOUBLE_SPACING / 2 + 0.16
# How tall a mast is, how wide, and how many red and white stripes it has, in cells; how long its
# arm is and how thick; how big its lamp's glasses are.
MAST = 0.38
MAST_WIDTH = 0.022
STRIPES = 6
ARM = 0.15
ARM_WIDTH = 0.028
GLASS = 0.022
# How far the arm rises at clear, in degrees.
RISE = 45.0
# How long a signal takes to change from danger to clear or back, in seconds.
SWING = 0.5
# How many screen pixels a cell must take before masts show; below it only the lamps do, and
# below the second no signals at all.
MASTS_FROM = 48.0
LAMPS_FROM = 10.0
RED = (0.95, 0.2, 0.14)
GREEN = (0.3, 0.95, 0.4)
SIGNAL_RED = (0.78, 0.12, 0.1)
SIGNAL_RED_SHADED = (0.55, 0.08, 0.07)
WHITE = (0.93, 0.92, 0.88)
WHITE_SHADED = (0.7, 0.69, 0.66)
OUTLINE = (0.09, 0.08, 0.08)
FOOTING = (0.14, 0.16, 0.22)
RIM = (0.72, 0.73, 0.75)
SHADOW = (0.0, 0.0, 0.0, 0.28)


def _u32(color: tuple[float, ...], alpha: float = 1.0) -> int:
    rgba = color if len(color) == 4 else (*color, alpha)
    return imgui.color_convert_float4_to_u32(imgui.ImVec4(*rgba))


class SignalLayer:
    def __init__(self, world: PixelWorld) -> None:
        self.world = world
        # How clear each signal shows, from 0 at danger to 1 clear, by where it stands.
        self._clear: dict[tuple, float] = {}

    def draw(
        self,
        draw_list: imgui.ImDrawList,
        railway: Railway,
        camera: IsoCamera,
        center: Vec2,
        near: bool,
    ) -> None:
        """Draws the signals on the far side of their track from the viewer, to show behind the
        trains drawn next, or with `near` those on the near side, drawn after the trains so they
        show in front of them."""
        spots = blocks_of(railway.constructions).spots
        projection = self.world.projection
        scale = camera.pixel_scale or camera.cell_width / PIXEL_CELL_WIDTH
        per_cell = PIXEL_CELL_WIDTH * scale
        if not spots or per_cell < LAMPS_FROM:
            return
        camera_pixel = projection.to_pixels(camera.center_x, camera.center_y)
        origin = (
            center[0] - float(camera_pixel[0]) * scale,
            center[1] - float(camera_pixel[1]) * scale,
        )
        xs = np.array([spot.at[0] for spot in spots])
        ys = np.array([spot.at[1] for spot in spots])
        lengths = np.array([math.hypot(*spot.heading) or 1.0 for spot in spots])
        hx = np.array([spot.heading[0] for spot in spots]) / lengths
        hy = np.array([spot.heading[1] for spot in spots]) / lengths
        ax, ay = projection.to_pixels(xs, ys)
        bx, by = projection.to_pixels(xs + hx * 0.1, ys + hy * 0.1)
        dx, dy = bx - ax, by - ay
        way = np.hypot(dx, dy)
        way[way == 0] = 1.0
        dx, dy = dx / way, dy / way
        side = SIDE * PIXEL_CELL_WIDTH
        lift = self.world.elevation.lift(xs, ys)
        sx = origin[0] + (ax - dy * side) * scale
        sy = origin[1] + (ay + dx * side - lift) * scale
        size = imgui.get_io().display_size
        reach = MAST * per_cell * 2
        shown = (sx > -reach) & (sx < size.x + reach) & (sy > -reach) & (sy < size.y + 2 * reach)
        shown &= (dx > 0) if near else (dx <= 0)
        indices = np.flatnonzero(shown)
        if not indices.size:
            return
        hidden = self.world.underground_all(xs[indices].tolist(), ys[indices].tolist())
        cleared = cleared_signals(railway.trains)
        step = min(1.0, imgui.get_io().delta_time / SWING)
        up = float(np.hypot(*projection.matrix[:, 0])) * HEIGHT_PER_WIDTH * scale
        masts = per_cell >= MASTS_FROM
        order = sorted(
            (float(sy[k]), float(sx[k]), k)
            for k, under in zip(indices.tolist(), hidden, strict=True)
            if not under
        )
        for y, x, k in order:
            spot = spots[k]
            key = (spot.cell, spot.at)
            wanted = 1.0 if key in cleared else 0.0
            clear = self._clear.get(key, wanted)
            clear += max(-step, min(step, wanted - clear))
            self._clear[key] = clear
            lamp = tuple(r + (g - r) * clear for r, g in zip(RED, GREEN, strict=True))
            if masts:
                towards = 1.0 if dy[k] >= 0 else -1.0
                _semaphore(draw_list, (x, y), per_cell, up, clear, towards, lamp)
            else:
                _glow(draw_list, (x, y - MAST * up * 0.8), max(1.6, GLASS * per_cell), lamp)


def _semaphore(
    draw_list: imgui.ImDrawList,
    foot: Vec2,
    per_cell: float,
    up: float,
    clear: float,
    towards: float,
    lamp: tuple[float, float, float],
) -> None:
    """One semaphore standing at `foot` on the screen, its arm reaching `towards` the right
    (1) or the left (-1) and risen `clear` of the way from danger to clear."""
    x, y = foot
    line = max(1.0, 0.005 * per_cell)
    draw_list.add_ellipse_filled(
        (x + 0.05 * per_cell, y + 0.008 * per_cell),
        (0.08 * per_cell, 0.022 * per_cell),
        _u32(SHADOW),
    )
    legs = 0.035 * per_cell
    knee = y - 0.05 * up
    for spread in (-legs, legs * 0.55, legs):
        draw_list.add_line((x, knee), (x + spread, y), _u32(OUTLINE), line * 2.4)
        draw_list.add_line((x, knee), (x + spread, y), _u32(FOOTING), line * 1.3)
    top = y - MAST * up
    half = max(0.8, MAST_WIDTH * per_cell / 2)
    draw_list.add_rect_filled((x - half - line, top), (x + half + line, knee), _u32(OUTLINE))
    band = (knee - top) / STRIPES
    for stripe in range(STRIPES):
        lit, shaded = (SIGNAL_RED, SIGNAL_RED_SHADED) if stripe % 2 else (WHITE, WHITE_SHADED)
        y0, y1 = top + stripe * band, top + (stripe + 1) * band
        draw_list.add_rect_filled((x - half, y0), (x, y1), _u32(lit))
        draw_list.add_rect_filled((x, y0), (x + half, y1), _u32(shaded))
    finial = max(1.2, 0.012 * per_cell)
    draw_list.add_circle_filled((x, top - finial * 0.6), finial + line, _u32(OUTLINE))
    draw_list.add_circle_filled((x, top - finial * 0.6), finial, _u32(WHITE))
    pivot = (x, top + band * 0.6)
    angle = math.radians(RISE * clear)
    ax, ay = towards * math.cos(angle), -math.sin(angle)
    nx, ny = -ay, ax
    length, thick = ARM * per_cell, max(1.0, ARM_WIDTH * per_cell / 2)

    def bar(start: float, end: float, color: tuple[float, float, float], grow: float) -> None:
        width = thick + grow
        a = (pivot[0] + ax * start, pivot[1] + ay * start)
        b = (pivot[0] + ax * end, pivot[1] + ay * end)
        draw_list.add_quad_filled(
            (a[0] + nx * width, a[1] + ny * width),
            (b[0] + nx * width, b[1] + ny * width),
            (b[0] - nx * width, b[1] - ny * width),
            (a[0] - nx * width, a[1] - ny * width),
            _u32(color),
        )

    bar(-length * 0.15 - line, length + line, OUTLINE, line)
    bar(-length * 0.15, length, SIGNAL_RED, 0.0)
    bar(length * 0.7, length * 0.82, WHITE, 0.0)
    glass = max(1.1, GLASS * per_cell)
    spectacle = (pivot[0] - towards * glass * 1.4, pivot[1] + glass * 1.6)
    draw_list.add_line(pivot, spectacle, _u32(OUTLINE), line * 1.5)
    for offset, color in ((-1.0, RED), (1.0, GREEN)):
        centre = (spectacle[0] + offset * towards * glass * 1.1, spectacle[1])
        draw_list.add_circle_filled(centre, glass + line, _u32(RIM))
        draw_list.add_circle_filled(centre, glass, _u32(tuple(c * 0.45 for c in color)))
    lit = (spectacle[0] + (2 * clear - 1) * towards * glass * 1.1, spectacle[1])
    _glow(draw_list, lit, glass * 1.05, lamp)


def _glow(
    draw_list: imgui.ImDrawList, at: Vec2, radius: float, color: tuple[float, float, float]
) -> None:
    """A lit lamp: a soft halo round a bright core."""
    draw_list.add_circle_filled(at, radius * 3.2, _u32(color, 0.12))
    draw_list.add_circle_filled(at, radius * 2.0, _u32(color, 0.28))
    draw_list.add_circle_filled(at, radius, _u32(color))
    draw_list.add_circle_filled(
        (at[0] - radius * 0.3, at[1] - radius * 0.3), radius * 0.4, _u32((1.0, 1.0, 1.0), 0.6)
    )
