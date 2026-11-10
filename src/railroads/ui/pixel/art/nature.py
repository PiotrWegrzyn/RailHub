"""Trees, rocks, and the hills and mountains drawn as small height fields."""

import numpy as np
from PIL import Image

from railroads.ui.pixel.art.painter import (
    RGB,
    ROCK,
    SNOW_LIT,
    SNOW_SHADE,
    TINT_GREYS,
    TRUNK,
    Painter,
    Sprite,
    shade,
)


def pine(colors: tuple[RGB, RGB, RGB]) -> Sprite:
    dark, mid, light = colors
    p = Painter(11, 16, (5, 14))
    p.shadow(3, 1.5)
    p.line([(0, 0), (0, -2)], TRUNK)
    for i, half in enumerate((4, 3, 2)):
        base = -2 - 3.5 * i
        p.polygon([(-half, base), (half, base), (0, base - 5)], mid, dark)
        p.line([(-half + 1, base - 1), (-1, base - 4)], light)
    return p.sprite()


def leafy(colors: tuple[RGB, RGB, RGB]) -> Sprite:
    dark, mid, light = colors
    p = Painter(11, 14, (5, 12))
    p.shadow(3, 1.5)
    p.line([(0, 0), (0, -3)], TRUNK)
    p.ellipse(-4, -10, 4, -3, mid)
    p.blob(-3, -9, 0, -6, light)
    p.point(2, -5, dark)
    p.point(-1, -4, dark)
    return p.sprite()


def rocks() -> Sprite:
    p = Painter(14, 10, (7, 7))
    p.shadow(5, 1.5)
    p.polygon([(-6, 1), (-4, -4), (0, -6), (3, -2), (2, 1)], ROCK, shade(ROCK, 0.55))
    p.polygon([(-4, -4), (0, -6), (-1, -2)], shade(ROCK, 1.25))
    p.polygon([(1, 1), (3, -3), (6, -1), (5, 2)], shade(ROCK, 0.9), shade(ROCK, 0.5))
    return p.sprite()


def relief(width: int, height: int, seed: int, *, peaked: bool) -> Sprite:
    """A hill, or a peaked mountain when `peaked`, as a height field seen from the view angle:
    every point of its ground is raised by its height and shaded by how much it faces the light
    from the top left, dithered between TINT_GREYS. The tallest mountains get snow on their tops,
    in colours that are not grey so they keep them when tinted."""
    rng = np.random.default_rng(seed)
    a = width / 2
    gx = np.arange(-a, a) + 0.5
    gy = np.arange(-a, a, 0.25) + 0.125
    x, y = np.meshgrid(gx / a, gy / a)
    envelope = np.clip(1 - x**2 - y**2, 0, 1)
    if peaked:
        lift = np.zeros_like(x)
        for _ in range(3):
            cx, cy = rng.uniform(-0.35, 0.35, 2)
            reach = rng.uniform(0.45, 0.7)
            cone = np.clip(1 - np.hypot(x - cx, (y - cy) * 1.3) / reach, 0, 1)
            lift = np.maximum(lift, rng.uniform(0.6, 1.0) * cone**1.2)
        ridges = 1 - np.abs(2 * _noise(rng, x.shape, 7) - 1)
        lift = lift * (0.8 + 0.35 * ridges) * np.sqrt(envelope)
    else:
        lift = np.zeros_like(x)
        for _ in range(2):
            cx, cy = rng.uniform(-0.3, 0.3, 2)
            reach = rng.uniform(0.45, 0.65)
            lift += rng.uniform(0.6, 1.0) * np.exp(-(((x - cx) ** 2 + (y - cy) ** 2) / reach**2))
        lift = lift * (0.9 + 0.2 * _noise(rng, x.shape, 4)) * envelope
    rise = lift / lift.max() * height
    slope_y, slope_x = np.gradient(rise, 0.25, 1.0)
    towards_light = np.array([-0.62, 0.1, 0.78]) / np.hypot(np.hypot(-0.62, 0.1), 0.78)
    facing = (towards_light[2] - towards_light[0] * slope_x - towards_light[1] * slope_y) / np.sqrt(
        1 + slope_x**2 + slope_y**2
    )
    grain = _noise(rng, x.shape, 12) - 0.5
    value = (
        2.6 + (4 if peaked else 3) * (facing - towards_light[2]) + (0.8 if peaked else 0.3) * grain
    )
    snowy = peaked and height >= 11
    snow = rise > 0.72 * height + 1.2 * grain

    anchor_y = int(np.ceil(height + a / 2)) + 1
    w, h = width + 2, anchor_y + int(np.ceil(a / 2)) + 1
    image = np.zeros((h, w, 4), dtype=np.uint8)
    rows = np.arange(h)[:, None]
    bayer = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]) / 16 - 0.5
    dither = bayer[rows % 4, np.arange(w) % 4]
    for i in range(len(gy)):
        top = anchor_y + gy[i] / 2 - rise[i]
        base = anchor_y + gy[i] / 2
        raised = rise[i] > 0.4
        cover = raised & (rows >= np.floor(top)) & (rows <= np.floor(base))
        cover = np.pad(cover, ((0, 0), (1, 1)))
        level = np.clip(np.rint(np.pad(value[i], 1)[None, :] + dither), 0, 5).astype(int)
        rgb = np.array(TINT_GREYS)[level][..., None].repeat(3, axis=-1)
        if snowy:
            lit = np.pad(value[i], 1)[None, :] + dither > 2.2
            white = np.where(lit[..., None], SNOW_LIT, SNOW_SHADE)
            rgb = np.where(np.pad(snow[i], 1)[None, :, None], white, rgb)
        image[cover, :3] = rgb[cover]
        image[cover, 3] = 255
    return Sprite(Image.fromarray(image), (w // 2, anchor_y))


def _noise(rng: np.random.Generator, shape: tuple[int, ...], cells: int) -> np.ndarray:
    """Smooth random values from 0 to 1 over `shape`, varying over about `cells` steps across."""
    coarse = rng.random((cells + 1, cells + 1))
    v = np.linspace(0, cells, shape[0])
    u = np.linspace(0, cells, shape[1])
    v0, u0 = np.minimum(v.astype(int), cells - 1), np.minimum(u.astype(int), cells - 1)
    fv, fu = (v - v0)[:, None], (u - u0)[None, :]
    fv, fu = fv * fv * (3 - 2 * fv), fu * fu * (3 - 2 * fu)
    top = coarse[v0][:, u0] * (1 - fu) + coarse[v0][:, u0 + 1] * fu
    bottom = coarse[v0 + 1][:, u0] * (1 - fu) + coarse[v0 + 1][:, u0 + 1] * fu
    return top * (1 - fv) + bottom * fv
