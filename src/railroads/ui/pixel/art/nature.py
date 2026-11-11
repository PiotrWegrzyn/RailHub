"""Trees and rocks. A tree's crown is a heap of small lit lumps of leaves inside a ragged outline,
darker deep inside and underneath, each lump its own shade of green, with a fine leafy grain:
leafy trees spread wide on a short trunk with branches, and pines are spruces, narrow cones of
jagged tiers."""

import math

import numpy as np
from PIL import Image

from railroads.ui.pixel.art.painter import (
    RGB,
    ROCK,
    TRUNK,
    Painter,
    Sprite,
    fineness,
    shade,
)
from railroads.ui.pixel.noise import table_hash

F32 = np.float32
# Where the light comes from, as a direction out of the picture: from the top left.
LIGHT = np.array((-0.55, -0.65, 0.52)) / np.linalg.norm((-0.55, -0.65, 0.52))
# Trees are drawn this many pixels to an art pixel and shrunk to the fineness asked for, so they
# keep the same shape at every zoom.
DRAWN_AT = 2


def _heap(
    width: float,
    height: float,
    lumps: int,
    rng: np.random.Generator,
    colors: tuple[RGB, RGB, RGB],
    k: int,
    flat: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """A crown `width` by `height` art pixels, `k` pixels to one, heaped from lumps of leaves
    gathered into a few lobes: its colours, where it is solid, and the middle of its box."""
    w, h = width * k, height * k
    cols, rows = int(w + 4 * k), int(h + 4 * k)
    yy, xx = np.mgrid[0:rows, 0:cols].astype(F32)
    cx, cy = cols / 2, rows / 2
    best = np.full((rows, cols), -1e9, F32)
    nx, ny, tint = np.zeros_like(best), np.zeros_like(best), np.zeros_like(best)
    lobes = [(rng.uniform(0, 2 * math.pi), rng.uniform(0.55, 0.85)) for _ in range(5)]
    for i in range(lumps):
        if i % 3:
            angle, out = lobes[i % len(lobes)]
            angle += rng.normal(0, 0.35)
            reach = min(1.0, out + rng.normal(0, 0.18))
        else:
            angle, reach = rng.uniform(0, 2 * math.pi), math.sqrt(rng.uniform(0, 1)) * 0.7
        ox, oy = math.cos(angle) * reach * w / 2, math.sin(angle) * reach * h / 2
        lift = math.sqrt(max(0.0, 1 - reach * reach)) * (1 - flat)
        oy -= lift * h * 0.1
        radius = rng.uniform(0.12, 0.22) * min(w, h) * (0.85 + 0.4 * lift)
        dx, dy = (xx - cx - ox) / radius, (yy - cy - oy) / radius
        d2 = dx * dx + dy * dy
        z = np.sqrt(np.clip(1 - d2, 0, 1)) * 0.6 + lift
        top = (d2 < 1) & (z > best)
        best = np.where(top, z, best)
        nx, ny = np.where(top, dx, nx), np.where(top, dy, ny)
        tint = np.where(top, rng.uniform(-1, 1), tint)
    solid = best > -1e8
    nz = np.sqrt(np.clip(1 - nx * nx - ny * ny, 0, 1))
    light = np.clip(nx * LIGHT[0] + ny * LIGHT[1] + nz * LIGHT[2], 0, 1)
    depth = np.clip(best, 0, 1.6) / 1.6
    under = np.clip((yy - cy) / (h / 2), 0, 1)
    grain = table_hash(xx.astype(np.int64), yy.astype(np.int64), int(rng.integers(1000))) - 0.5
    level = light * (0.5 + 0.5 * depth) * (1 - 0.35 * under) + 0.18 * grain
    dark, mid, high = (np.array(c, F32) for c in colors)
    color = np.where(
        (level < 0.5)[..., None],
        dark + (mid - dark) * (level / 0.5)[..., None],
        mid + (high - mid) * np.clip((level - 0.5) / 0.5, 0, 1)[..., None],
    )
    color = color * (1 + 0.1 * tint[..., None])
    edge = solid & ~(
        np.roll(solid, 1, 0) & np.roll(solid, -1, 0) & np.roll(solid, 1, 1) & np.roll(solid, -1, 1)
    )
    color = np.where(edge[..., None], color * 0.8, color)
    return color, solid, cx, cy


def _standing(
    crown: np.ndarray,
    solid: np.ndarray,
    stem: np.ndarray,
    bark: np.ndarray,
    foot: tuple[float, float],
    k: int,
) -> Sprite:
    """The tree's picture, its stem under its crown, standing on a soft shadow at its foot."""
    rows, cols = stem.shape
    picture = np.zeros((rows, cols, 4), np.uint8)
    picture[stem, :3] = np.clip(bark[stem], 0, 255)
    picture[stem, 3] = 255
    crown_rows = crown.shape[0]
    picture[:crown_rows][solid, :3] = np.clip(crown[solid], 0, 255)
    picture[:crown_rows][solid, 3] = 255
    image = Image.fromarray(picture, "RGBA")
    unit = fineness()
    shrink = unit / k
    fx, fy = foot[0] * shrink, foot[1] * shrink
    if shrink != 1:
        size = (max(1, round(cols * shrink)), max(1, round(rows * shrink)))
        image = image.resize(size, Image.Resampling.BOX)
    p = Painter(image.width / unit + 4, image.height / unit + 2, (fx / unit + 2, fy / unit))
    p.shadow(image.width / unit / 4, 1.2)
    p.image.alpha_composite(image, (p.anchor[0] - round(fx), p.anchor[1] - round(fy)))
    return p.sprite()


def leafy(colors: tuple[RGB, RGB, RGB], seed: int = 0, height: float = 9.0) -> Sprite:
    """A leafy tree like an oak: a broad ragged crown of leaf clusters on a short dark trunk
    with branches reaching up into it."""
    k = DRAWN_AT
    rng = np.random.default_rng(seed)
    width = height * rng.uniform(1.15, 1.4)
    crown, solid, cx, cy = _heap(width, height * 0.66, int(40 + width * 3), rng, colors, k)
    rows, cols = crown.shape[0] + 3 * k, crown.shape[1]
    yy, xx = np.mgrid[0:rows, 0:cols].astype(F32)
    tx = cx + rng.uniform(-0.5, 0.5) * k
    half_height = height * 0.66 * k / 2
    foot = cy + half_height + 2.6 * k
    stem = (np.abs(xx - tx) < 0.75 * k) & (yy > cy) & (yy < foot)
    for side in (-1, 1):
        start = cy + half_height * rng.uniform(0.0, 0.5)
        slope = rng.uniform(1.2, 2.2)
        reach = start - yy
        stem |= (
            (np.abs(xx - tx - side * reach * slope) < 0.5 * k)
            & (yy < start)
            & (reach < half_height * 0.6)
        )
    lit = xx < tx
    bark = np.where(lit[..., None], np.array(shade(TRUNK, 1.2), F32), np.array(TRUNK, F32))
    return _standing(crown, solid, stem, bark, (tx, foot), k)


def pine(colors: tuple[RGB, RGB, RGB], seed: int = 0, height: float = 13.0) -> Sprite:
    """A spruce: a dark cone of drooping, jagged tiers lit on its left, a little taller or
    broader from tree to tree."""
    rng = np.random.default_rng(seed)
    return _spruce(colors, seed, height * rng.uniform(0.75, 0.95), DRAWN_AT, rng.uniform(0.5, 0.62))


def _spruce(
    colors: tuple[RGB, RGB, RGB], seed: int, height: float, k: int, broad: float = 0.5
) -> Sprite:
    dark, mid, high = (np.array(c, F32) * 0.85 for c in colors)
    tall = height * k
    cols, rows = int(height * broad * k) + 4, int(tall) + 4
    yy, xx = np.mgrid[0:rows, 0:cols].astype(F32)
    cx = cols / 2
    t = (yy - 2) / tall
    phase = (t * (5 + int(height / 4))) % 1
    half = (0.12 + 0.88 * t) * height * broad / 2 * k * (0.75 + 0.35 * phase)
    jag = (
        (table_hash(xx.astype(np.int64), (yy // 2).astype(np.int64), seed) - 0.5)
        * k
        * 1.4
        * np.clip(t * 2.5, 0, 1)
    )
    solid = (np.abs(xx - cx) <= half + jag) & (t >= 0) & (t <= 1)
    side = (xx - cx) / np.maximum(half, 1e-3)
    level = np.clip(0.55 - 0.4 * side + 0.25 * phase - 0.25 * (1 - phase) ** 4, 0, 1)
    level = np.clip(
        level + 0.2 * (table_hash(xx.astype(np.int64), yy.astype(np.int64), seed + 1) - 0.5), 0, 1
    )
    crown = np.where(
        (level < 0.5)[..., None],
        dark + (mid - dark) * (level / 0.5)[..., None],
        mid + (high - mid) * ((level - 0.5) / 0.5)[..., None],
    )
    total = rows + k
    stem = np.zeros((total, cols), bool)
    stem[rows - 2 : total, int(cx) - k // 2 : int(cx) + k // 2 + 1] = True
    bark = np.broadcast_to(np.array(TRUNK, F32), (total, cols, 3))
    return _standing(crown, solid, stem, bark, (cx, total - 1), k)


def rocks() -> Sprite:
    p = Painter(14, 10, (7, 7))
    p.shadow(5, 1.5)
    p.polygon([(-6, 1), (-4, -4), (0, -6), (3, -2), (2, 1)], ROCK, shade(ROCK, 0.55))
    p.polygon([(-4, -4), (0, -6), (-1, -2)], shade(ROCK, 1.25))
    p.polygon([(1, 1), (3, -3), (6, -1), (5, 2)], shade(ROCK, 0.9), shade(ROCK, 0.5))
    return p.sprite()
