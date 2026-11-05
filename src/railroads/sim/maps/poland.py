"""A hand-traced, approximate Poland: Baltic coast, main rivers, lake districts and mountains.

Coordinates are (longitude, latitude) in degrees.
"""

import numpy as np

from railroads.sim.geo import LonLat, Projection, distance_to_polyline, polygon_mask
from railroads.sim.world import Terrain, World, fractal_noise

PROJECTION = Projection(west=13.6, east=24.4, south=48.9, north=55.0)

BALTIC_SEA: list[LonLat] = [
    (13.6, 55.0),
    (13.6, 54.25),
    (14.2, 53.95),
    (14.8, 54.03),
    (15.6, 54.19),
    (16.4, 54.44),
    (16.9, 54.6),
    (17.6, 54.77),
    (18.35, 54.84),
    (18.55, 54.65),
    (18.55, 54.5),
    (18.68, 54.38),
    (19.0, 54.35),
    (19.6, 54.42),
    (19.95, 54.6),
    (19.95, 54.95),
    (21.25, 54.95),
    (21.25, 55.0),
]

SZCZECIN_LAGOON: list[LonLat] = [
    (14.2, 53.9),
    (14.0, 53.75),
    (14.3, 53.68),
    (14.62, 53.72),
    (14.55, 53.88),
]

VISTULA_LAGOON: list[LonLat] = [
    (19.3, 54.36),
    (19.6, 54.38),
    (20.0, 54.6),
    (20.3, 54.65),
    (19.9, 54.45),
    (19.4, 54.3),
]

RIVERS: list[list[LonLat]] = [
    [
        (18.97, 49.62),
        (19.4, 50.0),
        (19.94, 50.05),
        (20.8, 50.3),
        (21.75, 50.68),
        (21.85, 51.56),
        (21.25, 52.0),
        (21.02, 52.23),
        (20.4, 52.4),
        (19.7, 52.55),
        (18.6, 53.01),
        (18.75, 53.48),
        (18.8, 54.08),
        (18.95, 54.35),
    ],
    [
        (18.3, 49.9),
        (17.93, 50.67),
        (17.03, 51.11),
        (16.08, 51.66),
        (15.73, 51.8),
        (15.0, 52.2),
        (14.63, 52.59),
        (14.4, 53.1),
        (14.55, 53.43),
        (14.4, 53.72),
    ],
    [
        (19.4, 50.6),
        (18.7, 51.8),
        (17.8, 52.0),
        (16.93, 52.41),
        (15.6, 52.73),
        (14.63, 52.59),
    ],
    [
        (23.6, 50.4),
        (24.1, 51.3),
        (23.6, 52.1),
        (22.6, 52.4),
        (21.5, 52.5),
        (21.02, 52.23),
    ],
]

MOUNTAIN_RIDGES: list[tuple[list[LonLat], float, float]] = [
    ([(15.0, 50.88), (15.8, 50.75), (16.3, 50.6), (16.7, 50.25), (17.4, 50.25)], 12.0, 0.85),
    ([(18.8, 49.6), (19.5, 49.45), (20.0, 49.25), (20.7, 49.4), (21.6, 49.4)], 14.0, 1.0),
    ([(21.6, 49.4), (22.4, 49.2), (22.9, 49.05), (24.4, 48.95)], 14.0, 0.8),
]

UPLANDS: list[tuple[LonLat, float, float]] = [
    ((19.4, 50.45), 35.0, 0.42),
    ((20.9, 50.85), 22.0, 0.55),
    ((22.6, 50.95), 35.0, 0.36),
    ((18.0, 54.25), 28.0, 0.42),
    ((16.0, 50.9), 30.0, 0.4),
]

LAKE_DISTRICTS: list[tuple[LonLat, float]] = [
    ((21.6, 53.85), 55.0),
    ((17.2, 53.85), 60.0),
    ((15.6, 53.4), 40.0),
]

RIVER_HALF_WIDTH_TILES = 0.6
LAKE_SIZE_TILES = 5.0
LAKE_THRESHOLD = 0.8


def generate(seed: int = 1) -> World:
    xs, ys = PROJECTION.tile_centers()
    width, height = PROJECTION.width_tiles, PROJECTION.height_tiles
    rng = np.random.default_rng(seed)
    noise = fractal_noise(width, height, rng)
    noise = (noise - noise.min()) / (noise.max() - noise.min())

    elevation = 0.25 * noise
    for ridge, half_width, peak in MOUNTAIN_RIDGES:
        distance = distance_to_polyline(xs, ys, PROJECTION, ridge)
        elevation += peak * np.exp(-((distance / half_width) ** 2))
    for (lon, lat), radius, height_gain in UPLANDS:
        cx, cy = PROJECTION.to_tile(lon, lat)
        distance = np.hypot(xs - cx, ys - cy)
        elevation += height_gain * np.exp(-((distance / radius) ** 2))

    terrain = np.full((height, width), Terrain.PLAINS, dtype=np.uint8)
    terrain[elevation > 0.42] = Terrain.HILLS
    terrain[elevation > 0.75] = Terrain.MOUNTAINS

    water = polygon_mask(xs, ys, PROJECTION, BALTIC_SEA)
    water |= polygon_mask(xs, ys, PROJECTION, SZCZECIN_LAGOON)
    water |= polygon_mask(xs, ys, PROJECTION, VISTULA_LAGOON)
    lake_noise = fractal_noise(width, height, rng, octaves=2, first_cell=LAKE_SIZE_TILES)
    lake_noise = (lake_noise - lake_noise.min()) / (lake_noise.max() - lake_noise.min())
    for (lon, lat), radius in LAKE_DISTRICTS:
        cx, cy = PROJECTION.to_tile(lon, lat)
        in_district = np.hypot(xs - cx, ys - cy) < radius
        water |= in_district & (lake_noise > LAKE_THRESHOLD)
    for river in RIVERS:
        water |= distance_to_polyline(xs, ys, PROJECTION, river) < RIVER_HALF_WIDTH_TILES
    terrain[water] = Terrain.WATER

    return World(width=width, height=height, terrain=terrain)
