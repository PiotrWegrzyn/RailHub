"""Ready-made strategies for maps of real places, built from public-domain data: NOAA ETOPO1
elevation, and Natural Earth sea, lakes, rivers and deserts."""

from dataclasses import dataclass
from pathlib import Path
from typing import Self

from railroads.mapgen.sources import Elevation, Shape, read_geojson
from railroads.mapgen.strategies import (
    HeightFromElevation,
    ReliefFromSlopes,
    Strategy,
    TemperatureFromClimate,
    VegetationPlaceholder,
    WaterFromShapes,
)

OCEAN_FILE = "ne_10m_ocean.geojson"
LAKE_FILES = ("ne_10m_lakes.geojson", "ne_10m_lakes_europe.geojson")
RIVER_FILES = ("ne_10m_rivers_lake_centerlines.geojson", "ne_10m_rivers_europe.geojson")
REGIONS_FILE = "ne_10m_geography_regions_polys.geojson"
MIN_RIVER_KM = 200.0
ETOPO_SOURCE = "Elevation: NOAA ETOPO1, 1 arc-minute, public domain"
NATURAL_EARTH_SOURCE = "Sea, lakes, rivers and deserts: Natural Earth 1:10m, public domain"


@dataclass(frozen=True)
class NaturalEarth:
    ocean: list[Shape]
    lakes: list[Shape]
    rivers: list[Shape]
    deserts: list[Shape]

    @classmethod
    def load(cls, raw_dir: Path) -> Self:
        def shapes(*names: str) -> list[Shape]:
            return [shape for name in names for shape in read_geojson(raw_dir / name)]

        regions = shapes(REGIONS_FILE)
        return cls(
            ocean=shapes(OCEAN_FILE),
            lakes=shapes(*LAKE_FILES),
            rivers=shapes(*RIVER_FILES),
            deserts=[shape for shape in regions if shape.properties.get("FEATURECLA") == "Desert"],
        )


def real_world_strategies(
    data: NaturalEarth, elevation: Elevation, seed: int
) -> dict[str, Strategy]:
    return {
        "height_m": HeightFromElevation(elevation),
        "terrain": WaterFromShapes(
            lakes=data.lakes, rivers=data.rivers, sea=data.ocean, min_river_km=MIN_RIVER_KM
        ),
        "relief": ReliefFromSlopes(),
        "temperature": TemperatureFromClimate(deserts=data.deserts),
        "vegetation": VegetationPlaceholder(deserts=data.deserts, seed=seed),
    }
