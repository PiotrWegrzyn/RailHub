"""Bake the world's land into src/railroads/maps/world-land.npz, for the minimap.

Inputs in data/raw/ (download commands in README.md): Natural Earth ne_10m_ocean and lakes.

Run with: uv run python tools/build_world_map.py
"""

from pathlib import Path

from railroads import maps
from railroads.mapgen.presets import NaturalEarth
from railroads.mapgen.world import WorldLand

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
DEGREES_PER_CELL = 0.25


def main() -> None:
    world = NaturalEarth.load(RAW)
    land = WorldLand.bake(world.ocean + world.lakes, DEGREES_PER_CELL)
    land.save(maps.WORLD_LAND_FILE)
    rows, cols = land.land.shape
    share = land.land.mean() / 255
    print(f"wrote {maps.WORLD_LAND_FILE.name}: {rows} x {cols} cells, {share:.0%} land")


if __name__ == "__main__":
    main()
