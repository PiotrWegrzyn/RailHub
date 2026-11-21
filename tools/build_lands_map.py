"""Bake the made-up map of every land (railroads.mapgen.lands) into src/railroads/maps/.

Run with: uv run python tools/build_lands_map.py
"""

from railroads import maps
from railroads.mapgen.lands import lands_map


def main() -> None:
    lands_map().save(maps.map_file("lands", "hex"))


if __name__ == "__main__":
    main()
