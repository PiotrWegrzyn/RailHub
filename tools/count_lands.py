"""Count what grows on each land of the made-up map of every land (railroads.mapgen.lands).

Starts a game there and prints, for every kind of building, how many stand on each of the nine
lands when play begins, and again every so many years up to a given year. Played on, every
building gets the share of what it needs given by --share, standing in for the railway.

Run with: uv run python tools/count_lands.py [--seed 1] [--until 1850] [--every 50] [--share 1.0]
"""

import argparse

import numpy as np

from railroads.mapgen.lands import BANDS, COVERS
from railroads.newgame import MapChoice, Settings, make_game
from railroads.sim.constructions import BuildingKind
from railroads.sim.growth import Growth
from railroads.sim.tilemap import Terrain

NAMES = {
    (0, 0): "ice",
    (0, 1): "tundra",
    (0, 2): "taiga",
    (1, 0): "steppe",
    (1, 1): "farmland",
    (1, 2): "woods",
    (2, 0): "desert",
    (2, 1): "savanna",
    (2, 2): "jungle",
}


def table(growth: Growth) -> str:
    """Every kind of building standing, how many on each land and in all, as a Markdown table."""
    tile_map = growth.constructions.tile_map
    land = tile_map.terrain == Terrain.LAND
    lands = [
        land & (tile_map.temperature == band) & (tile_map.vegetation == cover)
        for band in BANDS
        for cover in COVERS
    ]
    names = [NAMES[(b, c)] for b in range(len(BANDS)) for c in range(len(COVERS))]
    lines = [
        "| building | " + " | ".join(names) + " | all |",
        "|---" * (len(names) + 2) + "|",
    ]
    for kind in BuildingKind:
        counts = growth.count(kind) if kind in growth.counts else None
        if counts is None or not counts.any():
            continue
        each = [int(counts[where].sum()) for where in lands]
        lines.append(f"| {kind.value} | " + " | ".join(map(str, each)) + f" | {sum(each)} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--until", type=int, default=None)
    parser.add_argument("--every", type=int, default=50)
    parser.add_argument("--share", type=float, default=1.0)
    args = parser.parse_args()
    settings = Settings(seed=args.seed, map=MapChoice.LANDS)
    growth = make_game(settings, report=lambda text: None).game.growth
    print(f"## {growth.year}\n\n{table(growth)}\n", flush=True)
    if args.until is None:
        return
    growth.assumed_shares = np.full(growth.grid.rows * growth.grid.cols, args.share)
    rng = np.random.default_rng(args.seed)
    next_print = growth.year + args.every
    while growth.year < args.until:
        growth.month(rng)
        due = growth.year >= next_print or growth.year >= args.until
        if due and growth.months % 12 == 0:
            print(f"## {growth.year}\n\n{table(growth)}\n", flush=True)
            next_print += args.every


if __name__ == "__main__":
    main()
