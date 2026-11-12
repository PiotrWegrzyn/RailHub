"""Grows the food sandbox (railroads.sim.sandbox): a 50x50 piece of a start map with only houses,
farms, orchards and fishing grounds, from the first year of history, and prints every decade how
many of each stand, how well fed the houses are, how much food is made and eaten and how many
buildings died out.

Run with: uv run python tools/food_sandbox.py [--map poland] [--seed 1] [--years 300]
"""

import argparse
import time
from dataclasses import replace

import numpy as np

from railroads import maps
from railroads.mapgen.deposits import place_deposits
from railroads.sim.clock import HISTORY_START
from railroads.sim.constructions import BuildingKind, Constructions
from railroads.sim.economy import Satisfaction
from railroads.sim.goods import Need
from railroads.sim.growth import Growth
from railroads.sim.sandbox import FOOD_ONLY, small_map
from railroads.sim.tilemap import Deposit

KINDS = (BuildingKind.HOUSE, BuildingKind.FARM, BuildingKind.ORCHARD, BuildingKind.FISHERY)


def report(growth: Growth, seconds: float) -> str:
    counts = [int(growth.counts[kind].sum()) for kind in KINDS]
    trade = growth.trade
    fed = "-"
    hungry = "-"
    food = "-"
    if trade is not None and BuildingKind.HOUSE in trade.got:
        houses = growth.counts[BuildingKind.HOUSE].ravel()[trade.cells]
        share = trade.share((BuildingKind.HOUSE,), empty=0.0)
        weight = houses.sum()
        if weight:
            fed = f"{float((share * houses).sum() / weight):4.0%}"
            levels = trade.levels((BuildingKind.HOUSE,))
            hungry = f"{float(houses[levels == Satisfaction.UNSATISFIED].sum() / weight):4.0%}"
        got, wanted = trade.goods_got.get((BuildingKind.HOUSE, Need.FOOD), (None, None))
        if got is not None:
            food = f"{float(got.sum()):7.0f}/{float(wanted.sum()):7.0f}"
    made = (
        sum(float(given.sum()) for (_, _), (given, _) in trade.goods_gave.items()) if trade else 0
    )
    could = sum(float(most.sum()) for (_, _), (_, most) in trade.goods_gave.items()) if trade else 0
    sold = f"{made / could:4.0%}" if could else "-"
    return (
        f"{growth.year:5d} {counts[0]:6d} {counts[1]:5d} {counts[2]:7d} {counts[3]:8d}"
        f" {fed:>5s} {hungry:>6s} {food:>15s} {sold:>5s} {growth.died_out:6d} {seconds:6.2f}s"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", default="poland")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--years", type=int, default=300)
    parser.add_argument("--every", type=int, default=10, help="years between lines")
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    tile_map = small_map(maps.load(args.map, "hex"))
    tile_map = replace(tile_map, deposits=place_deposits(tile_map, rng))
    fishing = int((tile_map.deposits == Deposit.FISH).sum())
    print(f"{tile_map.name}: {tile_map.grid.shape}, {fishing} fishing ground cells")
    growth = Growth(Constructions(tile_map), FOOD_ONLY, start_year=HISTORY_START)
    print(" year houses farms orchards fisheries   fed hungry  house food got/need  sold  died")
    started = time.perf_counter()
    for _ in range(args.years // args.every):
        growth.run(12 * args.every, rng)
        print(report(growth, time.perf_counter() - started), flush=True)


if __name__ == "__main__":
    main()
