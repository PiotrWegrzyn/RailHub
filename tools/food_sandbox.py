"""Grows the food sandbox (railroads.sim.sandbox): a square piece of a start map with only houses,
farms, orchards and fishing grounds, from the first year of history, and prints every decade how
many of each stand, how well fed the houses are, how much food is made and eaten and how many
buildings died out.

Run with: uv run python tools/food_sandbox.py [--map poland] [--size 100] [--seed 1] [--years 300]

With --html PATH it also writes a page that shows the map at every line, to click through.
"""

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from railroads import maps
from railroads.mapgen.deposits import place_deposits
from railroads.sim.clock import HISTORY_START
from railroads.sim.constructions import BuildingKind, Constructions
from railroads.sim.economy import Satisfaction
from railroads.sim.goods import Need
from railroads.sim.growth import Growth
from railroads.sim.sandbox import FOOD_ONLY, SANDBOX_SIZE, small_map
from railroads.sim.tilemap import Deposit

KINDS = (
    BuildingKind.HOUSE,
    BuildingKind.FARM,
    BuildingKind.ORCHARD,
    BuildingKind.FISHERY,
    BuildingKind.POTATO_FARM,
    BuildingKind.BAKERY,
)


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
        f" {counts[4]:7d} {counts[5]:7d}"
        f" {fed:>5s} {hungry:>6s} {food:>15s} {sold:>5s} {growth.died_out:6d} {seconds:6.2f}s"
    )


def snapshot(growth: Growth) -> dict:
    """What the page shows of one moment: every cell's buildings, how well fed its houses are and
    how much of its farms' and orchards' food found a buyer."""
    cells = growth.grid.rows * growth.grid.cols
    fed = np.full(cells, -1.0)
    sold = np.full(cells, -1.0)
    trade = growth.trade
    if trade is not None:
        for (kind, need), (got, wanted) in trade.goods_got.items():
            if kind == BuildingKind.HOUSE and need == Need.FOOD:
                fed[trade.cells] = np.where(wanted > 0, got / np.where(wanted > 0, wanted, 1), -1)
        given = np.zeros(trade.cells.size)
        most = np.zeros(trade.cells.size)
        for (kind, _), (gave, could) in trade.goods_gave.items():
            if kind in KINDS:
                given += gave
                most += could
        sold[trade.cells] = np.where(most > 0, given / np.where(most > 0, most, 1), -1)
    return {
        "year": growth.year,
        "kinds": [growth.counts[kind].ravel().tolist() for kind in KINDS],
        "fed": np.round(fed, 2).tolist(),
        "sold": np.round(sold, 2).tolist(),
        "died": growth.died_out,
    }


def write_page(path: Path, tile_map, frames: list[dict], lines: list[str]) -> None:
    data = {
        "rows": tile_map.grid.rows,
        "cols": tile_map.grid.cols,
        "terrain": tile_map.terrain.ravel().tolist(),
        "vegetation": tile_map.vegetation.ravel().tolist(),
        "fish": (tile_map.deposits == Deposit.FISH).ravel().astype(int).tolist(),
        "frames": frames,
        "lines": lines,
        "title": tile_map.name,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(PAGE.replace("__DATA__", json.dumps(data)))


PAGE = (Path(__file__).with_name("food_sandbox.html")).read_text()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", default="poland")
    parser.add_argument("--size", type=int, default=SANDBOX_SIZE, help="cells along each side")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--years", type=int, default=300)
    parser.add_argument("--every", type=int, default=10, help="years between lines")
    parser.add_argument("--html", type=Path, help="write a page to click through the years")
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    tile_map = small_map(maps.load(args.map, "hex"), args.size)
    tile_map = replace(tile_map, deposits=place_deposits(tile_map, rng))
    fishing = int((tile_map.deposits == Deposit.FISH).sum())
    print(f"{tile_map.name}: {tile_map.grid.shape}, {fishing} fishing ground cells")
    growth = Growth(Constructions(tile_map), FOOD_ONLY, start_year=HISTORY_START)
    header = (
        " year houses farms orchards fisheries potatoes bakeries   fed hungry"
        "  house food got/need  sold  died"
    )
    print(header)
    started = time.perf_counter()
    frames = [snapshot(growth)]
    lines = [report(growth, 0.0)]
    for _ in range(args.years // args.every):
        growth.run(12 * args.every, rng)
        lines.append(report(growth, time.perf_counter() - started))
        frames.append(snapshot(growth))
        print(lines[-1], flush=True)
    if args.html is not None:
        write_page(args.html, tile_map, frames, [header, *lines])
        print(f"Wrote {args.html}")


if __name__ == "__main__":
    main()
