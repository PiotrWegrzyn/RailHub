"""Where a good is made and where it is wanted, from the last month of trade
(railroads.sim.trade.Trade): what each cell's buildings made of it and what was left that nothing
took, what they wanted of the needs it meets and what they were short of, and the biggest makers
and buyers on the map.

Buildings want needs, not goods: a bakery wants flour and homes want food, which wheat, apples or
fish all meet. So a good's buyers are the buildings with a need it meets, each counted with what
it wanted and was short of in that need, whichever good would fill it."""

from dataclasses import dataclass

import numpy as np

from railroads.sim.constructions import BuildingKind
from railroads.sim.goods import FITS, Good, Need
from railroads.sim.grids import Cell, FloatArray
from railroads.sim.trade import Trade

# How many of the biggest makers and buyers a good's map lists.
MOST_LISTED = 8


@dataclass(frozen=True, slots=True)
class Spot:
    """One kind of building in one cell with an amount of a good a month: what it made or what
    it was left with, or what it wanted or was short of, and the need it wants the good for."""

    cell: Cell
    kind: BuildingKind
    amount: float
    need: Need | None = None


@dataclass(frozen=True, slots=True)
class GoodMap:
    """A good on the whole map last month, every array a flat cell's buildings summed."""

    good: Good
    made: FloatArray
    left: FloatArray
    wanted: FloatArray
    short: FloatArray
    # The kinds and cells left with the most and short of the most, the most first, and those
    # that made and wanted the most.
    most_left: tuple[Spot, ...]
    most_short: tuple[Spot, ...]
    most_made: tuple[Spot, ...]
    most_wanted: tuple[Spot, ...]


@dataclass(frozen=True, slots=True)
class GoodTotals:
    """A good on the whole map last month: made, left over, wanted and short of."""

    good: Good
    made: float
    left: float
    wanted: float
    short: float


def needs_met_by(good: Good) -> tuple[Need, ...]:
    """Every need a good meets."""
    return tuple(need for need in Need if good in FITS[need])


def good_map(trade: Trade, size: int, cols: int, good: Good) -> GoodMap:
    """A good on a map of `size` flat cells, `cols` wide, from a month of trade."""
    made, left, wanted, short = (np.zeros(size) for _ in range(4))
    makers: list[tuple[Spot, Spot]] = []
    buyers: list[tuple[Spot, Spot]] = []
    for (kind, given_good), (given, _) in trade.goods_gave.items():
        if given_good != good:
            continue
        unsold = trade.goods_left.get((kind, good), np.zeros_like(given))
        here = given + unsold
        made[trade.cells] += here
        left[trade.cells] += unsold
        makers += _spots(trade, cols, kind, here, unsold, None)
    needs = needs_met_by(good)
    for (kind, need), (got, want) in trade.goods_got.items():
        if need not in needs:
            continue
        lacking = np.maximum(want - got, 0.0)
        wanted[trade.cells] += want
        short[trade.cells] += lacking
        buyers += _spots(trade, cols, kind, want, lacking, need)
    return GoodMap(
        good=good,
        made=made,
        left=left,
        wanted=wanted,
        short=short,
        most_left=_most([gap for _, gap in makers]),
        most_short=_most([gap for _, gap in buyers]),
        most_made=_most([whole for whole, _ in makers]),
        most_wanted=_most([whole for whole, _ in buyers]),
    )


def goods_totals(trade: Trade) -> dict[Good, GoodTotals]:
    """Every good made or wanted last month, with its totals on the map."""
    made: dict[Good, float] = {}
    left: dict[Good, float] = {}
    for (kind, good), (given, _) in trade.goods_gave.items():
        unsold = trade.goods_left.get((kind, good))
        unsold_sum = 0.0 if unsold is None else float(unsold.sum())
        made[good] = made.get(good, 0.0) + float(given.sum()) + unsold_sum
        left[good] = left.get(good, 0.0) + unsold_sum
    wanted: dict[Need, float] = {}
    short: dict[Need, float] = {}
    for (_, need), (got, want) in trade.goods_got.items():
        wanted[need] = wanted.get(need, 0.0) + float(want.sum())
        short[need] = short.get(need, 0.0) + float(np.maximum(want - got, 0.0).sum())
    goods = set(made) | {good for need in wanted for good in FITS[need]}
    return {
        good: GoodTotals(
            good=good,
            made=made.get(good, 0.0),
            left=left.get(good, 0.0),
            wanted=sum(wanted.get(need, 0.0) for need in needs_met_by(good)),
            short=sum(short.get(need, 0.0) for need in needs_met_by(good)),
        )
        for good in goods
    }


def _spots(
    trade: Trade,
    cols: int,
    kind: BuildingKind,
    whole: FloatArray,
    gap: FloatArray,
    need: Need | None,
) -> list[tuple[Spot, Spot]]:
    """Each traded cell with any of the whole, as the whole and as the gap."""
    spots = []
    for at in np.flatnonzero(whole > 0).tolist():
        cell = divmod(int(trade.cells[at]), cols)
        spots.append(
            (
                Spot(cell, kind, float(whole[at]), need),
                Spot(cell, kind, float(gap[at]), need),
            )
        )
    return spots


def _most(spots: list[Spot]) -> tuple[Spot, ...]:
    """The spots with any amount, the most first, at most MOST_LISTED of them."""
    kept = sorted((spot for spot in spots if spot.amount > 0), key=lambda spot: -spot.amount)
    return tuple(kept[:MOST_LISTED])


@dataclass(frozen=True, slots=True)
class CellGood:
    """A good in one cell last month: each kind making it with what it made and what was left,
    each kind wanting it with the need, what it wanted and got, and per need what the cell's
    buildings got from sellers nearby rather than from stations."""

    makers: tuple[tuple[BuildingKind, float, float], ...]
    buyers: tuple[tuple[BuildingKind, Need, float, float], ...]
    nearby: tuple[tuple[Need, float, float], ...]


def cell_good(trade: Trade, flat: int, good: Good) -> CellGood:
    """A good in the flat cell, from a month of trade; nothing where nothing traded."""
    at = int(np.searchsorted(trade.cells, flat))
    if at >= trade.cells.size or trade.cells[at] != flat:
        return CellGood((), (), ())
    makers = []
    for (kind, given_good), (given, _) in trade.goods_gave.items():
        if given_good == good:
            unsold = trade.goods_left.get((kind, good))
            left = 0.0 if unsold is None else float(unsold[at])
            made = float(given[at]) + left
            if made > 0:
                makers.append((kind, made, left))
    needs = needs_met_by(good)
    buyers = []
    got_of: dict[Need, float] = {}
    for (kind, need), (got, want) in trade.goods_got.items():
        if need in needs and want[at] > 0:
            buyers.append((kind, need, float(want[at]), float(got[at])))
            got_of[need] = got_of.get(need, 0.0) + float(got[at])
    nearby = tuple(
        (need, got, float(trade.local[need][at]) if need in trade.local else 0.0)
        for need, got in got_of.items()
    )
    return CellGood(tuple(makers), tuple(buyers), nearby)
