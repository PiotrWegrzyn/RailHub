"""Buying land next to the map in play (railroads.sim.lots).

A lot bought is cut from the map of the game's place (railroads.newgame.place_map) and grown on
its own from HISTORY_START to the year it is bought, under the same rules as the map's history,
so it arrives about as built up as the rest. Of the buildings the map keeps at least one of
(Growth.least_of) it keeps only a quarter, picked at random, and each of the game's novelties
not yet built comes up in it at a quarter of the chance it had over the years played, at most.
The game then moves onto the map with the lot (railroads.sim.regrid)."""

from collections.abc import Callable
from dataclasses import replace

import numpy as np

from railroads.mapgen.seeding import seed_growth_steps
from railroads.newgame import place_map
from railroads.saves import SavedGame
from railroads.sim.clock import HISTORY_START, MONTHS_PER_YEAR
from railroads.sim.constructions import ONE_OFFS, Building
from railroads.sim.game import Game
from railroads.sim.grids import Cell
from railroads.sim.growth import KEY_BUILDINGS, Growth
from railroads.sim.ladders import GROWTH
from railroads.sim.lots import Lot, Lots
from railroads.sim.novelties import YEARLY_CHANCE
from railroads.sim.regrid import regrid
from railroads.sim.steps import Steps
from railroads.sim.summaries import MonthSummary, StationSummary
from railroads.sim.tilemap import LAYERS, Terrain, TileMap
from railroads.sim.tiles import kind_era

# The share of the buildings the map keeps at least one of that land bought keeps too, and of
# the chance a novelty not yet built had over the years played that it comes up there.
TOP_UP_SHARE = 0.25
NOVELTY_SHARE = 0.25


def offered_lots(saved: SavedGame) -> list[Lot]:
    """The lots the player can buy, none in a game without lots or a place to cut them from."""
    lots = saved.game.lots
    if lots is None or saved.settings is None:
        return []
    return lots.offers


def lot_middle(lots: Lots, lot: Lot) -> Cell:
    """The map cell in the middle of a lot."""
    return lots.middle_cell(lot)


def buy_lot_steps(
    saved: SavedGame,
    lot: Lot,
    report: Callable[[str], None] = print,
    progress: Callable[[float], None] = lambda done: None,
) -> Steps[SavedGame]:
    """Buys a lot and grows it, a step at a time, telling `progress` the share of its history
    grown; returns the game on the map with the lot. Refuses a lot not offered or not afforded."""
    game, settings = saved.game, saved.settings
    lots = game.lots
    if lots is None or settings is None or lot not in offered_lots(saved):
        raise ValueError(f"Lot {lot} isn't for sale")
    price = lots.price()
    if not game.finances.pay(price, "land"):
        raise ValueError(f"Lot {lot} costs more than the cash there is")
    place = place_map(settings)
    year = game.growth.year
    report(f"Growing the land bought from year {HISTORY_START} to {year}")
    window = lots.window_of({lot})
    inside = lots.in_lots(window, {lot})
    grown = yield from seed_growth_steps(
        replace(place.window(*window), locked=~inside),
        saved.rng,
        replace(GROWTH, top_ups=quarter_of_top_ups(saved.rng), orders_share=TOP_UP_SHARE),
        months=(year - HISTORY_START) * MONTHS_PER_YEAR,
        progress=progress,
    )
    yield
    _grow_novelties(game, grown, saved.rng)
    old = lots.window()
    lots.buy(lot)
    lots.offer(_land(place))
    new = lots.window()
    rows, cols = old[0] - new[0], old[1] - new[1]
    at = window[0] - new[0], window[1] - new[1]
    lot_map = grown.constructions.tile_map
    tile_map = _joined(place, game.tile_map, (rows, cols), lot_map, at, inside, lots)
    built = [
        Building((building.cell[0] + at[0], building.cell[1] + at[1]), building.kind)
        for building in grown.constructions
        if isinstance(building, Building)
    ]
    report("Joining the land to the map")
    yield
    regrid(game, tile_map, rows, cols, built)
    return replace(
        saved,
        summaries=[
            (
                _moved_summary(summary, rows, cols)
                if isinstance(summary, MonthSummary)
                else summary,
                seen,
            )
            for summary, seen in saved.summaries
        ],
        camera=None,
    )


def quarter_of_top_ups(rng: np.random.Generator) -> frozenset:
    """TOP_UP_SHARE of the kinds the map keeps at least one of, picked at random."""
    kinds = sorted(
        {
            ladder.kinds[0]
            for ladder in GROWTH.ladders
            if ladder.kinds[0] in KEY_BUILDINGS
            or (
                (ladder.from_year > 0 or kind_era(ladder.kinds[0]) > 1)
                and ladder.kinds[0] not in ONE_OFFS
            )
        },
        key=lambda kind: kind.value,
    )
    picked = rng.choice(len(kinds), round(len(kinds) * TOP_UP_SHARE), replace=False)
    return frozenset(kinds[index] for index in picked.tolist())


def novelty_chance(years: int) -> float:
    """The chance a novelty not built yet comes up in land bought: NOVELTY_SHARE of the chance
    it had to come up over `years` played in its era."""
    if years <= 0:
        return 0.0
    return NOVELTY_SHARE * (1 - (1 - YEARLY_CHANCE) ** years)


def _grow_novelties(game: Game, grown: Growth, rng: np.random.Generator) -> None:
    """Puts up in the land bought, each at NOVELTY_SHARE of the chance it had over the years
    played, the game's novelties not built yet."""
    novelties = game.growth.novelties
    if novelties is None:
        return
    now, first = game.growth.year, game.railway.first_year
    for novelty in novelties.picked:
        if novelty.kind in novelties.built:
            continue
        years = min(now, novelty.until_year) - max(first, novelty.from_year)
        if rng.random() >= novelty_chance(years):
            continue
        cell = novelty.place(grown, novelty.kind, rng)
        if cell is not None:
            grown.build(Building(cell, novelty.kind))
            novelties.built.add(novelty.kind)


def _land(tile_map: TileMap) -> np.ndarray:
    return tile_map.terrain == Terrain.LAND


def _joined(
    place: TileMap,
    old: TileMap,
    old_at: Cell,
    lot: TileMap,
    lot_at: Cell,
    inside: np.ndarray,
    lots: Lots,
) -> TileMap:
    """The map of the place over the lots' window, with the old map at `old_at` and the cells
    `inside` the lot bought of its map at `lot_at` laid on it as they grew, and the lots not
    owned locked."""
    base = place.window(*lots.window())
    layers = {layer: getattr(base, layer).copy() for layer in LAYERS}
    deposits = np.zeros(base.grid.shape, dtype=np.uint8)
    everywhere = np.ones(old.grid.shape, dtype=bool)
    for part, (row, col), keep in ((old, old_at, everywhere), (lot, lot_at, inside)):
        rows, cols = part.grid.shape
        area = slice(row, row + rows), slice(col, col + cols)
        for layer in LAYERS:
            layers[layer][area] = np.where(keep, getattr(part, layer), layers[layer][area])
        if part.deposits is not None:
            deposits[area] = np.where(keep, part.deposits, deposits[area])
    return replace(
        old,
        grid=base.grid,
        bounds=base.bounds,
        deposits=deposits,
        locked=lots.locked(),
        **layers,
    )


def _moved_summary(summary: MonthSummary, rows: int, cols: int) -> MonthSummary:
    def moved(cell: Cell) -> Cell:
        return cell[0] + rows, cell[1] + cols

    return replace(
        summary,
        stations=tuple(
            StationSummary(moved(s.station), s.changes, tuple(moved(c) for c in s.cells))
            for s in summary.stations
        ),
    )
