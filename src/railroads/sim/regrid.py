"""Moves a running game onto a bigger map, as when the player buys land next to it
(railroads.sim.lots): every cell moves down and right by the rows and columns added above and to
the left, and everything the game keeps by cell or in arrays over the map moves with it.

What can be worked out again from what is built (the counts, neighbourhoods, towns, chances and
caches) is worked out afresh on the bigger map; what the game remembers (satisfaction,
stockpiles, starving buildings, the player's shares, stations, trains and their goods) is moved."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import numpy.typing as npt

from railroads.sim.constructions import Building, Construction, Constructions, Station
from railroads.sim.economy import Economy
from railroads.sim.game import Game
from railroads.sim.grids import HEX_ROW_STEP, Cell, Grid, HexGrid
from railroads.sim.growth import Growth
from railroads.sim.summaries import Reach
from railroads.sim.tilemap import TileMap
from railroads.sim.track_network import TrackNetwork
from railroads.sim.trade import Stockpiles, Trade
from railroads.sim.trains import Train


@dataclass(frozen=True)
class Move:
    """Where the cells of the `old` grid lie on the `new`: `rows` further down and `cols`
    further right."""

    old: Grid
    new: Grid
    rows: int
    cols: int

    def __post_init__(self) -> None:
        if self.rows % 2 and isinstance(self.old, HexGrid):
            raise ValueError("Hex rows move in twos, so every hex keeps its neighbours")
        if (
            self.rows < 0
            or self.cols < 0
            or self.rows + self.old.rows > self.new.rows
            or self.cols + self.old.cols > self.new.cols
        ):
            raise ValueError(f"{self.old.shape} moved by {self.rows, self.cols} is off the new map")

    def cell(self, cell: Cell) -> Cell:
        return cell[0] + self.rows, cell[1] + self.cols

    def cells(self, cells: Iterable[Cell]) -> set[Cell]:
        return {self.cell(cell) for cell in cells}

    def keys[V](self, by_cell: Mapping[Cell, V]) -> dict[Cell, V]:
        return {self.cell(cell): value for cell, value in by_cell.items()}

    def parcels[V](self, parcels: Mapping[Cell | None, V]) -> dict[Cell | None, V]:
        return {None if cell is None else self.cell(cell): value for cell, value in parcels.items()}

    def flat(self, index: int) -> int:
        row, col = divmod(index, self.old.cols)
        return (row + self.rows) * self.new.cols + col + self.cols

    def flats(self, indices: npt.NDArray[np.intp]) -> npt.NDArray[np.intp]:
        rows, cols = np.divmod(np.asarray(indices, dtype=np.intp), self.old.cols)
        return (rows + self.rows) * self.new.cols + cols + self.cols

    def grid[T: np.generic](self, array: npt.NDArray[T], fill: Any = 0) -> npt.NDArray[T]:
        """An array over the old grid's rows and columns, the first two axes, on the new grid."""
        moved = np.full((*self.new.shape, *array.shape[2:]), fill, dtype=array.dtype)
        moved[self.rows : self.rows + self.old.rows, self.cols : self.cols + self.old.cols] = array
        return moved

    def flat_cells[T: np.generic](self, array: npt.NDArray[T], fill: Any = 0) -> npt.NDArray[T]:
        """An array over the old grid's flat cells, its last axis, on the new grid's."""
        lead = array.shape[:-1]
        on_grid = np.moveaxis(array.reshape(*lead, *self.old.shape), (-2, -1), (0, 1))
        moved = np.moveaxis(self.grid(on_grid, fill), (0, 1), (-2, -1))
        return np.ascontiguousarray(moved.reshape(*lead, self.new.rows * self.new.cols))

    def point(self, point: tuple[float, float]) -> tuple[float, float]:
        """A point on the map in cell units (railroads.sim.grids.Grid.center)."""
        step = HEX_ROW_STEP if isinstance(self.old, HexGrid) else 1.0
        return point[0] + self.cols, point[1] + self.rows * step

    def construction(self, built: Construction) -> Construction:
        if isinstance(built, Station) and built.main is not None:
            return replace(built, cell=self.cell(built.cell), main=self.cell(built.main))
        return replace(built, cell=self.cell(built.cell))


def regrid(
    game: Game,
    tile_map: TileMap,
    rows: int,
    cols: int,
    buildings: Iterable[Building] = (),
) -> None:
    """Moves the game onto `tile_map`, its old map `rows` down and `cols` right on it, and puts
    up `buildings` there, already on the new map's cells, as land bought arrives grown: those
    that can stand there as the map now is."""
    move = Move(game.tile_map.grid, tile_map.grid, rows, cols)
    _move_constructions(game.constructions, tile_map, move)
    for building in buildings:
        if game.constructions.why_not(building) is None:
            game.constructions.build(building)
    game.tile_map = tile_map
    old = game.growth
    growth = _moved_growth(old, move)
    game.growth = growth
    _move_railway(game, move)
    game._reach = Reach(tile_map.grid)
    if game.stand_in is not None:
        game.stand_in.growth = growth
        game.stand_in.stations = [move.cell(station) for station in game.stand_in.stations]
        game.stand_in._grid = tile_map.grid


def _move_constructions(constructions: Constructions, tile_map: TileMap, move: Move) -> None:
    constructions.tile_map = tile_map
    constructions._cells = {
        move.cell(cell): [move.construction(built) for built in here]
        for cell, here in constructions._cells.items()
    }
    constructions._stations = move.cells(constructions._stations)
    constructions._platforms = {
        move.cell(main): move.cells(added) for main, added in constructions._platforms.items()
    }
    constructions._track = move.keys(constructions._track)
    constructions.version += 1
    constructions.railway_version += 1


def _moved_growth(old: Growth, move: Move) -> Growth:
    """A growth for the moved constructions, worked out afresh and given what the old one
    remembered, without the month of trade a new one starts with."""
    growth = Growth(old.constructions, replace(old.rule, trades=False), old.start_year, old.history)
    growth.rule = old.rule
    growth._months = old._months
    growth._served = {points: move.flat_cells(served) for points, served in old._served.items()}
    if old._starving is not None:
        growth._starving = {
            kind: move.flat_cells(starving) for kind, starving in old._starving.items()
        }
    growth._traded = move.flats(old._traded)
    if old.assumed_shares is not None:
        growth.assumed_shares = move.flat_cells(old.assumed_shares, np.nan)
    growth.forests_changed = move.cells(old.forests_changed)
    growth.economy = _moved_economy(old.economy, growth.economy, move)
    growth.last_trade = None if old.last_trade is None else _moved_trade(old.last_trade, move)
    growth.owners._shares = {kind: move.grid(shares) for kind, shares in old.owners._shares.items()}
    growth.owners.company = old.owners.company
    novelties = old.novelties
    if novelties is not None:
        novelties.hosts = [move.cell(cell) for cell in novelties.hosts]
        novelties.works = [move.cell(cell) for cell in getattr(novelties, "works", [])]
    growth.novelties = novelties
    growth.railway = old.railway
    growth._stack()
    growth._reckon_everywhere()
    for place in growth._places:
        place.highest = float(place.chances.max()) if place.chances.size else 0.0
    return growth


def _moved_economy(old: Economy, new: Economy, move: Move) -> Economy:
    new.levels = move.flat_cells(old.levels, old.levels.dtype.type(new.levels[0]))
    new.year = old.year
    new.bonus_per_good = old.bonus_per_good
    stock = Stockpiles(new.cells)
    for key, held in old.stock.items():
        stock[key] = move.flat_cells(held)
    new.stock = stock
    new.months = [move.flat_cells(month, 1.0) for month in old.months]
    if old.taken is not None:
        new.taken = {
            key: {move.flat(cell): amount for cell, amount in taken.items()}
            for key, taken in old.taken.items()
        }
    return new


def _moved_trade(trade: Trade, move: Move) -> Trade:
    """The last month's trade on the moved cells: its arrays follow its cells, or the map."""
    size = move.old.rows * move.old.cols

    def arrays[K](by_key: Mapping[K, Any]) -> dict[K, Any]:
        return {key: _moved_values(value, size, move) for key, value in by_key.items()}

    return replace(
        trade,
        cells=move.flats(trade.cells),
        got=arrays(trade.got),
        needed=arrays(trade.needed),
        gave=arrays(trade.gave),
        could_make=arrays(trade.could_make),
        handed=move.keys(trade.handed),
        gathered=move.keys(trade.gathered),
        goods_got=arrays(trade.goods_got),
        goods_gave=arrays(trade.goods_gave),
        handed_for=move.keys(trade.handed_for),
        bonus=arrays(trade.bonus),
        local=arrays(trade.local),
        worth=arrays(trade.worth),
        owned_brought=move.keys(trade.owned_brought),
    )


def _moved_values(value: Any, size: int, move: Move) -> Any:
    """An array over the map's flat cells moved onto the new map; one over a trade's cells, or a
    tuple of either, as it is."""
    if isinstance(value, tuple):
        return tuple(_moved_values(part, size, move) for part in value)
    if isinstance(value, np.ndarray) and value.shape[-1:] == (size,):
        return move.flat_cells(value)
    return value


def _move_railway(game: Game, move: Move) -> None:
    railway = game.railway
    growth = game.growth
    growth.railway = railway
    railway.station_trade = growth.station_trade
    railway.owners = growth.owners
    railway.goods = move.keys(railway.goods)
    for goods in railway.goods.values():
        goods.origins = {good: move.parcels(parcels) for good, parcels in goods.origins.items()}
    railway.demand = move.keys(railway.demand)
    railway.levels = move.keys(railway.levels)
    if railway.facilities is not None:
        railway.facilities = move.keys(railway.facilities)
    if railway.supplied is not None:
        railway.supplied = move.keys(railway.supplied)
    railway._towns = {}
    names = railway.name_of
    names.tile_map = game.tile_map
    names._names = move.keys(names._names)
    for train in railway.trains:
        _move_train(train, move)
    railway.network = TrackNetwork(game.constructions)
    growth.station_trade.update_reach()


def _move_train(train: Train, move: Move) -> None:
    for stop in train.stops:
        stop.cell = move.cell(stop.cell)
    if train.loaded_at is not None:
        train.loaded_at = move.cell(train.loaded_at)
    if train.platform is not None:
        train.platform = move.cell(train.platform)
    for car in train.cars:
        car.origins = move.parcels(car.origins)
    train.path = [move.point(point) for point in train.path]
    if train.path_ends is not None:
        points, lengths = train.path_ends
        train.path_ends = ([move.point(point) for point in points], lengths)
