"""Prints a fingerprint of everything the game works out, to check that a refactor changed none of
it: grows a seeded map of Poland, builds stations and track, runs a train, bulldozes, moves a
station and lets the stand-in player play, all through the map view's own clicks, then renders a
few chunks and plans a few routes. Each part gets its own hash, so a difference points at where
it came from. Run it before and after a change; every line must match.

Run with: uv run python tools/fingerprint.py
"""

import contextlib
import hashlib
import io
from collections.abc import Callable

import numpy as np

from railroads import maps
from railroads.mapgen.seeding import play_begins, seed_buildings
from railroads.sim.clock import GameClock
from railroads.sim.constructions import BuildingKind, Station
from railroads.sim.game import Game, place_names
from railroads.sim.goods import Good
from railroads.sim.grids import Cell
from railroads.sim.ladders import Points
from railroads.sim.novelties import Novelties
from railroads.sim.railway import Stop
from railroads.sim.routes import plan_route
from railroads.sim.stats import History
from railroads.ui.build_menu import BuildMenu, Tool
from railroads.ui.map_view import MapView
from railroads.ui.pixel.sheet import FINE
from railroads.ui.pixel.world import CHUNK

SEED = 7
YEARS = 1000
STAND_IN_YEARS = 10
STEPS_PER_MONTH = 10


def digest(value: object) -> str:
    data = value.tobytes() if isinstance(value, np.ndarray) else repr(value).encode()
    return hashlib.sha256(data).hexdigest()[:16]


def towns(view: MapView, count: int) -> list[Cell]:
    """The most crowded cells, each at least 8 cells from the others."""
    people = view.growth.points(Points.POPULATION)
    picked: list[Cell] = []
    for flat in np.argsort(people, axis=None)[::-1].tolist():
        cell = divmod(flat, people.shape[1])
        if all(abs(cell[0] - r) + abs(cell[1] - c) >= 8 for r, c in picked):
            picked.append((int(cell[0]), int(cell[1])))
        if len(picked) == count:
            return picked
    return picked


def main() -> None:
    said: list[str] = []
    original_say = BuildMenu.say

    def say(menu: BuildMenu, message: str) -> None:
        said.append(message)
        original_say(menu, message)

    BuildMenu.say = say  # type: ignore[method-assign]
    rng = np.random.default_rng(SEED)
    months = YEARS * 12
    history = History()
    seeded = seed_buildings(maps.load("poland", "hex"), rng, months=months, history=history)
    game = Game(
        seeded, start_year=play_begins(months), history=history, names=place_names(seeded.tile_map)
    )
    game.growth.novelties = Novelties.pick(game.tile_map, np.random.default_rng(SEED + 3))
    view = MapView(game)
    clock = GameClock(first_year=view.growth.year)
    play_rng = np.random.default_rng(SEED + 1)

    def play_months(count: int) -> None:
        for _ in range(count):
            for _ in range(STEPS_PER_MONTH):
                view.time_passed(1 / STEPS_PER_MONTH)
            clock.months += 1
            view.month_passed(play_rng, clock.date_of(clock.months))

    def click(tool: Tool, *cells: Cell) -> None:
        view.build.choose(tool)
        for cell in cells:
            view._click(cell)

    view.railway.finances.cash = 5_000_000.0
    a, b, c = towns(view, 3)
    click(Tool.STATION, a, b, c, a)
    click(Tool.TRACK, a, b, c)
    train = view.railway.buy()
    assert train is not None
    train.stops = [Stop(a, [Good.PASSENGERS, Good.MAIL]), Stop(b, [Good.PASSENGERS])]
    play_months(24)
    beside = view.tile_map.grid.neighbours(*a)[0]
    click(Tool.BULLDOZE, beside, (0, 0))
    view.start_moving(c)
    view._click(view.tile_map.grid.neighbours(*c)[1])
    play_months(12)
    with contextlib.redirect_stdout(io.StringIO()) as printed:
        view.simulate(STAND_IN_YEARS, np.random.default_rng(SEED + 2))
    play_months(12)

    growth, railway = view.growth, view.railway
    world = view._pixel_layer().world
    stations = [x for x in view.constructions if isinstance(x, Station)]
    chunks = sorted(
        {
            (int(px) // CHUNK, int(py) // CHUNK)
            for px, py in (
                world.projection.to_pixels(*view.tile_map.grid.center(*s.cell)) for s in stations
            )
        }
    )[:4]
    pairs = [(a, b), (b, c), (a, c)]
    routes = [plan_route(view.constructions, start, end) for start, end in pairs]
    parts: dict[str, Callable[[], object]] = {
        "buildings": lambda: sorted(repr(x) for x in view.constructions),
        "counts": lambda: [
            (k.name, growth.count(k).tobytes()) for k in BuildingKind if k in growth.counts
        ],
        "satisfaction": lambda: growth.satisfaction(),
        "owned": lambda: sorted(
            (k.name, growth.owners.owned(k).sum()) for k in growth.owners.kinds()
        ),
        "cash": lambda: (railway.finances.cash, sorted(railway.finances.last_month.items())),
        "trains": lambda: railway.trains,
        "station goods": lambda: sorted(railway.goods.items()),
        "levels and names": lambda: (sorted(railway.levels.items()), railway.stations()),
        "research": lambda: railway.research,
        "history": lambda: history,
        "messages": lambda: said,
        "stand-in report": lambda: printed.getvalue(),
        "chunks 1x": lambda: [world.render(*chunk) for chunk in chunks],
        f"chunks {FINE}x": lambda: [world.render(*chunk, FINE) for chunk in chunks],
        "routes": lambda: [None if r is None else (r.cells, r.pieces) for r in routes],
    }
    for name, part in parts.items():
        value = part()
        if isinstance(value, list) and value and isinstance(value[0], np.ndarray):
            value = [digest(v) for v in value]
        print(f"{name:18} {digest(value)}")
    print(f"{'year':18} {growth.year}, {len(stations)} stations, {len(railway.trains)} trains")


if __name__ == "__main__":
    main()
