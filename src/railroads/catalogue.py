"""What the help window and the README say of every building, good and engine, worked out
without drawing anything: who makes and wants what, when it opens, and an engine's figures."""

from dataclasses import dataclass
from functools import cache

from railroads.sim.constructions import BUILDING_RULES, BuildingKind
from railroads.sim.fuel_market import fuel_of
from railroads.sim.goods import (
    FITS,
    MAKES,
    NEEDS,
    Good,
    Need,
    freight_of,
    price_of,
)
from railroads.sim.ladders import PASSENGERS_PER_POINT, POINTS, ladder_of
from railroads.sim.points import Points
from railroads.sim.tiles import Tile, good_era, kind_era, opened_from
from railroads.sim.trains import (
    ENGINES,
    EXPRESS_FARE,
    MAGLEV_FARE,
    OPENED_BY,
    SLOWDOWN,
    Engine,
    successor,
)
from railroads.ui.engine_origins import origins
from railroads.words import dollars, tonnes_text


def fuel_use(engine: Engine, per_cell: float | None = None) -> str:
    """How much fuel an engine burns a cell, and what that costs now if given."""
    text = f"{engine.burn:.2g} {fuel_of(engine).value} a cell half loaded"
    return text if per_cell is None else f"{text}, about {dollars(per_cell)} a cell now"


def engine_range(engine: Engine) -> str:
    """How far an engine runs on the fuel it takes on setting off."""
    if engine.tank:
        return f"{engine.tank:.0f} cells between stops"
    return "As far as the wires go" if engine.electric else "As far as the guideway goes"


def cars_pulled(engine: Engine) -> str:
    """The cars an engine pulls at most: "5 cars", or an express's coaches and dining car."""
    return f"{engine.cars} coaches and a dining car" if engine.express else f"{engine.cars} cars"


def most_pulled(engine: Engine) -> str:
    """The most an engine pulls, and how much slower it runs pulling that."""
    return f"{tonnes_text(engine.tonnes)} t, {SLOWDOWN[engine.kind]:.0%} slower at that"


def fares_note(engine: Engine) -> str | None:
    """What an express is paid more for its passengers; None for other engines."""
    if not engine.express:
        return None
    fare = MAGLEV_FARE if engine.maglev else EXPRESS_FARE
    return f"Passengers only, paying {fare - 1:.0%} more"


def track_needed(engine: Engine) -> list[str]:
    """The only track an engine runs on, if it needs track of its own."""
    return [
        track
        for track, needed in (("Electrified", engine.electric), ("Maglev", engine.maglev))
        if needed
    ]


def engine_figures(
    engine: Engine, per_cell: float | None = None, opened: tuple[Tile, int] | None = None
) -> list[tuple[str, str]]:
    """What an engine can do, as labels and values for its table on the screen, with what its
    fuel for a cell costs now and the supply tile level that put it on sale if given."""
    figures = []
    if opened is not None:
        tile, level = opened
        figures.append(("On sale from", f"{tile.value.capitalize()} supply {level}"))
    figures += [
        ("First built", str(engine.year)),
        ("Power", engine.power.value.capitalize()),
        ("Top speed", f"{engine.kmh:.0f} km/h"),
        ("Pulls up to", cars_pulled(engine)),
        ("Pulls at most", most_pulled(engine)),
        ("Up to speed in", f"{engine.pickup:g} s"),
        ("Price", dollars(engine.price)),
        ("Burns", fuel_use(engine, per_cell)),
        ("Range", engine_range(engine)),
    ]
    figures.append(("Cars", f"Up to generation {engine.top_generation}"))
    fares = fares_note(engine)
    if fares is not None:
        figures.append(("Fares", fares))
    figures += [("Runs on", f"{track} track only") for track in track_needed(engine)]
    return figures


@dataclass(frozen=True)
class BuildingHelp:
    kind: BuildingKind
    group: Points
    makes: tuple[tuple[Good, float], ...]
    # Each need, how much of it a month, and from when it is felt ("" for always): a year, or the
    # demand tile level that opens it.
    wants: tuple[tuple[Need, float, str], ...]
    travellers: int
    space: int
    # From when it comes up: a year, its era, or "" for always.
    since: str


@dataclass(frozen=True)
class GoodHelp:
    good: Good
    price: float
    freight: float
    # From when it is made: its era, or "" for always.
    since: str
    made_by: tuple[str, ...]
    wanted_by: tuple[str, ...]


@dataclass(frozen=True)
class EngineHelp:
    name: str
    engine: Engine
    # The year its sale ends and the engine that ends it, if one does.
    sold_until: tuple[int, str] | None
    # The supply tile and level that open it for sale, in words.
    opened_by: str
    # The real engines it is drawn after, as names and Wikipedia links.
    origins: tuple[tuple[str, str], ...]


def _opening(kind: BuildingKind) -> str:
    era = kind_era(kind)
    if era > 1:
        return f"era {era}"
    ladder = ladder_of(kind)
    year = 0 if ladder is None else ladder.from_year
    return str(year) if year else ""


@cache
def building_help() -> tuple[BuildingHelp, ...]:
    """Every kind of building, grouped by the points it gives; worked out once, as the open
    window draws them all every frame."""
    rows = [
        BuildingHelp(
            kind,
            POINTS[kind][0],
            tuple(MAKES.get(kind, {}).items()),
            tuple(
                (need, amount, opened_from(kind, need))
                for need, amount in NEEDS.get(kind, {}).items()
            ),
            PASSENGERS_PER_POINT[POINTS[kind][0]] * POINTS[kind][1],
            BUILDING_RULES[kind].space,
            _opening(kind),
        )
        for kind in BuildingKind
    ]
    order = list(Points)
    return tuple(sorted(rows, key=lambda row: order.index(row.group)))


@cache
def good_help() -> tuple[GoodHelp, ...]:
    """Every good, with its market price and what carrying it earns; worked out once."""
    rows = []
    for good in Good:
        needs = {need for need, goods in FITS.items() if good in goods}
        rows.append(
            GoodHelp(
                good,
                price_of(good),
                freight_of(good),
                f"era {good_era(good)}" if good_era(good) > 1 else "",
                tuple(kind.value for kind, made in MAKES.items() if good in made),
                tuple(kind.value for kind, wanted in NEEDS.items() if needs & set(wanted)),
            )
        )
    return tuple(rows)


def opened_by(name: str) -> str:
    """The supply tile and level that open an engine for sale, in words."""
    tile, level = OPENED_BY[name]
    return f"{tile.value} supply level {level}"


@cache
def engine_help() -> tuple[EngineHelp, ...]:
    """Every engine by the year it is first sold; worked out once."""
    rows = []
    for name, engine in sorted(ENGINES.items(), key=lambda item: item[1].year):
        newer = successor(name)
        rows.append(
            EngineHelp(
                name,
                engine,
                (ENGINES[newer].year, ENGINES[newer].title) if newer else None,
                opened_by(name),
                origins(name),
            )
        )
    return tuple(rows)
