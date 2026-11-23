"""What the player sees a train called: the stations its route runs between and the category of
its engine, numbered from the second train with the same name on, or the name the player gave it.
A train's own `name` stays as it was bought, as what the simulation knows it by."""

from collections import Counter
from collections.abc import Callable, Iterable
from enum import Enum

from railroads.sim.grids import Cell
from railroads.sim.trains import ENGINES, Cargo, Kind, Train


class Category(Enum):
    """What a train is at a glance, by its engine: an express, a local train behind a light engine
    or railcar, or a goods train behind a heavy freight engine. Pioneers and general engines pull
    normal trains, which have none."""

    EXPRESS = "Express"
    LOCAL = "Local"
    CARGO = "Cargo"


CATEGORIES = {
    Kind.EXPRESS: Category.EXPRESS,
    Kind.LIGHT: Category.LOCAL,
    Kind.HEAVY: Category.CARGO,
}
# Between the first and last station of a train's name.
DASH = " – "


def category(train: Train) -> Category | None:
    return CATEGORIES.get(ENGINES[train.engine].kind)


def first_station(train: Train) -> Cell | None:
    return train.stops[0].cell if train.stops else None


def last_station(train: Train) -> Cell | None:
    """The last stop on the route that is not its first station, as a route that ends where it
    began still runs to somewhere; None with fewer than two stations."""
    first = first_station(train)
    return next((s.cell for s in reversed(train.stops) if s.cell != first), None)


def route_name(train: Train, name_of: Callable[[Cell], str]) -> str:
    """The stations a train runs between and its category, before any number: its engine's name
    while it has no stops, its one station's with only that."""
    first, last = first_station(train), last_station(train)
    if first is None:
        stations = ENGINES[train.engine].title
    elif last is None:
        stations = name_of(first)
    else:
        stations = f"{name_of(first)}{DASH}{name_of(last)}"
    kind = category(train)
    return stations if kind is None else f"{stations} {kind.value}"


def train_titles(trains: Iterable[Train], name_of: Callable[[Cell], str]) -> dict[str, str]:
    """What each train is called, by its own name: the name the player gave it, or its route's
    name, with a number from 2 on for each later train whose route's name is the same."""
    seen: Counter[str] = Counter()
    titles = {}
    for train in trains:
        if train.label:
            titles[train.name] = train.label
            continue
        name = route_name(train, name_of)
        seen[name] += 1
        titles[train.name] = name if seen[name] == 1 else f"{name} {seen[name]}"
    return titles


def cars_used_most(train: Train, most: int = 3) -> list[tuple[Cargo, int]]:
    """The cargos a train's route takes on the most cars of, counted over all its stops, most
    first, and how many cars of each."""
    counts = Counter(cargo for stop in train.stops for cargo in stop.cars)
    return counts.most_common(most)
