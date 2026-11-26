"""What the player sees a train called: its engine's name and the number it was bought as, or the
name the player gave it; and the category of its engine. A train's own `name` stays as it was
bought, as what the simulation knows it by."""

from collections import Counter
from collections.abc import Iterable

from railroads.sim.keyed import Keyed
from railroads.sim.trains import ENGINES, Cargo, Kind, Train


class Category(Keyed):
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


def category(train: Train) -> Category | None:
    return CATEGORIES.get(ENGINES[train.engine].kind)


def train_titles(trains: Iterable[Train]) -> dict[str, str]:
    """What each train is called, by its own name: the name the player gave it, or its engine's
    name and the number it was bought as."""
    return {
        train.name: train.label or f"{ENGINES[train.engine].title} {number_of(train)}"
        for train in trains
    }


def number_of(train: Train) -> str:
    """The number a train was bought as, the last word of its own name."""
    return train.name.rsplit(" ", 1)[-1]


def cars_used_most(train: Train, most: int = 3) -> list[tuple[Cargo, int]]:
    """The cargos a train's route takes on the most cars of, counted over all its stops, most
    first, and how many cars of each."""
    counts = Counter(cargo for stop in train.stops for cargo in stop.cars)
    return counts.most_common(most)
