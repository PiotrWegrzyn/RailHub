"""What the trains window says of trains, their cars and their rides, worked out without drawing."""

from railroads.catalogue import (
    cars_pulled,
    engine_range,
    fares_note,
    fuel_use,
    most_pulled,
    track_needed,
)
from railroads.sim.amounts import EPSILON
from railroads.sim.goods import Good
from railroads.sim.railway import NO_WAY_ON, Railway, Ride
from railroads.sim.trade import SCARCE_WORTH
from railroads.sim.trains import (
    ENGINES,
    Engine,
    Kind,
    Train,
)
from railroads.words import dollars, tonnes_text

# How a cargo's worth at a stop is worked out, as freight is paid there.
CARGO_WORTH = (
    "Goods earn their freight for what the buildings around would take at once, up to "
    f"{SCARCE_WORTH:g} times as much where they went short; the rest waits at the station. "
    "Passengers and mail pay by the demand left there."
)


def picker_good(train: Train) -> Good:
    """The good whose cars the generation picker lists: coaches behind an express."""
    return Good.PASSENGERS if ENGINES[train.engine].express else Good.COAL


FIRST_TIP = (
    "Puts it first in every queue at the signals until it reaches its next stop: it gets the\n"
    "way on before any train waiting there, and takes back the way set for a train that has\n"
    "yet to come to it. It still never runs into a train standing on the track."
)


def adding_tip(full: bool, tonnes: float, spare: float) -> str:
    """The last line of a good's tooltip on a stop card: whether a car of it can be added."""
    if full:
        return "\nThe train is full: remove a car first"
    if tonnes > spare + EPSILON:
        return (
            f"\nToo heavy: a full car weighs {tonnes_text(tonnes)} t and the engine pulls only "
            f"{tonnes_text(max(0.0, spare))} t more"
        )
    return f"\nClick to add a car, {tonnes_text(tonnes)} t full"


def train_warning(railway: Railway, train: Train) -> str:
    """What the train window warns of first: that no track the train can run on reaches its next
    stop, or that a way between two of its stops is longer than its fuel lasts, so it waits where
    it is; empty when it can go all the way round."""
    destination = railway.destination(train)
    if train.status in NO_WAY_ON and destination is not None:
        return (
            f"{train.status}, {railway.name_of(destination)}: it waits here until track joins them"
        )
    legs = railway.long_legs(train)
    if not legs:
        return ""
    index, leg = legs[0]
    return f"{leg_warning(railway, train, index, leg)}. Add a stop between them to refuel at."


def leg_warning(railway: Railway, train: Train, index: int, leg: float) -> str:
    """That the way from a train's stop to the next is longer than its engine carries fuel for."""
    there = train.stops[(index + 1) % len(train.stops)].cell
    tank = ENGINES[train.engine].tank
    return (
        f"From {railway.name_of(train.stops[index].cell)} to {railway.name_of(there)} is "
        f"{leg:.0f} cells, but it carries fuel for only {tank:.0f}"
    )


def wagons_note(railway: Railway, name: str) -> list[str]:
    """A line saying the engine's cars stay the size of today's wagons until newer ones come,
    when it pulls newer ones than there are yet; none otherwise."""
    era = railway.wagon_era()
    top = ENGINES[name].top_generation
    if top <= era:
        return []
    note = f"Its cars stay generation {era}, the newest there are, until a new era brings "
    return [f"{note}generation {era + 1}"]


def engines_by_kind(names: list[str]) -> list[tuple[Kind, list[str]]]:
    """Engines grouped by what they are for, in the order of Kind, each group newest first."""
    groups = [
        (
            kind,
            sorted((n for n in names if ENGINES[n].kind == kind), key=lambda n: -ENGINES[n].year),
        )
        for kind in Kind
    ]
    return [(kind, group) for kind, group in groups if group]


def engine_summary(engine: Engine, per_cell: float | None = None) -> str:
    """An engine's speed, cars, fuel and what it runs on, for its tooltip, with what its fuel for
    a cell costs now if given."""
    lines = [
        (
            f"{engine.year}, {engine.power.value}: {engine.kmh:.0f} km/h, "
            f"pulls up to {cars_pulled(engine)}"
        ),
        f"Pulls at most {most_pulled(engine)}",
        f"Up to speed in {engine.pickup:g} s",
        f"Burns {fuel_use(engine, per_cell)}",
        f"Range: {engine_range(engine)}",
    ]
    lines.append(f"Pulls cars up to generation {engine.top_generation}")
    fares = fares_note(engine)
    if fares is not None:
        lines.append(fares)
    lines += [f"Runs on {track.lower()} track only" for track in track_needed(engine)]
    return "\n".join(lines)


def row_cars(cars: int, width: float) -> tuple[float, int]:
    """How wide a train's cars are drawn in its row of the list, and how many go on a line of
    that width: ROW_CAR wide over up to ROW_CAR_LINES lines, or narrowed to fit in them."""
    per_line = max(1, int((width + ROW_CAR_GAP) // (ROW_CAR + ROW_CAR_GAP)))
    if cars <= per_line * ROW_CAR_LINES:
        return ROW_CAR, per_line
    per_line = -(-cars // ROW_CAR_LINES)
    return max(1.0, (width + ROW_CAR_GAP) / per_line - ROW_CAR_GAP), per_line


def carried_in(train: Train, index: int) -> list[Good]:
    """The goods a train's cars bring into its stop at `index`: what it takes on at the stop
    before, past any it only refuels at, so a train can carry them on past a station that has no
    use for them."""
    if not train.stops:
        return []
    before = train.stops[train.before(index)]
    return list(dict.fromkeys(cargo for cargo in before.cars if isinstance(cargo, Good)))


def signed(amount: float) -> str:
    """Money earned with a plus, money paid with a minus."""
    return f"+{dollars(amount)}" if amount >= 0.5 else dollars(amount)


def ride_rows(ride: Ride) -> list[tuple[str, float, str]]:
    """A ride's bill, a line at a time: what it is for, how much, earned or paid, and what it
    means."""
    return [
        (
            "Cargo",
            ride.cargo,
            f"What its cargo would earn if it were unloaded there now.\n{CARGO_WORTH}",
        ),
        (
            f"Fuel for {ride.cells:.0f} cells",
            -ride.fuel,
            "Bought as it sets off, at retail; fuel the station has in stock comes cheaper",
        ),
        (
            f"{ride.loaded} loaded cars",
            -ride.loaded_cars,
            "Every car costs its trip each time the train leaves a stop with it",
        ),
        (
            f"{ride.empty} empty cars",
            -ride.empty_cars,
            "Empty cars cost their trip too: take only the cars the stop can fill",
        ),
        ("Ride earns", ride.net, ""),
    ]


def ride_lines(ride: Ride) -> str:
    """A ride's bill as lines of text."""
    return "\n".join(f"{label}: {signed(amount)}" for label, amount, _ in ride_rows(ride))


def ride_tiles(ride: Ride, onward: str) -> list[tuple[str, float, str]]:
    """The three figures over a train's route: what its cargo is worth at the next stop, what
    the ride there costs and what it earns after that."""
    costs = ride.fuel + ride.loaded_cars + ride.empty_cars
    return [
        (f"Cargo worth at {onward}", ride.cargo, ride_rows(ride)[0][2]),
        (
            f"Ride costs, {ride.cells:.0f} cells",
            -costs,
            "Fuel for the way and the trip of every car, loaded or empty",
        ),
        ("This ride earns", ride.net, "What its cargo earns there, less what the ride costs"),
    ]


# The size of a car in a train's row, and the gap between cars and between their lines.
ROW_CAR = 16.0


ROW_CAR_GAP = 2.0


# The most lines a train's cars take in its row; a train too long for them has its cars narrowed.
ROW_CAR_LINES = 3
