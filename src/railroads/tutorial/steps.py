"""The tutorial's steps, in order. Each explains something, names what to do and where to click,
and says when it is done: from the game itself, or from what the player has open on the screen
(Seen), which the window tells it every frame. A step with nothing to do is done once read.

The player's progress (Tutorial) is kept with the saved game, so a tutorial saved, loaded or
moved onto a bigger map by land bought goes on where it was."""

from collections.abc import Callable
from dataclasses import dataclass, field

from railroads.sim.economy import BONUS_PER_GOOD, COLLECTION_RADIUS, LOCAL_REACH, SCARCE_WORTH
from railroads.sim.finance import TRACK_PRICE, dollars
from railroads.sim.game import Game
from railroads.sim.goods import FOODS, FREIGHT_SHARE, Good
from railroads.sim.grids import Cell
from railroads.sim.lots import FIRST_FIBONACCI, PRICE_UNIT, Lots, fibonacci
from railroads.sim.ownership import HOLD_MONTHS, LOTS, SHARES
from railroads.sim.trains import ENGINES, FIRST_ENGINE
from railroads.tutorial.place import ASHFORD_STATION, BROOKBY_STATION, SHOWN_FARM

# How far from the cell the tutorial marks a station may stand and still count.
NEAR = 2
PLANET_RANGE = ENGINES[FIRST_ENGINE].tank
FIRST_LOT = fibonacci(FIRST_FIBONACCI) * PRICE_UNIT
# How many shares can be bought at once, in words: "100, 200 or 600".
SHARE_LOTS = ", ".join(f"{lot:,}" for lot in LOTS[:-1]) + f" or {LOTS[-1]:,}"


@dataclass(frozen=True)
class Seen:
    """What the player has open and turned on in the window."""

    # The lens tinting the map, by its name, or "satisfaction" while Shift is held.
    lens: str | None = None
    # The cell whose details are open.
    selected: Cell | None = None
    # Whether the supply and demand boards are open.
    boards_open: bool = False
    # Whether the screen for buying a train is open.
    buying_train: bool = False


def near(game: Game, cell: Cell) -> list[Cell]:
    """The player's stations within NEAR cells of a cell, nearest first."""
    around = game.tile_map.grid.within(cell, NEAR)
    return sorted(
        (station for station in game.railway.stations() if station in around),
        key=lambda station: around[station],
    )


def joined(game: Game) -> bool:
    """Whether track joins a station at Ashford to one at Brookby."""
    connected = game.railway.connected()
    return any(station in connected for station in near(game, ASHFORD_STATION)) and any(
        station in connected for station in near(game, BROOKBY_STATION)
    )


def train_set_off(game: Game) -> bool:
    """Whether a train with two stops or more, carrying passengers and some food out of one of
    them, has been let go."""
    return any(
        len(train.stops) >= 2
        and not train.held
        and any(
            Good.PASSENGERS in stop.cars and any(cargo in FOODS for cargo in stop.cars)
            for stop in train.stops
        )
        for train in game.railway.trains
    )


def cars_both_ways(game: Game) -> bool:
    """Whether a train pulls cars out of two of its stops or more."""
    return any(sum(1 for stop in train.stops if stop.cars) >= 2 for train in game.railway.trains)


def lot_offers(lots: Lots | None) -> list[Cell]:
    """The middle of every lot on offer, on the map."""
    if lots is None:
        return []
    cells = []
    for lot in lots.offers:
        cells.append(lots.middle_cell(lot))
    return cells


def money_for_a_lot(game: Game) -> None:
    """Gives the price of a lot, so the tutorial can show buying one."""
    if game.lots is not None:
        game.finances.earn(game.lots.price(), "tutorial")


@dataclass(frozen=True)
class Step:
    title: str
    # What it explains, a paragraph each.
    text: tuple[str, ...]
    # What to do, and when it is done; nothing to do for a step that only explains.
    task: str = ""
    done: Callable[[Game, Seen], bool] | None = None
    # The cells to click on the map and the buttons to click (railroads.ui.spotlight).
    cells: Callable[[Game], list[Cell]] = lambda game: []
    spots: tuple[str, ...] = ()
    # What the step does to the game as it begins.
    begin: Callable[[Game], None] = lambda game: None


STEPS: tuple[Step, ...] = (
    Step(
        "Welcome to RailHub",
        (
            (
                "It is 1850. Towns grow on their own, but only as well as they are supplied. Your "
                "trains bring them what they lack: they pay you for it, and the towns grow faster."
            ),
            "Drag the map to look around and scroll to zoom in and out.",
        ),
    ),
    Step(
        "1. A station",
        (
            (
                "A station is where goods meet trains. It hands goods out to the buildings a cell "
                "around it (shaded orange) and gathers what the buildings within "
                f"{COLLECTION_RADIUS} cells have left over (the blue line). Each level reaches a "
                "cell further."
            ),
            "A station does nothing on its own: track has to join it to another station.",
        ),
        "Pick Station in the bar at the bottom (or press 1) and click the marked cell in Ashford.",
        lambda game, seen: bool(near(game, ASHFORD_STATION)),
        lambda game: [ASHFORD_STATION],
        ("tool:Station",),
    ),
    Step(
        "1. A second station",
        (
            (
                "Ashford has farms, but Brookby has none near, so its houses go short of food. A "
                "station there can hand them the food your trains bring."
            ),
        ),
        "Build a station on the marked cell in Brookby.",
        lambda game, seen: bool(near(game, BROOKBY_STATION)),
        lambda game: [BROOKBY_STATION],
        ("tool:Station",),
    ),
    Step(
        "1. Track",
        (
            (
                f"Track costs {dollars(TRACK_PRICE)} a cell on the flat, more in hills and much more over water or "
                "through mountains. Track joins a station only at its two ends."
            ),
        ),
        "Pick Track (2), click the Ashford station, then the Brookby station.",
        lambda game, seen: joined(game),
        lambda game: near(game, ASHFORD_STATION)[:1] + near(game, BROOKBY_STATION)[:1],
        ("tool:Track",),
    ),
    Step(
        "2. A train",
        (
            (
                "A train runs from stop to stop. At each stop it unloads, then takes on the cars "
                "set for the way to the next stop and fills them from the station."
            ),
            (
                "Each car carries one good. Every car costs a little each time it leaves a stop, "
                "full or not, so take only cars that will fill."
            ),
        ),
        "Open Trains (T), press Buy a train and buy the Planet. Add Ashford and Brookby as "
        "stops, give Ashford's cars a passenger car and a food car, then press Done.",
        lambda game, seen: train_set_off(game),
        lambda game: [],
        (
            "trains",
            "buy_train",
            f"engine:{FIRST_ENGINE}",
            "buy_engine",
            "add_stop",
            "cargo:0:passengers",
            "cargo:0:wheat",
            "cargo:0:bread",
            "done_buying",
        ),
    ),
    Step(
        "3. Cars and hubs",
        (
            "The way back can carry other goods than the way out: each stop has its own cars.",
            (
                "From the third era, about 1873, general cars take whichever good of their kind "
                "pays best at the next stop, such as any food."
            ),
            (
                "What a train brings beyond what the buildings around can use in a month stays at "
                "the station, and another train can take it on from there: a station can be a hub. "
                "Goods taken back to the station they were loaded at only go back into its stock."
            ),
        ),
        "Give Brookby's cars a passenger car too.",
        lambda game, seen: cars_both_ways(game),
        lambda game: [],
        ("trains", "cargo:1:passengers"),
    ),
    Step(
        "4. Making and using goods",
        (
            (
                "Every month each building first hands what it makes to the buildings around it, "
                f"the nearest first, up to {LOCAL_REACH} cells away, and less the further they are."
            ),
            (
                f"Only what is left goes into a station within {COLLECTION_RADIUS} cells. A station "
                "hands out all it holds, what trains brought and what it gathered, nearest first: "
                "to your own buildings first, and what buildings must have before any bonus."
            ),
            (
                "You earn freight on every unit buildings take out of your stations. Goods handed "
                "out on their own earn nothing."
            ),
        ),
        "Right-click the marked farm to see what it made last month.",
        lambda game, seen: seen.selected == SHOWN_FARM,
        lambda game: [SHOWN_FARM],
    ),
    Step(
        "5. Satisfaction",
        (
            (
                "A building's satisfaction is the share of what it needs that it got. Bonus goods, "
                f"such as beer or newspapers, add {BONUS_PER_GOOD:.0%} each on top, as far as its "
                "needs are met."
            ),
            (
                "The more satisfied buildings are, the faster more of their kind come up around "
                "them, so the towns you serve grow and want more."
            ),
            (
                f"Freight pays {FREIGHT_SHARE:.0%} of a good's market price, up to {SCARCE_WORTH:g} "
                "times that to a building that got nothing last quarter. Bring goods where they "
                "are missing."
            ),
        ),
        "Turn on the Satisfaction lens above the minimap, or hold Shift.",
        lambda game, seen: seen.lens == "satisfaction",
        lambda game: [],
        ("lens:satisfaction",),
    ),
    Step(
        "6. Passengers and towns",
        (
            (
                "Every place that sends passengers belongs to a town, each in its own colour. Towns "
                "that grow into each other merge. A station belongs to the town around it."
            ),
            (
                "Passengers and mail only ride between stations of different towns. Those no train "
                "takes are gone the next month."
            ),
        ),
        "Turn on the Towns lens.",
        lambda game, seen: seen.lens == "towns",
        lambda game: [],
        ("lens:towns",),
    ),
    Step(
        "7. Shares",
        (
            (
                "You can buy shares of any building that makes something: farms, mines, mills and "
                f"works. A building has {SHARES:,} shares, bought {SHARE_LOTS} at a time, priced by how "
                f"satisfied it was last quarter. They can be sold {HOLD_MONTHS // 12} years after "
                "they are bought."
            ),
            (
                "Your part of what it makes goes into your stations around it first, and they buy "
                "it from you at the market price. Freight comes on top when buildings take it out "
                "of a station. With no station near it pays nothing."
            ),
        ),
        "Right-click the marked farm, press Shares, then Buy.",
        lambda game, seen: game.growth.owners.count() > 0,
        lambda game: [SHOWN_FARM],
        ("buy_shares",),
    ),
    Step(
        "8. Supply and demand",
        (
            (
                "Every January you raise two supply tiles and one demand tile: what the world "
                "makes grows faster than what it wants, so prices fall as the years go by."
            ),
            (
                "Supply levels bring new engines and eras, with new goods and bigger wagons; "
                "demand levels bring new buyers, such as mines taking planks."
            ),
        ),
        "Open the supply and demand boards (Y, or the scales at the top).",
        lambda game, seen: seen.boards_open,
        lambda game: [],
        ("boards",),
    ),
    Step(
        "9. Fuel and engines",
        (
            (
                "Steam engines burn coal (the American burns wood), diesels diesel fuel and "
                "electrics electricity. A train buys fuel for the way to its next stop as it sets "
                "off: cheaper out of your station's stock, dearer at retail."
            ),
            (
                "A coal or diesel engine carries only so much: a train won't set off for a stop "
                f"further than its range. The Planet goes {PLANET_RANGE:.0f} cells, so long lines need "
                "stops to take on fuel. Electrics run as far as the wires go."
            ),
            (
                "Light engines are cheap for short lines, heavy ones pull more freight, and express "
                "ones carry passengers fast."
            ),
        ),
        "Press Buy a train and compare the engines' fuel and range. You don't have to buy one.",
        lambda game, seen: seen.buying_train,
        lambda game: [],
        ("trains", "buy_train"),
    ),
    Step(
        "10. More land",
        (
            (
                "The land around the map can be bought a lot at a time. Each lot next to yours shows "
                f"its land behind a padlock with its price: the first {dollars(FIRST_LOT)}, each next "
                "one dearer."
            ),
            (
                "A lot bought grows on its own up to today, so it comes as built up as the rest. For "
                "the tutorial you have been given the money for one."
            ),
        ),
        "Zoom out, click a padlocked lot past the map's edge and confirm.",
        lambda game, seen: game.lots is not None and game.lots.bought > 0,
        lambda game: lot_offers(game.lots),
        begin=money_for_a_lot,
    ),
    Step(
        "That's it",
        (
            (
                "Build more stations and track, and give your trains goods that are missing where "
                "they go. Help (H) lists every building, good and engine."
            ),
        ),
    ),
)


@dataclass
class Tutorial:
    """How far the player has got: the step shown, the steps done and the last step begun."""

    step: int = 0
    finished: set[int] = field(default_factory=set)
    begun: int = -1
    # Whether the player closed the tutorial.
    closed: bool = False

    @property
    def current(self) -> Step | None:
        if self.closed or self.step >= len(STEPS):
            return None
        return STEPS[self.step]

    @property
    def done(self) -> bool:
        """Whether the step shown is done: read, or its task done once."""
        step = self.current
        return step is not None and (step.done is None or self.step in self.finished)

    def update(self, game: Game, seen: Seen) -> None:
        """Begins the step shown, once, and notes whether its task is done."""
        step = self.current
        if step is None:
            return
        if self.begun != self.step:
            self.begun = self.step
            step.begin(game)
        if step.done is not None and step.done(game, seen):
            self.finished.add(self.step)

    def next(self) -> None:
        """Goes on to the next step, if the one shown is done."""
        if self.done:
            self.step += 1

    def skip(self) -> None:
        """Goes on to the next step, done or not."""
        if self.current is not None:
            self.step += 1
