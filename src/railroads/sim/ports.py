"""Ports (railroads.sim.growth keeps them): the goods each one draws, gives and takes, the good
the owner of a mega port picks, and what the player ships between the ports they control.

Every good a port may draw has a tier (TIER_GOODS), and every tier a value (TIER_VALUES): a port
gives or takes as many units of a good a month as the tier's value buys at the good's base price
(monthly), more as its supply grows, like what other buildings make (railroads.sim.events).

A port draws one good it gives and one it takes from the common and rare goods in the game that
year. It gives what is made within DRAW_RADIUS cells of it only a little or not at all, less than
the tier's value a month, and takes what is made there more. Two ports in a neighbourhood merge
into a big port that keeps all their goods and draws one more epic or legendary good to give;
two big ports into a mega port, which draws one more, and whose owner may pick one good more of
any tier for it to give.

A port the player controls ships goods to another of theirs (SHIPS of them by its size): it
buys each from trains into its store, STORE_MONTHS of what a ship carries, and every month a
ship carries SHIP_WAGONS wagons of the era's worth to the other port, free, which gives it to
trains. Ports that merge keep what they ship and hold, ships bound for them sail to the merged
port, and the player keeps their biggest stake in any of them (railroads.sim.growth).
"""

from __future__ import annotations

import weakref
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from railroads.sim.amounts import EPSILON
from railroads.sim.constructions import PORT_KINDS, BuildingKind
from railroads.sim.events import make_factor, rise_of
from railroads.sim.goods import PRICES, RECIPES, Good, Need, good_exists
from railroads.sim.grids import Cell, FloatArray, Grid, HexGrid
from railroads.sim.keyed import Keyed
from railroads.sim.move import Move
from railroads.sim.tiles import of_year
from railroads.sim.trade import CellGoods, Trade
from railroads.sim.trains import car_size
from railroads.sim.yearly import makes_of

if TYPE_CHECKING:
    from railroads.sim.growth import Growth


class Tier(Keyed):
    COMMON = "common"
    RARE = "rare"
    EPIC = "epic"
    LEGENDARY = "legendary"


# What a month of a good of each tier is worth to a port, in dollars at base prices.
TIER_VALUES: Mapping[Tier, float] = MappingProxyType(
    {Tier.COMMON: 600.0, Tier.RARE: 2400.0, Tier.EPIC: 4800.0, Tier.LEGENDARY: 10_000.0}
)

# The colour of each tier's dot, as RGB.
TIER_COLOURS: Mapping[Tier, tuple[int, int, int]] = MappingProxyType(
    {
        Tier.COMMON: (86, 190, 92),
        Tier.RARE: (78, 140, 232),
        Tier.EPIC: (168, 92, 222),
        Tier.LEGENDARY: (232, 186, 52),
    }
)

G = Good
TIER_GOODS: Mapping[Tier, tuple[Good, ...]] = MappingProxyType(
    {
        Tier.COMMON: (
            G.WHEAT,
            G.LOGS,
            G.POTATOES,
            G.RICE,
            G.CHICKENS,
            G.MILK,
            G.APPLES,
            G.BANANAS,
            G.DATES,
            G.ORANGES,
            G.SUGAR,
            G.FISH,
            G.SAND,
            G.SOYBEANS,
            G.TOFU,
            G.BRICKS,
            G.MEAT,
            G.DAIRY,
            G.PLANKS,
            G.WOOD_CHIPS,
            G.COAL,
            G.GAS,
            G.CLAY,
            G.LIMESTONE,
            G.MISO,
            G.CEMENT,
            G.STEEL,
        ),
        Tier.RARE: (
            G.REFINED_SUGAR,
            G.SAUSAGES,
            G.FERTILISER,
            G.BARRELS,
            G.SALT,
            G.IRON,
            G.CANS,
            G.WOOL,
            G.GRAPES,
            G.FUEL,
            G.SOY_SAUCE,
            G.HOPS,
            G.GLASS,
            G.PAPER,
            G.JUICE,
            G.OIL,
            G.COTTON,
            G.FLAX,
            G.CHEMICALS,
            G.LEATHER,
            G.STOVE_TILES,
            G.COPPER,
            G.WAX,
            G.CASTINGS,
            G.INCENSE,
            G.STEEL,
        ),
        Tier.EPIC: (
            G.RUBBER,
            G.TEA,
            G.PLASTICS,
            G.COFFEE,
            G.FURNITURE,
            G.BEER,
            G.PORCELAIN,
            G.CLOTHES,
            G.SAKE,
            G.SHOES,
            G.ELECTRONICS,
            G.GOODS,
            G.VODKA,
            G.TOOLS,
            G.WINE,
            G.GUNS,
            G.MACHINERY,
            G.APPLIANCES,
            G.LUXURIES,
            G.STEEL,
        ),
        Tier.LEGENDARY: (G.GOLD, G.DIAMONDS, G.CARS, G.TANKS, G.PLANES),
    }
)

# The tiers an ordinary port draws from, and those a big and a mega port draw their one more
# good from.
LOW_TIERS = (Tier.COMMON, Tier.RARE)
HIGH_TIERS = (Tier.EPIC, Tier.LEGENDARY)

# How many cells around a port count when it decides what is made near it.
DRAW_RADIUS = 300

# How many goods a port of each size may ship to another port.
SHIPS: Mapping[BuildingKind, int] = MappingProxyType(
    {BuildingKind.PORT: 1, BuildingKind.BIG_PORT: 2, BuildingKind.MEGA_PORT: 4}
)
# How many wagons of the era's a ship carries a month, and how many months of that a port keeps
# in store at each end.
SHIP_WAGONS = 2
STORE_MONTHS = 2


def monthly(good: Good, tier: Tier) -> float:
    """How many units of a good of a tier a port gives or takes in a month at base: the tier's
    value at the good's base price."""
    return TIER_VALUES[tier] / PRICES[good]


def tiers_of(good: Good) -> tuple[Tier, ...]:
    return tuple(tier for tier, goods in TIER_GOODS.items() if good in goods)


def need_of(good: Good) -> Need:
    """The need a port takes a good by."""
    return Need(good.value)


def ship_load(good: Good, wagons: int) -> float:
    """What a ship carries of a good in a month, with wagons of a generation."""
    return SHIP_WAGONS * car_size(good, wagons)


@dataclass(frozen=True, slots=True)
class Slot:
    """A good a port gives or takes, of a tier, drawn or picked by its owner."""

    good: Good
    tier: Tier
    gives: bool
    picked: bool = False


@dataclass(slots=True)
class Port:
    """What one port gives and takes, the goods it ships and to where, and what it holds of them:
    in `store` to ship, in `landed` from ships to give out."""

    slots: list[Slot] = field(default_factory=list)
    ships: dict[Good, Cell] = field(default_factory=dict)
    store: dict[Good, float] = field(default_factory=dict)
    landed: dict[Good, float] = field(default_factory=dict)

    @property
    def gives(self) -> list[Slot]:
        return [slot for slot in self.slots if slot.gives]

    @property
    def takes(self) -> list[Slot]:
        return [slot for slot in self.slots if not slot.gives]

    @property
    def goods(self) -> set[Good]:
        return {slot.good for slot in self.slots}

    @property
    def pick(self) -> Slot | None:
        return next((slot for slot in self.slots if slot.picked), None)

    def joined(self, other: Port) -> Port:
        """This port's goods and the other's, as one port that merged them: each good once,
        neither port's pick; what both ship, this port's first where both ship a good, and all
        that both hold in store and landed from ships."""
        slots = [slot for slot in self.slots if not slot.picked]
        held = {slot.good for slot in slots}
        slots += [slot for slot in other.slots if not slot.picked and slot.good not in held]
        return Port(
            slots,
            {**other.ships, **self.ships},
            _summed(self.store, other.store),
            _summed(self.landed, other.landed),
        )


def steps_from(
    grid: Grid, cell: Cell, rows: npt.NDArray[np.intp], cols: npt.NDArray[np.intp]
) -> npt.NDArray[np.intp]:
    """How many steps each of these cells is from a cell."""
    if isinstance(grid, HexGrid):
        q = cols - (rows - (rows & 1)) // 2
        q0 = cell[1] - (cell[0] - (cell[0] & 1)) // 2
        dq, dr = q - q0, rows - cell[0]
        return np.maximum(np.maximum(np.abs(dq), np.abs(dr)), np.abs(dq + dr))
    return np.maximum(np.abs(rows - cell[0]), np.abs(cols - cell[1]))


class Ports:
    """Every port on the map by its cell, and what the economy trades with them."""

    def __init__(self, growth: Growth) -> None:
        self._growth = growth
        self.at: dict[Cell, Port] = {}

    @property
    def year(self) -> int:
        return self._growth.year

    def kind_at(self, cell: Cell) -> BuildingKind | None:
        counts = self._growth.counts
        return next((kind for kind in PORT_KINDS if counts[kind][cell] > 0), None)

    def cells(self) -> list[Cell]:
        counts = self._growth.counts
        built = sum((counts[kind] for kind in PORT_KINDS), start=np.zeros(self._growth.grid.shape))
        return [(int(row), int(col)) for row, col in np.argwhere(built > 0)]

    def port(self, cell: Cell) -> Port | None:
        """The port in a cell, its goods drawn now if it went up since the last month's trade
        drew them (sync); None where no port stands."""
        if cell not in self.at and self.kind_at(cell) is not None:
            self.sync()
        return self.at.get(cell)

    def sync(self) -> None:
        """Draws the goods of every port that has none yet and forgets those pulled down."""
        standing = set(self.cells())
        for cell in [cell for cell in self.at if cell not in standing]:
            del self.at[cell]
        new = [cell for cell in standing if cell not in self.at]
        if not new:
            return
        made = _Makers(self._growth)
        for cell in sorted(new):
            kind = self.kind_at(cell)
            assert kind is not None
            self.at[cell] = self.drawn(kind, cell, made)

    def drawn(
        self, kind: BuildingKind, cell: Cell, made: _Makers | None = None, salt: int = 0
    ) -> Port:
        """A port of a size with all its goods drawn afresh, as if it had merged from ports
        drawn afresh."""
        made = made or _Makers(self._growth)
        near = made.near(cell)
        rng = self._rng(cell, kind, salt)
        if kind == BuildingKind.PORT:
            port = Port()
            self._draw(port, near, rng, LOW_TIERS, gives=True)
            self._draw(port, near, rng, LOW_TIERS, gives=False)
            return port
        smaller = PORT_KINDS[PORT_KINDS.index(kind) - 1]
        first = self.drawn(smaller, cell, made, 2 * salt + 1)
        port = first.joined(self.drawn(smaller, cell, made, 2 * salt + 2))
        self._draw(port, near, rng, HIGH_TIERS, gives=True)
        return port

    def merged(self, cell: Cell, kind: BuildingKind, parts: Mapping[Cell, Port]) -> None:
        """Puts up a port of a size merged from the ports in these cells: all their goods, what
        they ship and hold, and one more epic or legendary good it gives. Ships sailing to any of
        them sail to it instead; none sails from it to itself."""
        port = Port()
        for part in parts.values():
            port = port.joined(part)
        gone = {cell, *parts}
        port.ships = {good: to for good, to in port.ships.items() if to not in gone}
        for other in self.at.values():
            other.ships = {good: cell if to in gone else to for good, to in other.ships.items()}
        near = _Makers(self._growth).near(cell)
        self._draw(port, near, self._rng(cell, kind), HIGH_TIERS, gives=True)
        self.at[cell] = port

    def take(self, cell: Cell) -> Port | None:
        return self.at.pop(cell, None)

    def _rng(self, cell: Cell, kind: BuildingKind, salt: int = 0) -> np.random.Generator:
        """The draws of a port of a size put up in a cell this year, the same each time."""
        return np.random.default_rng((cell[0], cell[1], self.year, PORT_KINDS.index(kind), salt))

    def _draw(
        self,
        port: Port,
        near: Mapping[Good, float],
        rng: np.random.Generator,
        tiers: tuple[Tier, ...],
        gives: bool,
    ) -> None:
        """Draws one good of these tiers into a port: one to give that is made near it less
        than the tier's value a month, or one to take made there more. When none is, one to
        take that is made there at all, and else any good of the tiers."""
        held = port.goods
        pairs = [
            (good, tier)
            for tier in tiers
            for good in TIER_GOODS[tier]
            if good not in held and good_exists(good, self.year)
        ]
        fitting = [
            (good, tier)
            for good, tier in pairs
            if (near.get(good, 0.0) * PRICES[good] < TIER_VALUES[tier]) == gives
        ]
        if not fitting and not gives:
            fitting = [(good, tier) for good, tier in pairs if near.get(good, 0.0) > 0]
        if not fitting:
            fitting = pairs
        if not fitting:
            return
        good, tier = fitting[int(rng.integers(len(fitting)))]
        port.slots.append(Slot(good, tier, gives))

    def pick(self, cell: Cell, good: Good | None, tier: Tier | None = None) -> None:
        """The owner of a mega port picks a good of a tier in the game this year for it to give,
        or none."""
        port = self.port(cell)
        if port is None:
            return
        port.slots = [slot for slot in port.slots if not slot.picked]
        if good is None or tier is None or (good, tier) not in self.pick_choices():
            return
        if good not in port.goods:
            port.slots.append(Slot(good, tier, gives=True, picked=True))

    def pick_choices(self) -> list[tuple[Good, Tier]]:
        """Every good of every tier in the game this year, which a mega port's owner may pick."""
        return [
            (good, tier)
            for tier, goods in TIER_GOODS.items()
            for good in goods
            if good_exists(good, self.year)
        ]

    def ship_choices(self) -> list[Good]:
        """Every good of any tier in the game this year, which a port may ship."""
        return list(dict.fromkeys(good for good, _ in self.pick_choices()))

    def can_pick(self, cell: Cell) -> bool:
        return self.kind_at(cell) == BuildingKind.MEGA_PORT and self._growth.owners.controls(
            cell, BuildingKind.MEGA_PORT
        )

    def controlled(self) -> list[Cell]:
        """The ports the player controls."""
        owners = self._growth.owners
        return [
            cell
            for cell in self.cells()
            if (kind := self.kind_at(cell)) is not None and owners.controls(cell, kind)
        ]

    def ship(self, cell: Cell, good: Good, to: Cell | None) -> str | None:
        """Ships a good from a port the player controls to another of theirs, or stops shipping
        it; why not, if it can't."""
        port = self.port(cell)
        kind = self.kind_at(cell)
        if port is None or kind is None or cell not in self.controlled():
            return "You don't control this port"
        if to is None:
            port.ships.pop(good, None)
            return None
        if to == cell or to not in self.controlled():
            return "Ships sail only to another port you control"
        if good not in port.ships and len(port.ships) >= SHIPS[kind]:
            return f"A {kind.value} ships at most {SHIPS[kind]} goods"
        port.ships[good] = to
        return None

    def amount(self, slot: Slot) -> float:
        """What a port gives or takes of a good in a month this year."""
        base = monthly(slot.good, slot.tier)
        if slot.gives:
            return base * make_factor(slot.good, self.year)
        return base * rise_of(BuildingKind.PORT, need_of(slot.good), self.year)

    def load(self, good: Good) -> float:
        """What a ship carries of a good in a month this year."""
        return ship_load(good, of_year(self.year).wagons)

    def cell_goods(self) -> CellGoods:
        """What every port gives, takes and has room to store for shipping this month, cell by
        cell, for the month's trade."""
        cols = self._growth.grid.cols
        cells = sorted(self.at)
        makes: dict[Good, FloatArray] = {}
        takes: dict[Need, FloatArray] = {}
        stores: dict[Need, FloatArray] = {}

        def add[K](into: dict[K, FloatArray], amounts: Mapping[K, float], at: int) -> None:
            for key, amount in amounts.items():
                if amount > EPSILON:
                    into.setdefault(key, np.zeros(len(cells)))[at] += amount

        for at, cell in enumerate(cells):
            port = self.at[cell]
            add(makes, self.gives_at(cell), at)
            add(takes, self.takes_at(cell), at)
            room = {
                need_of(good): STORE_MONTHS * self.load(good) - port.store.get(good, 0.0)
                for good in port.ships
            }
            add(stores, room, at)
        month = CellGoods(
            frozenset(PORT_KINDS),
            np.array([row * cols + col for row, col in cells], dtype=np.intp),
            MappingProxyType(makes),
            MappingProxyType(takes),
            MappingProxyType(stores),
        )
        _MONTHS[self] = month
        return month

    def current(self) -> CellGoods:
        """What the ports trade this month: cell_goods as the month's trade began."""
        month = _MONTHS.get(self)
        return self.cell_goods() if month is None else month

    def wants(self, need: Need, cells: npt.NDArray[np.intp]) -> FloatArray | None:
        """What the port in each of these flat cells takes and has room to store of a need this
        month; None if no port wants any."""
        month = self.current()
        if need not in month.takes and need not in month.stores:
            return None
        took = month.over({need: month.takes[need]}, cells)[need] if need in month.takes else 0.0
        if need not in month.stores:
            return took
        return took + month.over({need: month.stores[need]}, cells)[need]

    def wants_in(self, cells: npt.NDArray[np.intp]) -> dict[Need, float]:
        """What the ports in these flat cells take and have room to store this month."""
        month = self.current()
        wants: dict[Need, float] = {}
        for by_need in (month.takes, month.stores):
            for need, wanted in month.over(by_need, cells).items():
                amount = float(wanted.sum())
                if amount > EPSILON:
                    wants[need] = wants.get(need, 0.0) + amount
        return wants

    def gives_at(self, cell: Cell) -> dict[Good, float]:
        """What the port in a cell gives a month this year, drawn and landed from ships."""
        port = self.port(cell)
        if port is None:
            return {}
        gives: dict[Good, float] = {}
        for slot in port.gives:
            if good_exists(slot.good, self.year):
                gives[slot.good] = gives.get(slot.good, 0.0) + self.amount(slot)
        for good, held in port.landed.items():
            if held > EPSILON:
                gives[good] = gives.get(good, 0.0) + held
        return gives

    def takes_at(self, cell: Cell) -> dict[Need, float]:
        """What the port in a cell takes a month this year."""
        port = self.port(cell)
        if port is None:
            return {}
        takes: dict[Need, float] = {}
        for slot in port.takes:
            if good_exists(slot.good, self.year):
                need = need_of(slot.good)
                takes[need] = takes.get(need, 0.0) + self.amount(slot)
        return takes

    def gives_in(self, cells: npt.NDArray[np.intp]) -> dict[Good, float]:
        """What the ports in these flat cells give this month."""
        month = self.current()
        gives = {good: float(made.sum()) for good, made in month.over(month.makes, cells).items()}
        return {good: amount for good, amount in gives.items() if amount > EPSILON}

    def settle(self, trade: Trade) -> None:
        """Stores what the ports shipping goods got of them, keeps what ships landed that no one
        took, and sails a month of each shipped good to where it goes."""
        cols = self._growth.grid.cols
        for cell, port in self.at.items():
            kind = self.kind_at(cell)
            position = _position(trade, cell[0] * cols + cell[1])
            if kind is None or position is None:
                continue
            for good in port.ships:
                got = trade.goods_got.get((kind, need_of(good)))
                if got is None:
                    continue
                received, _ = got
                took = sum(self.amount(slot) for slot in port.takes if slot.good == good)
                extra = max(float(received[position]) - took, 0.0)
                port.store[good] = port.store.get(good, 0.0) + extra
            for good, held in list(port.landed.items()):
                gave = trade.goods_gave.get((kind, good))
                if gave is None:
                    continue
                given, most = gave
                offered = float(most[position])
                went = min(float(given[position]) / offered, 1.0) if offered > EPSILON else 0.0
                port.landed[good] = held * (1.0 - went)
        self.sail()

    def sail(self) -> None:
        """Every ship carries a month of its good from the store of the port shipping it to the
        port it goes to, if the player still controls both, as much as that has room for."""
        controlled = set(self.controlled())
        for cell, port in self.at.items():
            for good, to in list(port.ships.items()):
                if cell not in controlled or to not in controlled:
                    continue
                target = self.at.get(to)
                if target is None:
                    continue
                load = self.load(good)
                room = STORE_MONTHS * load - target.landed.get(good, 0.0)
                carried = min(port.store.get(good, 0.0), load, max(room, 0.0))
                if carried <= EPSILON:
                    continue
                port.store[good] -= carried
                target.landed[good] = target.landed.get(good, 0.0) + carried

    def moved(self, growth: Growth, move: Move) -> Ports:
        ports = Ports(growth)
        ports.at = {
            move.cell(cell): replace(
                port,
                slots=list(port.slots),
                ships={good: move.cell(to) for good, to in port.ships.items()},
                store=dict(port.store),
                landed=dict(port.landed),
            )
            for cell, port in self.at.items()
        }
        return ports


# What each game's ports trade this month (Ports.current), kept apart from the game so it is
# never saved.
_MONTHS: weakref.WeakKeyDictionary[Ports, CellGoods] = weakref.WeakKeyDictionary()


def _summed(first: Mapping[Good, float], second: Mapping[Good, float]) -> dict[Good, float]:
    together = dict(first)
    for good, amount in second.items():
        together[good] = together.get(good, 0.0) + amount
    return together


def _position(trade: Trade, flat: int) -> int | None:
    found = int(np.searchsorted(trade.cells, flat))
    if found < trade.cells.size and int(trade.cells[found]) == flat:
        return found
    return None


class _Makers:
    """Where on the map buildings other than ports make goods, and how much of each a month."""

    def __init__(self, growth: Growth) -> None:
        self._grid = growth.grid
        year = growth.year
        self._parts: list[
            tuple[npt.NDArray[np.intp], npt.NDArray[np.intp], FloatArray, dict[Good, float]]
        ] = []
        for kind, count in growth.counts.items():
            if kind in PORT_KINDS:
                continue
            recipe = RECIPES.get(kind)
            if recipe is not None:
                also = {good: recipe.most * share for good, share in recipe.also.items()}
                made = {recipe.makes: recipe.most, **also}
            else:
                made = dict(makes_of(kind, year))
            if not made:
                continue
            rows, cols = np.nonzero(count)
            if not rows.size:
                continue
            self._parts.append((rows, cols, count[rows, cols].astype(np.float64), made))

    def near(self, cell: Cell) -> dict[Good, float]:
        """How much of each good buildings within DRAW_RADIUS cells of a cell make a month."""
        near: dict[Good, float] = {}
        for rows, cols, many, made in self._parts:
            within = float(many[steps_from(self._grid, cell, rows, cols) <= DRAW_RADIUS].sum())
            if within <= 0:
                continue
            for good, amount in made.items():
                near[good] = near.get(good, 0.0) + within * amount
        return near
