"""What engines cost to run each month (railroads.sim.railway charges it).

An engine's base running cost (Engine.running) is what its fuel costs at first. Fuel gets dearer
every year from when its kind of power came in: coal fast, so steam is cheap to run at first and
dear in the end, diesel oil and electricity slowly. The player's business makes running a little
cheaper: each building they own takes OWNED_CUT off, and every hundred cars of freight their trains
have delivered DELIVERED_CUT, each up to a limit, so in the long run only newer engines keep it
cheap. Each working power station beside a station takes POWER_STATION_CUT off electric and maglev
engines, up to MOST_POWER_STATION_CUT.
"""

from dataclasses import dataclass

from railroads.sim.trains import Engine, Power

# The year each kind of fuel's price starts rising from, and how much of its first price it rises
# by every year after.
FUEL_FROM = {Power.STEAM: 1840, Power.DIESEL: 1935, Power.ELECTRIC: 1925, Power.MAGLEV: 2040}
FUEL_RISE = {Power.STEAM: 0.04, Power.DIESEL: 0.01, Power.ELECTRIC: 0.005, Power.MAGLEV: 0.005}
OWNED_CUT = 0.005
MOST_OWNED_CUT = 0.2
DELIVERED_CUT = 0.01
DELIVERED_STEP = 100
MOST_DELIVERED_CUT = 0.2
POWER_STATION_CUT = 0.15
MOST_POWER_STATION_CUT = 0.6
ELECTRIC_POWER = frozenset({Power.ELECTRIC, Power.MAGLEV})


def fuel_price(power: Power, year: int) -> float:
    """How many times its first price a kind of fuel costs in a year."""
    return 1.0 + FUEL_RISE[power] * max(0, year - FUEL_FROM[power])


@dataclass(frozen=True, slots=True)
class Business:
    """What of the player's business makes running engines cheaper: the buildings they own, the
    cars of freight their trains have delivered, and their working power stations."""

    owned: int = 0
    delivered: int = 0
    power_stations: int = 0

    def cut(self, power: Power) -> float:
        """The share taken off the running cost of engines of a kind of power."""
        owned = min(MOST_OWNED_CUT, OWNED_CUT * self.owned)
        delivered = min(MOST_DELIVERED_CUT, DELIVERED_CUT * (self.delivered // DELIVERED_STEP))
        kept = (1.0 - owned) * (1.0 - delivered)
        if power in ELECTRIC_POWER:
            kept *= 1.0 - min(MOST_POWER_STATION_CUT, POWER_STATION_CUT * self.power_stations)
        return 1.0 - kept


NO_BUSINESS = Business()


def running_cost(engine: Engine, year: int, business: Business = NO_BUSINESS) -> float:
    """What an engine costs to run a month in a year, before tech cuts upkeep."""
    return engine.running * fuel_price(engine.power, year) * (1.0 - business.cut(engine.power))
