"""What the land means for building track and running trains. A cell of track costs
railroads.sim.finance.TRACK_PRICE times its price factor: half as much again in hills,
STREAM_FACTOR times that across a minor river, and far more for a bridge or a tunnel. The cost
factor is how much route planning avoids it, kept small so planning stays quick. Trains run at the speed factor of their top speed
through a cell, and burn the burn factor of their fuel there: hills slow them by 30% and make them
burn half as much again, tunnels slow them by 20%, and a bridge by what it is built of (BRIDGES):
wood, stone, steel or concrete, each carrying more tracks and letting trains run faster than the
one before, at a higher price. Steel and concrete bridges also go over track as flyovers, with a
ramp up to them on each side (RAMP_RULES)."""

import math
from dataclasses import dataclass, replace
from enum import IntEnum

from railroads.sim.tilemap import Relief, Terrain, Tile


class TrackBed(IntEnum):
    """What track is laid on: the ground, a bridge over a river or lake, or a tunnel through
    mountains."""

    GROUND = 0
    BRIDGE = 1
    TUNNEL = 2


@dataclass(frozen=True, slots=True)
class TrackRules:
    can_build: bool
    cost_factor: float
    speed_factor: float
    price_factor: float = 1.0
    burn_factor: float = 1.0


NO_TRACK = TrackRules(can_build=False, cost_factor=math.inf, speed_factor=0.0, price_factor=0.0)

RELIEF_RULES: dict[Relief, TrackRules] = {
    Relief.FLAT: TrackRules(can_build=True, cost_factor=1.0, speed_factor=1.0),
    Relief.HILLS: TrackRules(
        can_build=True, cost_factor=2.0, speed_factor=0.7, price_factor=1.5, burn_factor=1.5
    ),
    Relief.MOUNTAINS: NO_TRACK,
}
TUNNEL_RULES = TrackRules(can_build=True, cost_factor=8.0, speed_factor=0.8, price_factor=200.0 / 3)


class Bridge(IntEnum):
    """What a bridge is built of, from the oldest to the newest."""

    WOOD = 0
    STONE = 1
    STEEL = 2
    CONCRETE = 3


@dataclass(frozen=True, slots=True)
class BridgeKind:
    """What a kind of bridge is like: its name, the most tracks side by side it carries, what a
    cell of it costs against track on flat land, how fast trains run over it, whether it goes
    over track as a flyover and whether it takes overhead wires and a maglev guideway."""

    name: str
    tracks: int
    price_factor: float
    speed_factor: float
    over_track: bool = False
    electric: bool = False
    maglev: bool = False

    @property
    def rules(self) -> TrackRules:
        return TrackRules(
            can_build=True,
            cost_factor=5.0,
            speed_factor=self.speed_factor,
            price_factor=self.price_factor,
        )


BRIDGES: dict[Bridge, BridgeKind] = {
    Bridge.WOOD: BridgeKind("wooden", 1, 25.0, 0.3),
    Bridge.STONE: BridgeKind("stone", 2, 50.0, 0.4),
    Bridge.STEEL: BridgeKind("steel", 4, 75.0, 0.8, over_track=True, electric=True),
    Bridge.CONCRETE: BridgeKind(
        "concrete", 4, 125.0, 1.0, over_track=True, electric=True, maglev=True
    ),
}
# The embankment a flyover climbs on in the cell before and after it: track there costs half as
# much again, which the flyover's own price takes in for both, and trains run as on hills.
RAMP_RULES = TrackRules(can_build=True, cost_factor=2.0, speed_factor=0.7, price_factor=1.5)
RAMP_PRICE_FACTOR = 2 * (RAMP_RULES.price_factor - 1.0)


def bed_name(bed: TrackBed, bridge: Bridge, over_track: bool = False) -> str:
    """A bed in words: "ground", "tunnel", or the kind of bridge, "flyover" over track."""
    if bed != TrackBed.BRIDGE:
        return bed.name.lower()
    return f"{BRIDGES[bridge].name} {'flyover' if over_track else 'bridge'}"


def carries(bridge: Bridge, tracks: int, electric: bool = False, maglev: bool = False) -> bool:
    """Whether a bridge of this kind takes track this many tracks wide, electric or maglev."""
    kind = BRIDGES[bridge]
    return tracks <= kind.tracks and (kind.electric or not electric) and (kind.maglev or not maglev)


# Track across a minor river (railroads.mapgen.streams) costs this many times as much, culverts and
# all, and route planning shuns it as much more.
STREAM_FACTOR = 2.0


def track_rules_for(
    tile: Tile, bed: TrackBed = TrackBed.GROUND, bridge: Bridge = Bridge.WOOD
) -> TrackRules:
    """What laying track on this bed in this cell is like, on a bridge of this kind: over rivers
    and lakes, or over track on land as a flyover (railroads.sim.constructions checks the track
    under it), its price taking in the ramps on both sides."""
    if bed == TrackBed.BRIDGE:
        kind = BRIDGES[bridge]
        if tile.terrain in (Terrain.RIVER, Terrain.LAKE):
            return kind.rules
        if kind.over_track and tile.terrain == Terrain.LAND and tile.relief != Relief.MOUNTAINS:
            return replace(kind.rules, price_factor=kind.price_factor + RAMP_PRICE_FACTOR)
        return NO_TRACK
    if bed == TrackBed.TUNNEL:
        in_mountains = tile.terrain == Terrain.LAND and tile.relief == Relief.MOUNTAINS
        return TUNNEL_RULES if in_mountains else NO_TRACK
    if tile.terrain != Terrain.LAND:
        return NO_TRACK
    rules = RELIEF_RULES[tile.relief]
    if tile.stream and rules.can_build:
        return replace(
            rules,
            cost_factor=rules.cost_factor * STREAM_FACTOR,
            price_factor=rules.price_factor * STREAM_FACTOR,
        )
    return rules
