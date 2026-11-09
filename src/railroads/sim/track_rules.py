"""What the land means for building track and running trains. A cell of track costs
railroads.sim.finance.TRACK_PRICE times its price factor: $100 on the flat, $150 in hills,
$10,000 for a bridge and $20,000 for a tunnel. The cost factor is how much route planning
avoids it, kept small so planning stays quick."""

import math
from dataclasses import dataclass
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


NO_TRACK = TrackRules(can_build=False, cost_factor=math.inf, speed_factor=0.0, price_factor=0.0)

RELIEF_RULES: dict[Relief, TrackRules] = {
    Relief.FLAT: TrackRules(can_build=True, cost_factor=1.0, speed_factor=1.0),
    Relief.HILLS: TrackRules(can_build=True, cost_factor=2.0, speed_factor=0.6, price_factor=1.5),
    Relief.MOUNTAINS: NO_TRACK,
}
BRIDGE_RULES = TrackRules(can_build=True, cost_factor=5.0, speed_factor=1.0, price_factor=100.0)
TUNNEL_RULES = TrackRules(can_build=True, cost_factor=8.0, speed_factor=1.0, price_factor=200.0)


def track_rules_for(tile: Tile, bed: TrackBed = TrackBed.GROUND) -> TrackRules:
    """What laying track on this bed in this cell is like."""
    if bed == TrackBed.BRIDGE:
        over_water = tile.terrain in (Terrain.RIVER, Terrain.LAKE)
        return BRIDGE_RULES if over_water else NO_TRACK
    if bed == TrackBed.TUNNEL:
        in_mountains = tile.terrain == Terrain.LAND and tile.relief == Relief.MOUNTAINS
        return TUNNEL_RULES if in_mountains else NO_TRACK
    return RELIEF_RULES[tile.relief] if tile.terrain == Terrain.LAND else NO_TRACK
