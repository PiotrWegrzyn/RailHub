"""What the shape of the land means for building track and running trains."""

import math
from dataclasses import dataclass

from railroads.sim.tilemap import Relief


@dataclass(frozen=True, slots=True)
class TrackRules:
    can_build: bool
    cost_factor: float
    speed_factor: float


RELIEF_RULES: dict[Relief, TrackRules] = {
    Relief.FLAT: TrackRules(can_build=True, cost_factor=1.0, speed_factor=1.0),
    Relief.HILLS: TrackRules(can_build=True, cost_factor=2.0, speed_factor=0.6),
    Relief.MOUNTAINS: TrackRules(can_build=False, cost_factor=math.inf, speed_factor=0.0),
}
