"""Copies of a map's pixel world that paint its chunks somewhere else, as the web workers do in the
browser (railroads.ui.pixel.web_painter), so the frames never wait on painting.

A copy starts from the whole world pickled as it is, its heights, ground and scenery already worked
out, which is quicker than working them out again and needs far less memory on the way. It then
keeps up with the changes the world has since: buildings, track, stations and forests, taken up the
way the world takes them up (PixelWorld.buildings_changed), and the satisfaction the ground glows
with. Painted the same way from the same state, a copy's chunks are the very pixels the world
paints itself. Nothing here needs imgui, which a worker doesn't load."""

import pickle
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from railroads.ui.pixel.projection import Rgba
from railroads.ui.pixel.world import PixelWorld

type Drawing = tuple[int, int, int, bool, bool]


@dataclass(frozen=True)
class ChunkJob:
    """A chunk to paint (PixelWorld.render): where, at what resolution, with its raised ground
    worked out finely or not, and over ground the graphics card draws or not. Called, the world
    paints it."""

    world: PixelWorld = field(repr=False, compare=False)
    chunk_x: int
    chunk_y: int
    resolution: int
    fine_relief: bool = False
    over_ground: bool = False

    def __call__(self) -> Rgba:
        return self.world.render(*self.drawing)

    @property
    def drawing(self) -> Drawing:
        """What a copy of the world needs to paint the same chunk."""
        return (self.chunk_x, self.chunk_y, self.resolution, self.fine_relief, self.over_ground)


def world_message(world: PixelWorld) -> bytes:
    """The whole world as it is now, for a copy to start from."""
    return pickle.dumps(world, protocol=pickle.HIGHEST_PROTOCOL)


@dataclass(frozen=True)
class Seen:
    """How far a copy has caught up with its world: the world's counts of changes to buildings
    and to satisfaction when the copy last heard of them."""

    buildings: int
    satisfaction: int

    @classmethod
    def now(cls, world: PixelWorld) -> Seen:
        return cls(world.buildings_version, world.satisfaction_version)


def update_message(world: PixelWorld, seen: Seen) -> bytes | None:
    """What changed in the world since copies saw it as `seen`, for them to take up
    (WorldCopy.update); None if nothing did. Changes are sent as they stand now rather than one
    by one, so a copy that missed some catches up all the same."""
    update: dict[str, object] = {}
    if world.buildings_version != seen.buildings:
        constructions = vars(world.constructions)
        update["constructions"] = {k: v for k, v in constructions.items() if k != "tile_map"}
        update["vegetation"] = world.tile_map.vegetation
        update["station_levels"] = dict(world.station_levels)
    if world.satisfaction_version != seen.satisfaction:
        update["satisfaction"] = world.satisfaction
    if not update:
        return None
    return pickle.dumps(update, protocol=pickle.HIGHEST_PROTOCOL)


class WorldCopy:
    """A copy of one world at a time, numbered by whoever sends it, painting chunks of it."""

    def __init__(self) -> None:
        self.world: PixelWorld | None = None
        self.number: int | None = None

    def start(self, number: int, message: bytes) -> None:
        """Takes up a new world (world_message) in place of the one it had."""
        self.world = None
        self.world = pickle.loads(message)
        self.number = number

    def update(self, number: int, message: bytes) -> None:
        """Takes up the changes to its world (update_message); changes to another are old and
        left out."""
        world = self.world
        if world is None or number != self.number:
            return
        update = pickle.loads(message)
        if "constructions" in update:
            # The constructions take their new state in place, so whatever was worked out per
            # cell from them and is still right is kept (railroads.ui.building_spots).
            state = update["constructions"]
            state["tile_map"] = world.tile_map
            vars(world.constructions).update(state)
            vegetation: npt.NDArray[np.uint8] = update["vegetation"]
            vegetation.setflags(write=False)
            object.__setattr__(world.tile_map, "vegetation", vegetation)
            world.station_levels = update["station_levels"]
            world.buildings_changed(())
        if "satisfaction" in update:
            world.satisfaction = update["satisfaction"]

    def paint(self, number: int, drawing: Drawing) -> Rgba | None:
        """The chunk (ChunkJob.drawing) painted from its world, or None if the chunk is of a world
        it no longer has."""
        if self.world is None or number != self.number:
            return None
        return self.world.render(*drawing)
