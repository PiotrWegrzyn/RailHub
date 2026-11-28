"""Copies of a map's pixel world that paint its chunks somewhere else, as the web workers do in the
browser (railroads.ui.pixel.web_painter), so the frames never wait on painting.

A copy starts from the whole world pickled as it is, its heights, ground and scenery already worked
out, which is quicker than working them out again and needs far less memory on the way. It then
keeps up with the changes the world has since: buildings, track, stations and forests, taken up the
way the world takes them up (PixelWorld.buildings_changed). Painted the same way from the same state, a copy's chunks are the very pixels the world
paints itself. Nothing here needs imgui, which a worker doesn't load.

A worker also draws the fine pictures of a train's look (train_page), which take a frame of their
own to make on the page."""

import json
import pickle
from dataclasses import dataclass, field
from typing import Self

import numpy as np
import numpy.typing as npt

from railroads.ui.pixel.projection import Rgba
from railroads.ui.pixel.trains import fine_pictures, page_of
from railroads.ui.pixel.world import PixelWorld

type Drawing = tuple[int, int, int, bool, bool, bool, str]

# What a job paints: the chunk's picture (PixelWorld.render), the same without the track and
# everything standing on its ground (`standing`), or only its track, for the graphics card to draw
# (PixelWorld.track_shapes).
PICTURE = "picture"
GROUND = "ground"
TRACK = "track"


@dataclass(frozen=True)
class ChunkJob:
    """A chunk to paint (PixelWorld.render): where, at what resolution, with its raised ground
    worked out finely or not, over ground the graphics card draws or not, with what lies on its
    ground or without, which the graphics card then draws, and what of it (`what`): its whole
    picture, its ground alone, or its track as shapes `resolution` pixels to an art pixel.
    Called, the world paints it."""

    world: PixelWorld = field(repr=False, compare=False)
    chunk_x: int
    chunk_y: int
    resolution: int
    fine_relief: bool = False
    over_ground: bool = False
    surfaces: bool = True
    what: str = PICTURE

    def __call__(self) -> Rgba:
        return paint(self.world, self.drawing)

    @property
    def drawing(self) -> Drawing:
        """What a copy of the world needs to paint the same chunk."""
        return (
            self.chunk_x,
            self.chunk_y,
            self.resolution,
            self.fine_relief,
            self.over_ground,
            self.surfaces,
            self.what,
        )


def paint(world: PixelWorld, drawing: Drawing) -> Rgba:
    """What a job (ChunkJob.drawing) paints, painted from `world`."""
    *chunk, what = drawing
    if what == TRACK:
        return world.track_shapes(*chunk[:3])
    return world.render(*chunk, standing=what != GROUND)


def world_message(world: PixelWorld) -> bytes:
    """The whole world as it is now, for a copy to start from."""
    return pickle.dumps(world, protocol=pickle.HIGHEST_PROTOCOL)


@dataclass(frozen=True)
class Seen:
    """How far a copy has caught up with its world: the world's count of changes to buildings
    when the copy last heard of them."""

    buildings: int

    @classmethod
    def now(cls, world: PixelWorld) -> Self:
        return cls(world.buildings_version)


def update_message(world: PixelWorld, seen: Seen) -> bytes | None:
    """What changed in the world since copies saw it as `seen`, for them to take up
    (WorldCopy.update); None if nothing did. Changes are sent as they stand now rather than one
    by one, so a copy that missed some catches up all the same."""
    update: dict[str, object] = {}
    if world.buildings_version != seen.buildings:
        # As a save keeps them (Constructions.__getstate__): buildings as their kinds, which pickle
        # many times faster than the buildings themselves.
        constructions = world.constructions.__getstate__()
        update["constructions"] = {k: v for k, v in constructions.items() if k != "tile_map"}
        update["vegetation"] = world.tile_map.vegetation
        update["station_levels"] = dict(world.station_levels)
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
        # Let go of the old world first, so the new one can take its memory.
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
            world.constructions.__setstate__(state)
            vegetation: npt.NDArray[np.uint8] = update["vegetation"]
            vegetation.setflags(write=False)
            object.__setattr__(world.tile_map, "vegetation", vegetation)
            world.station_levels = update["station_levels"]
            world.buildings_changed(())

    def paint(self, number: int, drawing: Drawing) -> Rgba | None:
        """The chunk (ChunkJob.drawing) painted from its world, or None if the chunk is of a world
        it no longer has."""
        if self.world is None or number != self.number:
            return None
        return paint(self.world, tuple(drawing))


@dataclass(frozen=True)
class TrainPage:
    """A look's pictures every way round drawn finely (railroads.ui.pixel.trains.FinePainter) and
    packed on one page (page_of), as a worker hands them back: its RGBA pixels, and each
    picture's x, y, width, height and anchor on it, as JSON for the page to pass on."""

    pixels: npt.NDArray[np.uint8]
    spots_json: str


def train_page(message: bytes) -> TrainPage:
    """Draws the page a train layer asked for, pickled as its projection's matrix, the look, how
    many times as fine as the sheet and how wide a page is at the least."""
    matrix_key, look, times, width = pickle.loads(message)
    pixels, spots = page_of(fine_pictures(matrix_key, look, times), width)
    return TrainPage(np.ascontiguousarray(pixels), json.dumps(list(spots.values())))
