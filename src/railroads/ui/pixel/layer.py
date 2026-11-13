"""Shows the pixel art on screen: chunks are drawn as the view reaches them and kept as textures
blown up to the zoom's pixel scale. Where each art pixel covers an even number of screen pixels the
chunks are drawn finer (PixelWorld.render), so buildings, trees and track keep thin lines instead of
growing blocky. Where the ground shader runs, the graphics card blows the textures up, which saves
the frames repeating every pixel and sending the graphics card all of them.

Chunks are drawn away from the frames (railroads.ui.pixel.painter); until a chunk is ready, the
ground drawn beneath shows where it is missing, and a chunk that changed keeps its old texture.
Chunks where hills or mountains rise show first as they are quickly drawn, then are drawn again
with the heights, lift and colours of their raised ground at the full resolution, and the finer
drawing takes their place once it is ready.

In a browser, where threads cannot start, the page's web workers draw the chunks instead
(railroads.ui.pixel.web_painter), several at once, and frames carry on meanwhile as with a thread.
Until a worker is ready, or if none can be, one chunk is drawn on each frame, and raised chunks
are not drawn again finely, which would stall a frame for a third of a second. Then, when the
ground shader shows the ground beneath, chunks wait until the view and the mouse have stood still
for a moment, so moving the map or the mouse doesn't stall on drawing them.

Chunks nearest the middle of the view are asked for and turned into textures first, so the view
fills in from the middle out."""

import time
from collections import OrderedDict
from collections.abc import Callable
from typing import Protocol

import numpy as np
from imgui_bundle import hello_imgui, imgui

from railroads.ui.iso_camera import IsoCamera, Vec2
from railroads.ui.pixel.copies import ChunkJob
from railroads.ui.pixel.painter import Painter
from railroads.ui.pixel.projection import CHUNK, ChunkPlacement, Rgba
from railroads.ui.pixel.web_painter import WorkerPainter, page_workers
from railroads.ui.pixel.world import PixelWorld
from railroads.ui.workers import threads_start

# Time a frame may spend turning drawn chunks into textures; at least one is always made.
UPLOAD_BUDGET_S = 0.006
# Most chunks drawn while a game loads, enough to fill a large window.
PREPARED_CHUNKS = 48
# How long the view must stand still before chunks are drawn on frames, where they wait for it.
SETTLE_S = 0.3
# Most bytes of drawn chunks kept in memory; a fine chunk takes four times a plain one's.
CHUNK_BUDGET = 128 * 1024 * 1024
# Most bytes of chunk textures kept on the graphics card; the least recently shown go first.
TEXTURE_BUDGET = 512 * 1024 * 1024

type ChunkKey = tuple[int, int]


class Texture(Protocol):
    """A chunk's texture: one hello_imgui made, or one the ground shader blew up."""

    def texture_id(self) -> int: ...


def resolution_for(screen_pixels: int, resolutions: tuple[int, ...]) -> int:
    """The finest resolution a chunk can be drawn at that blows up evenly to `screen_pixels`
    for every art pixel."""
    return max(r for r in resolutions if screen_pixels % r == 0)


def painter_for(world: PixelWorld, threads: bool | None = None) -> Painter | WorkerPainter:
    """A painter of its own thread where threads can start; otherwise the page's web workers,
    if it started any, or else the frames."""
    if threads is None:
        threads = threads_start()
        workers = None if threads else page_workers()
        if workers is not None:
            return WorkerPainter(world, workers)
    return Painter(threads)


class PixelLayer:
    def __init__(self, world: PixelWorld, threads: bool | None = None) -> None:
        self.world = world
        # Drawn chunks by where they are and the resolution they were drawn at.
        self._chunks: OrderedDict[tuple[int, int, int], Rgba] = OrderedDict()
        self._textures: OrderedDict[tuple[int, int, int], Texture] = OrderedDict()
        self._stale: set[tuple[int, int, int]] = set()
        self._painter = painter_for(world, threads)
        # Chunks drawn finely, and how many times each chunk has been refreshed, so a drawing
        # asked for before a refresh is thrown away.
        self._refined: set[tuple[int, int, int]] = set()
        self._refreshes: dict[ChunkKey, int] = {}
        # Whether hills or mountains rise in each chunk, which never changes.
        self._rises: dict[ChunkKey, bool] = {}
        self._view: tuple | None = None
        self._still_since = 0.0
        # Whether the graphics card draws the ground beneath, so chunks where nothing rises are
        # drawn without theirs, and what blows textures up on it then, which makes the same
        # texels as blowing them up here (GroundShader.blow_up).
        self.over_ground = False
        self.blow_up: Callable[[Rgba, int], Texture] | None = None

    def refresh(self, chunks: set[ChunkKey]) -> None:
        """Draws these chunks again next time they show, keeping their old textures on screen
        until then."""
        for key in [key for key in self._chunks if key[:2] in chunks]:
            del self._chunks[key]
        self._stale |= {key for key in self._textures if key[:2] in chunks}
        for chunk in chunks:
            self._refreshes[chunk] = self._refreshes.get(chunk, 0) + 1
        self._refined = {key for key in self._refined if key[:2] not in chunks}

    def take_drawn(self) -> set[ChunkKey]:
        """Keeps the chunks the painter has finished, unless they changed since they were asked
        for; returns them."""
        taken: set[ChunkKey] = set()
        for (fine, drawn, refreshes), pixels, error in self._painter.finished():
            chunk = drawn[:2]
            if error is not None or refreshes != self._refreshes.get(chunk, 0):
                continue
            if fine:
                self._refined.add(drawn)
                self._stale |= {key for key in self._textures if key[:2] == chunk}
            elif drawn in self._chunks:
                continue
            self._chunks[drawn] = pixels
            taken.add(chunk)
        while len(self._chunks) > 1 and self._chunk_bytes() > CHUNK_BUDGET:
            self._chunks.popitem(last=False)
        return taken

    def draw(
        self,
        draw_list: imgui.ImDrawList,
        camera: IsoCamera,
        center: Vec2,
        clip: tuple[Vec2, Vec2],
        ground_below: bool = False,
    ) -> bool:
        """Draws the chunks on screen; returns whether all of them were ready. With
        `ground_below`, the ground already shows where a chunk is missing."""
        camera_pixel = self.world.projection.to_pixels(camera.center_x, camera.center_y)
        placement = ChunkPlacement.of(
            self.world.projection,
            (float(camera_pixel[0]), float(camera_pixel[1])),
            camera.pixel_scale,
            center,
            clip,
        )
        density = max(1, round(imgui.get_io().display_framebuffer_scale.x))
        scale = placement.scale * density
        resolution = resolution_for(scale, self.world.resolutions)
        self.take_drawn()
        now = time.perf_counter()
        io = imgui.get_io()
        pointer = (io.mouse_pos.x, io.mouse_pos.y, io.mouse_wheel, *io.mouse_down)
        waiting = ground_below and self._moved(camera, now, pointer)
        uploaded = False
        complete = True
        for chunk in middle_out(placement, center):
            key = (*chunk, scale)
            drawn = (*chunk, resolution)
            texture = self._textures.get(key)
            if texture is None or key in self._stale:
                late = uploaded and time.perf_counter() - now > UPLOAD_BUDGET_S
                if drawn in self._chunks and not late:
                    texture = self._upload(chunk, scale)
                    self._stale.discard(key)
                    uploaded = True
                else:
                    complete = False
                    self._ask(drawn, now)
                if texture is None:
                    continue
            self._refine(drawn, now)
            self._textures.move_to_end(key)
            top_left, bottom_right = placement.screen_rect(chunk)
            draw_list.add_image(imgui.ImTextureRef(texture.texture_id()), top_left, bottom_right)
        while len(self._textures) > 1 and self._texture_bytes() > TEXTURE_BUDGET:
            self._stale.discard(self._textures.popitem(last=False)[0])
        if not self._painter.threaded and not waiting:
            self._painter.work_once(now)
        return complete and not self._painter.waiting()

    def prepare(
        self, camera: IsoCamera, center: Vec2, clip: tuple[Vec2, Vec2], density: int
    ) -> None:
        """Draws the chunks this view would show, at most PREPARED_CHUNKS of them, on the calling
        thread, so they are ready for the first frame; for a game being loaded."""
        camera_pixel = self.world.projection.to_pixels(camera.center_x, camera.center_y)
        placement = ChunkPlacement.of(
            self.world.projection,
            (float(camera_pixel[0]), float(camera_pixel[1])),
            camera.pixel_scale,
            center,
            clip,
        )
        resolution = resolution_for(placement.scale * density, self.world.resolutions)
        for chunk in placement.chunks[:PREPARED_CHUNKS]:
            drawn = (*chunk, resolution)
            if drawn not in self._chunks:
                self._chunks[drawn] = self.world.render(*drawn)

    def _moved(self, camera: IsoCamera, now: float, pointer: tuple = ()) -> bool:
        """Whether chunks drawn on frames must wait because the view or the mouse (`pointer`:
        where it is and what it presses) moved a moment ago."""
        view = (camera.center_x, camera.center_y, camera.pixel_scale, *pointer)
        if view != self._view:
            self._view = view
            self._still_since = now
        return not self._painter.threaded and now - self._still_since < SETTLE_S

    def _ask(self, drawn: tuple[int, int, int], now: float) -> None:
        if drawn in self._chunks:
            return
        refreshes = self._refreshes.get(drawn[:2], 0)
        work = ChunkJob(self.world, *drawn, over_ground=self.over_ground)
        self._painter.want((False, drawn, refreshes), work, True, now)

    def _refine(self, drawn: tuple[int, int, int], now: float) -> None:
        """Asks for a chunk to be drawn again with fine raised ground, if it has any and is not
        drawn finely already."""
        resolution = drawn[2]
        if not self._painter.threaded or resolution == 1:
            return
        if drawn in self._refined or drawn not in self._chunks:
            return
        rises = self._rises.get(drawn[:2])
        if rises is None:
            rises = self._rises[drawn[:2]] = self.world.rises(*drawn[:2])
        if not rises:
            return
        refreshes = self._refreshes.get(drawn[:2], 0)
        work = ChunkJob(self.world, *drawn, fine_relief=True)
        self._painter.want((True, drawn, refreshes), work, False, now)

    def _chunk_bytes(self) -> int:
        return sum(pixels.nbytes for pixels in self._chunks.values())

    def _texture_bytes(self) -> int:
        return sum((CHUNK * key[2]) ** 2 * 4 for key in self._textures)

    def _upload(self, chunk: ChunkKey, scale: int) -> Texture:
        resolution = resolution_for(scale, self.world.resolutions)
        drawn = (*chunk, resolution)
        pixels = self._chunks[drawn]
        self._chunks.move_to_end(drawn)
        times = scale // resolution
        texture: Texture
        if times > 1 and self.blow_up is not None:
            texture = self.blow_up(pixels, times)
        else:
            big = blown_up(pixels, times)
            texture = hello_imgui.create_texture_gpu_from_rgba_data(np.ascontiguousarray(big))
        self._textures[(*chunk, scale)] = texture
        return texture


def blown_up(pixels: Rgba, times: int) -> Rgba:
    """The pixels with each repeated `times` times each way, as the texture of a chunk drawn at
    a finer scale than the screen's where the graphics card can't blow it up. Drawn at the
    screen's scale, as every chunk is at zoom 3 on a double-density screen, they are the pixels
    themselves: copying them anyway cost a browser more than a millisecond a chunk."""
    if times == 1:
        return pixels
    return np.repeat(np.repeat(pixels, times, axis=0), times, axis=1)


def middle_out(placement: ChunkPlacement, center: Vec2) -> list[ChunkKey]:
    """The chunks placed on screen, those nearest `center`, the middle of the view, first."""

    def off_middle(chunk: ChunkKey) -> float:
        (left, top), (right, bottom) = placement.screen_rect(chunk)
        return ((left + right) / 2 - center[0]) ** 2 + ((top + bottom) / 2 - center[1]) ** 2

    return sorted(placement.chunks, key=off_middle)
