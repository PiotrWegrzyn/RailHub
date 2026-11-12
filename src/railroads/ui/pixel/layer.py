"""Shows the pixel art on screen: chunks are drawn as the view reaches them and kept as textures
blown up to the zoom's pixel scale. Where each art pixel covers an even number of screen pixels the
chunks are drawn finer (PixelWorld.render), so buildings, trees and track keep thin lines instead of
growing blocky.

Chunks are drawn away from the frames (railroads.ui.pixel.painter); until a chunk is ready, the
ground drawn beneath shows where it is missing, and a chunk that changed keeps its old texture.
Chunks where hills or mountains rise show first as they are quickly drawn, then are drawn again
with the heights, lift and colours of their raised ground at the full resolution, and the finer
drawing takes their place once it is ready.

Where threads cannot start, as in a browser, one chunk is drawn on each frame. There, when the
ground shader shows the ground beneath, chunks wait until the view has stood still for a moment,
so moving the map doesn't stall on drawing them."""

import time
from collections import OrderedDict

import numpy as np
from imgui_bundle import hello_imgui, imgui

from railroads.ui.iso_camera import IsoCamera, Vec2
from railroads.ui.pixel.painter import Painter
from railroads.ui.pixel.projection import CHUNK, ChunkPlacement, Rgba
from railroads.ui.pixel.world import PixelWorld

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


def resolution_for(screen_pixels: int, resolutions: tuple[int, ...]) -> int:
    """The finest resolution a chunk can be drawn at that blows up evenly to `screen_pixels`
    for every art pixel."""
    return max(r for r in resolutions if screen_pixels % r == 0)


class PixelLayer:
    def __init__(self, world: PixelWorld, threads: bool | None = None) -> None:
        self.world = world
        # Drawn chunks by where they are and the resolution they were drawn at.
        self._chunks: OrderedDict[tuple[int, int, int], Rgba] = OrderedDict()
        self._textures: OrderedDict[tuple[int, int, int], hello_imgui.TextureGpu] = OrderedDict()
        self._stale: set[tuple[int, int, int]] = set()
        self._painter = Painter(threads)
        # Chunks drawn finely, and how many times each chunk has been refreshed, so a drawing
        # asked for before a refresh is thrown away.
        self._refined: set[tuple[int, int, int]] = set()
        self._refreshes: dict[ChunkKey, int] = {}
        # Whether hills or mountains rise in each chunk, which never changes.
        self._rises: dict[ChunkKey, bool] = {}
        self._view: tuple[float, float, int] | None = None
        self._still_since = 0.0

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
        waiting = ground_below and self._moved(camera, now)
        uploaded = False
        complete = True
        for chunk in placement.chunks:
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

    def _moved(self, camera: IsoCamera, now: float) -> bool:
        """Whether chunks drawn on frames must wait because the view moved a moment ago."""
        view = (camera.center_x, camera.center_y, camera.pixel_scale)
        if view != self._view:
            self._view = view
            self._still_since = now
        return not self._painter.threaded and now - self._still_since < SETTLE_S

    def _ask(self, drawn: tuple[int, int, int], now: float) -> None:
        if drawn in self._chunks:
            return
        refreshes = self._refreshes.get(drawn[:2], 0)
        work = lambda: self.world.render(*drawn)
        self._painter.want((False, drawn, refreshes), work, True, now)

    def _refine(self, drawn: tuple[int, int, int], now: float) -> None:
        """Asks for a chunk to be drawn again with fine raised ground, if it has any and is not
        drawn finely already."""
        resolution = drawn[2]
        if resolution == 1 or drawn in self._refined or drawn not in self._chunks:
            return
        rises = self._rises.get(drawn[:2])
        if rises is None:
            rises = self._rises[drawn[:2]] = self.world.rises(*drawn[:2])
        if not rises:
            return
        refreshes = self._refreshes.get(drawn[:2], 0)
        work = lambda: self.world.render(*drawn, True)
        self._painter.want((True, drawn, refreshes), work, False, now)

    def _chunk_bytes(self) -> int:
        return sum(pixels.nbytes for pixels in self._chunks.values())

    def _texture_bytes(self) -> int:
        return sum((CHUNK * key[2]) ** 2 * 4 for key in self._textures)

    def _upload(self, chunk: ChunkKey, scale: int) -> hello_imgui.TextureGpu:
        resolution = resolution_for(scale, self.world.resolutions)
        drawn = (*chunk, resolution)
        pixels = self._chunks[drawn]
        self._chunks.move_to_end(drawn)
        blow_up = scale // resolution
        big = np.repeat(np.repeat(pixels, blow_up, axis=0), blow_up, axis=1)
        texture = hello_imgui.create_texture_gpu_from_rgba_data(np.ascontiguousarray(big))
        self._textures[(*chunk, scale)] = texture
        return texture
