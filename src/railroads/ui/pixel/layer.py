"""Shows the pixel art on screen: chunks are drawn as the view reaches them, a few per frame, and
kept as textures blown up to the zoom's pixel scale. Where each art pixel covers an even number
of screen pixels the chunks are drawn finer (PixelWorld.render), so buildings, trees and track keep
thin lines instead of growing blocky."""

import time
from collections import OrderedDict

import numpy as np
from imgui_bundle import hello_imgui, imgui

from railroads.ui.iso_camera import IsoCamera, Vec2
from railroads.ui.pixel.world import CHUNK, ChunkPlacement, PixelWorld, Rgba

# Time a frame may spend drawing new chunks; at least one is always drawn.
FRAME_BUDGET_S = 0.025
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
    def __init__(self, world: PixelWorld) -> None:
        self.world = world
        # Drawn chunks by where they are and the resolution they were drawn at.
        self._chunks: OrderedDict[tuple[int, int, int], Rgba] = OrderedDict()
        self._textures: OrderedDict[tuple[int, int, int], hello_imgui.TextureGpu] = OrderedDict()
        self._stale: set[tuple[int, int, int]] = set()

    def refresh(self, chunks: set[ChunkKey]) -> None:
        """Draws these chunks again next time they show, keeping their old textures on screen
        until then."""
        for key in [key for key in self._chunks if key[:2] in chunks]:
            del self._chunks[key]
        self._stale |= {key for key in self._textures if key[:2] in chunks}

    def draw(
        self,
        draw_list: imgui.ImDrawList,
        camera: IsoCamera,
        center: Vec2,
        clip: tuple[Vec2, Vec2],
    ) -> bool:
        """Draws the chunks on screen; returns whether all of them were ready."""
        camera_pixel = self.world.projection.to_pixels(camera.center_x, camera.center_y)
        placement = ChunkPlacement.of(
            self.world.projection,
            (float(camera_pixel[0]), float(camera_pixel[1])),
            camera.pixel_scale,
            center,
            clip,
        )
        density = max(1, round(imgui.get_io().display_framebuffer_scale.x))
        started = time.perf_counter()
        drawn_new = False
        complete = True
        for chunk in placement.chunks:
            key = (*chunk, placement.scale * density)
            texture = self._textures.get(key)
            if texture is None or key in self._stale:
                if drawn_new and time.perf_counter() - started > FRAME_BUDGET_S:
                    complete = False
                    if texture is None:
                        continue
                else:
                    texture = self._upload(chunk, placement.scale * density)
                    self._stale.discard(key)
                    drawn_new = True
            self._textures.move_to_end(key)
            top_left, bottom_right = placement.screen_rect(chunk)
            draw_list.add_image(imgui.ImTextureRef(texture.texture_id()), top_left, bottom_right)
        while len(self._textures) > 1 and self._texture_bytes() > TEXTURE_BUDGET:
            self._stale.discard(self._textures.popitem(last=False)[0])
        return complete

    def _chunk_bytes(self) -> int:
        return sum(pixels.nbytes for pixels in self._chunks.values())

    def _texture_bytes(self) -> int:
        return sum((CHUNK * key[2]) ** 2 * 4 for key in self._textures)

    def _upload(self, chunk: ChunkKey, scale: int) -> hello_imgui.TextureGpu:
        resolution = resolution_for(scale, self.world.resolutions)
        drawn = (*chunk, resolution)
        pixels = self._chunks.get(drawn)
        if pixels is None:
            pixels = self.world.render(*chunk, resolution)
            self._chunks[drawn] = pixels
            while len(self._chunks) > 1 and self._chunk_bytes() > CHUNK_BUDGET:
                self._chunks.popitem(last=False)
        self._chunks.move_to_end(drawn)
        blow_up = scale // resolution
        big = np.repeat(np.repeat(pixels, blow_up, axis=0), blow_up, axis=1)
        texture = hello_imgui.create_texture_gpu_from_rgba_data(np.ascontiguousarray(big))
        self._textures[(*chunk, scale)] = texture
        return texture
