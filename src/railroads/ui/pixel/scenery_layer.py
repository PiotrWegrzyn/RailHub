"""Draws what stands on the ground, buildings and trees, over the chunks in close views, where the
chunks leave it out (railroads.ui.pixel.layer.chunk_resolution): each sprite drawn as fine as the
screen shows it, a building from its 3D model where it has one (railroads.ui.pixel.art.built), so
it keeps the painted look and gains detail and smooth edges instead of growing blocky.

Each kind of sprite is drawn once for the zoom and kept as a texture; a frame draws new ones only
while it has PAINT_TIME to spare, and until a kind is drawn it shows from the fine sprite sheet
blown up, as the chunks would have shown it."""

import time
from dataclasses import dataclass

import numpy as np
from imgui_bundle import hello_imgui, imgui

from railroads.ui.iso_camera import IsoCamera, Vec2
from railroads.ui.pixel.art import close_sprite
from railroads.ui.pixel.layer import chunk_resolution
from railroads.ui.pixel.world import CLOSE, PixelWorld

# Time a frame may spend drawing kinds of sprite as fine as the screen.
PAINT_TIME = 0.006
# How far in art pixels past the edges of the view sprites are still drawn: as far as one reaches.
MARGIN = 64


@dataclass(frozen=True)
class _Picture:
    """A sprite on a texture of its own: the texture, its size in pixels, its anchor and how many
    of its pixels an art pixel of the map takes."""

    texture: object
    size: tuple[int, int]
    anchor: tuple[int, int]
    per_art_pixel: int


class SceneryLayer:
    def __init__(self, world: PixelWorld) -> None:
        self.world = world
        self._pictures: dict[int, _Picture] = {}
        self._sheet_pictures: dict[int, _Picture] = {}
        self._device_scale = 0

    def draw(self, draw_list: imgui.ImDrawList, camera: IsoCamera, center: Vec2) -> None:
        """Draws the scenery in sight back to front, if the view is close."""
        scale = camera.pixel_scale
        density = max(1, round(imgui.get_io().display_framebuffer_scale.x))
        if not scale or chunk_resolution(scale, density, self.world.resolutions) != CLOSE:
            return
        device = scale * density
        if device != self._device_scale:
            self._pictures.clear()
            self._device_scale = device
        projection = self.world.projection
        camera_pixel = projection.to_pixels(camera.center_x, camera.center_y)
        origin = (
            round(center[0] - float(camera_pixel[0]) * scale),
            round(center[1] - float(camera_pixel[1]) * scale),
        )
        size = imgui.get_io().display_size
        scenery = self.world.scenery
        art_x0 = (-origin[0]) / scale - MARGIN
        art_x1 = (size.x - origin[0]) / scale + MARGIN
        art_y0 = (-origin[1]) / scale - MARGIN
        art_y1 = (size.y - origin[1]) / scale + MARGIN
        ys = scenery.y - scenery.lift
        seen = (scenery.x >= art_x0) & (scenery.x <= art_x1) & (ys >= art_y0) & (ys <= art_y1)
        until = time.perf_counter() + PAINT_TIME
        for i in np.flatnonzero(seen).tolist():
            kind = int(scenery.sprite[i])
            picture = self._pictures.get(kind)
            if picture is None and time.perf_counter() <= until:
                picture = self._pictures[kind] = self._paint(kind, device)
            if picture is None:
                picture = self._from_sheet(kind)
            per = scale / picture.per_art_pixel
            left = origin[0] + int(scenery.x[i]) * scale - picture.anchor[0] * per
            top = origin[1] + int(ys[i]) * scale - picture.anchor[1] * per
            draw_list.add_image(
                imgui.ImTextureRef(picture.texture.texture_id()),  # type: ignore[attr-defined]
                (left, top),
                (left + picture.size[0] * per, top + picture.size[1] * per),
            )

    def _paint(self, kind: int, device: int) -> _Picture:
        sprite = close_sprite(self.world.sheet.names[kind], device)
        pixels = np.ascontiguousarray(np.asarray(sprite.image.convert("RGBA")))
        texture = hello_imgui.create_texture_gpu_from_rgba_data(pixels)
        height, width = pixels.shape[:2]
        return _Picture(texture, (width, height), sprite.anchor, device)

    def _from_sheet(self, kind: int) -> _Picture:
        """The sprite as the finest sprite sheet holds it, for a kind not drawn finer yet."""
        picture = self._sheet_pictures.get(kind)
        if picture is None:
            resolution, pixels, anchor = self.world.sprite(kind)
            rgba = np.ascontiguousarray(np.clip(pixels * 255 + 0.5, 0, 255).astype(np.uint8))
            texture = hello_imgui.create_texture_gpu_from_rgba_data(rgba)
            height, width = rgba.shape[:2]
            picture = self._sheet_pictures[kind] = _Picture(
                texture, (width, height), anchor, resolution
            )
        return picture
