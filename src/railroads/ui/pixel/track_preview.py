"""Shows track the player is about to lay as the real track would look, greyed out, over the pixel
map: the planned pieces drawn like laid track (railroads.ui.pixel.track_art) into a picture of just the
area they cover, as finely as the map's chunks are drawn and lifted with the ground where hills and
mountains rise, faded to grey and see-through."""

import numpy as np
from imgui_bundle import hello_imgui, imgui

from railroads.sim.constructions import Track
from railroads.sim.routes import Route
from railroads.ui.iso_camera import PIXEL_CELL_WIDTH, IsoCamera, Vec2
from railroads.ui.pixel.layer import resolution_for
from railroads.ui.pixel.relief import MOST_LIFT
from railroads.ui.pixel.world import PixelWorld

# How much of the track's own colour is kept, and the greys its darkest and lightest parts fade
# to, so its sleepers and rails still show; and how see-through it is.
COLOUR_KEPT = 0.15
DARKEST = 0.12
LIGHTEST = 1.0
OPACITY = 0.85
# The most device pixels the blown-up picture may hold; past that it is blown up less.
MOST_PIXELS = 4_000_000


def ghost(pixels: np.ndarray) -> np.ndarray:
    """Track pixels faded to a see-through grey, as 0 to 255 RGBA."""
    light = pixels[..., :3] @ np.array([0.3, 0.59, 0.11], dtype=np.float32)
    grey = (DARKEST + (LIGHTEST - DARKEST) * light)[..., None]
    rgb = COLOUR_KEPT * pixels[..., :3] + (1 - COLOUR_KEPT) * grey
    alpha = pixels[..., 3:] * OPACITY
    return np.clip(np.concatenate([rgb, alpha], axis=-1) * 255, 0, 255).astype(np.uint8)


class TrackPreview:
    def __init__(self, world: PixelWorld) -> None:
        self.world = world
        self._shown: tuple[tuple[Track, ...], int] | None = None
        self._texture: hello_imgui.TextureGpu | None = None
        self._rect = (0, 0, 0, 0)
        self._factor = 1

    def draw(
        self, draw_list: imgui.ImDrawList, route: Route, camera: IsoCamera, center: Vec2
    ) -> None:
        scale = camera.pixel_scale
        density = max(1, round(imgui.get_io().display_framebuffer_scale.x))
        key = (tuple(route.pieces), scale * density)
        if key != self._shown:
            self._make(list(route.pieces), scale * density)
            self._shown = key
        if self._texture is None:
            return
        projection = self.world.projection
        camera_pixel = projection.to_pixels(camera.center_x, camera.center_y)
        origin_x = round(center[0] - float(camera_pixel[0]) * scale)
        origin_y = round(center[1] - float(camera_pixel[1]) * scale)
        left, top, width, height = self._rect
        draw_list.add_image(
            imgui.ImTextureRef(self._texture.texture_id()),
            (origin_x + left * scale, origin_y + top * scale),
            (origin_x + (left + width) * scale, origin_y + (top + height) * scale),
        )

    def _make(self, pieces: list[Track], device_scale: int) -> None:
        """Draws the pieces into a picture of the art pixels around them and uploads it."""
        self._texture = None
        if not pieces:
            return
        grid = self.world.tile_map.grid
        rows = np.array([piece.cell[0] for piece in pieces])
        cols = np.array([piece.cell[1] for piece in pieces])
        xs, ys = self.world.projection.to_pixels(*grid.center(rows, cols))
        reach = PIXEL_CELL_WIDTH
        left, top = int(np.min(xs)) - reach, int(np.min(ys)) - reach
        width, height = int(np.max(xs)) + reach - left, int(np.max(ys)) + reach - top
        resolution = resolution_for(device_scale, self.world.resolutions)
        layer = self.world.track.layer(pieces, left, top, width, height + MOST_LIFT, resolution)
        pixels = ghost(self.world.raised(layer, left, top, width, height, resolution))
        most = int((MOST_PIXELS / max(width * height * resolution**2, 1)) ** 0.5)
        factor = max(1, min(device_scale // resolution, most))
        big = np.repeat(np.repeat(pixels, factor, axis=0), factor, axis=1)
        self._texture = hello_imgui.create_texture_gpu_from_rgba_data(np.ascontiguousarray(big))
        self._rect = (left, top, width, height)
        self._factor = factor
