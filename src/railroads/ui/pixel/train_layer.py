"""Shows trains as pixel art over the pixel map: each engine, tender and car drawn from the train
sprite sheet (railroads.ui.pixel.trains), turned to the track, lined up on the map's own art
pixels and lifted with the ground where hills and mountains rise, with puffs of smoke from the
chimney."""

import math

import numpy as np
from imgui_bundle import hello_imgui, imgui

from railroads.sim.railway import Railway
from railroads.sim.trains import Train
from railroads.ui.iso_camera import IsoCamera, Vec2
from railroads.ui.pixel.models import HEIGHT_PER_WIDTH
from railroads.ui.pixel.trains import (
    CAR_HALF,
    COUPLING,
    DETAIL,
    ENGINES,
    SCALE,
    Look,
    TrainSprites,
    train_sprites,
)
from railroads.ui.pixel.world import PixelWorld

SMOKE = (0.88, 0.88, 0.9)
# The biggest texture a graphics card takes on each side; a bigger one draws black.
MAX_TEXTURE = 16384
PUFFS = 5


class TrainLayer:
    def __init__(self, world: PixelWorld) -> None:
        self.world = world
        matrix = world.projection.matrix.ravel()
        self.sprites: TrainSprites = train_sprites(tuple(float(v) for v in matrix))
        self._textures: dict[int, tuple[hello_imgui.TextureGpu, int]] = {}

    def draw(
        self, draw_list: imgui.ImDrawList, railway: Railway, camera: IsoCamera, center: Vec2
    ) -> None:
        projection = self.world.projection
        scale = camera.pixel_scale
        density = max(1, round(imgui.get_io().display_framebuffer_scale.x))
        camera_pixel = projection.to_pixels(camera.center_x, camera.center_y)
        origin = (
            round(center[0] - float(camera_pixel[0]) * scale),
            round(center[1] - float(camera_pixel[1]) * scale),
        )
        texture, _ = self._texture(scale * density)
        height, width = self.sprites.image.shape[:2]
        per_pixel = scale / DETAIL
        found = []
        for train in railway.trains:
            for look, back in self._parts(train):
                placed = railway.position(train, back)
                if placed is not None:
                    found.append((placed, look))
        parts = []
        if found:
            xs = np.array([placed[0][0] for placed, _ in found])
            ys = np.array([placed[0][1] for placed, _ in found])
            ax, ay = projection.to_pixels(xs, ys)
            ay = ay - self.world.elevation.lift(xs, ys)
            for k, (((_, _), (dx, dy)), look) in enumerate(found):
                sx = origin[0] + round(float(ax[k]) * DETAIL) * per_pixel
                sy = origin[1] + round(float(ay[k]) * DETAIL) * per_pixel
                parts.append((sy, sx, look, math.atan2(dy, dx)))
        parts.sort(key=lambda part: part[0])
        ref = imgui.ImTextureRef(texture.texture_id())
        for sy, sx, look, heading in parts:
            px, py, w, h, anchor_x, anchor_y = self.sprites.place(look, heading)
            left = sx - anchor_x * per_pixel
            top = sy - anchor_y * per_pixel
            draw_list.add_image(
                ref,
                (left, top),
                (left + w * per_pixel, top + h * per_pixel),
                (px / width, py / height),
                ((px + w) / width, (py + h) / height),
            )
        for train in railway.trains:
            self._smoke(draw_list, railway, train, origin, scale)

    def _parts(self, train: Train) -> list[tuple[object, float]]:
        """Each part of the train and how far its middle is behind the train's front, in cells."""
        model = ENGINES[train.engine]
        parts: list[tuple[object, float]] = [((Look.ENGINE, train.engine), model.half)]
        back = 2 * model.half
        if model.tender is not None:
            back += COUPLING + model.tender_half
            parts.append(((Look.TENDER, train.engine), back))
            back += model.tender_half
        for car in train.cars:
            back += COUPLING + CAR_HALF
            parts.append(((car.good, car.load > 0), back))
            back += CAR_HALF
        return [(look, middle * SCALE) for look, middle in parts]

    def _smoke(
        self,
        draw_list: imgui.ImDrawList,
        railway: Railway,
        train: Train,
        origin: tuple[int, int],
        scale: int,
    ) -> None:
        """Puffs of smoke rising from the chimney and drifting back over the train."""
        projection = self.world.projection
        model = ENGINES[train.engine]
        if model.chimney is None:
            return
        chimney = (model.chimney[0] * SCALE, model.chimney[1] * SCALE)
        placed = railway.position(train, (model.half - model.chimney[0]) * SCALE)
        if placed is None:
            return
        (x, y), (dx, dy) = placed
        lift = float(self.world.elevation.lift(x, y))
        length = math.hypot(dx, dy) or 1.0
        ux, uy = dx / length, dy / length
        up = float(np.hypot(*projection.matrix[:, 0])) * HEIGHT_PER_WIDTH
        phase = (train.along * 6.0) % 1.0
        for k in range(PUFFS):
            age = (k + phase) / PUFFS
            px, py = x - ux * age * 0.5 * SCALE, y - uy * age * 0.5 * SCALE
            ax, ay = projection.to_pixels(px, py)
            sx = origin[0] + float(ax) * scale
            sy = origin[1] + (float(ay) - lift - (chimney[1] + age * 0.35 * SCALE) * up) * scale
            radius = (1.5 + age * 4.0) * SCALE * scale
            alpha = 0.75 * (1 - age)
            color = imgui.color_convert_float4_to_u32(imgui.ImVec4(*SMOKE, alpha))
            draw_list.add_circle_filled((sx, sy), radius, color)

    def _texture(self, device_scale: int) -> tuple[hello_imgui.TextureGpu, int]:
        """The sprite sheet blown up by a whole number so it shows crisp at this many device
        pixels per art pixel, and by how much it was blown up."""
        height, width = self.sprites.image.shape[:2]
        fits = max(1, MAX_TEXTURE // max(height, width))
        factor = max(1, min(device_scale // DETAIL, fits))
        if factor not in self._textures:
            big = np.repeat(np.repeat(self.sprites.image, factor, axis=0), factor, axis=1)
            texture = hello_imgui.create_texture_gpu_from_rgba_data(np.ascontiguousarray(big))
            self._textures[factor] = (texture, factor)
        return self._textures[factor]
