"""The view gliding from where a new game opens to its biggest town: across at the zoom it opens at,
then closer a level at a time. Moving the view meanwhile ends the flight where the player left it."""

import math

from railroads.ui.iso_camera import IsoCamera, Vec2

PAN_SECONDS = 1.6
ZOOM_STEP_SECONDS = 0.4


class Flight:
    def __init__(self, camera: IsoCamera, to: Vec2, zoom: int) -> None:
        self.start = (camera.center_x, camera.center_y)
        self.to = to
        self.zoom_from = camera.zoom_level
        self.zoom_to = zoom
        self.seconds = 0.0
        self._set = self._view(camera)

    @property
    def seconds_long(self) -> float:
        return PAN_SECONDS + abs(self.zoom_to - self.zoom_from) * ZOOM_STEP_SECONDS

    def fly(self, camera: IsoCamera, seconds_passed: float) -> bool:
        """Moves the camera `seconds_passed` further on; False once it has arrived, or the view
        was moved by something else."""
        if self._view(camera) != self._set:
            return False
        self.seconds += seconds_passed
        share = min(self.seconds / PAN_SECONDS, 1.0)
        eased = (1 - math.cos(math.pi * share)) / 2
        camera.center_x = self.start[0] + (self.to[0] - self.start[0]) * eased
        camera.center_y = self.start[1] + (self.to[1] - self.start[1]) * eased
        steps = int(max(self.seconds - PAN_SECONDS, 0.0) / ZOOM_STEP_SECONDS)
        toward = 1 if self.zoom_to >= self.zoom_from else -1
        reached = min(steps, abs(self.zoom_to - self.zoom_from))
        camera.zoom_level = self.zoom_from + toward * reached
        self._set = self._view(camera)
        return self.seconds < self.seconds_long

    @staticmethod
    def _view(camera: IsoCamera) -> tuple[float, float, int]:
        return camera.center_x, camera.center_y, camera.zoom_level
