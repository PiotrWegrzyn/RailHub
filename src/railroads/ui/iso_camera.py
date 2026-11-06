import math
from dataclasses import dataclass

from railroads.sim.geo import Coord

# On-screen width in pixels of one cell at each zoom level.
ZOOM_CELL_WIDTHS = (4, 8, 16, 24, 32, 64)
# Screen y is squashed by this much to fake looking at the map from an angle.
TILT = 0.5

type Vec2 = tuple[float, float]


@dataclass(slots=True)
class IsoCamera:
    """A tilted view of the flat map, turnable in equal steps around the view centre.

    The map is turned by `angle`, scaled so one cell is `cell_width` pixels wide, and squashed
    vertically by TILT. With a base angle of 45 degrees, squares look like RT2's diamonds.
    Coordinates on the map are in cell units, as the grids define them.
    """

    center_x: float
    center_y: float
    turn_steps: int = 4
    base_angle: float = 45.0
    # On-screen width of one cell in map units: sqrt(2) for a square seen corner-on, 1 for a hexagon.
    cell_units: float = math.sqrt(2)
    zoom_level: int = 0
    rotation: int = 0

    @property
    def cell_width(self) -> int:
        return ZOOM_CELL_WIDTHS[self.zoom_level]

    @property
    def angle(self) -> float:
        return math.radians(self.base_angle + 360 * self.rotation / self.turn_steps)

    def to_screen(self, x: Coord, y: Coord, screen_center: Vec2) -> tuple[Coord, Coord]:
        scale = self._scale
        cos, sin = math.cos(self.angle), math.sin(self.angle)
        dx, dy = x - self.center_x, y - self.center_y
        return (
            screen_center[0] + (cos * dx - sin * dy) * scale,
            screen_center[1] + (sin * dx + cos * dy) * scale * TILT,
        )

    def to_plane(self, screen_x: float, screen_y: float, screen_center: Vec2) -> Vec2:
        dx, dy = self._unproject(screen_x - screen_center[0], screen_y - screen_center[1])
        return self.center_x + dx, self.center_y + dy

    def pan(self, pixels_x: float, pixels_y: float) -> None:
        dx, dy = self._unproject(pixels_x, pixels_y)
        self.center_x -= dx
        self.center_y -= dy

    def zoom(self, steps: int, anchor: Vec2, screen_center: Vec2) -> None:
        before = self.to_plane(*anchor, screen_center)
        self.zoom_level = max(0, min(len(ZOOM_CELL_WIDTHS) - 1, self.zoom_level + steps))
        after = self.to_plane(*anchor, screen_center)
        self.center_x += before[0] - after[0]
        self.center_y += before[1] - after[1]

    def turn(self, steps: int) -> None:
        self.rotation = (self.rotation + steps) % self.turn_steps

    @property
    def _scale(self) -> float:
        return self.cell_width / self.cell_units

    def _unproject(self, pixels_x: float, pixels_y: float) -> Vec2:
        u, v = pixels_x / self._scale, pixels_y / (self._scale * TILT)
        cos, sin = math.cos(self.angle), math.sin(self.angle)
        return cos * u + sin * v, -sin * u + cos * v
