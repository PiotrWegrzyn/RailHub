import math
from dataclasses import dataclass

from railroads.sim.geo import Coord

# On-screen width in pixels of one cell at each zoom level. The furthest levels draw the map as
# one smooth picture; from PIXEL_CELL_WIDTH on it is pixel art, shown 1 to 4 times as big.
ZOOM_CELL_WIDTHS = (4, 8, 16, 32, 64, 96, 128)
PIXEL_CELL_WIDTH = 32
# Screen y is squashed by this much to fake looking at the map from an angle.
TILT = 0.5

type Vec2 = tuple[float, float]


@dataclass(slots=True)
class IsoCamera:
    """A tilted view of the flat map from a fixed direction.

    The map is turned by `angle_deg`, scaled so one cell is `cell_width` pixels wide, and squashed
    vertically by TILT. Turned by -45 degrees, squares look like RT2's diamonds, with north at the
    top left. Coordinates on the map are in cell units, as the grids define them.
    """

    center_x: float
    center_y: float
    angle_deg: float = -45.0
    # On-screen width of one cell in map units: sqrt(2) for a square seen corner-on, 1 for a hexagon.
    cell_units: float = math.sqrt(2)
    zoom_level: int = 0

    @property
    def cell_width(self) -> int:
        return ZOOM_CELL_WIDTHS[self.zoom_level]

    @property
    def pixel_scale(self) -> int:
        """How many screen pixels one pixel of the pixel art covers, or 0 when zoomed out too far
        for pixel art."""
        return self.cell_width // PIXEL_CELL_WIDTH

    @property
    def angle(self) -> float:
        return math.radians(self.angle_deg)

    def to_screen(self, x: Coord, y: Coord, screen_center: Vec2) -> tuple[Coord, Coord]:
        scale = self._scale
        cos, sin = math.cos(self.angle), math.sin(self.angle)
        dx, dy = x - self.center_x, y - self.center_y
        return (
            screen_center[0] + (cos * dx - sin * dy) * scale,
            screen_center[1] + (sin * dx + cos * dy) * scale * TILT,
        )

    def to_plane(
        self, screen_x: Coord, screen_y: Coord, screen_center: Vec2
    ) -> tuple[Coord, Coord]:
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

    def keep_within(self, width: float, height: float) -> None:
        """Keeps the centre of the view on a map this wide and high, so panning cannot lose it,
        however long and thin it is."""
        self.center_x = min(max(self.center_x, 0.0), width)
        self.center_y = min(max(self.center_y, 0.0), height)

    def map_direction(self, screen_dx: float, screen_dy: float) -> Vec2:
        """The direction on the map, as a unit vector, that points that way on screen."""
        x, y = self._unproject(screen_dx, screen_dy)
        length = math.hypot(x, y)
        return x / length, y / length

    @property
    def _scale(self) -> float:
        return self.cell_width / self.cell_units

    def _unproject(self, pixels_x: Coord, pixels_y: Coord) -> tuple[Coord, Coord]:
        u, v = pixels_x / self._scale, pixels_y / (self._scale * TILT)
        cos, sin = math.cos(self.angle), math.sin(self.angle)
        return cos * u + sin * v, -sin * u + cos * v
