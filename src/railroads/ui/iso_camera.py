from dataclasses import dataclass

# Width in screen pixels of one tile's diamond at each zoom level. Diamonds are half as tall as wide.
ZOOM_TILE_WIDTHS = (4, 8, 16, 32, 64)
ROTATIONS = 4

Vec2 = tuple[float, float]


@dataclass(slots=True)
class IsoCamera:
    """Isometric view of the tile grid, turnable in 90-degree steps around the view centre.

    At rotation 0, tile x runs down-right and tile y runs down-left on screen.
    """

    center_x: float
    center_y: float
    zoom_level: int = 0
    rotation: int = 0

    @property
    def tile_width(self) -> int:
        return ZOOM_TILE_WIDTHS[self.zoom_level]

    def to_screen(self, tile_x: float, tile_y: float, screen_center: Vec2) -> Vec2:
        dx, dy = self._rotate(tile_x - self.center_x, tile_y - self.center_y, self.rotation)
        half_w, half_h = self.tile_width / 2, self.tile_width / 4
        return screen_center[0] + (dx - dy) * half_w, screen_center[1] + (dx + dy) * half_h

    def to_tile(self, screen_x: float, screen_y: float, screen_center: Vec2) -> Vec2:
        dx, dy = self._pixels_to_tiles(screen_x - screen_center[0], screen_y - screen_center[1])
        return self.center_x + dx, self.center_y + dy

    def pan(self, pixels_x: float, pixels_y: float) -> None:
        dx, dy = self._pixels_to_tiles(pixels_x, pixels_y)
        self.center_x -= dx
        self.center_y -= dy

    def zoom(self, steps: int, anchor: Vec2, screen_center: Vec2) -> None:
        before = self.to_tile(*anchor, screen_center)
        self.zoom_level = max(0, min(len(ZOOM_TILE_WIDTHS) - 1, self.zoom_level + steps))
        after = self.to_tile(*anchor, screen_center)
        self.center_x += before[0] - after[0]
        self.center_y += before[1] - after[1]

    def turn(self, quarter_turns: int) -> None:
        self.rotation = (self.rotation + quarter_turns) % ROTATIONS

    def _pixels_to_tiles(self, pixels_x: float, pixels_y: float) -> Vec2:
        a = pixels_x / (self.tile_width / 2)
        b = pixels_y / (self.tile_width / 4)
        return self._rotate((a + b) / 2, (b - a) / 2, -self.rotation)

    @staticmethod
    def _rotate(dx: float, dy: float, quarter_turns: int) -> Vec2:
        for _ in range(quarter_turns % ROTATIONS):
            dx, dy = -dy, dx
        return dx, dy
