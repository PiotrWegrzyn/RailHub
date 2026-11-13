"""The search window: finds a kind of building by its name or by a good it makes or wants, outlines
every one on the map and steps from one to the next."""

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cache

import numpy as np
import numpy.typing as npt
from imgui_bundle import icons_fontawesome_4 as fa
from imgui_bundle import imgui

from railroads.sim.constructions import BuildingKind
from railroads.sim.goods import FITS, MAKES, NEEDS
from railroads.sim.grids import Cell
from railroads.sim.growth import Growth
from railroads.ui.iso_camera import Vec2
from railroads.ui.theme import TEXT, TEXT_DIM, color_u32, draw_good_icon

WINDOW_SIZE = (360.0, 440.0)
WINDOW_POS = (16.0, 96.0)
ROW_HEIGHT = 24.0
ROW_ICON = 18.0
FOOTER_HEIGHT = 92.0
MOST_ROW_ICONS = 3


@cache
def goods_of(kind: BuildingKind) -> frozenset[str]:
    """The names of every good a kind of building makes or could take, worked out once: the
    search and the help match every kind against what is typed each frame."""
    made = {good.value for good in MAKES.get(kind, {})}
    wanted = {need.value for need in NEEDS.get(kind, {})}
    meeting = {good.value for need in NEEDS.get(kind, {}) for good in FITS[need]}
    return frozenset(made | wanted | meeting)


def kinds_matching(query: str) -> list[BuildingKind]:
    """The kinds of building a search finds: those named by it first, then those trading a good
    named by it, each in alphabetical order."""
    words = query.lower().split()
    named, trading = [], []
    for kind in sorted(BuildingKind, key=lambda k: k.value):
        name = kind.value.lower()
        if all(word in name for word in words):
            named.append(kind)
        elif all(any(word in good for good in goods_of(kind)) for word in words):
            trading.append(kind)
    return named + trading


@dataclass
class BuildingSearch:
    open: bool = False
    query: str = ""
    # The kind of building outlined on the map, if one was picked.
    kind: BuildingKind | None = None
    # Where each one stands, the nearest to the view when it was picked first.
    cells: list[Cell] = field(default_factory=list)
    # The position in `cells` the view was last sent to, once stepping has begun.
    index: int | None = None
    counts: dict[BuildingKind, int] = field(default_factory=dict)
    # The rows and columns of `cells`, for drawing them.
    rows: npt.NDArray[np.intp] = field(default_factory=lambda: np.empty(0, dtype=np.intp))
    cols: npt.NDArray[np.intp] = field(default_factory=lambda: np.empty(0, dtype=np.intp))

    @property
    def current(self) -> Cell | None:
        return None if self.index is None else self.cells[self.index]

    def refresh(self, growth: Growth) -> None:
        """Catches up with what has been built and pulled down, keeping the order of the
        buildings already found and the place reached among them."""
        self.counts = {kind: int(growth.count(kind).sum()) for kind in BuildingKind}
        if self.kind is None:
            return
        standing = _cells_of(growth, self.kind)
        present = set(standing)
        current = self.current
        known = set(self.cells)
        kept = [cell for cell in self.cells if cell in present]
        self.cells = kept + [cell for cell in standing if cell not in known]
        if current in present:
            self.index = self.cells.index(current)
        elif self.index is not None:
            self.index = min(self.index, len(self.cells) - 1) if self.cells else None
        self._index_cells()

    def choose(self, kind: BuildingKind, growth: Growth, around: Vec2) -> None:
        """Outlines every building of a kind, ordering them outwards from a point on the map."""
        self.kind = kind
        self.index = None
        cells = _cells_of(growth, kind)
        if cells:
            xs, ys = growth.constructions.tile_map.grid.center(*np.array(cells).T)
            order = np.argsort(np.hypot(xs - around[0], ys - around[1]), kind="stable")
            cells = [cells[i] for i in order.tolist()]
        self.cells = cells
        self._index_cells()

    def step(self, by: int) -> Cell | None:
        """Moves on to the next building, or back; the first step forward goes to the nearest
        and the first step back to the farthest."""
        if not self.cells:
            return None
        start = -1 if by > 0 else 0
        self.index = ((start if self.index is None else self.index) + by) % len(self.cells)
        return self.cells[self.index]

    def clear(self) -> None:
        self.kind, self.index = None, None
        self.cells = []
        self._index_cells()

    def close(self) -> None:
        self.open = False
        self.clear()

    def _index_cells(self) -> None:
        located = np.array(self.cells, dtype=np.intp).reshape(-1, 2)
        self.rows, self.cols = located[:, 0], located[:, 1]

    def draw(
        self,
        growth: Growth,
        visit: Callable[[Cell], None],
        around: Callable[[], Vec2],
    ) -> None:
        """The window, if it is open: `visit` shows a cell and `around` is the view's centre."""
        if not self.open:
            return
        imgui.set_next_window_pos(WINDOW_POS, imgui.Cond_.first_use_ever.value)
        imgui.set_next_window_size(WINDOW_SIZE, imgui.Cond_.first_use_ever.value)
        expanded, still_open = imgui.begin(f"{fa.ICON_FA_SEARCH}  Find buildings###find", True)
        if expanded:
            self._draw_query(growth, around)
            self._draw_results(growth, around)
            if self.kind is not None:
                self._draw_stepper(visit)
        imgui.end()
        if not still_open:
            self.close()

    def _draw_query(self, growth: Growth, around: Callable[[], Vec2]) -> None:
        if imgui.is_window_appearing():
            imgui.set_keyboard_focus_here()
        imgui.set_next_item_width(-1)
        flags = imgui.InputTextFlags_.enter_returns_true.value
        hint = f"{fa.ICON_FA_SEARCH}  A building, or a good it trades"
        entered, self.query = imgui.input_text_with_hint("##find", hint, self.query, flags)
        if entered:
            found = (k for k in kinds_matching(self.query) if self.counts.get(k, 0))
            if (first := next(found, None)) is not None:
                self.choose(first, growth, around())

    def _draw_results(self, growth: Growth, around: Callable[[], Vec2]) -> None:
        height = -FOOTER_HEIGHT if self.kind is not None else 0.0
        imgui.begin_child("results", (0, height), imgui.ChildFlags_.borders.value)
        matches = kinds_matching(self.query)
        if not matches:
            imgui.text_disabled("Nothing goes by that name.")
        matches.sort(key=lambda kind: not self.counts.get(kind, 0))
        for kind in matches:
            count = self.counts.get(kind, 0)
            imgui.begin_disabled(count == 0)
            clicked = imgui.selectable(f"##{kind.value}", kind == self.kind, 0, (0, ROW_HEIGHT))[0]
            imgui.end_disabled()
            if clicked:
                self.choose(kind, growth, around())
            _draw_row(kind, count)
        imgui.end_child()

    def _draw_stepper(self, visit: Callable[[Cell], None]) -> None:
        assert self.kind is not None
        total = len(self.cells)
        where = "none on the map" if not total else f"{total:,} outlined on the map"
        imgui.text(self.kind.value.capitalize())
        imgui.same_line()
        imgui.text_disabled(where)
        imgui.begin_disabled(not total)
        moved = None
        if imgui.button(f"{fa.ICON_FA_ARROW_LEFT}##previous", (44, 0)):
            moved = self.step(-1)
        imgui.same_line()
        imgui.align_text_to_frame_padding()
        imgui.text("-" if self.index is None else f"{self.index + 1:,} of {total:,}")
        imgui.same_line()
        if imgui.button(f"{fa.ICON_FA_ARROW_RIGHT}##next", (44, 0)):
            moved = self.step(1)
        imgui.end_disabled()
        if moved is not None:
            visit(moved)
        imgui.same_line()
        if imgui.button(f"{fa.ICON_FA_TIMES}  Clear"):
            self.clear()
            return
        imgui.text_disabled("The left and right arrow keys step through them too.")


def _draw_row(kind: BuildingKind, count: int) -> None:
    """Draws a result over its selectable: the goods it makes, its name and how many stand."""
    low, high = imgui.get_item_rect_min(), imgui.get_item_rect_max()
    draw_list = imgui.get_window_draw_list()
    text_y = low.y + (ROW_HEIGHT - imgui.get_text_line_height()) / 2
    icon_y = low.y + (ROW_HEIGHT - ROW_ICON) / 2
    color = color_u32(TEXT if count else TEXT_DIM)
    goods = list(MAKES.get(kind, {})) or list(NEEDS.get(kind, {}))
    for index, good in enumerate(goods[:MOST_ROW_ICONS]):
        draw_good_icon(draw_list, good, (low.x + 2 + index * (ROW_ICON + 1), icon_y), ROW_ICON)
    left = low.x + 4 + MOST_ROW_ICONS * (ROW_ICON + 1) + 4
    draw_list.add_text((left, text_y), color, kind.value)
    number = f"{count:,}"
    width = imgui.calc_text_size(number).x
    draw_list.add_text((high.x - width - 6, text_y), color_u32(TEXT_DIM), number)


def _cells_of(growth: Growth, kind: BuildingKind) -> list[Cell]:
    return [(int(row), int(col)) for row, col in np.argwhere(growth.count(kind) > 0).tolist()]


def pulse(time: float) -> float:
    """Swings between 0 and 1, to make the building being visited stand out."""
    return 0.5 + 0.5 * math.sin(time * 6.0)
