"""Monthly summaries along the right edge of the map: a card for each month that changed
something within reach of a station; clicking one opens what was built and pulled down there,
station by station, and outlines those cells on the map."""

from collections.abc import Callable
from dataclasses import dataclass, field

from imgui_bundle import icons_fontawesome_4 as fa
from imgui_bundle import imgui

from railroads.sim.grids import Cell
from railroads.sim.summaries import MonthSummary
from railroads.ui.iso_camera import Vec2
from railroads.ui.theme import (
    BORDER,
    BRASS,
    CARDS_WIDTH,
    GONE,
    GROWN,
    PANEL,
    TEXT,
    TEXT_DIM,
    color_u32,
)
from railroads.ui.trains_window import station_name

CARD_SIZE = (250.0, 50.0)
CARD_GAP = 6.0
CARDS_TOP = 70.0
MOST_CARDS = 5
KEPT = 24
WINDOW_WIDTH = 360.0
WINDOW_MAX_HEIGHT = 520.0


@dataclass
class Note:
    summary: MonthSummary
    seen: bool = False


@dataclass
class SummariesPanel:
    notes: list[Note] = field(default_factory=list)
    # The summary whose window is open.
    open: MonthSummary | None = None

    def add(self, summary: MonthSummary) -> None:
        self.notes.insert(0, Note(summary))
        del self.notes[KEPT:]

    def outlined(self) -> tuple[Cell, ...]:
        """The cells to outline on the map: those that changed in the open summary."""
        if self.open is None:
            return ()
        return tuple(cell for station in self.open.stations for cell in station.cells)

    def draw(self, top_left: Vec2, bottom_right: Vec2, look_at: Callable[[Cell], None]) -> None:
        if self.notes:
            self._draw_cards(top_left, bottom_right)
        if self.open is not None:
            self._draw_window(look_at)

    def _draw_cards(self, top_left: Vec2, bottom_right: Vec2) -> None:
        imgui.set_next_window_pos(
            (bottom_right[0] - 12, top_left[1] + CARDS_TOP), imgui.Cond_.always.value, (1.0, 0.0)
        )
        flags = (
            imgui.WindowFlags_.no_decoration.value
            | imgui.WindowFlags_.always_auto_resize.value
            | imgui.WindowFlags_.no_move.value
            | imgui.WindowFlags_.no_saved_settings.value
            | imgui.WindowFlags_.no_focus_on_appearing.value
            | imgui.WindowFlags_.no_nav.value
            | imgui.WindowFlags_.no_background.value
        )
        imgui.push_style_var(imgui.StyleVar_.window_padding.value, imgui.ImVec2(0, 0))
        imgui.push_style_var(imgui.StyleVar_.item_spacing.value, imgui.ImVec2(0, CARD_GAP))
        imgui.begin("##summaries", flags=flags)
        for index, note in enumerate(list(self.notes[:MOST_CARDS])):
            self._draw_card(index, note)
        if len(self.notes) > MOST_CARDS:
            imgui.text_disabled(f"  {len(self.notes) - MOST_CARDS} older")
        imgui.end()
        imgui.pop_style_var(2)

    def _draw_card(self, index: int, note: Note) -> None:
        summary = note.summary
        clicked = imgui.invisible_button(f"card{index}", CARD_SIZE)
        dismissed = imgui.is_item_clicked(imgui.MouseButton_.right.value)
        hovered = imgui.is_item_hovered()
        low, high = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        draw_list = imgui.get_window_draw_list()
        picked = summary is self.open
        draw_list.add_rect_filled(low, high, color_u32(PANEL), 7.0)
        edge = BRASS if hovered or picked else BORDER
        draw_list.add_rect(low, high, color_u32(edge), 7.0, 1.0)
        if not note.seen:
            draw_list.add_rect_filled(
                (low.x + 4, low.y + 8), (low.x + 7, high.y - 8), color_u32(BRASS), 2.0
            )
        stations = len(summary.stations)
        where = "1 station" if stations == 1 else f"{stations} stations"
        draw_list.add_text(
            (low.x + 16, low.y + 7),
            color_u32(BRASS if not note.seen else TEXT),
            f"{fa.ICON_FA_HOME}  {summary.date}",
        )
        parts = []
        if summary.built:
            parts.append(f"{summary.built} built")
        if summary.pulled_down:
            parts.append(f"{summary.pulled_down} pulled down")
        draw_list.add_text(
            (low.x + 16, low.y + 27), color_u32(TEXT_DIM), f"{', '.join(parts)} near {where}"
        )
        if hovered:
            imgui.set_tooltip("Click to see what changed; right click to dismiss")
        if clicked:
            note.seen = True
            self.open = None if picked else summary
        if dismissed:
            self.notes.remove(note)
            if picked:
                self.open = None

    def _draw_window(self, look_at: Callable[[Cell], None]) -> None:
        summary = self.open
        assert summary is not None
        display = imgui.get_io().display_size
        imgui.set_next_window_pos(
            (display.x - CARDS_WIDTH - 12, 96), imgui.Cond_.appearing.value, (1.0, 0.0)
        )
        imgui.set_next_window_size_constraints((WINDOW_WIDTH, 0), (WINDOW_WIDTH, WINDOW_MAX_HEIGHT))
        title = f"{fa.ICON_FA_HOME}  {summary.date} near your stations###summary"
        flags = imgui.WindowFlags_.always_auto_resize.value
        expanded, still_open = imgui.begin(title, True, flags)
        if expanded:
            imgui.text_disabled("Outlined on the map. What changed within reach of:")
            for station in summary.stations:
                imgui.push_id(str(station.station))
                imgui.separator()
                imgui.align_text_to_frame_padding()
                imgui.text(station_name(station.station))
                imgui.same_line()
                if imgui.small_button(fa.ICON_FA_CROSSHAIRS):
                    look_at(station.station)
                if imgui.is_item_hovered():
                    imgui.set_tooltip("Look at this station")
                for kind, count in sorted(station.changes.items(), key=lambda kc: -kc[1]):
                    color = GROWN if count > 0 else GONE
                    imgui.text_colored(imgui.ImVec4(*color), f"   {count:+d}")
                    imgui.same_line()
                    imgui.text(kind.value)
                imgui.pop_id()
        imgui.end()
        if not still_open:
            self.open = None
