"""The saves window in play: a tab to save the game under a name, over an older save or as a new
one, and a tab to load any save, autosaves included."""

from dataclasses import dataclass, field
from typing import Literal

from imgui_bundle import imgui

from railroads.saves import SaveInfo, Saves, name_problem
from railroads.ui.save_files import BrowserFiles, DesktopFiles, draw_file_buttons, save_files
from railroads.ui.theme import BRASS, GONE

WINDOW_SIZE = (560.0, 420.0)
WINDOW_TOP = 56.0
LIST_HEIGHT = 240.0


@dataclass(frozen=True)
class SaveRequest:
    kind: Literal["save", "load"]
    name: str


def describe(info: SaveInfo) -> str:
    if info.problem is not None:
        return info.problem
    when = info.saved_at.strftime("%Y-%m-%d %H:%M") if info.saved_at else ""
    return "  ·  ".join(part for part in (info.date, info.title, when) if part)


def draw_save_list(
    listing: list[SaveInfo], selected: str | None, height: float = LIST_HEIGHT
) -> tuple[str | None, bool]:
    """The saves as a list to pick from; returns the one picked and whether it was
    double-clicked."""
    chosen = False
    imgui.begin_child("saves", (0, height), imgui.ChildFlags_.borders.value)
    if not listing:
        imgui.text_disabled("No saves yet.")
    for info in listing:
        label = f"{info.name}##{info.path}"
        if imgui.selectable(label, info.name == selected)[0]:
            selected = info.name
            chosen = imgui.is_mouse_double_clicked(imgui.MouseButton_.left)
        imgui.indent()
        if info.problem is not None:
            imgui.text_colored(imgui.ImVec4(*GONE), describe(info))
        else:
            imgui.text_disabled(describe(info))
        imgui.unindent()
    imgui.end_child()
    return selected, chosen


@dataclass
class SavesWindow:
    open: bool = False
    name: str = ""
    selected: str | None = None
    message: str = ""
    listing: list[SaveInfo] = field(default_factory=list)
    files: BrowserFiles | DesktopFiles = field(default_factory=save_files)

    def show(self, saves: Saves) -> None:
        self.open = True
        self.message = ""
        self.refresh(saves)

    def refresh(self, saves: Saves) -> None:
        self.listing = saves.listing()

    def _brought_in(self, saves: Saves, name: str) -> None:
        self.refresh(saves)
        self.selected = name

    def draw(self, saves: Saves) -> SaveRequest | None:
        if not self.open:
            return None
        width = imgui.get_io().display_size.x
        imgui.set_next_window_pos(
            ((width - WINDOW_SIZE[0]) / 2, WINDOW_TOP), imgui.Cond_.appearing.value
        )
        imgui.set_next_window_size(WINDOW_SIZE, imgui.Cond_.appearing.value)
        expanded, self.open = imgui.begin("Save and load", self.open)
        request = None
        if expanded and imgui.begin_tab_bar("saves tabs"):
            if imgui.begin_tab_item("Save")[0]:
                request = self._draw_save(saves)
                imgui.end_tab_item()
            if imgui.begin_tab_item("Load")[0]:
                request = self._draw_load(saves) or request
                imgui.end_tab_item()
            imgui.end_tab_bar()
            if self.message:
                imgui.text_wrapped(self.message)
        imgui.end()
        return request

    def _draw_save(self, saves: Saves) -> SaveRequest | None:
        mine = [info for info in self.listing if not info.autosave]
        picked, _ = draw_save_list(mine, self.selected)
        if picked != self.selected:
            self.selected = self.name = picked or ""
        imgui.set_next_item_width(-120)
        entered, self.name = imgui.input_text(
            "##save name", self.name, imgui.InputTextFlags_.enter_returns_true.value
        )
        problem = name_problem(self.name)
        imgui.same_line()
        label = "Overwrite" if not problem and saves.exists(self.name) else "Save"
        imgui.begin_disabled(problem is not None)
        clicked = imgui.button(label, (-1, 0))
        imgui.end_disabled()
        if problem and self.name:
            imgui.text_colored(imgui.ImVec4(*GONE), problem)
        else:
            imgui.text_disabled("Pick a save to write over it, or type a new name.")
        if (clicked or entered) and problem is None:
            return SaveRequest("save", self.name.strip())
        return None

    def _draw_load(self, saves: Saves) -> SaveRequest | None:
        self.selected, chosen = draw_save_list(self.listing, self.selected)
        loadable = any(info.name == self.selected and info.problem is None for info in self.listing)
        imgui.begin_disabled(not loadable)
        clicked = imgui.button("Load", (-1, 0))
        imgui.end_disabled()
        message = draw_file_buttons(
            self.files,
            saves,
            self.selected,
            imgui.get_content_region_avail().x,
            lambda name: self._brought_in(saves, name),
        )
        if message:
            self.message = message
        imgui.text_colored(
            imgui.ImVec4(*BRASS), "Loading leaves this game; save it first to keep it."
        )
        if loadable and self.selected and (clicked or chosen):
            return SaveRequest("load", self.selected)
        return None
