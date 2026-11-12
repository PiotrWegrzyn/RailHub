"""The menu the game opens on: start a new game with its settings chosen first, load a saved
one, or quit."""

import secrets
from dataclasses import dataclass, field
from typing import Literal

from imgui_bundle import icons_fontawesome_4 as fa
from imgui_bundle import imgui

from railroads.newgame import (
    EARLIEST_START_YEAR,
    MapChoice,
    Settings,
    allowed_start_year,
    world_maps_available,
)
from railroads.saves import SaveInfo, Saves
from railroads.sim.clock import END_YEAR, HISTORY_START, START_YEAR
from railroads.sim.finance import START_CASH, dollars
from railroads.ui.saves_window import draw_save_list
from railroads.ui.theme import BRASS, GONE

MENU_WIDTH = 460.0
BUTTON_HEIGHT = 36.0
CASH_STEP = 10_000
MOST_CASH = 10_000_000
SEEDS = 1_000_000
TITLE_SCALE = 2.0

Page = Literal["main", "new", "load"]


@dataclass(frozen=True)
class NewGameChoice:
    settings: Settings


@dataclass(frozen=True)
class LoadChoice:
    name: str


@dataclass(frozen=True)
class QuitChoice:
    pass


MenuChoice = NewGameChoice | LoadChoice | QuitChoice


def random_seed() -> int:
    """A seed from the system's randomness, so each start of the game offers a different one."""
    return secrets.randbelow(SEEDS)


def map_note(choice: MapChoice) -> str:
    """What the menu says under the chosen map: its size and what is on it."""
    baked = choice.start_map
    if baked is None:
        return "A mostly-land spot anywhere in the world, picked by the seed."
    bounds = baked.bounds
    return f"{bounds.width_km:,.0f} × {bounds.height_km:,.0f} km. {baked.about}"


@dataclass
class MainMenu:
    saves: Saves
    page: Page = "main"
    map: MapChoice = MapChoice.POLAND
    seed: int = field(default_factory=random_seed)
    start_year: int = START_YEAR
    cash: int = int(START_CASH)
    stand_in_years: int = 0
    selected: str | None = None
    listing: list[SaveInfo] = field(default_factory=list)
    # What went wrong with the last game started or loaded, if anything.
    error: str = ""

    def settings(self) -> Settings:
        return Settings(
            seed=self.seed,
            map=self.map,
            start_year=allowed_start_year(self.start_year),
            cash=float(self.cash),
            stand_in_years=self.stand_in_years,
        )

    def go_to(self, page: Page) -> None:
        self.page = page
        if page == "load":
            self.listing = self.saves.listing()
            if self.selected is None and self.listing:
                self.selected = self.listing[0].name

    def draw(self) -> MenuChoice | None:
        """Draws the menu centred in the current window, which in the game is the full-screen
        one, so nothing left over from play can stand in front of it."""
        width, height = imgui.get_window_size()
        imgui.set_cursor_pos(((width - MENU_WIDTH) / 2, max(24.0, height * 0.12)))
        imgui.begin_group()
        imgui.push_item_width(MENU_WIDTH)
        imgui.push_font(None, imgui.get_style().font_size_base * TITLE_SCALE)
        imgui.text_colored(imgui.ImVec4(*BRASS), f"{fa.ICON_FA_TRAIN}  Railroads")
        imgui.pop_font()
        imgui.spacing()
        if self.error:
            imgui.push_text_wrap_pos(imgui.get_cursor_pos_x() + MENU_WIDTH)
            imgui.text_colored(imgui.ImVec4(*GONE), self.error)
            imgui.pop_text_wrap_pos()
        imgui.separator()
        imgui.spacing()
        choice = {"main": self._draw_main, "new": self._draw_new, "load": self._draw_load}[
            self.page
        ]()
        imgui.pop_item_width()
        imgui.end_group()
        return choice

    def _button(self, label: str) -> bool:
        return imgui.button(label, (MENU_WIDTH, BUTTON_HEIGHT))

    def _draw_main(self) -> MenuChoice | None:
        if self._button(f"{fa.ICON_FA_PLAY}  New game"):
            self.go_to("new")
        if self._button(f"{fa.ICON_FA_FOLDER_OPEN}  Load game"):
            self.go_to("load")
        if self._button(f"{fa.ICON_FA_POWER_OFF}  Quit"):
            return QuitChoice()
        return None

    def _draw_new(self) -> MenuChoice | None:
        imgui.text("Map")
        world = world_maps_available()
        if imgui.begin_combo("##map", self.map.value, imgui.ComboFlags_.height_large.value):
            for choice in MapChoice:
                disabled = choice == MapChoice.WORLD and not world
                imgui.begin_disabled(disabled)
                if imgui.selectable(choice.value, self.map == choice)[0]:
                    self.map = choice
                imgui.end_disabled()
                hovered = imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled.value)
                if disabled and hovered:
                    imgui.set_tooltip(
                        "Needs the elevation database: uv run python tools/build_elevation_db.py"
                    )
            imgui.end_combo()
        imgui.push_text_wrap_pos(imgui.get_cursor_pos_x() + MENU_WIDTH)
        imgui.text_disabled(map_note(self.map))
        imgui.pop_text_wrap_pos()
        imgui.spacing()
        imgui.text("Seed")
        imgui.set_next_item_width(MENU_WIDTH - 40)
        _, self.seed = imgui.input_int("##seed", self.seed, 0)
        self.seed = min(max(self.seed, 0), SEEDS - 1)
        imgui.same_line(0, 4)
        if imgui.button(fa.ICON_FA_RANDOM, (36, 0)):
            self.seed = random_seed()
        if imgui.is_item_hovered():
            imgui.set_tooltip("A new random seed")
        imgui.text_disabled("The same seed and map grow the same towns.")
        imgui.spacing()
        imgui.text("Play begins in")
        _, year = imgui.slider_int(
            "##start year",
            self.start_year,
            EARLIEST_START_YEAR,
            END_YEAR - 1,
            flags=imgui.SliderFlags_.always_clamp.value,
        )
        self.start_year = allowed_start_year(year)
        imgui.text_colored(imgui.ImVec4(*BRASS), f"{START_YEAR} is suggested.")
        imgui.text_disabled(
            f"Towns grow for {self.start_year - HISTORY_START} years from {HISTORY_START} first; "
            "more years take longer to set up."
        )
        imgui.spacing()
        imgui.text("Starting money")
        _, self.cash = imgui.input_int("##cash", self.cash, CASH_STEP, CASH_STEP * 10)
        self.cash = min(max(self.cash, 0), MOST_CASH)
        imgui.text_disabled(f"{dollars(self.cash)}; the usual start is {dollars(START_CASH)}.")
        imgui.spacing()
        if imgui.collapsing_header("For testing"):
            imgui.text("A stand-in player runs the railway first for (years)")
            _, self.stand_in_years = imgui.slider_int(
                "##stand-in", self.stand_in_years, 0, END_YEAR - self.start_year
            )
        imgui.spacing()
        imgui.separator()
        if self._button(f"{fa.ICON_FA_PLAY}  Start"):
            self.error = ""
            return NewGameChoice(self.settings())
        if self._button("Back"):
            self.go_to("main")
        return None

    def _draw_load(self) -> MenuChoice | None:
        self.selected, chosen = draw_save_list(self.listing, self.selected, 320.0)
        loadable = any(info.name == self.selected and info.problem is None for info in self.listing)
        imgui.text_disabled(f"Saves are kept in {self.saves.folder}")
        imgui.begin_disabled(not loadable)
        clicked = self._button(f"{fa.ICON_FA_FOLDER_OPEN}  Load")
        imgui.end_disabled()
        if self._button("Back"):
            self.go_to("main")
        if loadable and self.selected and (clicked or chosen):
            self.error = ""
            return LoadChoice(self.selected)
        return None
