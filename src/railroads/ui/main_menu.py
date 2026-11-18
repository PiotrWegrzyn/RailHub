"""The menu the game opens on: start a new game with its settings chosen first, play the
tutorial, load a saved one, or quit."""

import secrets
from dataclasses import dataclass, field, replace
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
from railroads.sim.tilemap import START_SHARE
from railroads.ui.menu_scene import HORIZON, MenuScene
from railroads.ui.ornate import DEEP, bar_buttons, frame_rect, rule, spaced, spaced_text
from railroads.ui.save_files import (
    IN_BROWSER,
    BrowserFiles,
    DesktopFiles,
    draw_file_buttons,
    save_files,
)
from railroads.ui.saves_window import draw_save_list
from railroads.ui.theme import BRASS, GONE, TEXT_DIM, color_u32, tip
from railroads.version import VERSION

MENU_WIDTH = 460.0
BUTTON_HEIGHT = 36.0
CASH_STEP = 10_000
MOST_CASH = 10_000_000
SEEDS = 1_000_000
TITLE_SCALE = 2.0
# The room round the menu inside its frame, and the panel behind it.
PANEL_PAD = 20.0
# How fast the menu grows or shrinks to what it holds: the share of the way a second.
MENU_EASING = 14.0
PANEL_BG = (DEEP[0], DEEP[1], DEEP[2], 0.88)

Page = Literal["main", "new", "load"]


@dataclass(frozen=True)
class NewGameChoice:
    settings: Settings


@dataclass(frozen=True)
class LoadChoice:
    name: str


@dataclass(frozen=True)
class TutorialChoice:
    pass


@dataclass(frozen=True)
class QuitChoice:
    pass


MenuChoice = NewGameChoice | TutorialChoice | LoadChoice | QuitChoice


def draw_version() -> None:
    """The game's version, small in the bottom right corner of the current window."""
    label = f"v{VERSION}"
    corner = imgui.get_window_pos() + imgui.get_window_size()
    size = imgui.calc_text_size(label)
    padding = imgui.get_style().window_padding
    colour = imgui.get_color_u32(imgui.Col_.text_disabled.value)
    imgui.get_window_draw_list().add_text(corner - size - padding, colour, label)


def random_seed() -> int:
    """A seed from the system's randomness, so each start of the game offers a different one."""
    return secrets.randbelow(SEEDS)


def map_note(choice: MapChoice) -> str:
    """What the menu says under the chosen map: how much of it a game starts on, the seven lots
    in its middle (railroads.sim.lots), and what is on it."""
    baked = choice.start_map
    if baked is None:
        return "A mostly-land spot anywhere in the world, picked by the seed."
    bounds = baked.bounds
    area = round(bounds.width_km * bounds.height_km * START_SHARE, -3)
    return f"About {area:,.0f} km² to start, in seven hexagonal lots. {baked.about}"


@dataclass
class MainMenu:
    saves: Saves
    page: Page = "main"
    # One of the ready-made maps; a random place is started with its own button.
    map: MapChoice = MapChoice.POLAND
    seed: int = field(default_factory=random_seed)
    start_year: int = START_YEAR
    cash: int = int(START_CASH)
    stand_in_years: int = 0
    show_map_growing: bool = False
    selected: str | None = None
    listing: list[SaveInfo] = field(default_factory=list)
    # What went wrong with the last game started or loaded, if anything.
    error: str = ""
    files: BrowserFiles | DesktopFiles = field(default_factory=save_files)
    # What came of the last save file exported or imported.
    note: str = ""
    # The trains running behind the menu.
    scene: MenuScene = field(default_factory=MenuScene)
    # How tall the menu's first page was last drawn: every page hangs from where that one is
    # centred, so opening a page or a fold only lengthens the menu downwards.
    _height: float = 0.0
    # How tall the menu is drawn now, growing or shrinking smoothly to what it holds.
    _shown: float = 0.0
    # The folding headings on the new game page that are open.
    _unfolded: set[str] = field(default_factory=set)

    def settings(self) -> Settings:
        return Settings(
            seed=self.seed,
            map=self.map,
            start_year=allowed_start_year(self.start_year),
            cash=float(self.cash),
            stand_in_years=self.stand_in_years,
            show_map_growing=self.show_map_growing,
        )

    def random_settings(self) -> Settings:
        """The settings with a random place in the world for the map."""
        return replace(self.settings(), map=MapChoice.WORLD)

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
        low = imgui.get_window_pos()
        self.scene.draw(low, imgui.ImVec2(low.x + width, low.y + height))
        top = max(16.0, (height * HORIZON - self._height) / 2)
        imgui.set_cursor_pos(((width - MENU_WIDTH) / 2 - PANEL_PAD, top))
        # A child window of the menu's width, so headers and lines end where the buttons do.
        imgui.push_style_color(imgui.Col_.child_bg, imgui.ImVec4(*PANEL_BG))
        imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(PANEL_PAD, PANEL_PAD))
        flags = imgui.ChildFlags_.always_use_window_padding.value
        if not self._shown:
            flags |= imgui.ChildFlags_.auto_resize_y.value
        no_scroll = (
            imgui.WindowFlags_.no_scrollbar.value | imgui.WindowFlags_.no_scroll_with_mouse.value
        )
        imgui.begin_child("##menu", (MENU_WIDTH + 2 * PANEL_PAD, self._shown), flags, no_scroll)
        imgui.pop_style_var()
        imgui.push_item_width(MENU_WIDTH)
        _draw_title()
        imgui.spacing()
        if self.error:
            imgui.push_text_wrap_pos(imgui.get_cursor_pos_x() + MENU_WIDTH)
            imgui.text_colored(imgui.ImVec4(*GONE), self.error)
            imgui.pop_text_wrap_pos()
        imgui.spacing()
        with bar_buttons():
            choice = {"main": self._draw_main, "new": self._draw_new, "load": self._draw_load}[
                self.page
            ]()
        imgui.pop_item_width()
        content = imgui.get_cursor_pos_y() - imgui.get_style().item_spacing.y + PANEL_PAD
        imgui.end_child()
        imgui.pop_style_color()
        self._shown = grown_towards(self._shown, content, imgui.get_io().delta_time)
        if self.page == "main" or not self._height:
            self._height = content
        frame_rect(
            imgui.get_window_draw_list(),
            imgui.get_item_rect_min(),
            imgui.get_item_rect_max(),
            imgui.get_style().child_rounding,
        )
        draw_version()
        return choice

    def _button(self, label: str) -> bool:
        return imgui.button(label, (MENU_WIDTH, BUTTON_HEIGHT))

    def _fold(self, label: str) -> bool:
        """A heading the width of the menu, in spaced capitals with a caret and a fading rule,
        that opens or folds away what follows it; returns whether it is open."""
        opened = label in self._unfolded
        height = imgui.get_frame_height()
        if imgui.invisible_button(f"##fold{label}", (MENU_WIDTH, height)):
            self._unfolded ^= {label}
            opened = not opened
        low = imgui.get_item_rect_min()
        draw_list = imgui.get_window_draw_list()
        caret = fa.ICON_FA_CARET_DOWN if opened else fa.ICON_FA_CARET_RIGHT
        top = low.y + (height - imgui.get_font_size()) / 2
        colour = BRASS if opened or imgui.is_item_hovered() else TEXT_DIM
        draw_list.add_text((low.x, top), color_u32(colour), caret)
        name = label.upper()
        spaced_text(draw_list, (low.x + 16, top), name, colour, 0.9)
        end = low.x + 16 + spaced(name, 0.9) + 10
        rule(draw_list, end, low.x + MENU_WIDTH, low.y + height / 2, fade_left=False)
        return opened

    def _draw_main(self) -> MenuChoice | None:
        if self._button(f"{fa.ICON_FA_PLAY}  New game"):
            self.go_to("new")
        if self._button(f"{fa.ICON_FA_GRADUATION_CAP}  Tutorial"):
            self.error = ""
            return TutorialChoice()
        if self._button(f"{fa.ICON_FA_FOLDER_OPEN}  Load game"):
            self.go_to("load")
        quit_label = (
            f"{fa.ICON_FA_ARROW_LEFT}  Go back" if IN_BROWSER else f"{fa.ICON_FA_POWER_OFF}  Quit"
        )
        if self._button(quit_label):
            return QuitChoice()
        return None

    def _draw_new(self) -> MenuChoice | None:
        imgui.text("Map")
        if imgui.begin_combo("##map", self.map.value, imgui.ComboFlags_.height_large.value):
            for choice in MapChoice:
                if choice.offered and imgui.selectable(choice.value, self.map == choice)[0]:
                    self.map = choice
            imgui.end_combo()
        imgui.push_text_wrap_pos(imgui.get_cursor_pos_x() + MENU_WIDTH)
        imgui.text_disabled(map_note(self.map))
        imgui.pop_text_wrap_pos()
        imgui.spacing()
        imgui.text("If you don't like any of these:")
        world = world_maps_available()
        imgui.begin_disabled(not world)
        yolo = imgui.button(f"{fa.ICON_FA_RANDOM}  Yolo! random map", (MENU_WIDTH, 0))
        imgui.end_disabled()
        if not world and imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled.value):
            imgui.set_tooltip(
                "Needs the world's map data: uv run python tools/build_world_files.py"
            )
        if yolo:
            self.error = ""
            return NewGameChoice(self.random_settings())
        imgui.spacing()
        _, self.show_map_growing = imgui.checkbox("Show map generation", self.show_map_growing)
        imgui.push_text_wrap_pos(imgui.get_cursor_pos_x() + MENU_WIDTH)
        imgui.text_colored(
            imgui.ImVec4(*BRASS),
            "Shows the map as its towns grow; loading takes about a minute longer.",
        )
        imgui.pop_text_wrap_pos()
        imgui.spacing()
        self.start_year = allowed_start_year(self.start_year)
        if self._fold("Advanced options"):
            self._draw_advanced()
        if self._fold("Cheats"):
            self._draw_cheats()
        imgui.spacing()
        at = imgui.get_cursor_screen_pos()
        middle = at.y + 2
        draw_list = imgui.get_window_draw_list()
        rule(draw_list, at.x, at.x + MENU_WIDTH / 2, middle, fade_left=True)
        rule(draw_list, at.x + MENU_WIDTH / 2, at.x + MENU_WIDTH, middle, fade_left=False)
        imgui.dummy((MENU_WIDTH, 6))
        if self._button(f"{fa.ICON_FA_PLAY}  Start"):
            self.error = ""
            return NewGameChoice(self.settings())
        if self._button("Back"):
            self.go_to("main")
        return None

    def _draw_advanced(self) -> None:
        imgui.text("Seed")
        imgui.set_next_item_width(MENU_WIDTH - 40)
        _, self.seed = imgui.input_int("##seed", self.seed, 0)
        self.seed = min(max(self.seed, 0), SEEDS - 1)
        imgui.same_line(0, 4)
        if imgui.button(fa.ICON_FA_RANDOM, (36, 0)):
            self.seed = random_seed()
        tip("A new random seed")
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

    def _draw_cheats(self) -> None:
        imgui.text_disabled("For trying the game out while it is made.")
        imgui.text("Starting money")
        _, self.cash = imgui.input_int("##cash", self.cash, CASH_STEP, CASH_STEP * 10)
        self.cash = min(max(self.cash, 0), MOST_CASH)
        imgui.text_disabled(f"{dollars(self.cash)}; the usual start is {dollars(START_CASH)}.")
        imgui.text("A stand-in player runs the railway first for (years)")
        _, self.stand_in_years = imgui.slider_int(
            "##stand-in", self.stand_in_years, 0, END_YEAR - self.start_year
        )
        imgui.spacing()

    def _brought_in(self, name: str) -> None:
        self.listing = self.saves.listing()
        self.selected = name

    def _draw_load(self) -> MenuChoice | None:
        self.selected, chosen = draw_save_list(self.listing, self.selected, 320.0)
        loadable = any(info.name == self.selected and info.problem is None for info in self.listing)
        message = draw_file_buttons(
            self.files, self.saves, self.selected, MENU_WIDTH, self._brought_in
        )
        if message:
            self.note = message
        if self.note:
            imgui.push_text_wrap_pos(imgui.get_cursor_pos_x() + MENU_WIDTH)
            imgui.text_disabled(self.note)
            imgui.pop_text_wrap_pos()
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


def _draw_title() -> None:
    """The game's name in big spaced capitals between fading rules."""
    draw_list = imgui.get_window_draw_list()
    at = imgui.get_cursor_screen_pos()
    name = "RAILHUB"
    wide = spaced(name, TITLE_SCALE)
    left = at.x + (MENU_WIDTH - wide) / 2
    spaced_text(draw_list, (left, at.y), name, BRASS, TITLE_SCALE)
    middle = at.y + imgui.get_font_size() * TITLE_SCALE / 2
    rule(draw_list, at.x, left - 14, middle, fade_left=True)
    rule(draw_list, left + wide + 14, at.x + MENU_WIDTH, middle, fade_left=False)
    imgui.dummy((MENU_WIDTH, imgui.get_font_size() * TITLE_SCALE + 4))


def grown_towards(shown: float, wanted: float, seconds: float) -> float:
    """How tall the menu is drawn after `seconds` more of growing or shrinking from `shown`
    towards `wanted`, a share of the way each moment so it eases in, at `wanted` once within
    half a pixel; at once when nothing was shown yet."""
    if not shown:
        return wanted
    shown += (wanted - shown) * min(1.0, seconds * MENU_EASING)
    return wanted if abs(wanted - shown) < 0.5 else shown
