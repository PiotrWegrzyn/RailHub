"""The screen a train is bought on: the engines on sale in a list, the one picked shown big as
the map draws it with what it can do, and the schedule it is to run, a new one of its own or one
other trains run (railroads.sim.schedules), with a warning when that engine can't run it. A train
bought here sets off at once and the screen closes."""

from collections.abc import Callable
from dataclasses import dataclass, field

from imgui_bundle import icons_fontawesome_4 as fa
from imgui_bundle import imgui

from railroads.catalogue import engine_figures
from railroads.sim.railway import Railway
from railroads.sim.schedules import Schedule
from railroads.sim.trains import (
    ENGINES,
    OPENED_BY,
    Engine,
    Kind,
)
from railroads.ui.iso_camera import Vec2
from railroads.ui.ornate import begin_ornate, end_ornate
from railroads.ui.spotlight import spot
from railroads.ui.theme import BRASS, GONE, TEXT_DIM, color_u32, colored, tip
from railroads.words import dollars

SCREEN_SIZE = (860.0, 560.0)
ENGINES_WIDTH = 250.0
# How tall the engine's picture may be drawn, and how many times its art pixels it is blown up
# at the most.
PICTURE_HEIGHT = 190.0
MOST_BLOW_UP = 8
# Where the values in the table of an engine's figures start, after their labels.
FIGURES_LABEL = 130.0
PICTURE_BACK = (0.08, 0.07, 0.06, 1.0)

# An engine's picture (railroads.ui.pixel.train_layer.engine_picture) as a texture drawn so many
# times as fine each way, and its size in the sprite sheet's pixels.
type EnginePicture = Callable[[str, int], tuple[imgui.ImTextureRef, Vec2]]
# The engines on sale grouped by what they are for, and an engine's tooltip.
type EngineGroups = list[tuple[Kind, list[str]]]
type Summary = Callable[[Engine, float], str]


@dataclass
class BuyScreen:
    open: bool = False
    # The engine shown big.
    engine: str | None = None
    # The schedule a train bought here runs; None for a new one of its own.
    schedule: str | None = None
    # Why an engine can't run a schedule, by schedule, its version, what its refuelling was
    # planned for and engine, so it is worked out once rather than every frame.
    _cannot: dict[tuple, str] = field(default_factory=dict)

    def open_for(self, schedule: Schedule | None) -> None:
        """Opens the screen to pick an engine, for a train running `schedule` if it has stops."""
        self.open = True
        self.schedule = schedule.name if schedule is not None and schedule.stops else None

    def draw(
        self,
        railway: Railway,
        groups: EngineGroups,
        summary: Summary,
        picture: EnginePicture,
    ) -> str | None:
        """Draws the screen; returns the name of a train bought on it this frame."""
        if not self.open:
            return None
        display = imgui.get_io().display_size
        imgui.set_next_window_pos(
            (display.x / 2, display.y / 2), imgui.Cond_.appearing.value, (0.5, 0.5)
        )
        imgui.set_next_window_size(SCREEN_SIZE, imgui.Cond_.first_use_ever.value)
        title = f"{fa.ICON_FA_PLUS}  Buy a train###buy train"
        expanded, self.open = begin_ornate(title, imgui.WindowFlags_.no_collapse.value)
        bought = None
        if expanded:
            bought = self._draw_engines(railway, groups, summary, picture)
        end_ornate()
        if bought is not None:
            self.open = False
        return bought

    def _draw_engines(
        self, railway: Railway, groups: EngineGroups, summary: Summary, picture: EnginePicture
    ) -> str | None:
        names = [name for _, group in groups for name in group]
        if not names:
            imgui.text_disabled("No engine is on sale yet.")
            return None
        if self.engine not in names:
            self.engine = names[0]
        imgui.begin_child("engines", (ENGINES_WIDTH, 0), imgui.ChildFlags_.borders.value)
        for kind, group in groups:
            imgui.separator_text(kind.value.capitalize())
            for name in group:
                self._engine_row(railway, name, summary)
        imgui.end_child()
        imgui.same_line()
        imgui.begin_child("engine", (0, 0))
        bought = self._draw_engine(railway, self.engine, picture)
        imgui.end_child()
        return bought

    def _engine_row(self, railway: Railway, name: str, summary: Summary) -> None:
        engine = ENGINES[name]
        if imgui.selectable(f"{engine.title}##{name}", self.engine == name)[0]:
            self.engine = name
        spot(f"engine:{name}")
        tip(summary(engine, railway.fuel_cost(name)))
        price = dollars(engine.price)
        affordable = railway.finances.can_afford(engine.price)
        right = imgui.get_item_rect_max().x - imgui.calc_text_size(price).x - 4
        color = color_u32(BRASS if affordable else TEXT_DIM)
        imgui.get_window_draw_list().add_text((right, imgui.get_item_rect_min().y), color, price)

    def _draw_engine(self, railway: Railway, name: str, picture: EnginePicture) -> str | None:
        """The engine picked: its picture and figures, which scroll on their own, over the
        schedule it is to run and the Buy button, which stay at the bottom in sight."""
        engine = ENGINES[name]
        schedules = railway.schedules
        schedule = None if self.schedule is None else schedules.named(self.schedule)
        cannot = "" if schedule is None else self.cannot_run(railway, schedule, name)
        style = imgui.get_style()
        footer = 2 * imgui.get_frame_height_with_spacing() + style.item_spacing.y
        if cannot:
            footer += imgui.get_text_line_height_with_spacing()
        imgui.begin_child("figures", (0, -footer))
        self._draw_figures(railway, name, picture)
        imgui.end_child()
        imgui.align_text_to_frame_padding()
        imgui.text_disabled("Runs")
        imgui.same_line()
        imgui.set_next_item_width(320)
        new = "A new schedule of its own"
        if imgui.begin_combo("##runs", new if schedule is None else schedules.title(schedule)):
            if imgui.selectable(new, schedule is None)[0]:
                self.schedule = None
            for other in schedules.all:
                if (
                    other.stops
                    and imgui.selectable(
                        f"{schedules.title(other)}##{other.name}", other is schedule
                    )[0]
                ):
                    self.schedule = other.name
            imgui.end_combo()
        tip("The schedule it runs: its stops and the cars taken on at each, shared by its trains")
        if cannot:
            colored(GONE, f"{fa.ICON_FA_EXCLAMATION_TRIANGLE}  It would stand: {cannot}")
        affordable = railway.finances.can_afford(engine.price)
        imgui.begin_disabled(not affordable)
        clicked = imgui.button(f"{fa.ICON_FA_SHOPPING_CART}  Buy for {dollars(engine.price)}")
        spot("buy_engine")
        imgui.end_disabled()
        if not affordable:
            imgui.same_line()
            imgui.text_disabled("Not enough money")
        if not clicked:
            return None
        train = railway.buy(name, schedule=schedule)
        return None if train is None else train.name

    def cannot_run(self, railway: Railway, schedule: Schedule, engine: str) -> str:
        """Why a train behind this engine would stand rather than run the schedule; "" if it
        would run."""
        key = (schedule.name, schedule.version, schedule.planned_for, schedule.manual, engine)
        if key not in self._cannot:
            if len(self._cannot) > 64:
                self._cannot.clear()
            self._cannot[key] = railway.schedules.cannot_run(schedule, engine)
        return self._cannot[key]

    @staticmethod
    def _draw_figures(railway: Railway, name: str, picture: EnginePicture) -> None:
        """The engine's picture, its name and the table of what it can do."""
        engine = ENGINES[name]
        BuyScreen._draw_picture(name, picture)
        imgui.spacing()
        imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(*BRASS))
        imgui.text(engine.title)
        imgui.pop_style_color()
        imgui.separator()
        for label, value in engine_figures(engine, railway.fuel_cost(name), OPENED_BY.get(name)):
            imgui.text_disabled(label)
            imgui.same_line(FIGURES_LABEL)
            imgui.text(value)

    @staticmethod
    def _draw_picture(name: str, picture: EnginePicture) -> None:
        """The engine big, as many whole times as fit, drawn fine on a dark band."""
        density = max(1, round(imgui.get_io().display_framebuffer_scale.x))
        _, (width, height) = picture(name, density)
        room = imgui.get_content_region_avail().x
        times = max(1, min(MOST_BLOW_UP, int(room // width), int(PICTURE_HEIGHT // height)))
        texture, (width, height) = picture(name, times * density)
        low = imgui.get_cursor_screen_pos()
        band = PICTURE_HEIGHT + 16
        draw_list = imgui.get_window_draw_list()
        draw_list.add_rect_filled(low, (low.x + room, low.y + band), color_u32(PICTURE_BACK), 6.0)
        left = low.x + (room - width * times) / 2
        top = low.y + (band - height * times) / 2
        draw_list.add_image(texture, (left, top), (left + width * times, top + height * times))
        imgui.dummy((room, band))
