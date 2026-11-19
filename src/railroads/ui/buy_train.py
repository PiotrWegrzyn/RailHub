"""The screen a train is bought on: the engines on sale in a list, the one picked shown big as
the map draws it with what it can do, and once it is bought, its route set there and then. A train
bought here stands at its first stop until the screen is closed (Train.held)."""

from collections.abc import Callable
from dataclasses import dataclass

from imgui_bundle import icons_fontawesome_4 as fa
from imgui_bundle import imgui

from railroads.sim.finance import dollars
from railroads.sim.fuel_market import fuel_of
from railroads.sim.railway import Railway
from railroads.sim.trains import ENGINES, EXPRESS_FARE, MAGLEV_FARE, SLOWDOWN, Engine, Kind, Train
from railroads.ui.iso_camera import Vec2
from railroads.ui.ornate import begin_ornate, end_ornate
from railroads.ui.spotlight import spot
from railroads.ui.theme import BRASS, TEXT_DIM, color_u32

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
type DrawRoute = Callable[[Train], None]
# The engines on sale grouped by what they are for, and an engine's tooltip.
type EngineGroups = list[tuple[Kind, list[str]]]
type Summary = Callable[[Engine, float], str]


@dataclass
class BuyScreen:
    open: bool = False
    # The engine shown big.
    engine: str | None = None
    # The train just bought here, whose route is being set; None while an engine is being picked.
    bought: str | None = None
    # The train whose stops a train bought here calls at, if any, and whether it is to.
    route_of: str | None = None
    same_stops: bool = True

    def open_for(self, route_of: Train | None) -> None:
        """Opens the screen to pick an engine, offering the stops of `route_of`."""
        self.open = True
        self.bought = None
        self.route_of = route_of.name if route_of is not None and route_of.stops else None

    def draw(
        self,
        railway: Railway,
        groups: EngineGroups,
        summary: Summary,
        picture: EnginePicture,
        draw_route: DrawRoute,
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
            train = next((t for t in railway.trains if t.name == self.bought), None)
            if train is None:
                self.bought = None
                bought = self._draw_engines(railway, groups, summary, picture)
            else:
                self._draw_route(train, draw_route)
        end_ornate()
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
        if imgui.is_item_hovered():
            imgui.set_tooltip(summary(engine, railway.fuel_cost(name)))
        price = dollars(engine.price)
        affordable = railway.finances.can_afford(engine.price)
        right = imgui.get_item_rect_max().x - imgui.calc_text_size(price).x - 4
        color = color_u32(BRASS if affordable else TEXT_DIM)
        imgui.get_window_draw_list().add_text((right, imgui.get_item_rect_min().y), color, price)

    def _draw_engine(self, railway: Railway, name: str, picture: EnginePicture) -> str | None:
        engine = ENGINES[name]
        self._draw_picture(name, picture)
        imgui.spacing()
        imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(*BRASS))
        imgui.text(engine.title)
        imgui.pop_style_color()
        imgui.separator()
        for label, value in engine_figures(engine, railway.fuel_cost(name)):
            imgui.text_disabled(label)
            imgui.same_line(FIGURES_LABEL)
            imgui.text(value)
        imgui.spacing()
        route_of = next((t for t in railway.trains if t.name == self.route_of), None)
        if route_of is not None:
            _, self.same_stops = imgui.checkbox(
                f"Call at the stops of {route_of.name}", self.same_stops
            )
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
        train = railway.buy(name, route_of=route_of if self.same_stops else None)
        if train is None:
            return None
        train.held = True
        self.bought = train.name
        return train.name

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

    def _draw_route(self, train: Train, draw_route: DrawRoute) -> None:
        imgui.align_text_to_frame_padding()
        imgui.text_disabled("Give it its stops and cars; it sets off once this window is closed.")
        imgui.same_line()
        right = imgui.get_cursor_pos_x() + imgui.get_content_region_avail().x
        imgui.same_line(max(imgui.get_cursor_pos_x(), right - 190))
        if imgui.button(f"{fa.ICON_FA_PLUS}  Buy another"):
            self.bought = None
        imgui.same_line()
        if imgui.button("Done"):
            self.open = False
        spot("done_buying")
        imgui.separator()
        draw_route(train)


def fuel_use(engine: Engine, per_cell: float | None = None) -> str:
    """How much fuel an engine burns a cell, and what that costs now if given."""
    text = f"{engine.burn:.2g} {fuel_of(engine).value} a cell half loaded"
    return text if per_cell is None else f"{text}, about {dollars(per_cell)} a cell now"


def engine_range(engine: Engine) -> str:
    """How far an engine runs on the fuel it takes on setting off."""
    if engine.tank:
        return f"{engine.tank:.0f} cells between stops"
    return "As far as the wires go" if engine.electric else "As far as the guideway goes"


def engine_figures(engine: Engine, per_cell: float | None = None) -> list[tuple[str, str]]:
    """What an engine can do, as labels and values for its table on the screen, with what its
    fuel for a cell costs now if given."""
    cars = f"{engine.cars} coaches and a dining car" if engine.express else f"{engine.cars} cars"
    figures = [
        ("Built from", str(engine.year)),
        ("Power", engine.power.value.capitalize()),
        ("Top speed", f"{engine.kmh:.0f} km/h"),
        ("Pulls up to", cars),
        ("Pulls at most", f"{engine.tonnes:.0f} t, {SLOWDOWN[engine.kind]:.0%} slower at that"),
        ("Up to speed in", f"{engine.pickup:g} s"),
        ("Price", dollars(engine.price)),
        ("Burns", fuel_use(engine, per_cell)),
        ("Range", engine_range(engine)),
    ]
    figures.append(("Cars", f"Up to generation {engine.top_generation}"))
    if engine.express:
        fare = MAGLEV_FARE if engine.maglev else EXPRESS_FARE
        figures.append(("Fares", f"Passengers only, paying {fare - 1:.0%} more"))
    if engine.electric:
        figures.append(("Runs on", "Electrified track only"))
    if engine.maglev:
        figures.append(("Runs on", "Maglev track only"))
    return figures
