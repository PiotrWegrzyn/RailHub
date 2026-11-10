"""The trains window: every train in a list on the left; the chosen one's cars and route, stop by
stop with the cars it pulls on to the next, on the right."""

from dataclasses import dataclass

from imgui_bundle import icons_fontawesome_4 as fa
from imgui_bundle import imgui

from railroads.sim.finance import TRAIN_RESALE, dollars
from railroads.sim.goods import GOOD_YEARS, Good
from railroads.sim.railway import Railway
from railroads.sim.trains import ENGINES, Stop, Train
from railroads.ui.theme import BRASS, CARDS_WIDTH, TEXT_DIM, car_chip, danger_button, good_icon

WINDOW_SIZE = (660.0, 300.0)
WINDOW_TOP = 96.0
LIST_WIDTH = 190.0


@dataclass
class TrainsWindow:
    open: bool = False
    # The name of the train shown on the right.
    chosen: str | None = None

    def draw(self, railway: Railway) -> None:
        if not self.open:
            return
        display = imgui.get_io().display_size
        imgui.set_next_window_pos(
            (display.x - WINDOW_SIZE[0] - CARDS_WIDTH - 16, WINDOW_TOP),
            imgui.Cond_.first_use_ever.value,
        )
        imgui.set_next_window_size(WINDOW_SIZE, imgui.Cond_.first_use_ever.value)
        expanded, self.open = imgui.begin(f"{fa.ICON_FA_TRAIN}  Trains###trains", self.open)
        if expanded:
            train = self._draw_list(railway)
            imgui.same_line()
            imgui.begin_child("train", (0, 0))
            if train is None:
                _centered_hint("Buy a train, then give it two stations joined by track.")
            else:
                self._draw_train(railway, train)
            imgui.end_child()
        imgui.end()

    def _draw_list(self, railway: Railway) -> Train | None:
        imgui.begin_child("list", (LIST_WIDTH, 0), imgui.ChildFlags_.borders.value)
        if imgui.button(f"{fa.ICON_FA_PLUS}  Buy a train", (-1, 0)):
            imgui.open_popup("buy train")
        if imgui.begin_popup("buy train"):
            imgui.text_disabled("Buy a train with")
            for name in railway.engines():
                engine = ENGINES[name]
                affordable = railway.finances.can_afford(engine.price)
                imgui.begin_disabled(not affordable)
                label = f"{engine.title}  {dollars(engine.price)}##{name}"
                if imgui.selectable(label, False)[0]:
                    bought = railway.buy(name)
                    if bought is not None:
                        self.chosen = bought.name
                imgui.end_disabled()
                if imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled.value):
                    track = ", electrified track only" if engine.electric else ""
                    money = "" if affordable else " - not enough money"
                    imgui.set_tooltip(
                        f"{engine.speed:g}x speed, up to {engine.cars} cars{track}{money}"
                    )
            imgui.end_popup()
        imgui.spacing()
        chosen = None
        for train in railway.trains:
            picked = train.name == self.chosen
            if imgui.selectable(f"##{train.name}", picked, 0, (0, 40))[0]:
                self.chosen = train.name
            low = imgui.get_item_rect_min()
            draw_list = imgui.get_window_draw_list()
            draw_list.add_text(
                (low.x + 6, low.y + 3), imgui.get_color_u32(imgui.Col_.text), train.name
            )
            status_color = imgui.get_color_u32(imgui.ImVec4(*TEXT_DIM))
            draw_list.add_text((low.x + 6, low.y + 21), status_color, train.status)
            if picked:
                chosen = train
        imgui.end_child()
        if chosen is None and railway.trains and self.chosen is None:
            self.chosen = railway.trains[0].name
        return chosen

    def _draw_train(self, railway: Railway, train: Train) -> None:
        imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(*BRASS))
        imgui.text(train.name)
        imgui.pop_style_color()
        imgui.same_line()
        imgui.text_disabled(f"{ENGINES[train.engine].title} - {train.status}")
        width = imgui.get_content_region_avail().x
        imgui.same_line(max(0.0, imgui.get_cursor_pos_x() + width - 90))
        sold = danger_button(f"{fa.ICON_FA_TRASH}  Sell")
        if imgui.is_item_hovered():
            imgui.set_tooltip(f"Sell it for {dollars(ENGINES[train.engine].price * TRAIN_RESALE)}")
        if sold:
            railway.sell(train)
            self.chosen = None
            return
        imgui.separator()
        imgui.align_text_to_frame_padding()
        imgui.text_disabled("Pulling")
        if not train.cars:
            imgui.same_line()
            imgui.text_disabled("no cars")
        for index, car in enumerate(train.cars):
            imgui.same_line(0, 4 if index else 12)
            car_chip(
                f"pulled{index}",
                car.good,
                car.load,
                f"{car.good.value}: {car.load:.0f}/{car.capacity:.0f}",
            )
        imgui.spacing()
        imgui.text_disabled("Route")
        self._draw_stops(railway, train)

    def _draw_stops(self, railway: Railway, train: Train) -> None:
        flags = (
            imgui.TableFlags_.row_bg.value
            | imgui.TableFlags_.borders_inner_h.value
            | imgui.TableFlags_.sizing_fixed_fit.value
        )
        if train.stops and imgui.begin_table("stops", 5, flags):
            imgui.table_setup_column("#")
            imgui.table_setup_column("Station")
            imgui.table_setup_column(
                "Cars to the next stop", imgui.TableColumnFlags_.width_stretch.value
            )
            imgui.table_setup_column("Wait")
            imgui.table_setup_column("")
            imgui.table_headers_row()
            for index, stop in enumerate(list(train.stops)):
                imgui.push_id(index)
                if self._draw_stop(railway, train, index, stop):
                    imgui.pop_id()
                    break
                imgui.pop_id()
            imgui.end_table()
        choices = [s for s in railway.stations() if not train.stops or train.stops[-1].cell != s]
        if not choices:
            imgui.text_disabled("Build stations to give it stops.")
            return
        imgui.set_next_item_width(220)
        if imgui.begin_combo("##add stop", f"{fa.ICON_FA_PLUS}  Add a stop"):
            for station in choices:
                if imgui.selectable(railway.name_of(station), False)[0]:
                    cars = list(train.stops[-1].cars) if train.stops else []
                    train.stops.append(Stop(station, cars))
                    train.restart()
            imgui.end_combo()

    def _draw_stop(self, railway: Railway, train: Train, index: int, stop: Stop) -> bool:
        """One stop as a table row; returns whether it was removed."""
        name = railway.name_of(stop.cell)
        imgui.table_next_row(0, 34)
        imgui.table_next_column()
        imgui.align_text_to_frame_padding()
        stops = len(train.stops)
        current = train.heading % stops if train.path else (train.heading - 1) % stops
        imgui.text_disabled(f"{index + 1}")
        imgui.table_next_column()
        imgui.align_text_to_frame_padding()
        if index == current and stops > 1:
            imgui.text_colored(imgui.ImVec4(*BRASS), f"{fa.ICON_FA_TRAIN} {name}")
            if imgui.is_item_hovered():
                imgui.set_tooltip("Heading here" if train.path else "Standing here")
        else:
            imgui.text(name)
        imgui.table_next_column()
        for car_index, good in enumerate(list(stop.cars)):
            if car_index:
                imgui.same_line(0, 4)
            if car_chip(f"car{car_index}", good, tooltip=f"{good.value} car - click to remove"):
                stop.cars.pop(car_index)
                train.cars_changed()
        most = railway.most_cars(train)
        if len(stop.cars) < most:
            if stop.cars:
                imgui.same_line(0, 8)
            if imgui.small_button(f"{fa.ICON_FA_PLUS}##add car"):
                imgui.open_popup("add car")
            if imgui.is_item_hovered():
                imgui.set_tooltip(f"Add a car ({len(stop.cars)}/{most})")
        if imgui.begin_popup("add car"):
            imgui.text_disabled("Add a car for")
            for good in Good:
                if GOOD_YEARS.get(good, 0) > railway.year:
                    continue
                good_icon(good)
                imgui.same_line(0, 6)
                if imgui.selectable(good.value, False)[0] and stop.add_car(good, most):
                    train.cars_changed()
            imgui.end_popup()
        imgui.table_next_column()
        _, stop.wait_for_load = imgui.checkbox("##wait", stop.wait_for_load)
        if imgui.is_item_hovered():
            imgui.set_tooltip("Wait here, up to a month, for every car to fill")
        imgui.table_next_column()
        if imgui.small_button(fa.ICON_FA_TIMES):
            train.stops.pop(index)
            train.restart()
            return True
        if imgui.is_item_hovered():
            imgui.set_tooltip("Remove this stop")
        return False


def _centered_hint(text: str) -> None:
    width, height = imgui.get_content_region_avail()
    size = imgui.calc_text_size(text)
    imgui.set_cursor_pos(
        (
            imgui.get_cursor_pos_x() + max(0.0, (width - size.x) / 2),
            imgui.get_cursor_pos_y() + height / 2 - size.y,
        )
    )
    imgui.text_disabled(text)
