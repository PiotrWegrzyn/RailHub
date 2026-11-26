"""The port window, opened by right clicking a port: its size, the goods it gives and takes with
the tier of each, how much a month and whether drawn or picked, the player's shares, and for a
port the player controls the good a mega port's owner picks and what it ships to their other
ports (railroads.sim.ports)."""

from collections.abc import Callable
from dataclasses import dataclass

from imgui_bundle import icons_fontawesome_4 as fa
from imgui_bundle import imgui

from railroads.sim.constructions import PORT_KINDS, BuildingKind
from railroads.sim.goods import Good
from railroads.sim.grids import Cell
from railroads.sim.growth import Growth
from railroads.sim.ownership import SHARES
from railroads.sim.ports import (
    SHIP_WAGONS,
    SHIPS,
    STORE_MONTHS,
    TIER_COLOURS,
    Port,
    Slot,
    Tier,
)
from railroads.ui.ornate import begin_ornate, end_ornate, keep_in_front, section
from railroads.ui.stations_window import look_at_button
from railroads.ui.theme import BRASS, SIDE_WINDOW_POS, colored, good_icon, rgba_u32, tip

WINDOW_WIDTH = 440.0
AMOUNT_AT = 190.0
SOURCE_AT = 290.0
DOT_RADIUS = 5.0

MERGES_INTO = {
    BuildingKind.PORT: "Two ports in a neighbourhood merge into a big port",
    BuildingKind.BIG_PORT: "Two big ports in a neighbourhood merge into a mega port",
    BuildingKind.MEGA_PORT: "The largest port",
}


def tier_dot(tier: Tier) -> None:
    """A dot in the tier's colour, inline and named on hover."""
    height = imgui.get_text_line_height()
    imgui.dummy((2 * DOT_RADIUS + 2, height))
    low = imgui.get_item_rect_min()
    centre = (low.x + DOT_RADIUS + 1, low.y + height / 2)
    colour = (*(channel / 255 for channel in TIER_COLOURS[tier]), 1.0)
    imgui.get_window_draw_list().add_circle_filled(centre, DOT_RADIUS, rgba_u32(colour))
    tip(f"{tier.value.title()} good")


def port_name(growth: Growth, cell: Cell) -> str:
    kind = growth.ports.kind_at(cell)
    name = kind.value if kind is not None else "port"
    return f"{name[0].upper()}{name[1:]} at {cell[0]}, {cell[1]}"


@dataclass
class PortWindow:
    # The port shown, if the window is open.
    cell: Cell | None = None
    # What the shipping row is set to, and what the last order said.
    _good: int = 0
    _target: int = 0
    _said: str = ""

    def draw(
        self,
        growth: Growth,
        look_at: Callable[[Cell], None],
        shares: Callable[[Cell, BuildingKind], None],
    ) -> None:
        """`shares` opens the shares window of the port's kind in its cell."""
        cell = self.cell
        if cell is None:
            return
        kind = growth.ports.kind_at(cell)
        port = growth.ports.at.get(cell)
        if kind is None or port is None:
            self.cell = None
            return
        imgui.set_next_window_pos(SIDE_WINDOW_POS, imgui.Cond_.first_use_ever.value)
        imgui.set_next_window_size_constraints((WINDOW_WIDTH, 0), (WINDOW_WIDTH, 2000))
        flags = imgui.WindowFlags_.always_auto_resize.value
        title = f"{fa.ICON_FA_ANCHOR}  {port_name(growth, cell)}###port"
        expanded, still_open = begin_ornate(title, flags)
        keep_in_front()
        if expanded:
            self._draw_body(growth, cell, kind, port, look_at, shares)
        end_ornate()
        if not still_open:
            self.cell = None

    def _draw_body(
        self,
        growth: Growth,
        cell: Cell,
        kind: BuildingKind,
        port: Port,
        look_at: Callable[[Cell], None],
        shares: Callable[[Cell, BuildingKind], None],
    ) -> None:
        ports = growth.ports
        imgui.align_text_to_frame_padding()
        colored(BRASS, f"Level {PORT_KINDS.index(kind) + 1} of {len(PORT_KINDS)}")
        imgui.same_line()
        imgui.text_disabled(MERGES_INTO[kind])
        imgui.same_line()
        look_at_button(cell, look_at)
        held = int(growth.owners.shares(kind)[cell])
        imgui.align_text_to_frame_padding()
        imgui.text_disabled(f"You own {held} of {SHARES} shares")
        imgui.same_line()
        if imgui.small_button(f"{fa.ICON_FA_CHART_PIE}  Shares"):
            shares(cell, kind)
        section("Gives")
        for slot in port.gives:
            _good_row(slot.good, slot.tier, f"{ports.amount(slot):,.0f} a month", _source(slot))
        for good, landed in port.landed.items():
            if landed > 0:
                _good_row(good, None, f"{landed:,.0f} landed", "by ship")
        if not port.gives and not port.landed:
            imgui.text_disabled("Nothing")
        section("Takes")
        for slot in port.takes:
            _good_row(slot.good, slot.tier, f"{ports.amount(slot):,.0f} a month", _source(slot))
        for good in port.ships:
            room = STORE_MONTHS * ports.load(good)
            held = port.store.get(good, 0.0)
            _good_row(good, None, f"{held:,.0f} of {room:,.0f}", "in store to ship")
        if not port.takes and not port.ships:
            imgui.text_disabled("Nothing")
        if cell not in ports.controlled():
            imgui.spacing()
            imgui.text_disabled("Control it to pick goods for it and ship them")
            return
        if ports.can_pick(cell):
            self._draw_pick(growth, cell, port)
        self._draw_ships(growth, cell, kind, port)

    def _draw_pick(self, growth: Growth, cell: Cell, port: Port) -> None:
        section("Your pick")
        choices = growth.ports.pick_choices()
        picked = port.pick
        label = "none" if picked is None else f"{picked.good.value} ({picked.tier.value})"
        imgui.set_next_item_width(WINDOW_WIDTH - 40)
        if imgui.begin_combo("##pick", label):
            if imgui.selectable("none", picked is None)[0]:
                growth.ports.pick(cell, None)
            for good, tier in choices:
                if good in port.goods and (picked is None or picked.good != good):
                    continue
                chosen = picked is not None and (picked.good, picked.tier) == (good, tier)
                if imgui.selectable(f"{good.value} ({tier.value})##{tier.value}", chosen)[0]:
                    growth.ports.pick(cell, good, tier)
            imgui.end_combo()
        tip("A good of any tier this mega port gives as well, as much as its tier is worth")

    def _draw_ships(self, growth: Growth, cell: Cell, kind: BuildingKind, port: Port) -> None:
        ports = growth.ports
        most = SHIPS[kind]
        section(f"Ships ({len(port.ships)} of {most})")
        for good, to in list(port.ships.items()):
            imgui.align_text_to_frame_padding()
            good_icon(good, framed=True)
            imgui.same_line()
            imgui.text(f"{good.value} to {port_name(growth, to)}")
            imgui.same_line()
            if imgui.small_button(f"{fa.ICON_FA_TIMES}##stop{good.value}"):
                ports.ship(cell, good, None)
            tip("Stop shipping it")
        targets = [other for other in ports.controlled() if other != cell]
        if not targets:
            imgui.text_disabled("Control another port to ship goods to it")
            return
        if len(port.ships) < most:
            goods = [good for good in ports.ship_choices() if good not in port.ships]
            self._good = min(self._good, len(goods) - 1)
            self._target = min(self._target, len(targets) - 1)
            imgui.set_next_item_width(150)
            _, self._good = imgui.combo("##ship good", self._good, [g.value for g in goods])
            imgui.same_line()
            imgui.set_next_item_width(170)
            names = [port_name(growth, other) for other in targets]
            _, self._target = imgui.combo("##ship to", self._target, names)
            imgui.same_line()
            if imgui.button(f"{fa.ICON_FA_SHIP}  Ship"):
                said = ports.ship(cell, goods[self._good], targets[self._target])
                self._said = said or ""
            tip(
                f"It buys the good from trains into its store, and every month a ship carries\n"
                f"{SHIP_WAGONS} wagons of it to the other port, free, which gives it to trains"
            )
        if self._said:
            imgui.text_disabled(self._said)


def _source(slot: Slot) -> str:
    return "picked" if slot.picked else "drawn"


def _good_row(good: Good, tier: Tier | None, amount: str, source: str) -> None:
    if tier is not None:
        tier_dot(tier)
    else:
        imgui.dummy((2 * DOT_RADIUS + 2, imgui.get_text_line_height()))
    imgui.same_line()
    good_icon(good)
    imgui.same_line()
    imgui.text(good.value)
    imgui.same_line(AMOUNT_AT)
    imgui.text(amount)
    imgui.same_line(SOURCE_AT)
    imgui.text_disabled(source)
