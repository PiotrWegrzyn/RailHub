"""What the buildings around a station want, need by need: how much more of it they take, and
up to COINS coins for how well it pays now, the share of its demand left; hovering one shows the
share and what a unit earns."""

from imgui_bundle import imgui

from railroads.sim.demand import DemandRecord, StationDemand
from railroads.sim.goods import FITS, HANDED_OUT, TRAVELLERS, Need, freight_of
from railroads.sim.trade import SCARCE_WORTH
from railroads.ui.theme import (
    BORDER,
    BRASS,
    BRASS_DARK,
    ICON_SIZE,
    color_u32,
    draw_good_icon,
    good_icon,
    next_good_slot,
)
from railroads.words import dollars

COINS = 3
COIN_SIZE = 10.0
COIN_STEP = 11.5
# How wide one need takes on its row, icon, amount and coins together.
NEED_WIDTH = 116.0
# How far each of a need's icons in draw_wants sits right of the one before, a pile of as many
# as it has coins, and how wide its place is with its amount.
PILE_STEP = 8.0
WANT_WIDTH = ICON_SIZE + PILE_STEP * (COINS - 1) + 44.0


def coins_for(share: float) -> int:
    """How many coins a share of demand left is worth: none once it is nearly used up, then one
    more for every 1 / COINS of it left, up to COINS."""
    if share < 0.05:
        return 0
    return min(COINS, 1 + int(share * COINS))


def draw_demand(demand: StationDemand) -> None:
    wanted = [need for need in Need if demand.capacity.get(need, 0) > 0]
    if not wanted:
        imgui.text_disabled("none")
        return
    left = imgui.get_cursor_pos_x()
    per_row = max(1, int(imgui.get_content_region_avail().x // NEED_WIDTH))
    colors = coin_colors()
    for index, need in enumerate(wanted):
        record = demand.record(need)
        next_good_slot(index, left, per_row)
        imgui.begin_group()
        imgui.align_text_to_frame_padding()
        good_icon(need, framed=True)
        imgui.same_line(0, 3)
        imgui.text(f"{record.left:.0f}")
        imgui.same_line(0, 4)
        _draw_coins(coins_for(record.share), colors)
        imgui.end_group()
        if imgui.is_item_hovered():
            _demand_tooltip(demand, need, record)


def wants_in_order(demand: StationDemand) -> list[tuple[Need, DemandRecord]]:
    """The needs the buildings round a station want, the one they take most of first."""
    wanted = [(need, demand.record(need)) for need in Need if demand.capacity.get(need, 0) > 0]
    return sorted(wanted, key=lambda pair: -pair[1].left)


def draw_wants(demand: StationDemand) -> None:
    """What a station wants, the need taken most of first, each as a pile of its icon, one for
    each coin of how well it pays, with how much more is taken. Hovering one tells how much of
    its demand is left and what it pays."""
    wanted = wants_in_order(demand)
    if not wanted:
        imgui.text_disabled("none")
        return
    left = imgui.get_cursor_pos_x()
    per_row = max(1, int(imgui.get_content_region_avail().x // WANT_WIDTH))
    for index, (need, record) in enumerate(wanted):
        next_good_slot(index, left, per_row)
        imgui.begin_group()
        _draw_pile(need, max(1, coins_for(record.share)))
        imgui.same_line(0, 3)
        imgui.align_text_to_frame_padding()
        imgui.text(f"{record.left:.0f}")
        imgui.end_group()
        if imgui.is_item_hovered():
            _demand_tooltip(demand, need, record)


def _draw_pile(need: Need, count: int) -> None:
    """A need's icon `count` times, each a little right of the one before, in the room of
    COINS of them."""
    height = imgui.get_frame_height()
    imgui.dummy((ICON_SIZE + PILE_STEP * (COINS - 1), height))
    low = imgui.get_item_rect_min()
    top = low.y + (height - ICON_SIZE) / 2
    draw_list = imgui.get_window_draw_list()
    for place in range(count):
        draw_good_icon(draw_list, need, (low.x + place * PILE_STEP, top))


def _draw_coins(lit: int, colors: tuple[int, int, int, int]) -> None:
    """The coins of a need, in the colours of coin_colors."""
    gold, rim, shine, border = colors
    width = COIN_STEP * (COINS - 1) + COIN_SIZE
    height = imgui.get_frame_height()
    imgui.dummy((width, height))
    low = imgui.get_item_rect_min()
    draw_list = imgui.get_window_draw_list()
    radius = COIN_SIZE / 2
    for coin in range(COINS):
        center = (low.x + radius + coin * COIN_STEP, low.y + height / 2)
        if coin < lit:
            draw_list.add_circle_filled(center, radius, gold)
            draw_list.add_circle(center, radius, rim, 0, 1.2)
            draw_list.add_circle_filled((center[0] - 1.2, center[1] - 1.2), radius * 0.35, shine)
        else:
            draw_list.add_circle(center, radius - 0.5, border, 0, 1.0)


def coin_colors() -> tuple[int, int, int, int]:
    """A lit coin's gold, rim and shine, and an unlit one's border, as the style has them now."""
    return (
        color_u32(BRASS),
        color_u32(BRASS_DARK),
        color_u32((1, 0.9, 0.6, 0.8)),
        color_u32(BORDER),
    )


TRAVELLER_NEEDS = frozenset(Need(good.value) for good in TRAVELLERS)


def demand_lines(need: Need, record: DemandRecord) -> list[str]:
    """What a unit delivered for a need earns, good by good. Passengers and mail pay their freight
    times the share of demand left; goods pay their freight, up to SCARCE_WORTH times as much to a
    building that went short, and buildings take the most valuable goods first."""
    if need in TRAVELLER_NEEDS:
        good = FITS[need][0]
        return [f"Each one getting off now earns {dollars(freight_of(good) * record.share)}"]
    lines = [f"Each unit earns its freight, up to {SCARCE_WORTH:g}x where buildings went short;"]
    lines.append("buildings take the most valuable first:")
    for good in HANDED_OUT[need]:
        low = freight_of(good)
        lines.append(f"  {good.value} {dollars(low)} to {dollars(low * SCARCE_WORTH)}")
    return lines


def _demand_tooltip(demand: StationDemand, need: Need, record: DemandRecord) -> None:
    imgui.begin_tooltip()
    imgui.text(
        f"{need.value}: takes about {record.left:.0f} more, {record.share:.0%} of demand left"
    )
    for line in demand_lines(need, record):
        imgui.text(line)
    last = demand.last_month.get(need)
    if last is not None:
        imgui.text_disabled(f"Last month ended with {last.share:.0%} left")
    imgui.end_tooltip()
