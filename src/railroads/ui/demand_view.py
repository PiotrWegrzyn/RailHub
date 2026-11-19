"""What the buildings around a station want, need by need: how much more of it they take, and
up to COINS coins for how well it pays now, the share of its demand left; hovering one shows the
share and what a unit earns."""

from imgui_bundle import imgui

from railroads.sim.demand import DemandRecord, StationDemand
from railroads.sim.economy import SCARCE_WORTH
from railroads.sim.finance import dollars
from railroads.sim.goods import FITS, HANDED_OUT, TRAVELLERS, Need, freight_of
from railroads.ui.theme import BORDER, BRASS, BRASS_DARK, color_u32, good_icon, next_good_slot

COINS = 3
COIN_SIZE = 10.0
COIN_STEP = 11.5
# How wide one need takes on its row, icon, amount and coins together.
NEED_WIDTH = 116.0


def coins_for(share: float) -> int:
    """How many coins a share of demand left is worth: none once it is nearly used up, one, two
    from a third left, and all three from two thirds."""
    if share < 0.05:
        return 0
    return 3 if share >= 2 / 3 else 2 if share >= 1 / 3 else 1


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
        return [
            f"Takes about {record.left:.0f} more before it pays nothing",
            f"Each one getting off now earns {dollars(freight_of(good) * record.share)}",
        ]
    lines = [f"Each unit earns its freight, up to {SCARCE_WORTH:g}x where buildings went short;"]
    lines.append("buildings take the most valuable first:")
    for good in HANDED_OUT[need]:
        low = freight_of(good)
        lines.append(f"  {good.value} {dollars(low)} to {dollars(low * SCARCE_WORTH)}")
    return lines


def _demand_tooltip(demand: StationDemand, need: Need, record: DemandRecord) -> None:
    imgui.begin_tooltip()
    imgui.text(f"{need.value}: {record.share:.0%} of demand left")
    for line in demand_lines(need, record):
        imgui.text(line)
    last = demand.last_month.get(need)
    if last is not None:
        imgui.text_disabled(f"Last month ended with {last.share:.0%} left")
    imgui.end_tooltip()
