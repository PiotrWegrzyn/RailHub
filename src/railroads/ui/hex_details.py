"""The panel that opens when a cell is clicked: its land, its points, and every building on it with
what it makes, demands and takes as bonus goods, and how last month's trade went for it, what it
must get apart from the bonus goods and boosts it can do without."""

from imgui_bundle import imgui

from railroads.sim.constructions import BuildingKind
from railroads.sim.ownership import LOT, LOTS, MOST_PRICED
from railroads.sim.trade import Satisfaction
from railroads.ui.hex_rows import (
    HELD,
    BuildingRow,
    Flow,
    HexDetails,
    mood_text,
    result_lines,
    sell_lots,
    share_lines,
)
from railroads.ui.ornate import badge, bar, card, section
from railroads.ui.spotlight import spot
from railroads.ui.theme import (
    BRASS,
    GONE,
    GROWN,
    ICON_SIZE,
    TEXT_DIM,
    colored,
    good_amount,
    next_good_slot,
    tip,
)
from railroads.words import dollars

# Where the goods of a card's lines start, after their labels.
CARD_LABEL = 56.0
# How wide the bar after each good on a card's line is, and the gap between goods.
FLOW_BAR = 36.0
SLOT_GAP = 14.0


def draw_hex_details(details: HexDetails) -> BuildingKind | None:
    """Draws the details inside the current imgui window: the land, its points folded away, and
    a card for each kind of building with its satisfaction, what it made and got and the
    player's shares of it; returns the kind whose Shares button was pressed, if any."""
    shares: BuildingKind | None = None
    imgui.text(details.title)
    imgui.text_disabled(details.land)
    if details.points:
        summary = ", ".join(f"{name} {here} / {nearby}" for name, here, nearby in details.points)
        if imgui.tree_node_ex(f"Points: {summary}###points"):
            imgui.text_disabled("here / neighbourhood")
            for name, here, nearby in details.points:
                imgui.bullet_text(f"{name}: {here} / {nearby}")
            imgui.tree_pop()
    else:
        imgui.text_disabled("No points nearby")
    imgui.spacing()
    if not details.buildings and not details.others:
        imgui.text_disabled("Nothing built here")
        return None
    for row in details.buildings:
        if _draw_card(row):
            shares = row.kind
    for other in details.others:
        imgui.bullet_text(other)
    _draw_handed(details)
    return shares


# How many of the places a cell's goods went to its details list, the most first.
HANDED_SHOWN = 8


def _draw_handed(details: HexDetails) -> None:
    """Where what the buildings here made went last month: each good, how much and to whom."""
    if not details.handed:
        return
    section("Handed out to")
    for good, to, amount in details.handed[:HANDED_SHOWN]:
        good_amount(good, f"{amount:,.1f}" if amount < 10 else f"{amount:,.0f}")
        imgui.same_line(0, 8)
        imgui.text_disabled(to)
    hidden = len(details.handed) - HANDED_SHOWN
    if hidden > 0:
        imgui.text_disabled(f"and {hidden} more")


def _draw_card(row: BuildingRow) -> bool:
    """One kind of building as a card: its name and count, its satisfaction, what it made and
    gave away, what it got, and the player's shares with a button opening their window; returns
    whether that was pressed."""
    pressed = False
    with card(f"card{row.name}"):
        pressed = _draw_card_inside(row)
    imgui.spacing()
    return pressed


def _draw_card_inside(row: BuildingRow) -> bool:
    pressed = False
    right = imgui.get_window_width() - imgui.get_style().window_padding.x
    shown = row.makes[0][0] if row.makes else row.needs[0][0] if row.needs else None
    badge(shown)
    imgui.same_line(0, 10)
    imgui.begin_group()
    colored(BRASS, f"{row.name.capitalize()} x {row.count}")
    mood = mood_text(row)
    imgui.same_line()
    imgui.set_cursor_pos_x(
        imgui.get_cursor_pos_x() + imgui.get_content_region_avail().x - imgui.calc_text_size(mood).x
    )
    colored(mood_colour(row), mood)
    if row.percent is not None:
        width = right - imgui.get_cursor_pos_x()
        bar(row.percent / MOST_PRICED, mood_colour(row), width, 5.0)
        tip(
            f"Satisfied {row.percent:.0%} last month: 100% with all it must have, more "
            "with bonus goods"
        )
    imgui.end_group()
    imgui.spacing()
    _draw_line("Made", row.gave, "given away of made")
    if row.needs or row.got:
        _draw_line("Got", row.got, "got of needed")
    if row.bonus_got:
        _draw_line("Bonus", row.bonus_got, "got of wanted")
    if row.stock:
        imgui.text_disabled("In stock")
        for need, amount in row.stock:
            imgui.same_line(0, 8)
            good_amount(need, f"{amount:.0f}")
    if row.price is None:
        imgui.text_disabled("Not for sale")
    else:
        imgui.align_text_to_frame_padding()
        if row.owned:
            imgui.text(f"Yours: {row.owned:,} of {row.total:,} shares")
        else:
            imgui.text_disabled(f"{row.total:,} shares, none yours")
        label = "Shares"
        imgui.same_line(
            right - imgui.calc_text_size(label).x - 2 * imgui.get_style().frame_padding.x
        )
        pressed = imgui.small_button(f"{label}##shares{row.name}")
        spot("buy_shares")
    return pressed


def _draw_line(label: str, flows: list[Flow], meaning: str) -> None:
    """A line of a card: what it is about, then each good's icon, how it went and a bar of it,
    in as many rows of equal slots as the card's width takes, as a station's goods are."""
    imgui.text_disabled(label)
    tip(f"Last month: {meaning}")
    imgui.same_line(CARD_LABEL)
    if not flows:
        imgui.text_disabled("-")
        return
    widest = max(imgui.calc_text_size(flow.text).x for flow in flows)
    slot = ICON_SIZE + 3 + widest + 6 + FLOW_BAR
    per_row = max(1, int((imgui.get_content_region_avail().x + SLOT_GAP) // (slot + SLOT_GAP)))
    left = imgui.get_cursor_pos_x()
    for index, flow in enumerate(flows):
        next_good_slot(index, left, per_row, SLOT_GAP)
        imgui.begin_group()
        good_amount(flow.good, flow.text)
        imgui.same_line(0, 6)
        pad = widest - imgui.calc_text_size(flow.text).x
        if pad > 0:
            imgui.dummy((pad, 1))
            imgui.same_line(0, 0)
        bar(flow.done / flow.wanted if flow.wanted > 0 else 0.0, BRASS, FLOW_BAR, 4.0)
        imgui.end_group()


def mood_colour(row: BuildingRow) -> tuple[float, float, float, float]:
    """Green for a satisfied building, red for one that is not, dim where it does not trade."""
    if row.percent is None:
        return TEXT_DIM
    return GONE if row.satisfaction == Satisfaction.UNSATISFIED.name.lower() else GROWN


def draw_shares(row: BuildingRow, where: str) -> tuple[bool, int] | None:
    """The shares window's contents for one kind of building in a cell: how many shares there
    are, the player's, how many they may sell now, the price of LOT and the buttons to buy and
    sell LOTS of them, and the buildings' money last year and forecast for the next; returns
    whether a Buy or a Sell button was pressed and for how many shares, None if none was."""
    pressed: tuple[bool, int] | None = None
    badge(row.makes[0][0] if row.makes else None)
    imgui.same_line(0, 10)
    imgui.begin_group()
    colored(BRASS, f"{row.name.capitalize()} x {row.count}")
    imgui.text_disabled(f"at {where}")
    imgui.end_group()
    section("Shares")
    if row.total:
        bar(row.owned / row.total, BRASS, imgui.get_content_region_avail().x, 6.0)
        tip(f"Yours: {row.owned:,} of {row.total:,}")
    table = imgui.TableFlags_.sizing_fixed_fit.value
    if imgui.begin_table("shares", 2, table):
        for label, value, hint in share_lines(row):
            imgui.table_next_row()
            imgui.table_next_column()
            imgui.text_disabled(label)
            if hint:
                tip(hint)
            imgui.table_next_column()
            imgui.text(value)
        imgui.end_table()
    if row.price is not None:
        for index, lot in enumerate(LOTS):
            if index:
                imgui.same_line()
            imgui.begin_disabled(row.owned + lot > row.total)
            if imgui.button(f"Buy {lot} for {dollars(row.price * lot / LOT)}"):
                pressed = (True, lot)
            imgui.end_disabled()
            if not index:
                spot("buy_shares")
        for index, lot in enumerate(sell_lots(row.sellable)):
            if index:
                imgui.same_line()
            imgui.begin_disabled(not lot)
            if imgui.button(f"Sell {lot or LOT} for {dollars(row.price * (lot or LOT) / LOT)}"):
                pressed = (False, lot)
            imgui.end_disabled()
            if not lot and row.sell_from is not None:
                tip(f"{HELD}: from {row.sell_from}", disabled=True)
    section("Financial results")
    if imgui.begin_table("results", 3, table | imgui.TableFlags_.borders_inner_h.value):
        imgui.table_setup_column("")
        imgui.table_setup_column("Last year")
        imgui.table_setup_column("Next year")
        imgui.table_next_row()
        for heading in ("", "Last year", "Next year"):
            imgui.table_next_column()
            imgui.text_disabled(heading)
        for label, last, forecast, hint in result_lines(row):
            imgui.table_next_row()
            imgui.table_next_column()
            colour = TEXT_DIM if hint else BRASS
            colored(colour, label)
            if hint:
                tip(hint)
            for value in (last, forecast):
                imgui.table_next_column()
                if tip:
                    imgui.text(value)
                else:
                    colored(BRASS, value)
        imgui.end_table()
    return pressed
