"""The tech window: a tab for every tree with its level, experience and points, and its nodes drawn
as a tree of cards, to take with a click: each node in a row by how many others it builds on,
under the nodes it needs, with a line in right angles from each of those to it. A card shows the
node's kind on a round badge, glows once taken and is edged in brass while it can be taken."""

from dataclasses import dataclass

from imgui_bundle import icons_fontawesome_4 as fa
from imgui_bundle import imgui

from railroads.sim.finance import dollars
from railroads.sim.goods import OWNED_BOOSTS
from railroads.sim.ownership import CONTROL
from railroads.sim.tech import NODES, TREES, Effect, Node, Research, Tree, xp_for
from railroads.sim.trains import ENGINES
from railroads.ui.ornate import (
    BADGE,
    CARD,
    CARD_HEAD,
    EDGE,
    GLOW,
    TRACK,
    bar,
    frame_rect,
    ornate_window,
    section,
)
from railroads.ui.theme import BRASS, GROWN, TEXT_DIM, WINDOW_TOP, color_u32

WINDOW_SIZE = (940.0, 560.0)
NODE_SIZE = (200.0, 58.0)
# The space between nodes side by side, and between rows.
NODE_GAP = (16.0, 34.0)
LINK_WIDTH = 1.5
LINK = (0.36, 0.28, 0.18, 1.0)
# The icon on the badge of each kind of node.
EFFECT_ICONS = {
    Effect.PRODUCTION: fa.ICON_FA_INDUSTRY,
    Effect.PRICE: fa.ICON_FA_MONEY_BILL,
    Effect.STORAGE: fa.ICON_FA_ARCHIVE,
    Effect.DISCOUNT: fa.ICON_FA_TAG,
    Effect.SPEED: fa.ICON_FA_TRAIN,
    Effect.PULL: fa.ICON_FA_LINK,
    Effect.UPKEEP: fa.ICON_FA_WRENCH,
    Effect.GROWTH: fa.ICON_FA_LEAF,
    Effect.COACHES: fa.ICON_FA_USERS,
    Effect.WAITING: fa.ICON_FA_HOURGLASS_HALF,
    Effect.BONUS_GOODS: fa.ICON_FA_STAR,
    Effect.FUEL_MARKUP: fa.ICON_FA_FIRE,
    Effect.CAR_SIZE: fa.ICON_FA_EXPAND,
    Effect.UNLOCK: fa.ICON_FA_UNLOCK,
    Effect.TAKES: fa.ICON_FA_PLUS,
    Effect.STATION_UPKEEP: fa.ICON_FA_HOME,
    Effect.FOOD_FUEL: fa.ICON_FA_TRUCK,
    Effect.TO_COME: fa.ICON_FA_GIFT,
}
LOWERING = frozenset(
    {Effect.UPKEEP, Effect.DISCOUNT, Effect.FUEL_MARKUP, Effect.STATION_UPKEEP, Effect.FOOD_FUEL}
)
PERCENT_EFFECTS = frozenset(
    {
        Effect.PRODUCTION,
        Effect.PRICE,
        Effect.DISCOUNT,
        Effect.SPEED,
        Effect.UPKEEP,
        Effect.GROWTH,
        Effect.COACHES,
        Effect.BONUS_GOODS,
        Effect.FUEL_MARKUP,
        Effect.STATION_UPKEEP,
        Effect.FOOD_FUEL,
    }
)


@dataclass(frozen=True)
class TreeProgress:
    level: int
    xp: float
    # Experience the current level started at and the next one needs.
    floor: float
    next: float
    points: int


def progress(research: Research, tree: Tree) -> TreeProgress:
    level = research.level(tree)
    return TreeProgress(
        level, research.xp[tree], xp_for(level), xp_for(level + 1), research.points(tree)
    )


def depth(key: str) -> int:
    """How many nodes stand before this one, along its longest line of requirements."""
    node = NODES[key][1]
    return 1 + max((depth(need) for need in node.requires), default=-1)


def layout(tree: Tree) -> dict[str, tuple[float, int]]:
    """Where each node of a tree stands: its column, in node widths, and its row. Every node hangs
    from the first node it needs, so each branch takes columns of its own, its tips side by side
    and every node above the middle of what hangs from it."""
    nodes = TREES[tree]
    hanging: dict[str, list[str]] = {node.key: [] for node in nodes}
    for node in nodes:
        if node.requires:
            hanging[node.requires[0]].append(node.key)
    columns: dict[str, float] = {}
    free = 0.0

    def place(key: str) -> float:
        nonlocal free
        below = [place(child) for child in hanging[key]]
        if below:
            columns[key] = sum(below) / len(below)
        else:
            columns[key] = free
            free += 1.0
        return columns[key]

    for node in nodes:
        if not node.requires:
            place(node.key)
    return {node.key: (columns[node.key], depth(node.key)) for node in nodes}


def links(tree: Tree) -> list[tuple[str, str]]:
    """Every line of the tree: from each node to each node needing it."""
    return [(need, node.key) for node in TREES[tree] for need in node.requires]


def describe(node: Node) -> str:
    """What a node does, in a few words."""
    if node.effect == Effect.UNLOCK:
        years = [engine.year for engine in ENGINES.values() if node.key in engine.needs]
        return f"opens {node.opens} (from {min(years)})" if years else f"opens {node.opens}"
    if node.effect in PERCENT_EFFECTS:
        sign = "-" if node.effect in LOWERING else "+"
        reach = ", everywhere" if node.everywhere else ""
        return f"{node.effect.value} {sign}{node.amount:.0%}{reach}"
    if node.effect == Effect.TO_COME:
        return "a bonus still to come"
    if node.effect == Effect.TAKES and node.takes:
        kinds = " and ".join(f"{kind.value}s" for kind, _ in node.takes)
        return f"{kinds} you control take {node.takes[0][1].value}"
    if node.effect == Effect.CAR_SIZE:
        reach = "every car" if node.everywhere else "its cars"
        return f"{reach} {node.amount:g} size bigger"
    return f"{node.effect.value} +{node.amount:g}"


def _takes(node: Node) -> str:
    return " ".join(
        f"Every {kind.value} you control, with {CONTROL:.0%} of its shares, takes {need.value} "
        f"and makes up to {OWNED_BOOSTS[(kind, need)]:.0%} more with it."
        for kind, need in node.takes
    )


def explain(tree: Tree, node: Node) -> str:
    """What a node does, in a sentence: whose buildings or goods it reaches and how."""
    owned = "Every building you own" if node.everywhere else f"Every {tree.value} building you own"
    load = "Every load" if node.everywhere else f"Every load of {tree.value}"
    whose = "every building" if node.everywhere else f"every {tree.value} building"
    amount = f"{node.amount:.0%}"
    sentences = {
        Effect.PRODUCTION: f"{owned} makes and takes in {amount} more.",
        Effect.PRICE: f"{load} earns {amount} more.",
        Effect.STORAGE: f"Every station's stockpile holds {node.amount:g} more goods.",
        Effect.DISCOUNT: f"Shares of {whose} cost {amount} less.",
        Effect.SPEED: f"Every train runs {amount} faster.",
        Effect.PULL: f"Every train pulls {node.amount:g} more car.",
        Effect.UPKEEP: f"Stations, track, car trips and fuel cost {amount} less.",
        Effect.GROWTH: f"{owned} counts {amount} more towards growth around it.",
        Effect.COACHES: f"Passenger coaches and mail vans carry {amount} more.",
        Effect.WAITING: f"Every station holds {node.amount:g} more passengers and mail.",
        Effect.BONUS_GOODS: f"Each bonus good adds {amount} more satisfaction.",
        Effect.FUEL_MARKUP: f"Fuel bought outside your stations costs {amount} less over market.",
        Effect.CAR_SIZE: f"{'Every car' if node.everywhere else f'Every car of {tree.value}'} holds "
        f"{node.amount:g} size more.",
        Effect.UNLOCK: f"O{describe(node)[1:]}.",
        Effect.TAKES: _takes(node),
        Effect.STATION_UPKEEP: f"Every station costs {amount} less to keep.",
        Effect.FOOD_FUEL: (
            f"A train pays up to {amount} less for its fuel, in proportion to its cars carrying food."
        ),
        Effect.TO_COME: "A bonus for the buildings you own, still to be decided; it does nothing yet.",
    }
    return sentences[node.effect]


class TechWindow:
    def __init__(self) -> None:
        self.open = False

    def draw(self, research: Research) -> None:
        if not self.open:
            return
        imgui.set_next_window_pos((40.0, WINDOW_TOP), imgui.Cond_.first_use_ever.value)
        imgui.set_next_window_size(WINDOW_SIZE, imgui.Cond_.first_use_ever.value)
        with ornate_window("Tech", "tech") as (expanded, still_open):
            if expanded and imgui.begin_tab_bar("trees"):
                for tree in Tree:
                    done = progress(research, tree)
                    label = f"{tree.value.capitalize()} {done.level}"
                    if done.points:
                        label += f" (+{done.points})"
                    if imgui.begin_tab_item(f"{label}###{tree.name}")[0]:
                        self._draw_tree(research, tree, done)
                        imgui.end_tab_item()
                imgui.end_tab_bar()
        self.open = still_open

    def _draw_tree(self, research: Research, tree: Tree, done: TreeProgress) -> None:
        section(f"{tree.value} tree")
        width = imgui.get_content_region_avail().x
        imgui.text_colored(imgui.ImVec4(*BRASS), f"Level {done.level}")
        imgui.same_line(0, 16)
        points = f"{done.points} point{'s' if done.points != 1 else ''} to spend"
        if done.points:
            imgui.text_colored(imgui.ImVec4(*GROWN), points)
        else:
            imgui.text_disabled(points)
        imgui.same_line(0, 16)
        share = (done.xp - done.floor) / max(done.next - done.floor, 1.0)
        imgui.text_disabled(
            f"{dollars(done.xp)} of {dollars(done.next)} for level {done.level + 1}"
        )
        bar(share, BRASS, width, 6.0)
        imgui.spacing()
        imgui.begin_child("tree", (0, 0), 0, imgui.WindowFlags_.horizontal_scrollbar.value)
        spots = layout(tree)
        origin = imgui.get_cursor_screen_pos()
        width, height = NODE_SIZE
        step_x, step_y = width + NODE_GAP[0], height + NODE_GAP[1]

        def corner(key: str) -> tuple[float, float]:
            column, row = spots[key]
            return origin.x + column * step_x, origin.y + row * step_y

        draw_list = imgui.get_window_draw_list()
        for need, key in links(tree):
            top_x, top_y = corner(need)
            low_x, low_y = corner(key)
            _draw_link(
                draw_list,
                (top_x + width / 2, top_y + height),
                (low_x + width / 2, low_y),
                need in research.taken,
            )
        for node in TREES[tree]:
            imgui.set_cursor_screen_pos(corner(node.key))
            self._draw_node(research, tree, node)
        columns = max(column for column, _ in spots.values()) + 1
        rows = max(row for _, row in spots.values()) + 1
        imgui.set_cursor_screen_pos(origin)
        imgui.dummy((columns * step_x - NODE_GAP[0], rows * step_y - NODE_GAP[1]))
        imgui.end_child()

    @staticmethod
    def _draw_node(research: Research, tree: Tree, node: Node) -> None:
        """A node as a card: a round badge with its kind's icon, its name and what it does,
        glowing once taken, edged in brass while it can be taken and dim while it cannot."""
        taken = node.key in research.taken
        can = research.can_take(node.key)
        if imgui.invisible_button(f"##{node.key}", NODE_SIZE) and can:
            research.take(node.key)
        hovered = imgui.is_item_hovered()
        low, high = imgui.get_item_rect_min(), imgui.get_item_rect_max()
        draw_list = imgui.get_window_draw_list()
        draw_list.add_rect_filled(low, high, color_u32(CARD_HEAD if hovered else CARD), 6.0)
        if taken:
            frame_rect(draw_list, low, high, 6.0, bracket=8.0)
        else:
            edge = EDGE if can else (*EDGE[:3], 0.3)
            draw_list.add_rect(low, high, color_u32(edge), 6.0, thickness=1.5 if can else 1.0)
        middle = low.y + NODE_SIZE[1] / 2
        centre = (low.x + 10 + BADGE / 2, middle)
        draw_list.add_circle_filled(centre, BADGE / 2, color_u32(TRACK))
        ring = (*GLOW, 0.9 if taken or can else 0.3)
        draw_list.add_circle(centre, BADGE / 2, color_u32(ring), 0, 1.2)
        icon = fa.ICON_FA_CHECK if taken else EFFECT_ICONS.get(node.effect, fa.ICON_FA_STAR)
        if not taken and not can:
            icon = fa.ICON_FA_LOCK
        size = imgui.calc_text_size(icon)
        tint = GROWN if taken else BRASS if can else TEXT_DIM
        draw_list.add_text((centre[0] - size.x / 2, centre[1] - size.y / 2), color_u32(tint), icon)
        left = low.x + 18 + BADGE
        line = imgui.get_text_line_height()
        title = BRASS if can or taken else TEXT_DIM
        draw_list.push_clip_rect(low, (high.x - 6, high.y), True)
        draw_list.add_text((left, middle - line), color_u32(title), node.name)
        draw_list.add_text((left, middle + 1), color_u32(TEXT_DIM), describe(node))
        draw_list.pop_clip_rect()
        if hovered:
            text = f"{node.name}: {explain(tree, node)}"
            if node.requires:
                text += "\nNeeds " + ", ".join(NODES[key][1].name for key in node.requires)
            imgui.set_tooltip(text)


def _draw_link(
    draw_list: imgui.ImDrawList, start: tuple[float, float], end: tuple[float, float], lit: bool
) -> None:
    """A line from a node down to one needing it, in right angles with an arrowhead at its end:
    brass once the node it comes from is taken."""
    colour = color_u32((*GLOW, 0.85) if lit else LINK)
    bend = (start[1] + end[1]) / 2
    points = [start, (start[0], bend), (end[0], bend), (end[0], end[1] - 4)]
    draw_list.add_polyline([imgui.ImVec2(*point) for point in points], colour, LINK_WIDTH, 0)
    tip = 5.0
    draw_list.add_triangle_filled(
        (end[0], end[1]), (end[0] - tip, end[1] - tip - 1), (end[0] + tip, end[1] - tip - 1), colour
    )
