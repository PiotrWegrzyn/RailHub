"""The tech window: a tab for every tree with its level, experience and points, and its nodes drawn
as a tree, to take with a click: each node in a row by how many others it builds on, under the
nodes it needs, with a line from each of those to it."""

from dataclasses import dataclass

from imgui_bundle import imgui

from railroads.sim.finance import dollars
from railroads.sim.tech import NODES, TREES, Effect, Node, Research, Tree, xp_for
from railroads.sim.trains import ENGINES
from railroads.ui.theme import BRASS

WINDOW_SIZE = (940.0, 560.0)
WINDOW_TOP = 56.0
NODE_SIZE = (170.0, 44.0)
# The space between nodes side by side, and between rows.
NODE_GAP = (14.0, 34.0)
LINK_WIDTH = 2.0
TAKEN = (0.36, 0.56, 0.3, 1.0)
OPEN = (0.3, 0.34, 0.42, 1.0)
SHUT = (0.18, 0.18, 0.2, 1.0)
LINK = (0.45, 0.45, 0.5, 1.0)
PERCENT_EFFECTS = frozenset(
    {
        Effect.PRODUCTION,
        Effect.SHARE,
        Effect.PRICE,
        Effect.DISCOUNT,
        Effect.SPEED,
        Effect.UPKEEP,
        Effect.GROWTH,
        Effect.COACHES,
        Effect.BONUS_GOODS,
        Effect.FUEL_MARKUP,
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


def rows_of(tree: Tree) -> list[list[Node]]:
    """A tree's nodes in rows: those needing nothing first, then each row building on the last."""
    rows: list[list[Node]] = []
    for node in TREES[tree]:
        at = depth(node.key)
        while len(rows) <= at:
            rows.append([])
        rows[at].append(node)
    return rows


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
        sign = "-" if node.effect in (Effect.UPKEEP, Effect.DISCOUNT, Effect.FUEL_MARKUP) else "+"
        reach = ", everywhere" if node.everywhere else ""
        return f"{node.effect.value} {sign}{node.amount:.0%}{reach}"
    unit = " cars" if node.effect == Effect.PULL else ""
    return f"{node.effect.value} +{node.amount:g}{unit}"


class TechWindow:
    def __init__(self) -> None:
        self.open = False

    def draw(self, research: Research) -> None:
        if not self.open:
            return
        imgui.set_next_window_pos((40.0, WINDOW_TOP), imgui.Cond_.first_use_ever.value)
        imgui.set_next_window_size(WINDOW_SIZE, imgui.Cond_.first_use_ever.value)
        expanded, self.open = imgui.begin("Tech", self.open)
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
        imgui.end()

    def _draw_tree(self, research: Research, tree: Tree, done: TreeProgress) -> None:
        imgui.text_colored(imgui.ImVec4(*BRASS), f"Level {done.level}")
        imgui.same_line(0, 16)
        imgui.text(f"{done.points} point{'s' if done.points != 1 else ''} to spend")
        share = (done.xp - done.floor) / max(done.next - done.floor, 1.0)
        imgui.progress_bar(
            min(max(share, 0.0), 1.0),
            (-1.0, 0.0),
            f"{dollars(done.xp)} of {dollars(done.next)} for level {done.level + 1}",
        )
        imgui.separator()
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
            start = imgui.ImVec2(top_x + width / 2, top_y + height)
            end = imgui.ImVec2(low_x + width / 2, low_y)
            bend = (end.y - start.y) / 2
            color = BRASS if need in research.taken else LINK
            draw_list.add_bezier_cubic(
                start,
                imgui.ImVec2(start.x, start.y + bend),
                imgui.ImVec2(end.x, end.y - bend),
                end,
                imgui.get_color_u32(imgui.ImVec4(*color)),
                LINK_WIDTH,
            )
        for node in TREES[tree]:
            imgui.set_cursor_screen_pos(corner(node.key))
            self._draw_node(research, node)
        columns = max(column for column, _ in spots.values()) + 1
        rows = max(row for _, row in spots.values()) + 1
        imgui.set_cursor_screen_pos(origin)
        imgui.dummy((columns * step_x - NODE_GAP[0], rows * step_y - NODE_GAP[1]))
        imgui.end_child()

    @staticmethod
    def _draw_node(research: Research, node: Node) -> None:
        taken = node.key in research.taken
        can = research.can_take(node.key)
        color = TAKEN if taken else OPEN if can else SHUT
        imgui.push_style_color(imgui.Col_.button.value, imgui.ImVec4(*color))
        if not can and not taken:
            imgui.push_style_var(imgui.StyleVar_.alpha.value, 0.6)
        if imgui.button(f"{node.name}\n{describe(node)}##{node.key}", NODE_SIZE) and can:
            research.take(node.key)
        if not can and not taken:
            imgui.pop_style_var()
        imgui.pop_style_color()
        if imgui.is_item_hovered():
            text = f"{node.name}: {describe(node)}"
            if node.requires:
                text += "\nNeeds " + ", ".join(NODES[key][1].name for key in node.requires)
            imgui.set_tooltip(text)
