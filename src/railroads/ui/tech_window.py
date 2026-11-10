"""The tech window: a tab for every tree with its level, experience and points, and its nodes in
rows by how many others they build on, to take with a click."""

from dataclasses import dataclass

from imgui_bundle import imgui

from railroads.sim.finance import dollars
from railroads.sim.tech import NODES, TREES, Effect, Node, Research, Tree, xp_for
from railroads.sim.trains import ENGINES
from railroads.ui.theme import BRASS

WINDOW_SIZE = (760.0, 520.0)
WINDOW_TOP = 56.0
NODE_SIZE = (170.0, 44.0)
TAKEN = (0.36, 0.56, 0.3, 1.0)
OPEN = (0.3, 0.34, 0.42, 1.0)
SHUT = (0.18, 0.18, 0.2, 1.0)
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


def describe(node: Node) -> str:
    """What a node does, in a few words."""
    if node.effect == Effect.UNLOCK:
        engine = ENGINES[node.engine]
        return f"opens the {engine.title} (from {engine.year})"
    if node.effect in PERCENT_EFFECTS:
        sign = "-" if node.effect in (Effect.UPKEEP, Effect.DISCOUNT) else "+"
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
        for row in rows_of(tree):
            for index, node in enumerate(row):
                if index:
                    imgui.same_line(0, 8)
                self._draw_node(research, node)

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
        if imgui.is_item_hovered() and node.requires:
            needs = ", ".join(NODES[key][1].name for key in node.requires)
            imgui.set_tooltip(f"Needs {needs}")
