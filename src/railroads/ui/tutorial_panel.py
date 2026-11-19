"""The tutorial's window (railroads.tutorial.steps): the step shown, what it explains and what to
do, with Next once that is done. What to click is framed on the screen and marked on the map
(railroads.ui.spotlight)."""

from imgui_bundle import icons_fontawesome_4 as fa
from imgui_bundle import imgui

from railroads.tutorial.steps import STEPS, Tutorial
from railroads.ui import spotlight
from railroads.ui.buy_train import SCREEN_SIZE as BUY_SCREEN
from railroads.ui.ornate import begin_ornate, end_ornate, keep_in_front
from railroads.ui.theme import BRASS, CARDS_WIDTH, GROWN

# The window sits at first in the column along the right edge that the trains window leaves free,
# MARGIN from the edge, and stays in front of the other windows while no popup is open. Its top
# is at least TOP down, and below the row of buttons along the top of the screen for buying a
# train, which may reach under it: BUTTONS_ROW from that screen's top.
WIDTH = CARDS_WIDTH
MARGIN = 16.0
TOP = 96.0
BUTTONS_ROW = 72.0


def draw_tutorial(tutorial: Tutorial) -> None:
    """Draws the step shown, if the tutorial is open; Next goes on once it is done."""
    step = tutorial.current
    if step is None:
        return
    display = imgui.get_io().display_size
    top = max(TOP, (display.y - BUY_SCREEN[1]) / 2 + BUTTONS_ROW)
    imgui.set_next_window_pos(
        (display.x - MARGIN, top), imgui.Cond_.first_use_ever.value, (1.0, 0.0)
    )
    imgui.set_next_window_size_constraints((WIDTH, 0), (WIDTH, 1e6))
    title = f"{fa.ICON_FA_GRADUATION_CAP}  Tutorial {tutorial.step + 1} of {len(STEPS)}###tutorial"
    flags = (
        imgui.WindowFlags_.no_collapse.value
        | imgui.WindowFlags_.no_saved_settings.value
        | imgui.WindowFlags_.always_auto_resize.value
    )
    expanded, still_open = begin_ornate(title, flags)
    keep_in_front()
    if expanded:
        imgui.push_text_wrap_pos(0.0)
        imgui.text_colored(imgui.ImVec4(*BRASS), step.title)
        imgui.spacing()
        for paragraph in step.text:
            imgui.text(paragraph)
            imgui.spacing()
        if step.task:
            imgui.separator()
            if tutorial.done:
                imgui.text_colored(imgui.ImVec4(*GROWN), f"{fa.ICON_FA_CHECK}  {step.task}")
            else:
                imgui.text_colored(
                    imgui.ImVec4(*BRASS), f"{fa.ICON_FA_HAND_POINT_RIGHT}  {step.task}"
                )
        imgui.pop_text_wrap_pos()
        imgui.spacing()
        last = tutorial.step == len(STEPS) - 1
        imgui.begin_disabled(not tutorial.done)
        if imgui.button("Finish" if last else f"Next  {fa.ICON_FA_ARROW_RIGHT}", (120, 0)):
            if last:
                tutorial.closed = True
            else:
                tutorial.next()
        imgui.end_disabled()
        if step.task and tutorial.done:
            spotlight.frame(imgui.get_item_rect_min(), imgui.get_item_rect_max())
        if step.task and not tutorial.done:
            imgui.same_line()
            if imgui.small_button("Skip this step"):
                tutorial.skip()
    end_ornate()
    if not still_open:
        tutorial.closed = True
