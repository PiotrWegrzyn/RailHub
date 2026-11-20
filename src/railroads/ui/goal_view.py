"""The company's worth against its target in the top bar, the countdown to bankruptcy while the
cash is below nothing, and the window telling the player the target is reached, met or missed,
that the cash went below nothing, or that the company went bankrupt (railroads.sim.company)."""

from typing import TYPE_CHECKING

from imgui_bundle import icons_fontawesome_4 as fa
from imgui_bundle import imgui

from railroads.sim.clock import END_YEAR
from railroads.sim.company import BOOK_SHARE, CompanyValue, Goal, company_value
from railroads.sim.finance import dollars
from railroads.sim.game import Game
from railroads.ui.theme import BRASS, GONE, GROWN, TEXT_DIM, danger_button

if TYPE_CHECKING:
    from railroads.ui.app import App

NEWS_POPUP = "##goal news"
NEWS_WIDTH = 360.0


class Worth:
    """The company's worth now, worked out again only when the cash, the month, the trains or
    the stations change: everything that adds to it is paid for."""

    def __init__(self) -> None:
        self._key: tuple[object, ...] | None = None
        self._value = CompanyValue()

    def of(self, game: Game) -> CompanyValue:
        railway = game.railway
        key = (
            id(railway),
            railway.finances.cash,
            game.growth.months,
            len(railway.trains),
            len(railway.stations()),
        )
        if key != self._key:
            self._key = key
            self._value = company_value(railway)
        return self._value


WORTH = Worth()


def draw_worth(game: Game, target: bool = True) -> None:
    """The worth, and the target unless `target` is off, on the current line, what the worth is
    made of on hover, and how many months are left while the cash is below nothing."""
    goal = game.goal
    value = WORTH.of(game)
    imgui.align_text_to_frame_padding()
    colour = GROWN if value.total >= goal.target else BRASS
    imgui.text_colored(
        imgui.ImVec4(*colour),
        f"{fa.ICON_FA_UNIVERSITY}  {dollars(value.total)}"
        + (f" / {dollars(goal.target)}" if target else ""),
    )
    if imgui.is_item_hovered():
        _worth_tip(goal, value)
    if goal.bankrupt is not None:
        imgui.same_line(0, 10)
        imgui.text_colored(imgui.ImVec4(*GONE), f"{fa.ICON_FA_BAN}  Bankrupt")
    elif game.finances.cash < 0:
        imgui.same_line(0, 10)
        months = goal.months_left
        imgui.text_colored(
            imgui.ImVec4(*GONE),
            f"{fa.ICON_FA_EXCLAMATION_TRIANGLE}  Bankrupt in {months} "
            f"month{'' if months == 1 else 's'}",
        )
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                f"If {months} more month{'' if months == 1 else 's'} begin with your cash below "
                "nothing, the company goes bankrupt and the game ends."
            )


def _worth_tip(goal: Goal, value: CompanyValue) -> None:
    imgui.begin_tooltip()
    imgui.text(f"Company worth: {dollars(value.total)}")
    for name, amount in value.parts():
        imgui.text_colored(imgui.ImVec4(*(TEXT_DIM if amount >= 0 else GONE)), f"   {name}")
        imgui.same_line(110)
        imgui.text(dollars(amount))
    imgui.text_disabled(
        f"Stations, track and trains count for {BOOK_SHARE:.0%} of what they cost today."
    )
    imgui.separator()
    imgui.text(f"{goal.difficulty.value} target for 1 January {END_YEAR}: {dollars(goal.target)}")
    if goal.met is not None:
        imgui.text_colored(
            imgui.ImVec4(*(GROWN if goal.met else GONE)), "Met" if goal.met else "Missed"
        )
    elif goal.reached is not None:
        imgui.text_colored(imgui.ImVec4(*GROWN), f"First reached in {goal.reached}")
    imgui.end_tooltip()


def draw_goal_news(app: App) -> None:
    """The oldest news about the target or bankruptcy, in a window of its own until the player
    closes it. A bankrupt company's window offers only the main menu or a save to load."""
    goal = app.game.goal
    if not goal.news:
        return
    news = goal.news[0]
    if not imgui.is_popup_open(NEWS_POPUP):
        imgui.open_popup(NEWS_POPUP)
    imgui.set_next_window_size((NEWS_WIDTH, 0))
    flags = imgui.WindowFlags_.no_title_bar.value | imgui.WindowFlags_.no_move.value
    if not imgui.begin_popup_modal(NEWS_POPUP, None, flags)[0]:
        return
    bankrupt = news.ends_game
    colour = GONE if news.bad else BRASS
    imgui.text_colored(imgui.ImVec4(*colour), news.title.upper())
    imgui.separator()
    imgui.push_text_wrap_pos(0.0)
    imgui.text(news.text)
    imgui.pop_text_wrap_pos()
    imgui.spacing()
    if bankrupt:
        if danger_button("Main menu"):
            app.menu_wanted = True
            imgui.close_current_popup()
        if app.saves is not None:
            imgui.same_line()
            if imgui.button("Load a save"):
                goal.news.pop(0)
                app.toggle_saves()
                imgui.close_current_popup()
    elif imgui.button("OK", (80, 0)) or imgui.is_key_pressed(imgui.Key.enter):
        goal.news.pop(0)
        imgui.close_current_popup()
    imgui.end_popup()
