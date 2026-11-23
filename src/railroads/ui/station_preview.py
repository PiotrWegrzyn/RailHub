"""What a station reaches (railroads.sim.station_trade.StationPreview): the points of each kind of
building it hands goods out to, what they want of it each month and what is made around it. A card
in the map's top left below the status strip shows it while a station is being placed, and the
station window shows it for a station built."""

from imgui_bundle import imgui

from railroads.sim.station_trade import StationPreview
from railroads.ui.demand_view import draw_demand
from railroads.ui.iso_camera import Vec2
from railroads.ui.ornate import hover_card
from railroads.ui.theme import BRASS, colored, goods_inline
from railroads.words import plural, times_over

PREVIEW_WIDTH = 390.0


def preview_lines(preview: StationPreview) -> list[str]:
    """A line for each kind of building the station would hand goods out to: how many there
    are and the points they give together."""
    lines = []
    for kind, many, points, value in preview.reached:
        total = many * value
        named = times_over(kind.value, many)
        lines.append(f"{named}: {total} {points.value} {plural('point', total)}")
    return lines


def draw_station_preview(preview: StationPreview, top_left: Vec2) -> None:
    with hover_card("##station preview", top_left, PREVIEW_WIDTH) as shown:
        if shown:
            draw_preview_body(preview)


def draw_preview_body(preview: StationPreview, built: bool = False) -> None:
    """What a station reaches, worded for one about to be built or, if `built`, one standing."""
    heading = "It hands goods out to" if built else "A station here hands goods out to"
    colored(BRASS, heading)
    lines = preview_lines(preview)
    if not lines:
        imgui.text_disabled("no buildings")
    for line in lines:
        imgui.text(line)
    imgui.spacing()
    wanted = "They want each month" if built else "They would want each month"
    colored(BRASS, wanted)
    draw_demand(preview.demand)
    imgui.spacing()
    colored(BRASS, "Made around it each month")
    goods_inline(preview.made, "nothing")
