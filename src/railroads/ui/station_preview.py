"""The card shown while a station is being placed (railroads.sim.station_trade.StationPreview),
in the map's top left below the status strip: the points of each kind of building the station
would hand goods out to, what they would want of it each month and what is made around it."""

from imgui_bundle import imgui

from railroads.sim.station_trade import StationPreview
from railroads.ui.demand_view import draw_demand
from railroads.ui.iso_camera import Vec2
from railroads.ui.theme import BRASS, OVERLAY, OVERLAY_FLAGS, goods_inline

PREVIEW_WIDTH = 390.0


def preview_lines(preview: StationPreview) -> list[str]:
    """A line for each kind of building the station would hand goods out to: how many there
    are and the points they give together."""
    lines = []
    for kind, many, points, value in preview.reached:
        total = many * value
        named = kind.value if many == 1 else f"{kind.value} x{many}"
        lines.append(f"{named}: {total} {points.value} point{'' if total == 1 else 's'}")
    return lines


def draw_station_preview(preview: StationPreview, top_left: Vec2) -> None:
    imgui.set_next_window_pos(top_left)
    imgui.set_next_window_bg_alpha(OVERLAY[3])
    imgui.set_next_window_size_constraints((PREVIEW_WIDTH, 0), (PREVIEW_WIDTH, 1e6))
    flags = OVERLAY_FLAGS | imgui.WindowFlags_.no_inputs.value
    if imgui.begin("##station preview", None, flags):
        imgui.text_colored(imgui.ImVec4(*BRASS), "A station here hands goods out to")
        lines = preview_lines(preview)
        if not lines:
            imgui.text_disabled("no buildings")
        for line in lines:
            imgui.text(line)
        imgui.spacing()
        imgui.text_colored(imgui.ImVec4(*BRASS), "They would want each month")
        draw_demand(preview.demand)
        imgui.spacing()
        imgui.text_colored(imgui.ImVec4(*BRASS), "Made around it each month")
        goods_inline(preview.made, "nothing")
    imgui.end()
