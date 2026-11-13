"""Times the game the same way on desktop and in the browser: how long it takes to show its first
frame, how long each chunk of pixel art takes to draw (in the browser's web workers, those that
paint it there), and how long each frame takes while the camera pans steadily across the highest
ground at a pixel-art zoom. The view first rests on the middle of the map, so flat chunks are
timed too.

Desktop: uv run python web/measure.py --poland --seed 7
Browser: index.html?measure runs it in place of the plain game and prints the same summary to
the console."""

import json
import statistics
import sys
import time

import numpy as np
from imgui_bundle import hello_imgui, imgui

from railroads.ui import app
from railroads.ui.pixel.web_painter import Message, WorkerPainter
from railroads.ui.pixel.world import PixelWorld

ZOOM_LEVEL = 4
SETTLE_FRAMES = 120
PAN_FRAMES = 600
PAN_PIXELS = (6.0, 3.0)

started = time.perf_counter()
summary: dict[str, object] = {}
_frames: list[float] = []
_gui_times: list[float] = []
_renders: list[tuple[str, float]] = []
_state = {"frame": 0, "first": 0.0, "last": 0.0}


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    return {
        "count": len(values),
        "mean_ms": round(statistics.fmean(values) * 1000, 1),
        "p50_ms": round(ordered[len(ordered) // 2] * 1000, 1),
        "p95_ms": round(ordered[int(len(ordered) * 0.95)] * 1000, 1),
        "max_ms": round(ordered[-1] * 1000, 1),
    }


_plain_render = PixelWorld.render


def _timed_render(
    self: PixelWorld,
    chunk_x: int,
    chunk_y: int,
    resolution: int = 1,
    fine_relief: bool = False,
    over_ground: bool = False,
):
    before = time.perf_counter()
    pixels = _plain_render(self, chunk_x, chunk_y, resolution, fine_relief, over_ground)
    kind = _kind(self, chunk_x, chunk_y, resolution, fine_relief, over_ground)
    _renders.append((kind, time.perf_counter() - before))
    return pixels


def _kind(
    world: PixelWorld,
    chunk_x: int,
    chunk_y: int,
    resolution: int,
    fine_relief: bool,
    over_ground: bool,
) -> str:
    if fine_relief and resolution > 1:
        kind = "fine raised"
    elif world.rises(chunk_x, chunk_y):
        kind = "raised"
    else:
        kind = "flat over ground" if over_ground else "flat"
    return f"{kind} x{resolution}"


_plain_heard = WorkerPainter.heard


def _timed_heard(self: WorkerPainter, message: Message) -> None:
    """Notes how long a web worker took over a chunk it painted, as it says itself."""
    given = self._given.get(message.get("job"))
    if message["kind"] == "painted" and given is not None:
        job = given[1].work
        kind = _kind(job.world, *job.drawing)
        _renders.append((f"{kind} in a worker", message["ms"] / 1000))
    _plain_heard(self, message)


_plain_gui = app.App.gui


def _timed_gui(self: app.App) -> None:
    now = time.perf_counter()
    frame = _state["frame"]
    panning = 2 * SETTLE_FRAMES < frame <= 2 * SETTLE_FRAMES + PAN_FRAMES
    if frame == 0:
        _state["first"] = now
        summary["first_frame_s"] = round(now - started, 2)
        hello_imgui.get_runner_params().fps_idling.enable_idling = False
        self.view.camera.zoom_level = ZOOM_LEVEL
    elif frame == SETTLE_FRAMES:
        _aim_at_the_highest_ground(self)
    elif panning:
        _frames.append(now - _state["last"])
        self.view.camera.pan(*PAN_PIXELS)
    elif frame == 2 * SETTLE_FRAMES + PAN_FRAMES + 1:
        _report()
    _state["last"] = now
    _state["frame"] = frame + 1
    _plain_gui(self)
    if panning:
        _gui_times.append(time.perf_counter() - now)


def _aim_at_the_highest_ground(game: app.App) -> None:
    """Points the camera so its pan crosses the highest ground halfway through."""
    tile_map = game.view.tile_map
    row, col = np.unravel_index(np.argmax(tile_map.height_m), tile_map.height_m.shape)
    camera = game.view.camera
    camera.center_x, camera.center_y = (float(v) for v in tile_map.grid.center(row, col))
    camera.pan(-PAN_PIXELS[0] * PAN_FRAMES / 2, -PAN_PIXELS[1] * PAN_FRAMES / 2)


def _report() -> None:
    renders: dict[str, list[float]] = {}
    for kind, seconds in _renders:
        renders.setdefault(kind, []).append(seconds)
    io = imgui.get_io()
    summary["platform"] = sys.platform
    summary["window"] = [io.display_size.x, io.display_size.y]
    summary["framebuffer_scale"] = io.display_framebuffer_scale.x
    summary["pan_fps"] = round(len(_frames) / sum(_frames), 1)
    summary["pan_frame"] = _stats(_frames)
    summary["pan_gui"] = _stats(_gui_times)
    summary["chunk_render"] = {kind: _stats(times) for kind, times in sorted(renders.items())}
    print("MEASURE " + json.dumps(summary), flush=True)
    if sys.platform != "emscripten":
        hello_imgui.get_runner_params().app_shall_exit = True


PixelWorld.render = _timed_render
WorkerPainter.heard = _timed_heard
app.App.gui = _timed_gui

if __name__ == "__main__":
    app.main()
