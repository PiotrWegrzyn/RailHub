"""Grows a new game's towns in a web worker in the browser, where Python can't start threads, so
the page doesn't freeze for the twenty-odd seconds that takes and its waiting screen keeps showing
how far it got.

The page starts a worker of its own for each new game (web/index.html, web/painter.js) and stops
it once the game is handed over, so the memory growing took goes with it. The worker makes the
game (railroads.newgame.make_game), pickles and compresses it as a save holds it (grow) and hands
back the compressed bytes; the frames then take them up a few megabytes at a time (PageSetup.grow)
and unpickle the game. Bytes cross between Python and the page by their address in Python's
memory, as the painting workers' do (railroads.ui.pixel.web_painter), since Pyodide's own
conversions go wrong once that memory passes 2 GiB.

Nothing here needs imgui, which the worker doesn't load."""

import pickle
import sys
import zlib
from collections.abc import Callable
from functools import cache
from typing import Any, Protocol

import numpy as np

from railroads.newgame import Settings, make_game
from railroads.saves import SavedGame, snapshot
from railroads.sim.steps import Steps

if sys.platform == "emscripten":
    import js

# Most bytes of the game unpickled on one frame while the frames take up a grown game.
UNPACK_STEP = 8 * 1024 * 1024
# Compressed bytes fed to the unpacking at a time.
FEED_STEP = 1024 * 1024

type Message = dict[str, Any]


class Grown:
    """A grown game pickled and compressed, kept until the page has copied its bytes: where they
    are in Python's memory, how many, and how many the pickle takes once uncompressed."""

    def __init__(self, compressed: bytes, size: int) -> None:
        self._bytes = np.frombuffer(compressed, dtype=np.uint8)
        self.address = _address(self._bytes)
        self.length = len(compressed)
        self.size = size


def grow(settings: bytes, report: Callable[[str], None]) -> Grown:
    """In the worker: the game these pickled settings make, ready to hand back."""
    snap = snapshot(make_game(pickle.loads(settings), report))
    return Grown(zlib.compress(snap.data, 1), len(snap.data))


def unpacked(compressed: np.ndarray, size: int) -> Steps[SavedGame]:
    """The game a worker grew, from its compressed pickle of `size` bytes, uncompressed a few
    megabytes a step and unpickled in the last."""
    pickled = bytearray(size)
    view = memoryview(pickled)
    at = 0
    unpacking = zlib.decompressobj()
    for start in range(0, len(compressed), FEED_STEP):
        fed = compressed[start : start + FEED_STEP].tobytes()
        while fed:
            part = unpacking.decompress(fed, UNPACK_STEP)
            view[at : at + len(part)] = part
            at += len(part)
            fed = unpacking.unconsumed_tail
            yield
    tail = unpacking.flush()
    view[at : at + len(tail)] = tail
    return pickle.loads(pickled)


class Page(Protocol):
    """The page's way to a setup worker: start one growing a game from pickled settings, take
    what it sent since last asked (dicts naming their `kind`; a grown game's bytes as an array),
    and stop it."""

    def start(self, settings: bytes) -> None: ...

    def take(self) -> list[Message]: ...

    def stop(self) -> None: ...


class PageSetup:
    """New games grown in the page's setup worker."""

    def __init__(self, page: Page) -> None:
        self._page = page

    def grow(self, settings: Settings, report: Callable[[str], None]) -> Steps[SavedGame | None]:
        """The game these settings make, grown in a worker while the frames go on, a step a
        frame; None if the worker failed, for the caller to grow it itself."""
        self._page.start(pickle.dumps(settings, protocol=pickle.HIGHEST_PROTOCOL))
        try:
            while True:
                yield
                for message in self._page.take():
                    kind = message["kind"]
                    if kind == "progress":
                        report(message["text"])
                    elif kind in ("failed", "dead"):
                        print(f"The towns couldn't grow in a worker: {message.get('error')}")
                        return None
                    elif kind == "grown":
                        self._page.stop()
                        return (yield from unpacked(message["game"], message["size"]))
        finally:
            self._page.stop()


class _PageObject:
    """The page's own object for its setup worker (window.railroadsSetup), copying bytes from and
    into Python's memory by their address there."""

    def __init__(self, page: Any) -> None:
        self._page = page

    def start(self, settings: bytes) -> None:
        data = np.frombuffer(settings, dtype=np.uint8)
        self._page.start(_address(data), data.size)

    def take(self) -> list[Message]:
        return [self._taken(message) for message in self._page.take()]

    def stop(self) -> None:
        self._page.stop()

    def _taken(self, message: Any) -> Message:
        game = getattr(message, "game", None)
        if game is None:
            return message.to_py()
        del message.game
        taken: Message = message.to_py()
        try:
            bytes_ = np.empty(int(game.byteLength), dtype=np.uint8)
            self._page.copy(game, _address(bytes_))
        except Exception as error:  # noqa: BLE001 - the game is grown on the frames instead
            return {"kind": "failed", "error": f"its game could not be taken: {error}"}
        return taken | {"game": bytes_}


def _address(array: np.ndarray) -> int:
    """Where an array's data starts in Python's memory."""
    return int(array.__array_interface__["data"][0])


@cache
def page_setup() -> PageSetup | None:
    """The page's setup worker, if this runs in a page that can start one."""
    if sys.platform != "emscripten":
        return None
    page = getattr(js, "railroadsSetup", None)
    return None if page is None else PageSetup(_PageObject(page))
