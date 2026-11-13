import ctypes
import pickle
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from test_saves import played_game

from railroads.newgame import MapChoice, Settings
from railroads.saves import SavedGame
from railroads.sim.steps import finish
from railroads.ui import web_setup
from railroads.ui.web_setup import PageSetup, unpacked

SETTINGS = Settings(seed=7, map=MapChoice.POLAND, start_year=1850)


class FakePage:
    """A setup worker as the page runs one: it grows the game when told to work and its bytes
    are copied out of Python's memory by their address, as web/painter.js copies them."""

    def __init__(self, fails: bool = False) -> None:
        self.fails = fails
        self.settings: bytes | None = None
        self.inbox: list[dict[str, Any]] = []
        self.stopped = 0

    def start(self, settings: bytes) -> None:
        self.settings = settings

    def work(self) -> None:
        assert self.settings is not None
        if self.fails:
            self.inbox.append({"kind": "dead", "error": "out of memory"})
            return
        report = lambda text: self.inbox.append({"kind": "progress", "text": text})
        advance = lambda done: self.inbox.append({"kind": "advance", "done": done})
        grown = web_setup.grow(self.settings, report, advance)
        game = np.frombuffer(ctypes.string_at(grown.address, grown.length), dtype=np.uint8)
        self.inbox.append({"kind": "grown", "game": game, "size": grown.size})

    def take(self) -> list[dict[str, Any]]:
        taken, self.inbox = self.inbox, []
        return taken

    def stop(self) -> None:
        self.stopped += 1


def grows_as(saved: SavedGame) -> Callable[..., SavedGame]:
    def made(
        settings: Settings,
        report: Callable[[str], None],
        progress: Callable[[float], None] = lambda done: None,
    ) -> SavedGame:
        report(f"Growing towns for seed {settings.seed}")
        progress(0.5)
        return saved

    return made


def test_a_game_grown_in_a_worker_comes_back_as_it_was_grown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grown = played_game()
    monkeypatch.setattr(web_setup, "make_game", grows_as(grown))
    page = FakePage()
    progress: list[str] = []
    steps = PageSetup(page).grow(SETTINGS, progress.append)
    next(steps)
    page.work()

    saved = finish(steps)

    assert saved is not None
    assert progress == ["Growing towns for seed 7"]
    assert pickle.loads(page.settings or b"") == SETTINGS
    assert saved.clock == grown.clock
    assert saved.game.finances.cash == grown.game.finances.cash
    assert saved.game.railway.stations() == grown.game.railway.stations()
    assert page.stopped >= 1


def test_the_page_hears_how_far_a_worker_has_grown_the_towns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(web_setup, "make_game", grows_as(played_game()))
    page = FakePage()
    shares: list[float] = []
    steps = PageSetup(page).grow(SETTINGS, print, shares.append)
    next(steps)
    page.work()

    finish(steps)

    assert shares == [0.5]


def test_a_worker_that_fails_leaves_the_game_to_grow_elsewhere(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(web_setup, "make_game", grows_as(played_game()))
    page = FakePage(fails=True)
    steps = PageSetup(page).grow(SETTINGS, print)
    next(steps)
    page.work()

    saved = finish(steps)

    assert saved is None
    assert page.stopped >= 1


def test_a_grown_game_is_unpacked_a_few_megabytes_a_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(web_setup, "UNPACK_STEP", 4096)
    monkeypatch.setattr(web_setup, "FEED_STEP", 1024)
    grown = played_game()
    page = FakePage()
    monkeypatch.setattr(web_setup, "make_game", grows_as(grown))
    page.start(pickle.dumps(SETTINGS))
    page.work()
    message = next(m for m in page.take() if m["kind"] == "grown")
    steps = unpacked(message["game"], message["size"])

    frames = 0
    while True:
        try:
            next(steps)
            frames += 1
        except StopIteration as done:
            saved = done.value
            break

    assert frames > message["size"] // 4096
    assert saved.game.finances.cash == grown.game.finances.cash
