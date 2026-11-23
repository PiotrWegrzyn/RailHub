import pickle

import pytest

from builders import SETTINGS, FakePage, grows_as, played_game
from railroads.newgame import MapChoice, Settings
from railroads.sim.steps import finish
from railroads.ui import web_setup
from railroads.ui.web_setup import PageSetup, unpacked


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
    assert pickle.loads(page.settings or b"")[0] == SETTINGS
    assert saved.clock == grown.clock
    assert saved.game.finances.cash == grown.game.finances.cash
    assert saved.game.railway.stations() == grown.game.railway.stations()
    assert page.stopped >= 1


def test_a_game_grown_ahead_is_taken_up_when_asked_for_without_growing_it_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grown = played_game()
    monkeypatch.setattr(web_setup, "make_game", grows_as(grown))
    page = FakePage()
    setup = PageSetup(page)
    setup.grow_ahead(SETTINGS)
    page.work()

    saved = finish(setup.grow(SETTINGS, print))

    assert saved is not None
    assert saved.game.finances.cash == grown.game.finances.cash
    assert page.started == 1
    assert setup.ahead is None


def test_a_game_asked_for_with_other_settings_than_the_one_grown_ahead_is_grown_anew(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(web_setup, "make_game", grows_as(played_game()))
    page = FakePage()
    setup = PageSetup(page)
    setup.grow_ahead(SETTINGS)
    other = Settings(seed=8, map=MapChoice.POLAND, start_year=1850)

    steps = setup.grow(other, print)
    next(steps)
    page.work()
    finish(steps)

    assert page.started == 2
    assert pickle.loads(page.settings or b"")[0] == other


def test_growing_the_same_game_ahead_again_leaves_the_worker_at_it() -> None:
    page = FakePage()
    setup = PageSetup(page)
    setup.grow_ahead(SETTINGS)

    setup.grow_ahead(SETTINGS)

    assert page.started == 1


def test_a_game_grown_ahead_is_stopped_when_not_wanted() -> None:
    page = FakePage()
    setup = PageSetup(page)
    setup.grow_ahead(SETTINGS)

    setup.stop_ahead()

    assert page.stopped == 1
    assert setup.ahead is None


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


def test_a_game_grown_by_a_worker_of_another_version_is_left_to_grow_elsewhere(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(web_setup, "make_game", grows_as(played_game()))
    page = FakePage(version="0.5.47")
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
    page.start(pickle.dumps((SETTINGS, (64, 48), False)))
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


def test_the_page_sees_the_pictures_a_worker_takes_of_the_map_growing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(web_setup, "make_game", grows_as(played_game()))
    page = FakePage()
    seen: list[tuple[int, tuple[int, ...]]] = []
    steps = PageSetup(page).grow(
        SETTINGS,
        print,
        see=lambda picture: seen.append((picture.year, picture.pixels.shape)),
        size=(64, 48),
    )
    next(steps)
    page.work()

    finish(steps)

    assert seen == [(950, (48, 64, 4))]
