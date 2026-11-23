import importlib
import tomllib
from pathlib import Path

from railroads.ui.waiting_screen import TIPS

PAGE = Path(__file__).resolve().parents[1] / "web" / "index.html"
PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def test_the_page_shows_make_believe_work_while_it_loads_in_place_of_loading() -> None:
    page = PAGE.read_text()

    shown = 'id="loader">Turning cogs (0%)...<' in page

    assert shown
    assert "Shoveling cow shit" in page
    assert "Loading (" not in page


def test_no_line_on_the_page_calls_for_leeroy() -> None:
    page = PAGE.read_text().lower()

    called = "lero" in page or "jenkins" in page or "jennn" in page

    assert not called


def test_the_page_says_ready_in_place_of_the_work_once_it_has_loaded() -> None:
    page = PAGE.read_text()

    ready = 'reached >= 100 ? "Ready"' in page

    assert ready


def test_the_landing_page_offers_the_same_tips_as_the_game() -> None:
    page = PAGE.read_text()

    missing = [tip for tip in TIPS if f'"{tip}"' not in page]

    assert missing == []


def test_the_page_starts_the_game_from_the_same_main_as_the_railroads_command() -> None:
    script = tomllib.loads(PYPROJECT.read_text())["project"]["scripts"]["railroads"]
    module, function = script.split(":")
    page = PAGE.read_text()

    started = f"from {module} import {function}" in page

    assert started
    assert callable(getattr(importlib.import_module(module), function))
