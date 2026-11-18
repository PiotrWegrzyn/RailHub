from pathlib import Path

PAGE = Path(__file__).resolve().parents[1] / "web" / "index.html"


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
