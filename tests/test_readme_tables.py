import importlib.util
from pathlib import Path
from types import ModuleType

from railroads.sim.constructions import BuildingKind
from railroads.sim.finance import dollars
from railroads.sim.goods import Good, freight_of
from railroads.sim.trains import ENGINES
from railroads.ui.help_window import building_help, engine_help

ROOT = Path(__file__).resolve().parents[1]


def load_tables() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "readme_tables", ROOT / "tools" / "readme_tables.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tables = load_tables()


def test_the_readme_tables_show_the_games_own_figures() -> None:
    text = tables.README.read_text()

    written = tables.with_tables(text)

    assert written == text, "README.md is out of date: run uv run python tools/readme_tables.py"


def test_a_house_must_get_half_its_food_and_its_heating_and_takes_the_rest_as_a_bonus() -> None:
    house = next(row for row in building_help() if row.kind == BuildingKind.HOUSE)

    line = tables.building_row(house)

    _, name, makes, must, bonus, _ = line.split("|")
    assert (name.strip(), makes.strip()) == ("House", "")
    assert must.strip() == "10 food, 3 heating"
    assert bonus.strip().startswith("10 food,")
    assert "5 common goods" in bonus


def test_the_freight_table_puts_every_good_but_travellers_in_the_band_of_its_freight() -> None:
    wheat = f"wheat {freight_of(Good.WHEAT):g}"

    lines = tables.freight_table()

    assert any(line.startswith("| $10 or less |") and wheat in line for line in lines)
    assert not any("passengers" in line for line in lines)


def test_an_engine_line_gives_its_range_and_price_as_the_game_has_them() -> None:
    planet = next(row for row in engine_help() if row.name == "planet")

    line = tables.engine_row(planet)

    assert f"| {ENGINES['planet'].tank:.0f} cells |" in line
    assert f"| {dollars(ENGINES['planet'].price)} |" in line
