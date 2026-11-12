import argparse
import pickle
import random
from pathlib import Path

import pytest

from railroads import maps
from railroads.newgame import MapChoice, Settings, make_game
from railroads.saves import Saves
from railroads.sim.clock import HISTORY_START
from railroads.sim.geo import Bounds
from railroads.sim.grids import HexGrid
from railroads.ui.app import settings_from
from railroads.ui.main_menu import MainMenu, map_note, random_seed

# Settings(seed=5, map=MapChoice.POLAND, start_year=1850) as the game pickled them before there
# were more maps than Poland.
OLD_POLAND_SETTINGS = (
    b"\x80\x05\x95\x89\x00\x00\x00\x00\x00\x00\x00\x8c\x11railroads.newgame\x94\x8c\x08Settings"
    b"\x94\x93\x94)\x81\x94}\x94(\x8c\x04seed\x94K\x05\x8c\x03map\x94h\x00\x8c\tMapChoice\x94"
    b"\x93\x94\x8c\x06Poland\x94\x85\x94R\x94\x8c\nstart_year\x94M:\x07\x8c\x04cash\x94G@\xf3"
    b"\x88\x00\x00\x00\x00\x00\x8c\x0estand_in_years\x94K\x00ub."
)


def command_line(**given: object) -> argparse.Namespace:
    args = {"seed": None, "poland": False, "map": None, "years": None, "simulate": None}
    return argparse.Namespace(**(args | given))


def test_every_baked_map_is_a_menu_choice_and_every_choice_but_a_random_place_is_baked() -> None:
    titles = {start.title for start in maps.START_MAPS}

    choices = {choice.value for choice in MapChoice if choice != MapChoice.WORLD}

    assert choices == titles
    assert MapChoice.WORLD.start_map is None


@pytest.mark.parametrize("start", maps.START_MAPS, ids=lambda start: start.key)
def test_each_baked_map_ships_covering_its_place_at_two_km_a_cell(start: maps.StartMap) -> None:
    bounds = start.bounds
    expected = HexGrid.covering(bounds.width_km, bounds.height_km, 2.0)

    tile_map = maps.load(start.key, "hex")

    assert tile_map.bounds == bounds
    assert tile_map.grid == expected
    assert maps.map_file(start.key, "hex").stat().st_size < 300_000


def test_baked_maps_can_be_long_and_thin_either_way() -> None:
    chile = maps.load("chile", "hex").grid
    japan = maps.load("japan", "hex").grid

    chile_width, chile_height = chile.size()
    japan_width, japan_height = japan.size()

    assert chile_height > 2 * chile_width
    assert japan_width > 1.5 * japan_height


def test_a_start_map_is_found_by_its_key_or_its_menu_title() -> None:
    by_key = maps.start_map("japan")

    by_title = maps.start_map("Japan: Hiroshima to Tokyo")

    assert by_key is not None and by_key.key == "japan"
    assert by_title is by_key
    assert maps.start_map("atlantis") is None


def test_a_game_on_a_long_thin_map_is_set_up_and_loads_again(tmp_path: Path) -> None:
    saves = Saves(tmp_path)
    settings = Settings(seed=3, map=MapChoice.CHILE, start_year=HISTORY_START)

    saved = make_game(settings, report=lambda _: None)
    saves.save("Andes", saved)
    loaded = saves.load("Andes")

    assert saved.title == "Central Chile, seed 3"
    assert loaded.title == saved.title
    assert loaded.settings == settings
    assert loaded.game.tile_map.grid == maps.load("chile", "hex").grid
    assert loaded.game.tile_map.bounds == Bounds(west=-73.8, east=-69.6, south=-38.2, north=-30.0)


def test_settings_pickled_before_there_were_more_maps_still_load_as_poland() -> None:
    settings = pickle.loads(OLD_POLAND_SETTINGS)

    assert settings == Settings(seed=5, map=MapChoice.POLAND, start_year=1850)
    assert settings.map.start_map is maps.start_map("poland")


def test_the_command_line_picks_a_baked_map_by_its_key() -> None:
    settings = settings_from(command_line(map="japan", seed=4))

    assert settings is not None
    assert settings.map == MapChoice.JAPAN
    assert settings.seed == 4


@pytest.mark.parametrize(
    ("given", "expected"),
    [({"poland": True}, MapChoice.POLAND), ({"seed": 9}, MapChoice.WORLD)],
)
def test_the_command_line_still_offers_poland_and_a_random_place(
    given: dict, expected: MapChoice
) -> None:
    settings = settings_from(command_line(**given))

    assert settings is not None
    assert settings.map == expected


def test_the_menu_tells_a_baked_maps_size_and_what_is_on_it() -> None:
    note = map_note(MapChoice.BRITAIN)

    assert note.startswith("546 × 980 km. ")
    assert "Scotland" in note


def test_the_menu_tells_what_a_random_place_is() -> None:
    note = map_note(MapChoice.WORLD)

    assert "anywhere in the world" in note


def test_random_seeds_do_not_follow_pythons_random_module() -> None:
    seeds = []

    for _ in range(5):
        random.seed(1)
        seeds.append(random_seed())

    assert len(set(seeds)) > 1


def test_each_menu_opened_offers_its_own_seed(tmp_path: Path) -> None:
    menus = [MainMenu(Saves(tmp_path)) for _ in range(5)]

    seeds = {menu.seed for menu in menus}

    assert len(seeds) > 1
