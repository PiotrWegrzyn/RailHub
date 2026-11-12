import json
from enum import Enum
from pathlib import Path
from types import BuiltinFunctionType, FunctionType

import numpy as np
import pytest

from builders import flat_map
from railroads.newgame import MapChoice, Settings
from railroads.saves import MAGIC, SAVE_FORMAT, SavedGame, SaveError, Saves, name_problem
from railroads.sim.clock import GameClock
from railroads.sim.constructions import Building, BuildingKind, Constructions
from railroads.sim.game import Game
from railroads.sim.trains import Stop


def played_game(months: int = 3) -> SavedGame:
    game = Game(Constructions(flat_map(rows=16, cols=30)))
    game.finances.cash = 1_000_000.0
    for _ in range(4):
        game.growth.build(Building((8, 15), BuildingKind.HOUSE))
    game.build_station((8, 6), (3, 9))
    game.build_station((8, 24))
    game.lay_track((8, 6), (8, 24))
    train = game.railway.buy()
    assert train is not None
    train.stops = [Stop((8, 6)), Stop((8, 24))]
    clock = GameClock(first_year=1850)
    rng = np.random.default_rng(7)
    for _ in range(months):
        clock.advance(clock.seconds_per_month)
        game.month(rng, clock.date)
        game.time_passed(0.4)
    clock.advance(clock.seconds_per_month / 3)
    settings = Settings(seed=7, map=MapChoice.POLAND, start_year=1850)
    return SavedGame(game, clock, rng, "Test, seed 7", settings, camera=(4.5, 6.0, 2))


ATOMS = (Enum, type, FunctionType, BuiltinFunctionType, set, frozenset, str, bytes, int, float)


def differences(a: object, b: object, path: str = "saved", seen: set | None = None) -> list[str]:
    """Where two object graphs differ, field by field, arrays by value."""
    seen = set() if seen is None else seen
    if (id(a), id(b)) in seen:
        return []
    seen.add((id(a), id(b)))
    if type(a) is not type(b):
        return [f"{path}: {type(a).__name__} != {type(b).__name__}"]
    if isinstance(a, np.ndarray):
        numbers = a.dtype.kind in "fc"
        return [] if np.array_equal(a, b, equal_nan=numbers) else [path]
    if isinstance(a, np.random.Generator):
        same = a.bit_generator.state == b.bit_generator.state
        return [] if same else [path]
    if isinstance(a, dict):
        if set(a) != set(b):
            return [f"{path}: keys differ"]
        found: list[str] = []
        for key in a:
            found += differences(a[key], b[key], f"{path}[{key!r}]", seen)
        return found
    if isinstance(a, list | tuple):
        if len(a) != len(b):
            return [f"{path}: lengths differ"]
        found = []
        for index, (x, y) in enumerate(zip(a, b, strict=True)):
            found += differences(x, y, f"{path}[{index}]", seen)
        return found
    if isinstance(a, ATOMS) or a is None:
        return [] if a == b else [path]
    fields = dict(vars(a)) if hasattr(a, "__dict__") else {}
    for slot in getattr(type(a), "__slots__", ()):
        if hasattr(a, slot):
            fields[slot] = getattr(a, slot)
    found = []
    for name, value in fields.items():
        found += differences(value, getattr(b, name), f"{path}.{name}", seen)
    return found


def test_a_loaded_game_is_the_game_saved(tmp_path: Path) -> None:
    saves = Saves(tmp_path)
    saved = played_game()

    saves.save("My railway", saved)
    loaded = saves.load("My railway")

    assert differences(saved, loaded) == []
    assert loaded.game.constructions.station_ends((8, 6)) == (3, 9)
    assert loaded.game.railway.trains[0].stops == saved.game.railway.trains[0].stops
    assert loaded.clock == saved.clock
    assert loaded.camera == (4.5, 6.0, 2)


def test_a_loaded_game_plays_on_as_the_saved_one_would(tmp_path: Path) -> None:
    saves = Saves(tmp_path)
    saved = played_game()
    saves.save("My railway", saved)
    loaded = saves.load("My railway")

    for game in (saved, loaded):
        for _ in range(6):
            game.game.month(game.rng)
            game.game.time_passed(0.4)

    assert differences(saved, loaded) == []
    assert loaded.game.finances.cash == saved.game.finances.cash


def test_a_second_train_bought_after_loading_is_numbered_on(tmp_path: Path) -> None:
    saves = Saves(tmp_path)
    saves.save("My railway", played_game())
    loaded = saves.load("My railway")

    train = loaded.game.railway.buy()

    assert train is not None
    assert train.name.endswith(" 2")


def autosaved(saves: Saves, months: int) -> list[str]:
    dates = []
    for month in range(months):
        saved = played_game(months=0)
        saved.clock.months = month
        saves.autosave(saved)
        dates.append(saved.clock.date)
    return dates


def test_autosaves_keep_only_the_newest_few_newest_first(tmp_path: Path) -> None:
    saves = Saves(tmp_path, autosaves=3)

    dates = autosaved(saves, 5)

    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "autosave 1.save",
        "autosave 2.save",
        "autosave 3.save",
    ]
    kept = [saves.load(f"autosave {n}").clock.date for n in (1, 2, 3)]
    assert kept == dates[:-4:-1]


def test_autosaves_leave_named_saves_alone(tmp_path: Path) -> None:
    saves = Saves(tmp_path, autosaves=2)
    saves.save("Keep me", played_game(months=0))

    autosaved(saves, 4)

    assert saves.exists("Keep me")
    assert sorted(info.name for info in saves.listing()) == ["Keep me", "autosave 1", "autosave 2"]


def written(tmp_path: Path, name: str, content: bytes) -> Saves:
    tmp_path.joinpath(f"{name}.save").write_bytes(content)
    return Saves(tmp_path)


def header(format_number: int) -> bytes:
    return MAGIC + json.dumps({"format": format_number, "title": "Old"}).encode() + b"\n"


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (header(SAVE_FORMAT + 1) + b"x", f"saved in format {SAVE_FORMAT + 1}"),
        (header(0) + b"x", "saved in format 0"),
        (b"just some text", "is not a Railroads save"),
        (MAGIC + b"{broken\n", "is damaged"),
        (header(SAVE_FORMAT) + b"not compressed", "is damaged"),
    ],
)
def test_a_save_that_cant_be_read_is_refused_with_a_reason(
    tmp_path: Path, content: bytes, message: str
) -> None:
    saves = written(tmp_path, "Old", content)

    with pytest.raises(SaveError, match=message):
        saves.load("Old")


def test_a_save_that_cant_be_read_is_listed_with_its_reason(tmp_path: Path) -> None:
    saves = written(tmp_path, "Old", header(SAVE_FORMAT + 1))

    listing = saves.listing()

    assert [info.name for info in listing] == ["Old"]
    assert listing[0].problem is not None
    assert "can't be loaded" in listing[0].problem


def test_the_listing_shows_what_each_save_holds(tmp_path: Path) -> None:
    saves = Saves(tmp_path)
    saves.save("My railway", played_game())

    (info,) = saves.listing()

    assert (info.name, info.title, info.date, info.problem) == (
        "My railway",
        "Test, seed 7",
        "April 1850",
        None,
    )


@pytest.mark.parametrize(
    ("name", "allowed"),
    [
        ("My railway", True),
        ("Poland (1850) - take 2", True),
        ("", False),
        ("   ", False),
        ("../outside", False),
        ("a/b", False),
        (".hidden", False),
        ("autosave 1", False),
        ("Autosave mine", False),
        ("x" * 41, False),
    ],
)
def test_save_names_must_be_plain_and_not_an_autosave(name: str, allowed: bool) -> None:
    problem = name_problem(name)

    assert (problem is None) == allowed


def test_saving_under_a_refused_name_writes_nothing(tmp_path: Path) -> None:
    saves = Saves(tmp_path)

    with pytest.raises(SaveError):
        saves.save("../outside", played_game(months=0))

    assert list(tmp_path.iterdir()) == []


def test_loading_a_missing_save_says_so(tmp_path: Path) -> None:
    saves = Saves(tmp_path)

    with pytest.raises(SaveError, match="no save called 'Nothing'"):
        saves.load("Nothing")
