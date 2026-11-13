import tomllib
from pathlib import Path

from railroads.version import game_version

ROOT = Path(__file__).resolve().parents[1]


def test_the_game_version_is_the_one_in_pyproject() -> None:
    expected = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]

    found = game_version()

    assert found == expected
