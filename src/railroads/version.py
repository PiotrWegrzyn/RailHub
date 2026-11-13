"""The game's version, as pyproject.toml gives it and the installed package records it."""

from importlib.metadata import PackageNotFoundError, version


def game_version() -> str:
    try:
        return version("railroads")
    except PackageNotFoundError:
        return "dev"


VERSION = game_version()
