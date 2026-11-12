"""Saved games: the whole game, its clock and its random numbers, written to a file in the
player's data folder and read back exactly as they were.

A save is a header line naming the format and saying what is in it, then the game pickled and
compressed. The header is read on its own to list saves without loading them. Bump SAVE_FORMAT
whenever a change to the simulation's classes would make an older save load wrong, so that it is
refused with a clear message instead.

Named saves are the player's; AUTOSAVES more are written in turn at the start of every month,
"autosave 1" the newest, so one bad save can't lose the game.
"""

import json
import os
import pickle
import re
import sys
import zlib
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np

from railroads.sim.clock import GameClock
from railroads.sim.game import Game
from railroads.sim.summaries import MonthSummary

if TYPE_CHECKING:
    from railroads.newgame import Settings

SAVE_FORMAT = 1
MAGIC = b"RAILROADS SAVE\n"
SUFFIX = ".save"
AUTOSAVES = 3
AUTOSAVE_NAME = "autosave"
MAX_NAME = 40
NAME_PATTERN = re.compile(r"[\w ,.'()-]+")


def user_data_dir() -> Path:
    """Where this player's saves and other data go, as the system expects."""
    home = Path.home()
    if sys.platform == "darwin":
        return home / "Library" / "Application Support" / "Railroads"
    if sys.platform == "win32":
        return Path(os.environ.get("APPDATA", home / "AppData" / "Roaming")) / "Railroads"
    return Path(os.environ.get("XDG_DATA_HOME", home / ".local" / "share")) / "railroads"


SAVES_DIR = user_data_dir() / "saves"


class SaveError(Exception):
    """A save that can't be written or read, with a message for the player."""


@dataclass
class SavedGame:
    """Everything a game in progress needs to go on where it stopped."""

    game: Game
    clock: GameClock
    rng: np.random.Generator
    title: str
    settings: Settings | None = None
    # The month summaries on the map, newest first, and whether each was seen.
    summaries: list[tuple[MonthSummary, bool]] = field(default_factory=list)
    # Where the map was looked at: its centre and zoom level.
    camera: tuple[float, float, int] | None = None


@dataclass(frozen=True)
class Snapshot:
    """A game pickled at one moment, ready to be written while play goes on."""

    header: dict[str, Any]
    data: bytes


@dataclass(frozen=True)
class SaveInfo:
    name: str
    path: Path
    title: str = ""
    date: str = ""
    saved_at: datetime | None = None
    # Why it can't be loaded, if it can't.
    problem: str | None = None

    @property
    def autosave(self) -> bool:
        return is_autosave(self.name)


def _frozen(items: dict[Any, Any]) -> MappingProxyType[Any, Any]:
    return MappingProxyType(items)


class _Pickler(pickle.Pickler):
    def reducer_override(self, obj: object) -> Any:
        if type(obj) is MappingProxyType:
            return _frozen, (dict(obj),)
        return NotImplemented


def snapshot(saved: SavedGame) -> Snapshot:
    out = BytesIO()
    _Pickler(out, protocol=pickle.HIGHEST_PROTOCOL).dump(saved)
    header = {
        "format": SAVE_FORMAT,
        "title": saved.title,
        "date": saved.clock.date,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    return Snapshot(header, out.getvalue())


def is_autosave(name: str) -> bool:
    return name.casefold().startswith(AUTOSAVE_NAME)


def name_problem(name: str) -> str | None:
    """Why the player can't save under this name, if they can't."""
    name = name.strip()
    if not name:
        return "Give the save a name."
    if len(name) > MAX_NAME:
        return f"Keep the name to {MAX_NAME} letters."
    if not NAME_PATTERN.fullmatch(name) or name.startswith("."):
        return "Use letters, digits, spaces and - _ ' . , ( ) only."
    if is_autosave(name):
        return "Names starting with 'autosave' are kept for autosaves."
    return None


def _autosave_name(number: int) -> str:
    return f"{AUTOSAVE_NAME} {number}"


def read_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as file:
        if file.read(len(MAGIC)) != MAGIC:
            raise SaveError(f"'{path.stem}' is not a Railroads save.")
        try:
            header = json.loads(file.readline())
        except ValueError as error:
            raise SaveError(f"'{path.stem}' is damaged: its header can't be read.") from error
    if not isinstance(header, dict):
        raise SaveError(f"'{path.stem}' is damaged: its header can't be read.")
    found = header.get("format")
    if found != SAVE_FORMAT:
        raise SaveError(
            f"'{path.stem}' was saved in format {found}, and this version of the game reads only "
            f"format {SAVE_FORMAT}, so it can't be loaded."
        )
    return header


class Saves:
    """The saves in one folder."""

    def __init__(self, folder: Path = SAVES_DIR, autosaves: int = AUTOSAVES) -> None:
        self.folder = folder
        self.autosaves = autosaves

    def path(self, name: str) -> Path:
        return self.folder / f"{name.strip()}{SUFFIX}"

    def exists(self, name: str) -> bool:
        return self.path(name).exists()

    def save(self, name: str, saved: SavedGame | Snapshot) -> Path:
        """Saves the game under the player's name for it, over any save of that name."""
        problem = name_problem(name)
        if problem is not None:
            raise SaveError(problem)
        return self._write(self.path(name), saved)

    def autosave(self, saved: SavedGame | Snapshot) -> Path:
        """Saves the game as the newest autosave, each older one moving down a place and the
        oldest dropped."""
        self.folder.mkdir(parents=True, exist_ok=True)
        self.path(_autosave_name(self.autosaves)).unlink(missing_ok=True)
        for number in range(self.autosaves - 1, 0, -1):
            older = self.path(_autosave_name(number))
            if older.exists():
                older.replace(self.path(_autosave_name(number + 1)))
        return self._write(self.path(_autosave_name(1)), saved)

    def _write(self, path: Path, saved: SavedGame | Snapshot) -> Path:
        snap = snapshot(saved) if isinstance(saved, SavedGame) else saved
        header = json.dumps(snap.header).encode() + b"\n"
        data = zlib.compress(snap.data, 1)
        try:
            self.folder.mkdir(parents=True, exist_ok=True)
            partial = path.with_suffix(".partial")
            partial.write_bytes(MAGIC + header + data)
            partial.replace(path)
        except OSError as error:
            raise SaveError(f"The game couldn't be saved: {error.strerror or error}.") from error
        return path

    def load(self, name: str) -> SavedGame:
        path = self.path(name)
        if not path.exists():
            raise SaveError(f"There is no save called '{name}'.")
        read_header(path)
        with path.open("rb") as file:
            file.read(len(MAGIC))
            file.readline()
            compressed = file.read()
        try:
            saved = pickle.loads(zlib.decompress(compressed))
        except (zlib.error, pickle.UnpicklingError, EOFError) as error:
            raise SaveError(f"'{name}' is damaged and can't be loaded.") from error
        except (AttributeError, ImportError, TypeError, ValueError) as error:
            raise SaveError(
                f"'{name}' was saved by a different version of the game and can't be loaded."
            ) from error
        if not isinstance(saved, SavedGame):
            raise SaveError(f"'{name}' holds no game.")
        return saved

    def listing(self) -> list[SaveInfo]:
        """Every save in the folder, the latest saved first."""
        if not self.folder.is_dir():
            return []
        found = []
        for path in self.folder.glob(f"*{SUFFIX}"):
            try:
                header = read_header(path)
            except (SaveError, OSError) as error:
                found.append(SaveInfo(path.stem, path, problem=str(error)))
                continue
            saved_at = header.get("saved_at")
            found.append(
                SaveInfo(
                    path.stem,
                    path,
                    title=str(header.get("title", "")),
                    date=str(header.get("date", "")),
                    saved_at=datetime.fromisoformat(saved_at) if saved_at else None,
                )
            )
        return sorted(
            found,
            key=lambda info: (info.saved_at or datetime.min, info.name),
            reverse=True,
        )
