"""Saved games: the whole game, its clock and its random numbers, written to a file in the
player's data folder and read back exactly as they were.

A save is a header line naming the format and saying what is in it, then the game pickled and
compressed. The header is read on its own to list saves without loading them. Bump SAVE_FORMAT
whenever a change to the simulation's classes would make an older save load wrong, so that it is
refused with a clear message instead.

Named saves are the player's; AUTOSAVES more are written in turn at the start of every month,
"autosave 1" the newest, so one bad save can't lose the game. An autosave can be written in steps
(Saves.autosave_steps), compressing COMPRESS_STEP bytes of the game at a time, for a caller with
no thread to write it on to spread over frames.
"""

import json
import os
import pickle
import re
import sys
import zlib
from collections.abc import Buffer, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np

from railroads.sim.clock import GameClock
from railroads.sim.game import Game
from railroads.sim.steps import Steps, finish
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
# How much of a pickled game a step of an autosave compresses: a few milliseconds' work even in a
# browser, where the parts of a game that compress worst go at some 50 MB a second.
COMPRESS_STEP = 1 << 17
# The compression saves use: fast, and the games' many empty arrays shrink to almost nothing.
LEVEL = 1


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
    data: Buffer


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


class _Pieces:
    """What a pickler writes, kept as it comes rather than copied: its frames, and on their own
    the large buffers of arrays, which it hands over as they are (pickle protocol 5). Those are
    the game's own arrays, so the pieces hold the game only until it next changes."""

    def __init__(self) -> None:
        self.parts: list[memoryview] = []
        self.size = 0

    def write(self, data: Buffer) -> int:
        part = memoryview(data).cast("B")
        self.parts.append(part)
        self.size += len(part)
        return len(part)


class ReusedMemory:
    """Memory games are pickled into and kept for the next time, so a monthly autosave doesn't
    ask for a few hundred megabytes afresh; a snapshot held in it holds only until the next one
    is."""

    def __init__(self) -> None:
        self._memory = np.empty(0, dtype=np.uint8)

    def hold(self, parts: list[memoryview], size: int) -> memoryview:
        """Copies the parts one after another, `size` bytes in all, into the memory; returns
        them there."""
        if size > len(self._memory):
            # A new block rather than a larger one: an old snapshot may still look into this one.
            # A quarter more leaves the game room to grow; the system only hands the memory over
            # as it is written.
            self._memory = np.empty(size + size // 4, dtype=np.uint8)
        end = 0
        for part in parts:
            self._memory[end : end + len(part)] = part
            end += len(part)
        return memoryview(self._memory[:size])


def snapshot(saved: SavedGame, memory: ReusedMemory | None = None) -> Snapshot:
    """The game pickled now, into `memory` if given, which then mustn't be used again until the
    snapshot has been written. It is pickled before anything is copied, so the copy goes into
    memory of the size it needs."""
    pieces = _Pieces()
    _Pickler(pieces, protocol=pickle.HIGHEST_PROTOCOL).dump(saved)
    data: Buffer
    if memory is None:
        data = b"".join(pieces.parts)
    else:
        data = memory.hold(pieces.parts, pieces.size)
    header = {
        "format": SAVE_FORMAT,
        "title": saved.title,
        "date": saved.clock.date,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    return Snapshot(header, data)


def compressed(data: Buffer, step: int | None = None) -> Steps[bytes]:
    """The pickled game compressed as a save holds it: all at once, or `step` bytes at a time
    with a step yielded after each."""
    if step is None:
        return zlib.compress(data, LEVEL)
    compressor = zlib.compressobj(LEVEL)
    view = memoryview(data).cast("B")
    parts = []
    for start in range(0, len(view), step):
        parts.append(compressor.compress(view[start : start + step]))
        yield
    parts.append(compressor.flush())
    return b"".join(parts)


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
        return _header(file.read(len(MAGIC)), file.readline(), path.stem)


def _header(magic: bytes, line: bytes, name: str) -> dict[str, Any]:
    if magic != MAGIC:
        raise SaveError(f"'{name}' is not a Railroads save.")
    try:
        header = json.loads(line)
    except ValueError as error:
        raise SaveError(f"'{name}' is damaged: its header can't be read.") from error
    if not isinstance(header, dict):
        raise SaveError(f"'{name}' is damaged: its header can't be read.")
    found = header.get("format")
    if found != SAVE_FORMAT:
        raise SaveError(
            f"'{name}' was saved in format {found}, and this version of the game reads only "
            f"format {SAVE_FORMAT}, so it can't be loaded."
        )
    return header


def imported_name(file_name: str, taken: Callable[[str], bool]) -> str:
    """A save name for a file brought in from elsewhere: its own name without what a save name
    can't hold, made free of autosave names and of saves already there."""
    stem = Path(file_name.replace("\\", "/")).name.removesuffix(SUFFIX)
    base = " ".join("".join(NAME_PATTERN.findall(stem)).split())[:MAX_NAME].strip()
    if name_problem(base) is not None:
        base = f"Imported {base}"[:MAX_NAME].strip() if base else "Imported save"
    name, number = base, 2
    while taken(name):
        suffix = f" ({number})"
        name = base[: MAX_NAME - len(suffix)].strip() + suffix
        number += 1
    return name


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
        return finish(self.autosave_steps(saved, None))

    def autosave_steps(
        self, saved: SavedGame | Snapshot, step: int | None = COMPRESS_STEP
    ) -> Steps[Path]:
        """autosave in steps a caller can spread over frames, each compressing `step` bytes of
        the game (all of it at once if None); the older autosaves move down only once the new
        one is ready to write."""
        snap = snapshot(saved) if isinstance(saved, SavedGame) else saved
        data = yield from compressed(snap.data, step)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.path(_autosave_name(self.autosaves)).unlink(missing_ok=True)
        for number in range(self.autosaves - 1, 0, -1):
            older = self.path(_autosave_name(number))
            if older.exists():
                older.replace(self.path(_autosave_name(number + 1)))
        return self._store(self.path(_autosave_name(1)), snap.header, data)

    def _write(self, path: Path, saved: SavedGame | Snapshot) -> Path:
        snap = snapshot(saved) if isinstance(saved, SavedGame) else saved
        return self._store(path, snap.header, finish(compressed(snap.data)))

    def _store(self, path: Path, header: dict[str, Any], data: bytes) -> Path:
        """Writes a save's header and its compressed game, through a file of its own so a save
        cut short leaves the old one as it was."""
        line = json.dumps(header).encode() + b"\n"
        try:
            self.folder.mkdir(parents=True, exist_ok=True)
            partial = path.with_suffix(".partial")
            partial.write_bytes(MAGIC + line + data)
            partial.replace(path)
        except OSError as error:
            raise SaveError(f"The game couldn't be saved: {error.strerror or error}.") from error
        return path

    def export(self, name: str) -> bytes:
        """The save's file as it is, to keep or carry to the game elsewhere."""
        try:
            return self.path(name).read_bytes()
        except OSError as error:
            raise SaveError(f"'{name}' couldn't be read: {error.strerror or error}.") from error

    def bring_in(self, file_name: str, data: bytes) -> str:
        """Adds a save file from elsewhere, such as the game in a browser or on another
        computer, under a free name made from the file's own; returns that name."""
        line = data[len(MAGIC) :].split(b"\n", 1)[0]
        _header(data[: len(MAGIC)], line, Path(file_name).stem)
        name = imported_name(file_name, self.exists)
        path = self.path(name)
        try:
            self.folder.mkdir(parents=True, exist_ok=True)
            partial = path.with_suffix(".partial")
            partial.write_bytes(data)
            partial.replace(path)
        except OSError as error:
            raise SaveError(f"The save couldn't be added: {error.strerror or error}.") from error
        return name

    def load(self, name: str) -> SavedGame:
        path = self.path(name)
        if not path.exists():
            raise SaveError(f"There is no save called '{name}'.")
        read_header(path)
        with path.open("rb") as file:
            file.read(len(MAGIC))
            file.readline()
            packed = file.read()
        try:
            saved = pickle.loads(zlib.decompress(packed))
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
