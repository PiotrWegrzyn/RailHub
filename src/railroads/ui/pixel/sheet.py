"""Loads the sprite sheet the game draws its pixel art from."""

import io
import json
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Self, overload

import numpy as np
import numpy.typing as npt
from PIL import Image

SHEET_DIR = Path(__file__).parent
SHEET_PNG = SHEET_DIR / "sprites.png"
SHEET_INDEX = SHEET_DIR / "sprites.json"
# The same sprites drawn FINE times finer, for zooms that show each art pixel as two or more
# screen pixels.
FINE = 2
FINE_PNG = SHEET_DIR / "sprites@2x.png"
FINE_INDEX = SHEET_DIR / "sprites@2x.json"
# The same sprites drawn FINEST times finer, for the closest zooms, which draw chunks that fine
# (railroads.ui.pixel.world.CLOSE). It is kept as bytes until a sprite is drawn.
FINEST = 4
FINEST_PNG = SHEET_DIR / "sprites@4x.png"
FINEST_INDEX = SHEET_DIR / "sprites@4x.json"
# Every resolution the sprites are drawn at, in pixels per art pixel: one for each number of
# screen pixels an art pixel covers at a zoom where the graphics card stands them on the map, so
# each sprite pixel there is one screen pixel, whether the screen has one or two pixels to a
# point. Those the CPU draws chunks from too are sheets of PNG; the others are kept as each
# sprite's own PNG in a zip, so a sprite is unpacked only when a view shows it, rather than a
# sheet of up to a hundred megabytes of pixels at once.
DENSITIES = (1, 2, 3, 4, 6, 8, 12, 24)
SHEETS = (1, FINE, FINEST)


def sheet_files(resolution: int) -> tuple[Path, Path]:
    """The PNG, or for a resolution not in SHEETS the zip, and the JSON of the sprites drawn at
    `resolution`."""
    if resolution == 1:
        return SHEET_PNG, SHEET_INDEX
    kind = "png" if resolution in SHEETS else "zip"
    stem = f"sprites@{resolution}x"
    return SHEET_DIR / f"{stem}.{kind}", SHEET_DIR / f"{stem}.json"


Pixels = npt.NDArray[np.float32]


@dataclass(frozen=True)
class SpriteSheet:
    """Each sprite as RGBA pixels from 0 to 1, with the anchor that stands on the ground."""

    names: list[str]
    pixels: list[Pixels]
    anchors: npt.NDArray[np.int64]

    @classmethod
    def from_image(cls, sheet: Image.Image, places: dict[str, list[int]]) -> Self:
        rgba = np.asarray(sheet.convert("RGBA"), dtype=np.float32) / 255
        names = list(places)
        pixels = [rgba[y : y + h, x : x + w] for x, y, w, h, _, _ in places.values()]
        anchors = np.array([place[4:6] for place in places.values()], dtype=np.int64)
        return cls(names, pixels, anchors)

    @classmethod
    def load(cls, png: Path = SHEET_PNG, index: Path = SHEET_INDEX) -> Self:
        with Image.open(png) as sheet:
            return cls.from_image(sheet, json.loads(index.read_text()))

    @classmethod
    def load_fine(cls) -> Self:
        return cls.load(FINE_PNG, FINE_INDEX)

    @classmethod
    def load_finest(cls) -> ByteSheet:
        return cls.load_bytes(FINEST)

    @classmethod
    def load_bytes(cls, resolution: int) -> ByteSheet:
        """The sprites drawn at `resolution` (DENSITIES), as bytes: from a sheet, each a view of
        the whole sheet's, which is quicker to load than copying each out, or from a zip, each
        unpacked when it is asked for."""
        path, index = sheet_files(resolution)
        places = json.loads(index.read_text())
        if path.suffix == ".zip":
            return ByteSheet.zipped(path, places)
        with Image.open(path) as sheet:
            return ByteSheet.from_image(sheet, places, copied=False)

    def index(self, name: str) -> int:
        found = self._indices.get(name)
        if found is None:
            raise ValueError(f"No sprite is named {name!r}")
        return found

    @cached_property
    def _indices(self) -> dict[str, int]:
        """Each name's first sprite."""
        indices: dict[str, int] = {}
        for i, name in enumerate(self.names):
            indices.setdefault(name, i)
        return indices

    def variants(self, prefix: str) -> list[int]:
        """Indices of the sprites named prefix_0, prefix_1 and so on."""
        return [
            i
            for i, name in enumerate(self.names)
            if name.startswith(f"{prefix}_") and name.removeprefix(f"{prefix}_").isdigit()
        ]

    @property
    def sizes(self) -> npt.NDArray[np.int64]:
        """Width and height of each sprite."""
        return np.array([[p.shape[1], p.shape[0]] for p in self.pixels], dtype=np.int64)


@dataclass(frozen=True)
class ByteSheet:
    """Each sprite as RGBA bytes, with its anchor: a big sheet kept small in memory until its
    sprites are drawn."""

    names: list[str]
    pixels: Sequence[npt.NDArray[np.uint8]]
    anchors: npt.NDArray[np.int64]

    @classmethod
    def from_image(
        cls, sheet: Image.Image, places: dict[str, list[int]], copied: bool = True
    ) -> Self:
        rgba = np.asarray(sheet if sheet.mode == "RGBA" else sheet.convert("RGBA"))
        pixels = [rgba[y : y + h, x : x + w] for x, y, w, h, _, _ in places.values()]
        if copied:
            pixels = [sprite.copy() for sprite in pixels]
        anchors = np.array([place[4:6] for place in places.values()], dtype=np.int64)
        return cls(list(places), pixels, anchors)

    @classmethod
    def zipped(cls, path: Path, places: dict[str, list[int]]) -> Self:
        """The sprites in a zip of one PNG each, named by their place in `places`."""
        anchors = np.array([place[4:6] for place in places.values()], dtype=np.int64)
        sizes = np.array([place[2:4] for place in places.values()], dtype=np.int64)
        sheet = cls(list(places), _Zipped(path, len(places)), anchors)
        sheet.__dict__["sizes"] = sizes
        return sheet

    @cached_property
    def sizes(self) -> npt.NDArray[np.int64]:
        return np.array([[p.shape[1], p.shape[0]] for p in self.pixels], dtype=np.int64)


def sprite_name(index: int) -> str:
    """The name of the PNG holding a sprite in a zip of them (ByteSheet.zipped)."""
    return f"{index}.png"


class _Zipped(Sequence[npt.NDArray[np.uint8]]):
    """Sprites kept as PNGs in a zip, each unpacked to RGBA bytes whenever it is asked for."""

    def __init__(self, path: Path, count: int) -> None:
        self._zip = zipfile.ZipFile(path)
        self._count = count

    def __len__(self) -> int:
        return self._count

    @overload
    def __getitem__(self, index: int) -> npt.NDArray[np.uint8]: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[npt.NDArray[np.uint8]]: ...

    def __getitem__(
        self, index: int | slice
    ) -> npt.NDArray[np.uint8] | Sequence[npt.NDArray[np.uint8]]:
        if isinstance(index, slice):
            return [self[i] for i in range(len(self))[index]]
        if not 0 <= index < self._count:
            raise IndexError(index)
        with Image.open(io.BytesIO(self._zip.read(sprite_name(index)))) as sprite:
            return np.asarray(sprite.convert("RGBA"))
