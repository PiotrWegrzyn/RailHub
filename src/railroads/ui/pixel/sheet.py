"""Loads the sprite sheet the game draws its pixel art from."""

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Self

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
# point.
DENSITIES = (1, 2, 3, 4, 6, 8, 12)


def sheet_files(resolution: int) -> tuple[Path, Path]:
    """The PNG and JSON of the sheet drawn at `resolution`."""
    if resolution == 1:
        return SHEET_PNG, SHEET_INDEX
    return SHEET_DIR / f"sprites@{resolution}x.png", SHEET_DIR / f"sprites@{resolution}x.json"


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
        """The sheet drawn at `resolution` (DENSITIES), as bytes, each sprite a view of the
        whole sheet's, which is quicker to load than copying each out."""
        png, index = sheet_files(resolution)
        with Image.open(png) as sheet:
            return ByteSheet.from_image(sheet, json.loads(index.read_text()), copied=False)

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
    pixels: list[npt.NDArray[np.uint8]]
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

    @cached_property
    def sizes(self) -> npt.NDArray[np.int64]:
        return np.array([[p.shape[1], p.shape[0]] for p in self.pixels], dtype=np.int64)
