"""Loads the sprite sheet the game draws its pixel art from."""

import json
from dataclasses import dataclass
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

    def index(self, name: str) -> int:
        return self.names.index(name)

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
