"""Draw the game's sprites into src/railroads/ui/pixel/sprites.png, with sprites.json saying where
each one is and where it stands on the ground, the same sprites drawn twice and four times as
finely into sprites@2x.png and sprites@4x.png, and at every other resolution the graphics card
stands them at (DENSITIES) each into a PNG of its own in sprites@Nx.zip, each with its own JSON,
and the icons of goods into
good_icons.png. Edit the PNGs by hand to repaint a sprite; run this again only to start over from the drawing code.

Run with: uv run python tools/build_sprites.py
"""

import io
import json
import zipfile
from pathlib import Path

from PIL import Image

from railroads.ui.pixel.art.registry import draw_sheet
from railroads.ui.pixel.good_icons import ICONS_INDEX, ICONS_PNG, draw_icon_sheet
from railroads.ui.pixel.sheet import DENSITIES, sheet_files, sprite_name


def zip_sprites(sheet: Image.Image, places: dict[str, list[int]], path: Path) -> None:
    """Each sprite on the sheet as its own PNG in a zip (ByteSheet.zipped)."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as sprites:
        for index, (x, y, w, h, _, _) in enumerate(places.values()):
            png = io.BytesIO()
            sheet.crop((x, y, x + w, y + h)).save(png, "PNG", optimize=True)
            sprites.writestr(sprite_name(index), png.getvalue())


def main() -> None:
    for resolution in DENSITIES:
        path, index = sheet_files(resolution)
        sheet, places = draw_sheet(resolution=resolution)
        if path.suffix == ".zip":
            zip_sprites(sheet, places, path)
        else:
            sheet.save(path)
        index.write_text(json.dumps(places, indent=1) + "\n")
        print(f"wrote {path.name}: {len(places)} sprites drawn {resolution} times finer")
    icons, icon_places = draw_icon_sheet()
    icons.save(ICONS_PNG)
    ICONS_INDEX.write_text(json.dumps(icon_places, indent=1) + "\n")
    print(f"wrote {ICONS_PNG.name}: {len(icon_places)} icons")


if __name__ == "__main__":
    main()
