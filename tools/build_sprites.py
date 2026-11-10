"""Draw the game's sprites into src/railroads/ui/pixel/sprites.png, with sprites.json saying where
each one is and where it stands on the ground, the same sprites drawn twice as finely into
sprites@2x.png and sprites@2x.json, and the icons of goods into good_icons.png. Edit
the PNGs by hand to repaint a sprite; run this again only to start over from the drawing code.

Run with: uv run python tools/build_sprites.py
"""

import json

from railroads.ui.pixel.art import draw_sheet
from railroads.ui.pixel.good_icons import ICONS_INDEX, ICONS_PNG, draw_icon_sheet
from railroads.ui.pixel.sheet import FINE, FINE_INDEX, FINE_PNG, SHEET_INDEX, SHEET_PNG


def main() -> None:
    for resolution, png, index in ((1, SHEET_PNG, SHEET_INDEX), (FINE, FINE_PNG, FINE_INDEX)):
        sheet, places = draw_sheet(resolution=resolution)
        sheet.save(png)
        index.write_text(json.dumps(places, indent=1) + "\n")
        print(f"wrote {png.name}: {len(places)} sprites on {sheet.width} x {sheet.height} pixels")
    icons, icon_places = draw_icon_sheet()
    icons.save(ICONS_PNG)
    ICONS_INDEX.write_text(json.dumps(icon_places, indent=1) + "\n")
    print(f"wrote {ICONS_PNG.name}: {len(icon_places)} icons")


if __name__ == "__main__":
    main()
