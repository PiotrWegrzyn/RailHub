"""Draw the game's sprites into src/railroads/ui/pixel/sprites.png, with sprites.json saying where
each one is and where it stands on the ground. Edit the PNG by hand to repaint a sprite; run this
again only to start over from the drawing code.

Run with: uv run python tools/build_sprites.py
"""

import json

from railroads.ui.pixel.art import draw_sheet
from railroads.ui.pixel.sheet import SHEET_INDEX, SHEET_PNG


def main() -> None:
    sheet, places = draw_sheet()
    sheet.save(SHEET_PNG)
    SHEET_INDEX.write_text(json.dumps(places, indent=1) + "\n")
    print(f"wrote {SHEET_PNG.name}: {len(places)} sprites on {sheet.width} x {sheet.height} pixels")


if __name__ == "__main__":
    main()
