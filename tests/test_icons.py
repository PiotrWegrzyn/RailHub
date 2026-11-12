import re
from pathlib import Path

import imgui_bundle
import pytest
from imgui_bundle import icons_fontawesome_4 as fa
from PIL import ImageFont

UI = Path(__file__).resolve().parents[1] / "src" / "railroads" / "ui"
ICON_FONT = Path(imgui_bundle.__file__).parent / "assets" / "fonts" / "fontawesome-webfont.ttf"
USED = sorted(
    {name for path in UI.rglob("*.py") for name in re.findall(r"ICON_FA_\w+", path.read_text())}
)


@pytest.mark.parametrize("name", USED)
def test_every_icon_the_ui_shows_is_in_the_icon_font(name: str) -> None:
    font = ImageFont.truetype(str(ICON_FONT), 20)
    missing = font.getmask("\U0010fffd").getbbox()

    drawn = font.getmask(getattr(fa, name)).getbbox()

    assert drawn not in (None, missing)
