import json

import numpy as np
import pytest
from PIL import Image

from railroads.sim.goods import Good, Need
from railroads.ui.pixel.good_icons import ICON_PIXELS, ICONS, ICONS_INDEX, ICONS_PNG


@pytest.mark.parametrize("good", [*Good, *Need], ids=lambda good: good.value)
def test_every_good_and_need_has_a_picture_on_the_sheet(good: Good | Need) -> None:
    places = json.loads(ICONS_INDEX.read_text())
    with Image.open(ICONS_PNG) as sheet:
        rgba = np.asarray(sheet.convert("RGBA"))

    x, y = places[good.value]

    assert rgba[y : y + ICON_PIXELS, x : x + ICON_PIXELS, 3].any()


def test_no_two_icons_are_the_same_picture() -> None:
    drawn = [draw().image.tobytes() for draw in ICONS.values()]

    assert len(set(drawn)) == len(drawn)
