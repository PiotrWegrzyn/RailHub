import itertools

import numpy as np

from railroads.sim.goods import Good
from railroads.sim.grids import HexGrid
from railroads.sim.trains import ENGINES as ROSTER
from railroads.sim.trains import Car, Power, Train
from railroads.ui.pixel.projection import PixelProjection
from railroads.ui.pixel.train_layer import TrainLayer
from railroads.ui.pixel.trains import ENGINES, MODELS, Look, train_sprites


def sheet_key() -> tuple[float, ...]:
    projection = PixelProjection.of(HexGrid(rows=10, cols=10, km_per_cell=2.0), 0.0, 1.0)
    return tuple(float(v) for v in projection.matrix.ravel())


def test_every_engine_that_can_be_bought_has_a_look() -> None:
    looks = set(ENGINES)

    assert looks == set(ROSTER)


def test_steam_engines_and_only_they_have_a_chimney() -> None:
    steam = {name for name, engine in ROSTER.items() if engine.power == Power.STEAM}

    smoking = {name for name, model in ENGINES.items() if model.chimney is not None}

    assert smoking == steam


def test_engines_of_a_class_grow_longer_over_the_years() -> None:
    def kind_and_power(name: str) -> tuple[str, str]:
        return ROSTER[name].kind.value, ROSTER[name].power.value

    by_class = itertools.groupby(sorted(ROSTER, key=kind_and_power), key=kind_and_power)

    for _, names in by_class:
        lengths = [ENGINES[name].half for name in sorted(names, key=lambda name: ROSTER[name].year)]
        assert lengths == sorted(lengths)


def test_every_engine_looks_different() -> None:
    sprites = train_sprites(sheet_key())

    pictures = set()
    for name in ENGINES:
        x, y, width, height, _, _ = sprites.place((Look.ENGINE, name), 0.0)
        pictures.add(sprites.image[y : y + height, x : x + width].tobytes())

    assert len(pictures) == len(ENGINES)


def test_an_express_leads_its_coaches_with_the_dining_car() -> None:
    layer = TrainLayer.__new__(TrainLayer)
    train = Train("Express", engine="bullet")
    train.cars = [Car(Good.PASSENGERS, dining=True), Car(Good.PASSENGERS, load=10.0)]

    looks = [look for look, _ in layer._parts(train)]

    assert looks == [(Look.ENGINE, "bullet"), Look.DINING, (Good.PASSENGERS, True)]


def test_the_dining_car_is_on_the_sheet_and_unlike_a_coach() -> None:
    sprites = train_sprites(sheet_key())

    def picture(look: object) -> np.ndarray:
        x, y, width, height, _, _ = sprites.place(look, 0.0)
        return sprites.image[y : y + height, x : x + width]

    dining, coach = picture(Look.DINING), picture((Good.PASSENGERS, True))

    assert Look.DINING in MODELS
    assert dining.shape != coach.shape or not np.array_equal(dining, coach)
