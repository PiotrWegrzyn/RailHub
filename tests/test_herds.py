import numpy as np
import pytest

from railroads.sim.constructions import BuildingKind
from railroads.ui.pixel.ground import herd_parts
from railroads.ui.pixel.ground_palette import ART_METRE, GROVES


@pytest.mark.parametrize(
    "kind",
    [
        BuildingKind.SHEEP_FARM,
        BuildingKind.CATTLE_RANCH,
        BuildingKind.PIG_FARM,
        BuildingKind.CHICKEN_FARM,
    ],
)
def test_a_herds_animals_stand_at_their_true_length(kind: BuildingKind) -> None:
    grove = GROVES[kind]
    step = 0.01
    x, y = np.meshgrid(np.arange(0, 60, step), np.arange(0, 6, step))

    part, _ = herd_parts(x, y, grove)

    body = np.pad(part == 2, ((0, 0), (1, 1))).astype(np.int8)
    edges = np.diff(body, axis=1)
    starts, stops = np.nonzero(edges == 1), np.nonzero(edges == -1)
    longest = (stops[1] - starts[1]).max() * step
    assert longest == pytest.approx(grove.herd * ART_METRE, rel=0.1)


def test_a_herd_leaves_some_of_its_squares_empty() -> None:
    grove = GROVES[BuildingKind.PIG_FARM]
    gap = grove.gap * ART_METRE
    at = (np.arange(40 * 50) + 0.5) * gap / 50
    x, y = np.meshgrid(at, at)

    part, _ = herd_parts(x, y, grove)

    squares = (part >= 2).reshape(40, 50, 40, 50).any(axis=(1, 3))
    assert 0.3 < squares.mean() < 0.8
