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


@pytest.mark.parametrize(
    "kind",
    [
        BuildingKind.SHEEP_FARM,
        BuildingKind.CATTLE_RANCH,
        BuildingKind.PIG_FARM,
        BuildingKind.CHICKEN_FARM,
    ],
)
def test_a_herd_gathers_in_groups_each_round_its_shelter(kind: BuildingKind) -> None:
    grove = GROVES[kind]
    flock = grove.flock * ART_METRE
    at = (np.arange(12 * 60) + 0.5) * flock / 60
    x, y = np.meshgrid(at, at)

    part, _ = herd_parts(x, y, grove)

    squares = part.reshape(12, 60, 12, 60).transpose(0, 2, 1, 3)
    assert (squares == 5).any(axis=(2, 3)).all()
    assert (squares == 2).any(axis=(2, 3)).mean() > 0.9
    assert (squares[:, :, :3, :] == 2).sum() == 0
