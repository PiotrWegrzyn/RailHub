import numpy as np
import pytest

from railroads.mapgen.sources import ElevationGrid, bilinear
from railroads.sim.geo import Bounds


def test_elevation_grid_knows_what_it_covers() -> None:
    grid = ElevationGrid(
        lons=np.array([19.0, 20.0]), lats=np.array([49.0, 50.0]), heights=np.zeros((2, 2))
    )

    covered = grid.covers(Bounds(west=19.2, east=19.8, south=49.2, north=49.8))
    too_far_east = grid.covers(Bounds(west=19.2, east=20.5, south=49.2, north=49.8))

    assert covered
    assert not too_far_east


def test_bilinear_blends_the_four_nearest_values() -> None:
    values = np.array([[0.0, 10.0], [20.0, 30.0]])

    blended = bilinear(values, np.array([0.5, 0.0, 1.0]), np.array([0.5, 1.0, 0.0]))

    assert blended == pytest.approx([15.0, 10.0, 20.0])


def test_bilinear_holds_the_edge_value_outside_the_array() -> None:
    values = np.array([[0.0, 10.0], [20.0, 30.0]])

    blended = bilinear(values, np.array([-3.0]), np.array([9.0]))

    assert blended == pytest.approx([10.0])
