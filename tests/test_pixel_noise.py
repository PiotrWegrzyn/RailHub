import numpy as np
import numpy.typing as npt
import pytest

from railroads.ui.pixel.noise import _MASK, _TABLE, noise, ridged, table_hash


def noise_point_by_point(x: npt.ArrayLike, y: npt.ArrayLike, scale: float, seed: int):
    x, y = np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32)
    u = x * np.float32(1 / scale) + np.float32(seed * 37.1 + 500.0)
    v = y * np.float32(1 / scale) + np.float32(seed * 11.3 + 500.0)
    i, j = np.floor(u).astype(np.int32), np.floor(v).astype(np.int32)
    fu, fv = u - i, v - j
    fu, fv = fu * fu * (3 - 2 * fu), fv * fv * (3 - 2 * fv)
    i0, j0 = i & _MASK, j & _MASK
    i1, j1 = (i0 + 1) & _MASK, (j0 + 1) & _MASK
    top = _TABLE[j0, i0] * (1 - fu) + _TABLE[j0, i1] * fu
    bottom = _TABLE[j1, i0] * (1 - fu) + _TABLE[j1, i1] * fu
    return top * (1 - fv) + bottom * fv


def hash_point_by_point(a: npt.ArrayLike, b: npt.ArrayLike, seed: int):
    a, b = np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64)
    return _TABLE[(b * 7 + seed * 131) & _MASK, (a * 13 + seed * 71) & _MASK]


@pytest.mark.parametrize("shape", [(), (5,), (60, 70)], ids=["scalar", "row", "block"])
@pytest.mark.parametrize(("scale", "seed"), [(2.4, 41), (0.55, 61), (0.03, 86), (1.0, 2)])
def test_noise_matches_its_formula_to_the_last_bit(
    shape: tuple[int, ...], scale: float, seed: int
) -> None:
    rng = np.random.default_rng(seed)
    x, y = rng.uniform(-40, 900, shape), rng.uniform(-40, 900, shape)

    values = noise(x, y, scale, seed)

    expected = noise_point_by_point(x, y, scale, seed)
    assert np.shape(values) == np.shape(expected)
    assert np.asarray(values).dtype == np.asarray(expected).dtype
    assert np.array_equal(values, expected)


def test_noise_wraps_round_the_table_at_whole_steps_and_halves() -> None:
    steps = np.arange(-2048, 2048, 0.5)

    values = noise(steps, steps[::-1], 1.0, 3)

    assert np.array_equal(values, noise_point_by_point(steps, steps[::-1], 1.0, 3))


def test_noise_of_points_turned_to_float32_first_is_the_same() -> None:
    rng = np.random.default_rng(4)
    x, y = rng.uniform(0, 300, 1000), rng.uniform(0, 300, 1000)

    turned = noise(x.astype(np.float32), y.astype(np.float32), 0.16, 62)

    assert np.array_equal(turned, noise(x, y, 0.16, 62))


def test_ridged_noise_peaks_where_noise_crosses_its_middle() -> None:
    rng = np.random.default_rng(5)
    x, y = rng.uniform(0, 300, 1000), rng.uniform(0, 300, 1000)

    ridges = ridged(x, y, 2.6, 1)

    expected = (1 - np.abs(2 * noise_point_by_point(x, y, 2.6, 1) - 1)) ** 2
    assert np.array_equal(ridges, expected)


@pytest.mark.parametrize("seed", [11, 13, 51, 93])
def test_table_hash_matches_its_formula_for_any_integers(seed: int) -> None:
    rng = np.random.default_rng(seed)
    a, b = rng.integers(-100_000, 100_000, 5000), rng.integers(-100_000, 100_000, 5000)

    hashed = table_hash(a, b, seed)

    assert np.array_equal(hashed, hash_point_by_point(a, b, seed))


def test_table_hash_spreads_a_row_and_a_column_over_their_block() -> None:
    columns, rows = np.arange(-5, 9)[None, :], np.arange(3, 12)[:, None]

    hashed = table_hash(columns, rows, 51)

    assert hashed.shape == (9, 14)
    assert np.array_equal(hashed, hash_point_by_point(columns, rows, 51))
