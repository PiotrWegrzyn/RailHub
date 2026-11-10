"""Smooth random values for drawing the map, all read from one table of random numbers so each
costs a few lookups: value noise over the map's plane, ridged noise (sharp crests where value noise
crosses its middle), and repeatable random numbers for integer places. The table repeats every
TABLE_SIZE steps of any scale, too far apart to notice."""

import numpy as np
import numpy.typing as npt

TABLE_SIZE = 1024
_TABLE = np.random.default_rng(5).random((TABLE_SIZE, TABLE_SIZE)).astype(np.float32)
_MASK = TABLE_SIZE - 1

Floats = npt.NDArray[np.float32]


def noise(x: npt.ArrayLike, y: npt.ArrayLike, scale: float, seed: int) -> Floats:
    """Value noise from 0 to 1 at plane points, varying over about `scale` cells."""
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


def ridged(x: npt.ArrayLike, y: npt.ArrayLike, scale: float, seed: int) -> Floats:
    """Noise from 0 to 1 that peaks in sharp ridges and falls away into broad hollows."""
    return (1 - np.abs(2 * noise(x, y, scale, seed) - 1)) ** 2


def table_hash(a: npt.ArrayLike, b: npt.ArrayLike, seed: int) -> Floats:
    """A repeatable random number from 0 to 1 for each pair of integers."""
    a, b = np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64)
    return _TABLE[(b * 7 + seed * 131) & _MASK, (a * 13 + seed * 71) & _MASK]
