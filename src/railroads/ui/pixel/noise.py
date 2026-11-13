"""Smooth random values for drawing the map, all read from one table of random numbers so each
costs a few lookups: value noise over the map's plane, ridged noise (sharp crests where value noise
crosses its middle), and repeatable random numbers for integer places. The table repeats every
TABLE_SIZE steps of any scale, too far apart to notice.

Chunks of the map ask for these at tens of thousands of points at a time, so they are worked out
with as few passes over the points as the same results allow: lookups go through the table laid
out flat, and value noise reads a copy of the table with its first row and column repeated after
its last, so the four corners round a point sit at fixed steps from the first."""

import numpy as np
import numpy.typing as npt

TABLE_SIZE = 1024
_TABLE = np.random.default_rng(5).random((TABLE_SIZE, TABLE_SIZE)).astype(np.float32)
_MASK = TABLE_SIZE - 1
_FLAT = _TABLE.ravel()
_WRAPPED_SIZE = TABLE_SIZE + 1
_WRAPPED = np.pad(_TABLE, ((0, 1), (0, 1)), mode="wrap").ravel()

Floats = npt.NDArray[np.float32]


def noise(x: npt.ArrayLike, y: npt.ArrayLike, scale: float, seed: int) -> Floats:
    """Value noise from 0 to 1 at plane points, varying over about `scale` cells. Points given as
    float32 are used as they are, so points asked about at several scales can be turned to
    float32 once."""
    x, y = np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32)
    u = x * np.float32(1 / scale)
    u += np.float32(seed * 37.1 + 500.0)
    v = y * np.float32(1 / scale)
    v += np.float32(seed * 11.3 + 500.0)
    i, j = np.floor(u), np.floor(v)
    corner = (j.astype(np.intp) & _MASK) * _WRAPPED_SIZE
    corner += i.astype(np.intp) & _MASK
    # The part of a step past the corner is exact in float32, and the same in float64 as the
    # difference of the point and its whole steps.
    u -= i
    v -= j
    fu, fv = _eased(u), _eased(v)
    before = 1 - fu
    top = _WRAPPED[corner] * before
    corner += 1
    top += _WRAPPED[corner] * fu
    corner += TABLE_SIZE
    bottom = _WRAPPED[corner] * before
    corner += 1
    bottom += _WRAPPED[corner] * fu
    top *= 1 - fv
    bottom *= fv
    top += bottom
    return top


def _eased(part: Floats) -> npt.NDArray[np.float64]:
    """3t² - 2t³ of each part of a step, in float64."""
    t = part.astype(np.float64)
    rise = t * -2.0
    rise += 3
    t *= t
    t *= rise
    return t


def ridged(x: npt.ArrayLike, y: npt.ArrayLike, scale: float, seed: int) -> Floats:
    """Noise from 0 to 1 that peaks in sharp ridges and falls away into broad hollows."""
    return (1 - np.abs(2 * noise(x, y, scale, seed) - 1)) ** 2


def table_hash(a: npt.ArrayLike, b: npt.ArrayLike, seed: int) -> Floats:
    """A repeatable random number from 0 to 1 for each pair of integers."""
    a, b = np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64)
    row = b * 7
    row += seed * 131
    row &= _MASK
    row <<= 10
    column = a * 13
    column += seed * 71
    column &= _MASK
    return _FLAT[row | column]
