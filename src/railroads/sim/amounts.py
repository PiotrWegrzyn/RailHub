"""Small sums over amounts of goods and money that many modules make the same way."""

import numpy as np
import numpy.typing as npt

# Amounts closer together than this count as the same: what is left of a sum of floats that
# should come to nothing, or to all of something.
EPSILON = 1e-9

type Amounts = npt.ArrayLike


def ratio(part: Amounts, whole: Amounts, empty: float = 0.0) -> npt.NDArray[np.float64]:
    """`part` as a share of `whole`, cell by cell, and `empty` where the whole is nothing."""
    part, whole = np.broadcast_arrays(np.asarray(part, np.float64), np.asarray(whole, np.float64))
    return np.divide(part, whole, out=np.full(part.shape, empty), where=whole > 0)


def divided(part: Amounts, whole: Amounts) -> npt.NDArray[np.float64]:
    """`part` divided by `whole` where the whole is something, and `part` itself where it is
    nothing."""
    whole = np.asarray(whole, np.float64)
    return np.asarray(part, np.float64) / np.where(whole > 0, whole, 1.0)
