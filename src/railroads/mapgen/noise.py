import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def fractal_noise(
    x: FloatArray, y: FloatArray, rng: np.random.Generator, feature_size: float, octaves: int = 4
) -> FloatArray:
    """Smooth random values between 0 and 1 at the given points: blobs about `feature_size`
    across, with finer and fainter detail layered on top."""
    total = np.zeros(np.shape(x))
    weight, amplitude, size = 0.0, 1.0, feature_size
    for _ in range(octaves):
        total += amplitude * _value_noise(x / size, y / size, rng)
        weight += amplitude
        amplitude /= 2
        size /= 2
    return total / weight


def _value_noise(u: FloatArray, v: FloatArray, rng: np.random.Generator) -> FloatArray:
    i, j = np.floor(u).astype(np.int64), np.floor(v).astype(np.int64)
    i0, j0 = int(i.min()), int(j.min())
    lattice = rng.random((int(j.max()) - j0 + 2, int(i.max()) - i0 + 2))
    fu, fv = _smoothstep(u - i), _smoothstep(v - j)
    li, lj = i - i0, j - j0
    top = lattice[lj, li] * (1 - fu) + lattice[lj, li + 1] * fu
    bottom = lattice[lj + 1, li] * (1 - fu) + lattice[lj + 1, li + 1] * fu
    return top * (1 - fv) + bottom * fv


def _smoothstep(t: FloatArray) -> FloatArray:
    return t * t * (3 - 2 * t)
