import numpy as np

from railroads.sim.amounts import divided, ratio


def test_a_ratio_is_the_empty_share_where_the_whole_is_nothing() -> None:
    part = np.array([1.0, 2.0, 3.0])
    whole = np.array([4.0, 0.0, 6.0])

    shares = ratio(part, whole, empty=-1.0)

    assert shares.tolist() == [0.25, -1.0, 0.5]


def test_dividing_by_nothing_leaves_the_part_as_it_is() -> None:
    part = np.array([1.0, 2.0, 3.0])
    whole = np.array([4.0, 0.0, 6.0])

    each = divided(part, whole)

    assert each.tolist() == [0.25, 2.0, 0.5]
