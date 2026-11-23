import numpy as np

from railroads.kept import Fresh, Kept, KeptFor, same


def test_keys_of_plain_values_are_the_same_when_equal_and_others_only_when_the_same_object() -> (
    None
):
    trade = np.zeros(3)

    alike = same((1, "a", trade), (1, "a", trade))
    copied = same((1, "a", trade), (1, "a", trade.copy()))

    assert alike
    assert not copied


def test_a_kept_value_is_worked_out_again_only_once_its_key_changes() -> None:
    kept: Kept[int] = Kept()
    made: list[int] = []

    def make() -> int:
        made.append(len(made))
        return len(made)

    values = [kept.get(key, make) for key in (1, 1, 2, 2)]

    assert values == [1, 1, 2, 2]
    assert len(made) == 2


def test_values_kept_for_owners_are_forgotten_for_owners_left_out() -> None:
    kept: KeptFor[str, int] = KeptFor()
    kept.get("a", 0, lambda: 1)
    kept.get("b", 0, lambda: 2)

    kept.keep_only(["a"])
    again = kept.get("b", 0, lambda: 3)

    assert again == 3


def test_a_fresh_value_is_worked_out_again_once_it_is_old() -> None:
    clock = [0.0]
    fresh: Fresh[str, float] = Fresh(0.5, now=lambda: clock[0])
    first = fresh.get("train", lambda: clock[0])
    clock[0] = 0.4
    soon = fresh.get("train", lambda: clock[0])

    clock[0] = 1.0
    later = fresh.get("train", lambda: clock[0])

    assert (first, soon, later) == (0.0, 0.0, 1.0)
