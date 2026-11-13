from railroads.sim.goods import Good
from railroads.sim.trains import Stop, Train
from railroads.ui.trains_window import carried_in


def train_calling_at(*stops: Stop) -> Train:
    return Train("Test", list(stops))


def test_a_stop_offers_the_cars_the_train_brings_in_from_the_stop_before() -> None:
    train = train_calling_at(
        Stop((1, 1), [Good.CATTLE, Good.CATTLE, Good.COAL]), Stop((5, 5)), Stop((9, 9))
    )

    carried = carried_in(train, 1)

    assert carried == [Good.CATTLE, Good.COAL]


def test_the_first_stop_offers_what_the_train_brings_back_from_the_last() -> None:
    train = train_calling_at(Stop((1, 1)), Stop((5, 5), [Good.WHEAT]))

    carried = carried_in(train, 0)

    assert carried == [Good.WHEAT]
