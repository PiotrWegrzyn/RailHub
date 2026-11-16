import pytest
from test_railway import joined_stations

from railroads.sim.constructions import Track
from railroads.sim.trains import Train


def test_a_way_once_found_is_shared_by_every_train_taking_it() -> None:
    _, railway = joined_stations()

    first = railway.route((6, 3), (6, 20))
    second = railway.route((6, 3), (6, 20))

    assert first is not None
    assert second is first


def test_a_way_is_found_anew_once_its_track_is_pulled_up() -> None:
    _, railway = joined_stations()
    assert railway.route((6, 3), (6, 20)) is not None
    piece = next(p for p in railway.constructions.track() if isinstance(p, Track))

    railway.constructions.remove(piece)

    assert railway.route((6, 3), (6, 20)) is None


def test_a_train_is_placed_by_how_far_along_its_path_it_has_run() -> None:
    _, railway = joined_stations()
    train = Train("Test", path=[(0.0, 0.0), (2.0, 0.0), (2.0, 4.0)], speeds=[1.0, 0.5], along=3.0)

    placed = railway.position(train, back=0.5)

    assert placed == ((2.0, 0.5), (0.0, 4.0))
    assert railway.network.speed_at(train) == 0.5
    assert railway.network.length(train) == 6.0


def test_a_trains_body_runs_from_its_front_back_to_its_tail() -> None:
    _, railway = joined_stations()
    train = Train("Test", path=[(0.0, 0.0), (4.0, 0.0), (4.0, 4.0)], along=5.0)
    step = 0.25

    body = railway.network.body(train, step)

    points = [at for at, _, _ in body]
    count = int(train.length / step) + 1
    assert len(points) == count
    assert points[0] == pytest.approx((4.0, 1.0))
    assert points[-1] == pytest.approx(railway.position(train, back=step * (count - 1))[0])
