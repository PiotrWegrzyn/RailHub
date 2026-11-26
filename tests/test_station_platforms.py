import pytest

from builders import flat_map
from railroads.sim.constructions import Constructions
from railroads.sim.demolition import bulldoze
from railroads.sim.grids import flat_within_any
from railroads.sim.growth import Growth
from railroads.sim.ladders import GrowthRule
from railroads.sim.railway import Railway
from railroads.sim.routes import add_platform, build_station, lay_route, plan_route
from railroads.sim.signals import Signals
from railroads.sim.trains import Stop

WEST, HUB, NORTH = (8, 3), (8, 12), (1, 12)


def junction(level: int = 4) -> tuple[Growth, Railway]:
    constructions = Constructions(flat_map())
    growth = Growth(constructions, GrowthRule(trades=False))
    for cell in (WEST, HUB, NORTH):
        build_station(growth, cell)
    route = plan_route(constructions, WEST, HUB)
    assert route is not None
    lay_route(growth, route)
    railway = Railway(constructions)
    growth.railway = railway
    railway.finances.earn(10_000_000.0, "test")
    railway.levels[HUB] = level
    return growth, railway


def platform_cell(growth: Growth) -> tuple[int, int]:
    return next(cell for cell in growth.grid.within(HUB, 1) if cell[0] < HUB[0])


def test_a_station_gets_one_platform_at_level_4_and_two_at_level_5() -> None:
    counts = []
    for level in (3, 4, 5):
        _, railway = junction(level)
        counts.append(railway.platforms_left(HUB))

    assert counts == [0, 1, 2]


def test_a_platform_goes_only_in_a_cell_up_to_two_cells_from_its_station() -> None:
    growth, _ = junction()
    two_away = next(c for c, steps in growth.grid.within(HUB, 2).items() if steps == 2)

    far = add_platform(growth, HUB, (4, 12))
    near = add_platform(growth, HUB, two_away)

    assert far is None
    assert near is not None
    assert growth.constructions.platforms(HUB) == [two_away]
    assert growth.constructions.main_station(two_away) == HUB
    assert growth.constructions.stations() == sorted([WEST, HUB, NORTH])


def test_a_train_crosses_to_the_platform_its_next_line_leaves_from() -> None:
    growth, railway = junction()
    platform = platform_cell(growth)
    assert add_platform(growth, HUB, platform, (3, 9)) is not None
    north = plan_route(growth.constructions, platform, NORTH)
    assert north is not None
    lay_route(growth, north)

    way = railway.route(WEST, NORTH)

    assert way is not None
    cells = {railway.network.cell_at(point) for point in way[0]}
    assert {HUB, platform} <= cells


def test_a_train_reaching_a_station_by_its_platform_sets_off_again_from_there() -> None:
    growth, railway = junction()
    platform = platform_cell(growth)
    assert add_platform(growth, HUB, platform, (3, 9)) is not None
    north = plan_route(growth.constructions, platform, NORTH)
    assert north is not None
    lay_route(growth, north)
    train = railway.buy()
    assert train is not None
    train.stops = [Stop(NORTH, wait_for_load=False), Stop(HUB, wait_for_load=False)]

    for _ in range(400):
        railway.run(0.01)
        if train.arrivals:
            break

    assert train.platform == platform
    assert train.status != "No track to its next stop"


def test_bulldozing_a_station_takes_its_platforms_down_with_it() -> None:
    growth, _ = junction(5)
    platform = platform_cell(growth)
    assert add_platform(growth, HUB, platform) is not None

    changed = bulldoze(growth, HUB)

    assert platform in changed
    assert growth.constructions.station_at(platform) is None
    assert growth.constructions.platforms(HUB) == []


def test_a_station_hands_goods_out_around_its_platforms_too() -> None:
    growth, railway = junction()
    platform = platform_cell(growth)
    assert add_platform(growth, HUB, platform) is not None
    railway.levels[HUB] = 1
    beyond = next(
        cell for cell in growth.grid.within(platform, 1) if cell not in growth.grid.within(HUB, 1)
    )

    ring = growth.station_trade.inner_ring(HUB)

    reached = {divmod(int(index), growth.grid.cols) for index in ring}
    assert beyond in reached
    assert set(growth.grid.within(HUB, 1)) <= reached


def test_the_reach_of_a_station_with_platforms_counts_each_cell_from_the_nearest() -> None:
    growth, _ = junction()
    platform = platform_cell(growth)

    within = flat_within_any(growth.grid, (HUB, platform), 2)

    steps = dict(zip(within.cells.tolist(), within.steps.tolist(), strict=True))
    flat = growth.grid.cols * platform[0] + platform[1]
    assert steps[flat] == 0.0
    assert len(steps) == len(set(within.cells.tolist()))


def test_a_train_bound_for_a_station_takes_its_other_platform_when_the_way_to_one_is_held(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    growth, railway = junction()
    platform = (10, 12)
    assert add_platform(growth, HUB, platform) is not None
    line = plan_route(growth.constructions, WEST, HUB)
    branch = plan_route(growth.constructions, WEST, platform)
    assert line is not None and branch is not None
    lay_route(growth, branch)
    own = set(branch.cells) - set(line.cells)
    monkeypatch.setattr(
        Signals, "held_by_others", lambda _, __, cells: 0 if own & set(cells) else 1
    )
    train = railway.buy()
    assert train is not None
    train.stops = [Stop(WEST, wait_for_load=False), Stop(HUB, wait_for_load=False)]

    railway.run(1e-4)

    assert railway.network.cell_at(train.path[-1]) == platform


def test_the_way_on_from_a_station_is_measured_from_the_platform_a_train_pulls_into() -> None:
    growth, railway = junction()
    platform = platform_cell(growth)
    assert add_platform(growth, HUB, platform, (3, 9)) is not None
    north = plan_route(growth.constructions, platform, NORTH)
    assert north is not None
    lay_route(growth, north)
    stops = [Stop(NORTH), Stop(HUB)]

    legs = railway.leg_lengths(stops, "planet")

    assert legs[1] == railway.network.route_length(platform, NORTH)
    assert legs[1] < railway.network.route_length(HUB, NORTH)
