import numpy as np
import pytest

from builders import square_map
from railroads.sim.autoplay import PlayerRule, StandInPlayer, reach_sums
from railroads.sim.constructions import Building, BuildingKind, Constructions
from railroads.sim.grids import HexGrid
from railroads.sim.growth import Growth
from railroads.sim.ladders import GrowthRule
from railroads.sim.points import Points
from railroads.sim.railway import Railway
from railroads.sim.trade import Satisfaction

QUIET = GrowthRule(ladders=(), forests=None)
TOWNS = [(5, 5), (5, 20), (5, 35), (20, 5), (20, 20), (20, 35)]


def grown(towns: list[tuple[int, int]] = TOWNS, same: bool = False) -> Growth:
    constructions = Constructions(square_map())
    for houses, cell in enumerate(towns, start=1):
        for _ in range(1 if same else houses):
            constructions.build(Building(cell, BuildingKind.HOUSE))
    return Growth(constructions, QUIET)


def serving(player: StandInPlayer) -> set[tuple[int, int]]:
    grid = player.growth.constructions.tile_map.grid
    return {
        town for town in TOWNS if any(town in grid.within(s, player.reach) for s in player.stations)
    }


def test_the_first_five_stations_open_at_the_five_most_crowded_towns() -> None:
    growth = grown()

    player = StandInPlayer(growth, Railway(growth.constructions))

    assert serving(player) == set(TOWNS[1:])


def test_two_more_stations_open_each_decade() -> None:
    towns = [*TOWNS, (35, 5), (35, 20), (35, 35)]
    growth = grown(towns, same=True)
    for row, col in towns:
        growth.build(Building((row, col + 1), BuildingKind.FARM))
    player = StandInPlayer(growth, Railway(growth.constructions))

    for _ in range(120):
        player.month(np.random.default_rng(1))

    assert len(player.stations) == 7


def test_no_station_opens_where_nobody_lives() -> None:
    growth = grown(TOWNS[:2])

    player = StandInPlayer(growth, Railway(growth.constructions))

    assert len(player.stations) == 2


def test_a_station_starts_at_the_first_level_and_reaches_one_hex_out() -> None:
    growth = grown([(20, 20)])

    StandInPlayer(
        growth,
        Railway(growth.constructions),
        PlayerRule(stations_at_start=1),
    )

    assert growth.assumed_shares is not None
    assert np.count_nonzero(~np.isnan(growth.assumed_shares)) == 7


def test_every_cell_a_station_reaches_is_served_in_full() -> None:
    growth = grown([(20, 20)])

    StandInPlayer(growth, Railway(growth.constructions), PlayerRule(stations_at_start=1))

    assert growth.assumed_shares is not None
    shares = growth.assumed_shares[~np.isnan(growth.assumed_shares)]
    assert set(shares.tolist()) == {1.0}


@pytest.mark.parametrize(("year", "level"), [(1899, 1), (1900, 2), (1949, 2), (1950, 3)])
def test_stations_stand_at_the_second_level_from_1900_and_the_third_from_1950(
    year: int, level: int
) -> None:
    rule = PlayerRule()

    found = rule.level(year)

    assert found == level


def test_stations_are_raised_when_their_year_comes_and_reach_further() -> None:
    constructions = Constructions(square_map())
    constructions.build(Building((20, 20), BuildingKind.HOUSE))
    growth = Growth(constructions, QUIET, start_year=1899)
    player = StandInPlayer(growth, Railway(constructions), PlayerRule(stations_at_start=1))
    assert growth.assumed_shares is not None
    before = np.count_nonzero(~np.isnan(growth.assumed_shares))

    for _ in range(12):
        player.month(np.random.default_rng(0))

    assert player.railway.level(player.stations[0]) == 2
    assert np.count_nonzero(~np.isnan(growth.assumed_shares)) > before


def test_a_station_beside_a_served_town_leaves_its_cells_as_they_were() -> None:
    growth = grown([(20, 20), (20, 27)], same=True)
    player = StandInPlayer(
        growth,
        Railway(growth.constructions),
        PlayerRule(stations_at_start=1),
    )
    assert growth.assumed_shares is not None
    before = growth.assumed_shares.copy()

    player.open_stations(1)

    served = ~np.isnan(before)
    assert np.array_equal(growth.assumed_shares[served], before[served])


def test_a_thriving_cell_counts_as_served_without_any_trains() -> None:
    growth = grown([(20, 20)])
    flat = int(np.ravel_multi_index((20, 20), growth.constructions.tile_map.grid.shape))
    growth.assumed_shares = np.full(40 * 40, np.nan)
    growth.assumed_shares[flat] = 1.0

    growth.trade_goods()

    assert growth._served[Points.POPULATION][flat] > 0


def test_stations_stand_on_free_cells_beside_the_houses() -> None:
    growth = grown([(20, 20)])

    player = StandInPlayer(growth, Railway(growth.constructions))

    assert player.railway.stations() == player.stations
    assert player.stations[0] != (20, 20)
    assert player.railway.level(player.stations[0]) == 1


def test_served_cells_show_the_satisfaction_drawn_for_them() -> None:
    growth = grown([(20, 20)])
    flat = int(np.ravel_multi_index((20, 20), growth.constructions.tile_map.grid.shape))
    growth.assumed_shares = np.full(40 * 40, np.nan)
    growth.assumed_shares[flat] = 0.5

    growth.trade_goods()

    assert growth.satisfaction()[20, 20] == Satisfaction.CONTENT


def test_reach_sums_add_up_every_cell_within_reach() -> None:
    grid = HexGrid(rows=9, cols=11, km_per_cell=2.0)
    values = np.random.default_rng(3).integers(0, 5, grid.shape)

    sums = reach_sums(grid, values, 2)

    for cell in [(0, 0), (3, 4), (4, 4), (8, 10), (5, 1)]:
        assert sums[cell] == sum(int(values[c]) for c in grid.within(cell, 2))


def test_a_month_of_the_stand_in_is_done_in_steps_as_one_done_at_once() -> None:
    stepped, whole = grown(), grown()
    players = [StandInPlayer(growth, Railway(growth.constructions)) for growth in (stepped, whole)]
    steps = players[0].month_steps(np.random.default_rng(1))

    taken = 0
    while True:
        try:
            next(steps)
            taken += 1
        except StopIteration as done:
            changed = done.value
            break

    assert taken > 1
    assert changed == players[1].month(np.random.default_rng(1))
    assert np.array_equal(stepped.assumed_shares, whole.assumed_shares, equal_nan=True)
