import numpy as np
import pytest

from builders import flat_map
from railroads.sim.autoplay import PlayerRule, StandInPlayer, reach_sums
from railroads.sim.constructions import Building, BuildingKind, Constructions
from railroads.sim.economy import Satisfaction
from railroads.sim.grids import HexGrid
from railroads.sim.growth import Growth
from railroads.sim.ladders import GrowthRule, Points
from railroads.sim.railway import Railway
from railroads.sim.tilemap import TileMap

QUIET = GrowthRule(ladders=(), forests=None)
TOWNS = [(5, 5), (5, 20), (5, 35), (20, 5), (20, 20), (20, 35)]


def make_map(size: int = 40) -> TileMap:
    return flat_map(size, size)


def grown(towns: list[tuple[int, int]] = TOWNS, same: bool = False) -> Growth:
    constructions = Constructions(make_map())
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

    player = StandInPlayer(growth, Railway(growth.constructions), np.random.default_rng(0))

    assert serving(player) == set(TOWNS[1:])


def test_two_more_stations_open_each_decade() -> None:
    growth = grown([*TOWNS, (35, 5), (35, 20), (35, 35)], same=True)
    player = StandInPlayer(growth, Railway(growth.constructions), np.random.default_rng(0))

    for _ in range(120):
        player.month(np.random.default_rng(1))

    assert len(player.stations) == 7


def test_no_station_opens_where_nobody_lives() -> None:
    growth = grown(TOWNS[:2])

    player = StandInPlayer(growth, Railway(growth.constructions), np.random.default_rng(0))

    assert len(player.stations) == 2


def test_a_station_is_raised_a_level_and_reaches_three_hexes_out() -> None:
    growth = grown([(20, 20)])

    StandInPlayer(
        growth,
        Railway(growth.constructions),
        np.random.default_rng(0),
        PlayerRule(stations_at_start=1),
    )

    assert growth.assumed_shares is not None
    assert np.count_nonzero(~np.isnan(growth.assumed_shares)) == 37


def test_a_third_of_the_cells_reached_thrive_and_half_are_content() -> None:
    growth = grown([(20, 20)])
    rule = PlayerRule(stations_at_start=1, station_level=60)

    StandInPlayer(growth, Railway(growth.constructions), np.random.default_rng(0), rule)

    assert growth.assumed_shares is not None
    shares = growth.assumed_shares[~np.isnan(growth.assumed_shares)]
    assert np.mean(shares == 1.0) == pytest.approx(1 / 3, abs=0.03)
    assert np.mean(shares == 0.5) == pytest.approx(1 / 2, abs=0.03)


def test_a_station_beside_a_served_town_leaves_its_cells_as_they_were() -> None:
    growth = grown([(20, 20), (20, 27)], same=True)
    player = StandInPlayer(
        growth,
        Railway(growth.constructions),
        np.random.default_rng(0),
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

    player = StandInPlayer(growth, Railway(growth.constructions), np.random.default_rng(0))

    assert player.railway.stations() == player.stations
    assert player.stations[0] != (20, 20)
    assert player.railway.level(player.stations[0]) == 2


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
