import pytest

from railroads.sim.constructions import Track, TrackWidth
from railroads.sim.finance import START_CASH, Finances, dollars, track_price, track_upkeep
from railroads.sim.tilemap import Relief, Temperature, Terrain, Tile, Vegetation
from railroads.sim.track_rules import TrackBed

FLAT = Tile(Terrain.LAND, Vegetation.GRASS, Temperature.NORMAL, 0, Relief.FLAT)
RIVER = Tile(Terrain.RIVER, Vegetation.GRASS, Temperature.NORMAL, 0, Relief.FLAT)


def test_a_game_starts_with_eighty_thousand() -> None:
    finances = Finances()

    assert finances.cash == START_CASH == 80_000


def test_the_books_keep_this_month_and_last_by_what_money_was_for() -> None:
    finances = Finances()
    finances.spend(15_000, "stations")
    finances.earn(200, "cargo")

    finances.close_month()
    finances.spend(100, "station upkeep")

    assert finances.cash == pytest.approx(START_CASH - 15_000 + 200 - 100)
    assert finances.last_month == {"stations": -15_000, "cargo": 200}
    assert finances.profit == -100


def test_spending_past_the_cash_runs_into_debt_but_cannot_be_afforded() -> None:
    finances = Finances(cash=50)

    affordable = finances.can_afford(100)
    finances.spend(100, "station upkeep")

    assert not affordable
    assert finances.cash == -50


@pytest.mark.parametrize(("amount", "shown"), [(15000, "$15,000"), (-1200.4, "-$1,200"), (0, "$0")])
def test_money_shows_in_whole_dollars(amount: float, shown: str) -> None:
    assert dollars(amount) == shown


@pytest.mark.parametrize(
    ("piece", "tile", "price"),
    [
        (Track((0, 0), (0, 6)), FLAT, 300),
        (Track((0, 0), (0, 6), TrackWidth.DOUBLE), FLAT, 540),
        (Track((0, 0), (0, 6), electric=True), FLAT, 750),
        (Track((0, 0), (0, 6), TrackWidth.DOUBLE, electric=True), FLAT, 1350),
        (Track((0, 0), (0, 6), TrackWidth.DOUBLE, TrackBed.BRIDGE), RIVER, 18_000),
    ],
)
def test_double_track_costs_more_and_electric_track_more_again(
    piece: Track, tile: Tile, price: float
) -> None:
    assert track_price(piece, tile) == pytest.approx(price)


def test_each_cell_of_track_pays_upkeep_once_for_its_dearest_piece() -> None:
    pieces = [
        Track((0, 0), (0, 6)),
        Track((0, 1), (0, 6), TrackWidth.DOUBLE),
        Track((0, 1), (2, 8)),
        Track((0, 2), (0, 6), TrackWidth.DOUBLE, electric=True),
    ]

    upkeep = track_upkeep(pieces)

    assert upkeep == pytest.approx(2 + 3.6 + 9)
