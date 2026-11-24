from railroads.sim.goods import General, Good
from railroads.sim.grids import Cell
from railroads.sim.train_names import cars_used_most, train_titles
from railroads.sim.trains import ENGINES, Stop, Train

NAMES = {(0, 0): "Warka", (5, 5): "Grójec", (9, 9): "Magnuszew"}


def name_of(cell: Cell) -> str:
    return NAMES[cell]


def train(name: str, *cells: Cell, engine: str = "planet") -> Train:
    return Train(name, [Stop(cell) for cell in cells], engine=engine)


def test_a_train_is_called_after_its_engine_and_the_number_it_was_bought_as() -> None:
    trains = [train("Steam engine 1", (0, 0), (9, 9)), train("Steam engine 2", engine="goods")]

    titles = train_titles(trains)

    assert titles == {
        "Steam engine 1": f"{ENGINES['planet'].title} 1",
        "Steam engine 2": f"{ENGINES['goods'].title} 2",
    }


def test_a_name_the_player_gave_is_kept() -> None:
    named = train("Steam engine 1", (0, 0), (9, 9))
    named.label = "Night mail"

    titles = train_titles([named])

    assert titles == {"Steam engine 1": "Night mail"}


def test_the_cars_used_most_are_counted_over_every_stop_most_first() -> None:
    busy = Train(
        "a",
        [
            Stop((0, 0), [Good.COAL, Good.COAL, Good.MAIL]),
            Stop((5, 5), [Good.WHEAT, General.FOOD]),
            Stop((9, 9), [Good.COAL, Good.WHEAT]),
        ],
    )

    most = cars_used_most(busy)

    assert most == [(Good.COAL, 3), (Good.WHEAT, 2), (Good.MAIL, 1)]
