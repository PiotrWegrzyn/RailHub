from railroads.sim.goods import General, Good
from railroads.sim.grids import Cell
from railroads.sim.train_names import cars_used_most, route_name, train_titles
from railroads.sim.trains import ENGINES, Stop, Train

NAMES = {(0, 0): "Warka", (5, 5): "Grójec", (9, 9): "Magnuszew"}


def name_of(cell: Cell) -> str:
    return NAMES[cell]


def train(name: str, *cells: Cell, engine: str = "planet") -> Train:
    return Train(name, [Stop(cell) for cell in cells], engine=engine)


def test_a_normal_train_is_called_after_its_first_and_last_stations() -> None:
    normal = train("Steam engine 1", (0, 0), (5, 5), (9, 9))

    name = route_name(normal, name_of)

    assert name == "Warka – Magnuszew"


def test_express_light_and_heavy_engines_add_their_category_to_the_name() -> None:
    trains = [
        train("a", (0, 0), (9, 9), engine="single"),
        train("b", (0, 0), (9, 9), engine="tank"),
        train("c", (0, 0), (9, 9), engine="goods"),
        train("d", (0, 0), (9, 9), engine="rocket"),
    ]

    names = [route_name(each, name_of) for each in trains]

    assert names == [
        "Warka – Magnuszew Express",
        "Warka – Magnuszew Local",
        "Warka – Magnuszew Cargo",
        "Warka – Magnuszew",
    ]


def test_a_route_back_to_its_first_station_ends_at_the_last_other_one() -> None:
    round_trip = train("a", (0, 0), (9, 9), (5, 5), (0, 0))

    name = route_name(round_trip, name_of)

    assert name == "Warka – Grójec"


def test_a_train_without_stops_is_called_after_its_engine_and_with_one_after_its_station() -> None:
    bare, single = train("a"), train("b", (5, 5))

    names = route_name(bare, name_of), route_name(single, name_of)

    assert names == (ENGINES["planet"].title, "Grójec")


def test_later_trains_with_the_same_name_are_numbered_from_two() -> None:
    trains = [
        train("a", (0, 0), (9, 9)),
        train("b", (0, 0), (9, 9)),
        train("c", (0, 0), (9, 9), engine="goods"),
        train("d", (0, 0), (9, 9)),
    ]

    titles = train_titles(trains, name_of)

    assert titles == {
        "a": "Warka – Magnuszew",
        "b": "Warka – Magnuszew 2",
        "c": "Warka – Magnuszew Cargo",
        "d": "Warka – Magnuszew 3",
    }


def test_a_name_the_player_gave_is_kept_and_takes_no_number_from_the_route() -> None:
    named = train("a", (0, 0), (9, 9))
    named.label = "Night mail"
    other = train("b", (0, 0), (9, 9))

    titles = train_titles([named, other], name_of)

    assert titles == {"a": "Night mail", "b": "Warka – Magnuszew"}


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
