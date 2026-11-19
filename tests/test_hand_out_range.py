from builders import flat_map
from railroads.sim.constructions import Building, BuildingKind, Constructions
from railroads.ui.map_overlays import consumers_in_reach


def built(*buildings: tuple[tuple[int, int], BuildingKind]) -> Constructions:
    constructions = Constructions(flat_map(30, 30))
    for cell, kind in buildings:
        constructions.build(Building(cell, kind))
    return constructions


def test_a_farm_reaches_the_houses_up_to_six_hexes_away_and_no_further() -> None:
    constructions = built(
        ((10, 4), BuildingKind.FARM),
        ((10, 6), BuildingKind.HOUSE),
        ((10, 10), BuildingKind.HOUSE),
        ((10, 11), BuildingKind.HOUSE),
    )

    consumers = consumers_in_reach(constructions, (10, 4), 1850)

    assert set(consumers) == {(10, 6), (10, 10)}


def test_buildings_that_need_nothing_a_cell_makes_are_not_consumers() -> None:
    constructions = built(((10, 4), BuildingKind.FARM), ((10, 6), BuildingKind.LOGGING_CAMP))

    consumers = consumers_in_reach(constructions, (10, 4), 1850)

    assert consumers == ()


def test_a_cell_that_makes_nothing_has_no_consumers() -> None:
    constructions = built(((10, 4), BuildingKind.HOUSE), ((10, 5), BuildingKind.HOUSE))

    consumers = consumers_in_reach(constructions, (10, 4), 1850)

    assert consumers == ()
