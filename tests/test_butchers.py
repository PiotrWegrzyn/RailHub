from growing import build, make_map, only

from railroads.sim.breeding import ANIMAL_FARMS
from railroads.sim.constructions import BuildingKind, Constructions
from railroads.sim.growth import Growth
from railroads.sim.ladders import BUTCHERS

TOWN = (5, 5)
BESIDE = (5, 6)


def town(animals: BuildingKind | None) -> Growth:
    constructions = Constructions(make_map())
    build(constructions, TOWN, BuildingKind.HOUSE, 3)
    if animals is not None:
        build(constructions, BESIDE, animals, 1)
    return Growth(constructions, only(BUTCHERS))


def test_animal_farms_around_people_draw_butchers() -> None:
    plain = town(None).chances(BUTCHERS)[TOWN]

    with_animals = [town(kind).chances(BUTCHERS)[TOWN] for kind in ANIMAL_FARMS]

    assert all(chance > plain for chance in with_animals)


def test_fields_draw_no_butchers() -> None:
    plain = town(None).chances(BUTCHERS)[TOWN]

    with_wheat = town(BuildingKind.FARM).chances(BUTCHERS)[TOWN]

    assert with_wheat == plain
