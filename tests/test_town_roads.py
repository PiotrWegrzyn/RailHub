import pytest

from builders import flat_map
from railroads.sim.constructions import Building, BuildingKind, Constructions
from railroads.sim.grids import Cell
from railroads.ui.pixel import town_roads as town_roads_module
from railroads.ui.pixel.town_roads import Roads, corner_key, town_roads


def built(*placed: tuple[Cell, BuildingKind]) -> Constructions:
    constructions = Constructions(flat_map())
    for cell, kind in placed:
        constructions.build(Building(cell, kind))
    return constructions


def edges_of(roads: Roads) -> set[frozenset[tuple[int, int]]]:
    return {
        frozenset((corner_key(x0, y0), corner_key(x1, y1)))
        for x0, y0, x1, y1 in roads.segments.tolist()
    }


def edges_of_cell(constructions: Constructions, cell: Cell) -> set[frozenset[tuple[int, int]]]:
    corners = [corner_key(x, y) for x, y in constructions.tile_map.grid.corners(*cell)]
    return {frozenset((corners[i], corners[(i + 1) % 6])) for i in range(6)}


def pieces(roads: Roads) -> int:
    parent: dict[tuple[int, int], tuple[int, int]] = {}

    def find(corner: tuple[int, int]) -> tuple[int, int]:
        while parent.setdefault(corner, corner) != corner:
            corner = parent[corner]
        return corner

    for x0, y0, x1, y1 in roads.segments.tolist():
        parent[find(corner_key(x0, y0))] = find(corner_key(x1, y1))
    return len({find(corner) for corner in list(parent)})


def test_two_homes_side_by_side_share_a_road_along_their_common_edge() -> None:
    constructions = built(((4, 4), BuildingKind.HOUSE), ((4, 5), BuildingKind.HOUSE))

    roads = town_roads(constructions)

    shared = edges_of_cell(constructions, (4, 4)) & edges_of_cell(constructions, (4, 5))
    assert shared <= edges_of(roads)


def test_a_works_gets_a_road_along_every_edge_its_slots_face() -> None:
    constructions = built(((4, 4), BuildingKind.FACTORY), ((4, 5), BuildingKind.HOUSE))

    roads = town_roads(constructions)

    assert len(edges_of_cell(constructions, (4, 4)) & edges_of(roads)) >= 3


def test_a_building_standing_alone_gets_a_road_along_the_edges_its_slots_face() -> None:
    constructions = built(((4, 4), BuildingKind.FACTORY))

    roads = town_roads(constructions)

    assert len(edges_of_cell(constructions, (4, 4)) & edges_of(roads)) >= 3


def test_what_grows_from_the_land_gets_no_road() -> None:
    constructions = built(((4, 4), BuildingKind.MINE), ((4, 5), BuildingKind.MINE))

    roads = town_roads(constructions)

    assert len(roads.segments) == 0


def test_loose_pieces_of_road_join_round_the_cells(monkeypatch: pytest.MonkeyPatch) -> None:
    constructions = built(((4, 4), BuildingKind.HOUSE), ((4, 5), BuildingKind.HOUSE))
    monkeypatch.setattr(town_roads_module, "JOIN_STEPS", 0)
    apart = town_roads(constructions)
    monkeypatch.undo()

    joined = town_roads(constructions)

    assert pieces(apart) > 1
    assert pieces(joined) == 1
