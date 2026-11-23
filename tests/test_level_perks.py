from railroads.sim.level_perks import LevelPerk, Perk, perks
from railroads.sim.tiles import Board, Tile


def test_a_supply_level_that_opens_an_engine_names_it() -> None:
    found = perks(Board.SUPPLY, Tile.COAL, 13)

    assert found == [LevelPerk(Perk.ENGINE, "New engine for sale: Light tank engine")]


def test_the_level_of_an_era_brings_its_wagons_on_every_supply_tile() -> None:
    found = [perks(Board.SUPPLY, tile, 14) for tile in (Tile.FARMING, Tile.MAIL)]

    assert all(perk[0].kind == Perk.ERA for perk in found)
    assert "Wagons of generation 3" in found[0][0].text
    assert "New buildings: power plant." in found[0][0].text


def test_a_demand_level_that_opens_buyers_says_who_takes_what() -> None:
    found = perks(Board.DEMAND, Tile.WOOD, 2)

    assert found == [LevelPerk(Perk.BUYERS, "Mines take planks")]


def test_a_level_that_only_adds_its_share_brings_nothing_special() -> None:
    supply, demand = perks(Board.SUPPLY, Tile.COAL, 10), perks(Board.DEMAND, Tile.COAL, 13)

    assert (supply, demand) == ([], [])
