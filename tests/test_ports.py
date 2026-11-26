import numpy as np
import pytest
from growing import build, make_map, only

from builders import flat_map, layer
from railroads.sim.constructions import BuildingKind, Constructions
from railroads.sim.goods import PRICES, Good, Need
from railroads.sim.growth import Growth
from railroads.sim.growth_odds import _cap_reasons, _capped
from railroads.sim.ladders import PASSENGERS_PER_POINT, PORTS
from railroads.sim.ownership import SHARES
from railroads.sim.ports import (
    HIGH_TIERS,
    LOW_TIERS,
    STORE_MONTHS,
    TIER_GOODS,
    Port,
    Slot,
    Tier,
    monthly,
)
from railroads.sim.tilemap import Relief, Terrain

K = BuildingKind
COAST = (5, 1)
NEXT = (6, 1)
FAR = (1, 1)


def port_town(*ports: tuple[tuple[int, int], BuildingKind], year: int = 1900) -> Growth:
    constructions = Constructions(make_map())
    for cell, kind in ports:
        build(constructions, cell, kind, 1)
    return Growth(constructions, only(PORTS), start_year=year)


def test_a_ports_monthly_amount_is_its_tiers_value_at_the_goods_base_price() -> None:
    amounts = [monthly(Good.LIMESTONE, Tier.COMMON), monthly(Good.STEEL, Tier.EPIC)]

    assert amounts == [600 / PRICES[Good.LIMESTONE], 4800 / PRICES[Good.STEEL]]
    assert amounts[0] == 40


@pytest.mark.parametrize(
    "good",
    [Good.EGGS, Good.BREAD, Good.HONEY, Good.DRIED_FISH, Good.POTS, Good.SHEEP, Good.PIGS],
)
def test_some_goods_are_never_drawn(good: Good) -> None:
    tiers = [tier for tier, goods in TIER_GOODS.items() if good in goods]

    assert tiers == []


def test_steel_is_common_rare_and_epic_but_never_legendary() -> None:
    tiers = [tier for tier, goods in TIER_GOODS.items() if Good.STEEL in goods]

    assert tiers == [Tier.COMMON, Tier.RARE, Tier.EPIC]


def test_a_port_draws_a_common_or_rare_good_it_gives_and_one_it_takes() -> None:
    growth = port_town((COAST, K.PORT))

    growth.ports.sync()

    port = growth.ports.at[COAST]
    assert len(port.gives) == 1 and len(port.takes) == 1
    assert all(slot.tier in LOW_TIERS for slot in port.slots)


def test_a_port_takes_what_is_made_near_it_and_gives_what_is_not() -> None:
    constructions = Constructions(make_map())
    build(constructions, COAST, K.PORT, 1)
    build(constructions, (5, 5), K.MINE, 1)
    growth = Growth(constructions, only(PORTS), start_year=1900)

    growth.ports.sync()

    port = growth.ports.at[COAST]
    assert [slot.good for slot in port.takes] == [Good.COAL]
    assert Good.COAL not in [slot.good for slot in port.gives]


def test_a_port_draws_only_goods_already_in_the_game() -> None:
    growth = port_town((COAST, K.MEGA_PORT), year=1800)

    growth.ports.sync()

    goods = growth.ports.at[COAST].goods
    assert Good.PLANES not in goods and Good.TANKS not in goods


def test_two_ports_in_a_neighbourhood_merge_into_a_big_port_keeping_their_goods() -> None:
    growth = port_town((COAST, K.PORT), (NEXT, K.PORT))
    growth.ports.sync()
    before = growth.ports.at[COAST].goods | growth.ports.at[NEXT].goods

    growth._turn(COAST, PORTS, np.random.default_rng(0))

    port = growth.ports.at[COAST]
    assert growth.count(K.BIG_PORT)[COAST] == 1
    assert growth.count(K.PORT).sum() == 0
    assert before < port.goods
    extra = [slot for slot in port.slots if slot.good not in before]
    assert len(extra) == 1 and extra[0].gives and extra[0].tier in HIGH_TIERS


def test_two_big_ports_merge_into_a_mega_port_with_one_more_good() -> None:
    growth = port_town((COAST, K.BIG_PORT), (NEXT, K.BIG_PORT))
    growth.ports.sync()
    before = growth.ports.at[COAST].goods | growth.ports.at[NEXT].goods

    growth._turn(COAST, PORTS, np.random.default_rng(0))

    assert growth.count(K.MEGA_PORT)[COAST] == 1
    assert len(growth.ports.at[COAST].goods - before) == 1


def test_ports_merging_never_leave_two_ports_in_one_cell() -> None:
    growth = port_town((COAST, K.BIG_PORT), (NEXT, K.PORT), ((4, 1), K.PORT))

    growth._turn(COAST, PORTS, np.random.default_rng(0))

    assert growth.count(K.BIG_PORT)[COAST] == 1
    assert growth.count(K.PORT).sum() == 2


def test_ports_keep_no_distance_from_each_other() -> None:
    growth = port_town((NEXT, K.PORT))

    allowed = growth.ladder_allows(PORTS, COAST)

    assert allowed


def test_ports_grow_by_the_sea_lakes_and_rivers_but_not_on_mountains() -> None:
    terrain = layer(12, 12, Terrain.LAND)
    terrain[:, 0] = Terrain.SEA
    terrain[0, 4:8] = Terrain.LAKE
    terrain[11, 4:8] = Terrain.RIVER
    relief = layer(12, 12, Relief.FLAT)
    relief[6, 1] = Relief.MOUNTAINS
    tile_map = flat_map(12, 12, terrain=terrain, relief=relief)

    weights = PORTS.land.weights(tile_map)

    assert weights[5, 1] > 0 and weights[1, 5] > 0 and weights[10, 5] > 0
    assert weights[6, 1] == 0 and weights[6, 6] == 0


@pytest.mark.parametrize(("kind", "sent"), [(K.PORT, 11), (K.BIG_PORT, 22), (K.MEGA_PORT, 44)])
def test_ports_send_passengers_like_blocks_of_flats_more_for_bigger_ones(
    kind: BuildingKind, sent: int
) -> None:
    growth = port_town((COAST, kind))
    points, value = growth.rule.points[kind]

    passengers = PASSENGERS_PER_POINT[points] * value

    assert passengers == sent


def test_a_port_offers_what_it_gives_and_wants_what_it_takes_in_the_months_trade() -> None:
    growth = port_town((COAST, K.PORT))
    growth.ports.at[COAST] = Port(
        [Slot(Good.COAL, Tier.COMMON, True), Slot(Good.WOOL, Tier.RARE, False)]
    )

    goods = growth.ports.cell_goods()

    flat = np.array([COAST[0] * growth.grid.cols + COAST[1]])
    assert goods.over(goods.makes, flat)[Good.COAL][0] == pytest.approx(
        growth.ports.amount(Slot(Good.COAL, Tier.COMMON, True))
    )
    assert goods.over(goods.takes, flat)[Need.WOOL][0] == pytest.approx(
        growth.ports.amount(Slot(Good.WOOL, Tier.RARE, False))
    )


def owned(growth: Growth, *cells: tuple[int, int]) -> None:
    for cell in cells:
        kind = growth.ports.kind_at(cell)
        assert kind is not None
        growth.owners.own(cell, kind, SHARES)


def test_a_port_ships_only_to_another_port_the_player_controls() -> None:
    growth = port_town((COAST, K.PORT), (FAR, K.PORT))
    growth.ports.sync()
    owned(growth, COAST)

    refused = growth.ports.ship(COAST, Good.COAL, FAR)

    assert refused is not None
    assert growth.ports.at[COAST].ships == {}


@pytest.mark.parametrize(("kind", "most"), [(K.PORT, 1), (K.BIG_PORT, 2), (K.MEGA_PORT, 4)])
def test_a_port_ships_as_many_goods_as_its_size_allows(kind: BuildingKind, most: int) -> None:
    growth = port_town((COAST, kind), (FAR, K.PORT))
    growth.ports.sync()
    owned(growth, COAST, FAR)
    goods = [Good.COAL, Good.WOOL, Good.TEA, Good.GOLD, Good.CARS]

    refusals = [growth.ports.ship(COAST, good, FAR) for good in goods]

    assert refusals[:most] == [None] * most
    assert all(refused is not None for refused in refusals[most:])


def test_a_shipping_port_wants_room_in_its_store_and_ships_two_wagons_a_month() -> None:
    growth = port_town((COAST, K.PORT), (FAR, K.PORT))
    growth.ports.sync()
    owned(growth, COAST, FAR)
    growth.ports.ship(COAST, Good.COAL, FAR)
    load = growth.ports.load(Good.COAL)
    growth.ports.at[COAST].store[Good.COAL] = 3 * load

    stores = growth.ports.cell_goods().stores
    growth.ports.sail()

    assert Need.COAL not in stores
    assert growth.ports.at[FAR].landed[Good.COAL] == pytest.approx(load)
    assert growth.ports.at[COAST].store[Good.COAL] == pytest.approx(2 * load)
    assert growth.ports.gives_at(FAR)[Good.COAL] >= load
    assert STORE_MONTHS == 2


def test_the_owner_of_a_mega_port_picks_a_good_of_any_tier_for_it_to_give() -> None:
    growth = port_town((COAST, K.MEGA_PORT))
    growth.ports.at[COAST] = Port([Slot(Good.COAL, Tier.COMMON, True)])
    owned(growth, COAST)

    growth.ports.pick(COAST, Good.GOLD, Tier.LEGENDARY)

    assert growth.ports.can_pick(COAST)
    assert growth.ports.at[COAST].pick == Slot(Good.GOLD, Tier.LEGENDARY, gives=True, picked=True)


def test_a_mega_port_cannot_pick_a_good_not_yet_in_the_game() -> None:
    growth = port_town((COAST, K.MEGA_PORT), year=1800)
    growth.ports.at[COAST] = Port([Slot(Good.COAL, Tier.COMMON, True)])

    growth.ports.pick(COAST, Good.PLANES, Tier.LEGENDARY)

    assert growth.ports.at[COAST].pick is None


def test_a_month_of_trade_settles_what_ports_ship() -> None:
    growth = port_town((COAST, K.PORT), (FAR, K.PORT))
    growth.ports.sync()
    owned(growth, COAST, FAR)
    growth.ports.ship(COAST, Good.COAL, FAR)
    growth.ports.at[COAST].store[Good.COAL] = 100.0

    growth.rule = only(PORTS)
    growth.economy.year = growth.year
    trade = growth.economy.trade(growth.counts, cell_goods=growth.ports.cell_goods())
    growth.ports.settle(trade)

    assert growth.ports.at[FAR].landed[Good.COAL] == pytest.approx(100.0)


def test_the_growth_odds_count_a_big_port_against_another_port_in_its_cell() -> None:
    growth = port_town((COAST, K.BIG_PORT))

    reasons = _cap_reasons(growth, PORTS, COAST)

    assert reasons == ["1 stand here already, the most a cell holds."]


def test_the_growth_odds_leave_no_room_for_a_port_in_a_cell_holding_a_mega_port() -> None:
    growth = port_town((COAST, K.MEGA_PORT))

    capped = _capped(growth, PORTS)

    assert capped[COAST[0] * growth.grid.cols + COAST[1]]


def test_a_merged_port_keeps_the_players_biggest_stake_in_the_ports_it_merged() -> None:
    growth = port_town((COAST, K.PORT), (NEXT, K.PORT))
    growth.owners.own(NEXT, K.PORT, SHARES)
    growth.owners.own(COAST, K.PORT, SHARES // 4)

    growth._turn(COAST, PORTS, np.random.default_rng(0))

    assert int(growth.owners.shares(K.BIG_PORT)[COAST]) == SHARES


def test_a_merged_port_keeps_what_its_ports_shipped_and_held_and_ships_to_it_follow() -> None:
    growth = port_town((COAST, K.PORT), (NEXT, K.PORT), (FAR, K.PORT))
    growth.ports.sync()
    growth.ports.at[NEXT].ships = {Good.COAL: FAR}
    growth.ports.at[NEXT].store = {Good.COAL: 40.0}
    growth.ports.at[COAST].landed = {Good.WOOL: 25.0}
    growth.ports.at[FAR].ships = {Good.TEA: NEXT}

    growth._turn(COAST, PORTS, np.random.default_rng(0))

    big = growth.ports.at[COAST]
    assert big.ships == {Good.COAL: FAR}
    assert big.store == {Good.COAL: 40.0}
    assert big.landed == {Good.WOOL: 25.0}
    assert growth.ports.at[FAR].ships == {Good.TEA: COAST}
