# Test review (November 2026)

A pass over the ~1,920 tests in `tests/`, asking of each whether it checks something real, says
what it checks, repeats another, is pinned to a tuning number where a relation would test the rule,
or is slow. Marks: **done** was fixed on this branch; **open** is a safe change left for later;
**game** needs a change to the game first, so it is only listed.

The suite took 123 s, five tests 67 s of it; after this pass it takes 57 s for 1,899 tests (24
fewer: no-op, repeated and old-save tests deleted, six trade-rule tests added).

## Trade rules (`docs/principles.md`)

| Rule | Tests |
|---|---|
| Buildings hand out first, nearest first, a buyer `d` cells away takes at most (11-d)/10 | `test_economy`: tenth less per cell, ten cells then none, nearer house first; **done**: nothing lost on the way |
| Buildings hand out before a station hands out what trains brought | **done**: `test_economy::test_buildings_hand_out_their_own_goods_before_a_station_hands_out_what_trains_brought`, `..._a_station_hands_out_only_what_the_buildings_around_left_short` |
| Stations gather only the leftovers | `test_economy::test_a_station_gathers_only_what_the_buildings_around_left`, `test_railway` town-eats-its-own-farms, coal-heats-its-own-houses |
| Stations never hand out what they gathered | `test_railway::test_a_station_keeps_what_it_gathered_for_trains_and_hands_none_of_it_around`; **done**: `test_economy::test_a_station_never_hands_out_the_freight_it_gathered`, `test_hand_out_now::test_freight_a_station_gathered_is_never_handed_out_at_once` |
| The player earns only on goods taken out of stations | `test_railway::test_cargo_earns_money_only_when_the_buildings_around_take_it`; **done**: `test_station_trade::test_goods_buildings_hand_out_on_their_own_by_a_station_earn_the_player_nothing` |
| Station demand counts only what local trade left short | `test_railway::test_a_station_wants_only_what_its_houses_were_left_short_of_by_the_farms_around` (all or nothing); **done**: `test_station_trade::test_a_station_wants_more_food_the_less_the_farms_around_could_hand_its_houses` (partial) |

A deliberate break of the order (stations before local trade) fails three of these tests.

### Game behaviour that disagrees with the rules (game)

- **Owned buildings earn on local hand-outs.** `Owners.pay` pays the player's share of all of
  `trade.goods_gave`, so an owned farm beside houses earns with no station at all. The comment on
  `OWNER_SHARE` and `test_ownership::test_the_player_earns_half_of_what_their_farm_sells_to_a_joined_station`
  say "sells to their stations". Decided: they keep earning on hand-outs, at a price that should
  follow last quarter's satisfaction; whose satisfaction and how far the price moves are still
  open.
- **done: `StationTrade.supplies` counted everything made in reach.** Each station now keeps
  what it was given over the last three months (`StationGoods.last_quarter`), and the trains
  window, route map and station tooltip show that.
- **done: two hand-out orders.** Goods off a train now go to the least satisfied first, then the
  nearest, as in a month's trade (`test_hand_out_now`).
- **Track moves buildings.** The principle says track goes round buildings and never moves them;
  `lay_route` clears room. Being fixed in another session.
- **done: merging.** Food makers merge in threes and the other works in twos; the principle now
  says so. The factory line (factory, big, mega) still merges in threes.
- **done: help window years.** `goods.need_year` gives each building's own year for a need, which
  `needs_of` and the help both use.
- **Not a bug: works dying "a month early".** `Growth(...)` trades once as it is made, so the
  359th call was the 360th starving month; `test_deposits` now counts it.
- **77 of 78 `art.MODELS` are never drawn**; only `test_close_scenery::test_every_model_can_be_drawn`
  touches them.
- **`engine_summary` and `engine_figures`** build the same strings twice (trains window, buy screen).

## Slow tests

| Test | Time | Change |
|---|---|---|
| `test_growth::test_a_long_run_keeps_its_counts_and_chances_in_step_with_what_is_built` | 29 s | **done**: smaller map, fewer months |
| `test_growth::test_the_log_holds_every_change_months_make_to_the_counts` | 13 s | **done**: fewer months |
| `test_sandbox::test_the_sandbox_grows_only_houses_food_and_bakeries` | 10 s | **done**: start in 1750, 60 years |
| `test_military::test_outposts_open_only_on_military_ground` | 9 s | **done**: only the military ladder, no trade |
| `test_seeding` fixture | 6 s | open: `test_seeding_leaves_stations_to_the_player` seeds again; merge it with the track test |
| `test_close_scenery::test_close_chunks_stand_buildings_as_their_pixel_art_blown_up` | 2 s | **done**: stop at the first chunk that differs |
| `test_close_scenery::test_every_model_can_be_drawn` | 2.7 s | open: tied to the dead models above |
| `test_pixel_world` sheet tests | ~1 s | open: draw the sheet once per module |
| `test_world_copies` subprocess tests | 1 s | open: one subprocess for both imports |

## By file

### test_economy.py
- **done**: per-cell shares are written as `share * FARM_MONTH`, the farm's month from `makes_of`,
  not 10, 9, 7.
- **done**: new trade-rule tests above.
- open: `test_lumber_mills_only_grow_at_the_edge_of_a_forest` and `test_mines_like_hills_beside_mountains_best` test ladders, not trade; move them.
- open: `test_farms_want_limestone_always_and_machinery_and_diesel_only_from_1920` and `test_new_works_turn_their_inputs_into_their_goods` copy the goods tables; they will change with any tuning.

### test_railway.py
- **done** renames: `test_a_stop_sends_at_most_six_cars` (it is `MAX_CARS`, 5);
  `test_a_train_costs_forty_five_thousand_and_sells_for_half` (it is `TRAIN_PRICE`);
  `test_stations_hand_goods_to_the_hungriest_buildings_and_gather_spare_ones` (one buyer, so
  "hungriest" is never tested).
- **done** delete: `test_a_station_gathers_what_a_farm_makes` (repeats the gather check above it);
  `test_a_route_costs_three_hundred_a_cell_on_the_flat` (repeats `test_routes`);
  `test_a_station_saved_before_goods_could_be_kept_for_trains_keeps_none` (old saves).
- **done** relation: `test_station_demand_is_twice_the_needs_of_buildings_in_its_inner_ring` built from
  the house's needs, not a dict of twelve numbers.
- open: `test_a_trains_cargo_is_worth_what_its_destination_would_pay_for_it` and
  `test_cargo_going_back_where_it_came_from_is_worth_nothing` compute the expected value with the same
  `StationDemand.deliver` the code uses.
- open: storage tests L502/L514/L707 overlap; level prices and upkeep are literals.
- open, missing: a station no track joins gathers nothing.

### test_routes.py
- **done** rename: `test_no_route_into_the_sea` plans from a cell to itself on a map with no sea.
- open: a real sea test; track prices as `TRACK_PRICE * ...` instead of literals.
- game: track moving buildings (above).

### test_growth.py
- **done**: the two slow runs.
- open: `test_brewers_distillers_and_potters_need_a_town_around_them`,
  `test_gasworks_are_drawn_by_industry_and_dairies_by_people_twice` and
  `test_soldiers_never_starve_to_death` read config back; test chances or survival instead.

### test_dying_out.py
- **done** delete: `test_a_house_with_nothing_dies_out_after_ten_years` repeats the four-house test.
- open: the four 30-year "never die out" runs could be one run with every kind in its own cell.

### test_military.py
- **done**: slow outpost run; soldiers' travel as "headquarters > outpost > 0".

### test_churches.py
- **done** renames: St Paul's goes to the most crowded cathedral, not the first;
  churches only send passengers, "want" is never checked.
- **done** delete: `test_a_house_is_one_population...`, `test_buildings_take_up_their_share_of_a_cell`
  copy source tables.
- open: `test_hills_favour_orchards_over_farms` as a relation.

### test_tech.py
- **done** rename: `test_electrified_track_costs_twenty_one_times...`, as a relation of the upkeep
  constants.
- open: every building that makes something has a tree, over all of `MAKES`.

### test_novelties.py
- open: `isinstance(need, Need)` is a tautology; check every novelty need is met by a good some
  other building makes.

### test_saves.py
- **done** delete: `test_a_snapshot_is_the_games_big_parts_pickled_then_the_game` rebuilds the
  snapshot by hand; other tests cover it.

### test_seeding.py
- open: merge the stations test into the track test (one seeding fewer); drop `HISTORY_START == 850`.

### test_constructions.py, test_fuel_market.py
- **done** rename: the mountain test also allows limestone quarries.
- open: `1.25 / 1.5` from `RETAIL_MARKUP`.

### test_deposits.py
- **done** rename + relation: works with nothing to work from close only after thirty years; the
  old name said they stay open.
- open: gold never checked in the diamonds test; printing-works and fishing numbers.

### test_ownership.py
- open: rename the farm test once the owner-pay rule is decided (game, above).

### test_demolition.py, test_game.py
- open: prices as `demolition_price(...)` and an order `HOUSE < FARM < FACTORY < ...` instead of
  literals; `test_game::test_the_bulldozer_clears_a_cell_for_its_price` repeats `test_demolition`.

### test_start_year.py, test_start_maps.py, test_train_motion.py, test_track_network.py
- **done** delete: `test_the_earliest_start_is_1804_and_the_suggested_one_1850` (constants equal
  themselves), and the old-save tests `test_settings_pickled_before_there_were_more_maps_still_load_as_poland`,
  `test_a_train_saved_before_trains_picked_up_speed_still_runs`,
  `test_a_railway_saved_before_ways_and_held_trains_were_kept_still_runs`.

### test_poland_map.py, test_tilemap.py
- **done** delete: scale tests repeated by the start-map and load-back tests.

### test_forests.py, test_power_plants.py, test_facilities.py, test_diesel.py, test_maglev_track.py, test_map_trains.py
- **done** renames: the power-plant test has no year in it; the engine-shed test sets `fuelled` by
  hand.
- open: forests' 100× rule as a ratio; a missing test for a second power plant; a missing test
  that calling at a working shed fuels a train; config copies in the diesel and maglev tests.

### test_pixel_world.py
- open: draw the sheet once; house variants from `SPRITES`; `26` is not a cell width
  (`PIXEL_CELL_WIDTH` is 32); turned station houses by distinct pixels, not sizes.
- game: the woods and crown reference helpers copy tuning constants; the ground module should
  expose its per-pixel functions.

### test_app_frame.py
- **done**: `test_the_game_autosaves_as_a_month_begins` folded into the next test;
  `test_the_saves_window_draws_both_tabs` renamed (no tabs are checked).
- open: `test_the_waiting_screen_shows_how_far_a_new_game_has_grown` reads back what it set;
  `test_a_month_that_changes_colours_paints_the_overview_again_within_its_steps` would pass if
  nothing were repainted.

### test_engine_roster.py, test_trains_window.py, test_train_sprites.py
- **done** rename: `test_the_list_draws_a_long_express_with_its_dining_car` (the train is no
  express).
- open: planet price/speed/cars as comparisons; express tooltip strings from `ENGINES`; a missing
  maglev fare test; `OUTLINE` instead of `(22, 18, 18)`.

### test_route_map.py, test_building_drawing.py, test_hex_details.py, test_web_painter.py
- **done** rename: the route map test checks only centring.
- open: hex-details and house capacity numbers from `needs_of`/`makes_of`; one dot per kind.

### test_goods.py, test_finance.py, test_taverns.py, test_relief.py, test_stats.py, test_minimum.py, test_track_rules.py
- **done** delete: `test_goods::test_a_house_demands_food_common_goods_and_heating...` repeats the
  homes test; `test_taverns::test_taverns_open_from_three_people...` reads ladder constants;
  `test_relief::test_water_is_never_lightened` repeats `test_ground`.
- **done** rename: `test_a_processor_demands_every_input...` (coke is a boost, not a demand);
  `test_one_that_opens_later_goes_up_only_then`; `test_a_car_counts_as_delivered_where_it_leaves_its_load`.
- open, missing: the yearly books (`close_month`, `year_books`) have no test; track price over hills,
  rivers and mountains.
- open: `test_finance::test_a_game_starts_with_eighty_thousand` copies `START_CASH`.

### test_help_window.py, test_pixel_noise.py, test_desktop_build.py, the rest
- game: help-window need years (above).
- open: winery, prices and engine years from the tables; `ridged` noise re-implemented in the test;
  `release_tag` re-implemented in the test.
