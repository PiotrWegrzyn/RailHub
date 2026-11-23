# Refactoring audit (0.5.178, 2026-11-23)

A read-only pass over `src/railroads`, `tools` and `tests`. Nothing has been changed yet. Paths are
relative to `src/railroads/` unless they start with `tests/` or `tools/`.

Severity: **H** hides balance, must stay in sync by hand, or is a bug; **M** an unclear knob or a
real maintenance cost; **L** cosmetic. Effort: **S** under an hour, **M** a few hours, **L** a day
or more.

Files another session was rewriting during the audit (train traffic: `sim/railway.py`,
`sim/trains.py`, `sim/track_network.py`, `sim/traffic.py` → `sim/signals.py`,
`ui/pixel/track_art.py`) were left out of the plan. Their findings are under
[Later](#7-later-train-traffic-files). That rewrite reached staging in 0.5.179, so they are open now.

**Done (steps 1-6 of section 8):** B1-B4, B6-B8 and M13-M18, M20-M22, M24, M26-M34. B5 was not a
bug: the farm sprite is 13 px wide and a house 9 px, so the farmstead does stand bigger. Steps 1-6
changed no line of `tools/fingerprint.py` output. That tool had been broken since the tech tree
went, and is now fixed. The help, hint and message texts read word for word as before. New shared
pieces: `sim/clock.MONTHS_PER_YEAR` and `FIRST_RAILWAY_YEAR`, `sim/grids.KM_PER_CELL`,
`sim/tilemap.REFERENCE_LAND`, `sim/lots.land_price` and `Lots.in_middle_of`,
`ui/theme.SIDE_WINDOW_TOP`, `sim/year_caches.boards_changed`, `railroads/words.py` (counts, shares
and factors in words) and `sim/amounts.py` (`EPSILON`, `ratio`, `divided`). The short-form
divisions that keep the part where the whole is nothing became `divided`. They were left with that
meaning on purpose, since a `ratio` there would change results.

## 0. Bugs found on the way

| # | Where | What | Sev | Eff |
|---|---|---|---|---|
| B1 | sim/economy.py:125 | `input_stock` is a plain `@cache`, but it calls `rise_of`, which reads the year's demand boards and is `@year_cached`. When a tile levels up (`year_caches.forget()`), on a new game, and between tests, `input_stock` keeps the old stock limit. Fix: `@year_cached`. | H | S |
| B2 | ui/trains_window.py:793 | "All N trains" is enabled on `can_afford(price)`, where `price` is the swap price of *this* train, not the cheapest of `alike`. The tooltip ("as many as the cash allows") says one affordable swap should be enough, so use `min(swap_price(t) for t in alike)`. | L | S |
| B3 | ui/map_view.py:729 | `_consumers_of` is keyed on (cell, constructions.version, year), but `consumers_in_reach` depends on `needs_of`/`makes_of`, which change when a board tile levels up mid-year. Add the boards' generation (or `growth.trade`) to the key. | M | S |
| B4 | ui/pixel/scenery.py:211 | Comment points at `art.GROWN_SCALE`, which does not exist. | L | S |
| B5 | ui/pixel/art/__init__.py:183-192 | Comment says a farmstead is "a touch bigger than a house", but `FARMSTEAD_SCALE` 0.585 < `HOME_SCALE` 0.59. Comment or value is wrong. | L | S |
| B6 | sim/track_rules.py:2-3 | Docstring prices ($300 flat, $10,000 bridge…) are stale against `TRACK_PRICE = 600`. | M | S |
| B7 | sim/ladders.py:896, :1156; sim/constructions.py:422; sim/finance.py:32 | Comments disagree with values: butchers "from 1850" vs `from_year=1600`; filling stations "half as often as taverns" vs base 20 vs 60; "two of a kind merge" vs `FOOD_MERGING = 3`; "level 2 and then 3" vs four `LEVEL_PRICES`. | L | S |
| B8 | ui/theme.py:72, :84 | `Good.FURNITURE` and `Good.LOGS` have the same colour, so their chips can't be told apart. | M | S |

## 1. Magic numbers and arbitrary constants

### 1a. Hidden scaling factors (the project rule)

The rule: every per-building number sits at that building; the only global derivation is
pull = `BREEDING_TIMES` × base. These break it. Every one of them is a balance decision, so the
fix is "write the current values out at each building" (numbers unchanged), and only after the
user says which ones to unfold.

| # | Where | What is derived globally | Sev | Eff |
|---|---|---|---|---|
| M1 | sim/goods.py:585-603, sim/ladders.py:216-224 and :516-527, sim/constructions.py:436-443 | Big and mega works (~60 kinds): output = base × (`MERGING`/`FOOD_MERGING` × `MERGE_BONUS`)^step, points = base × merging^step, merge counts and room all derived from three constants. Any BIG_/MEGA_ row written into `_POINTS` is silently overwritten. | H | M |
| M2 | sim/growth.py:137-189, :222-226 | `OWN_KIND_TIMES = 10` and `CATEGORY_SHARE = 0.1` rewrite `drawn_by`/`served_by` of every industry-drawn ladder; `DRAWN_BY`, `DRAWN_BY_KIND` and `SHUNNED_BY` hold more per-kind weights. A ladder's real breeding weights live in four tables away from its `Ladder(...)`. | H | M |
| M3 | sim/ladders.py:75-88; used at sim/growth.py:938, :1391, sim/station_trade.py:529, ui/help_window.py:189 | Passengers = POINTS × `PASSENGERS_PER_POINT[category]`; mail adds `MAIL_PER_PASSENGERS = 5` (station_trade.py:55). An airport's passengers are never written at the airport. | H | M |
| M4 | sim/demolition.py:30-55 | Demolition price = POINTS × `DEMOLITION_PER_POINT[category]`, so changing points silently changes clearing cost. | H | S |
| M5 | sim/ownership.py:122 | Building price = 12 × `PAYBACK_YEARS` × monthly output. A global derivation; may be intended, needs a "yes, keep". | M | S |
| M6 | sim/goods.py:310-352 (merged into needs at :943-951) | `DIESEL_USERS` fuel per month, `KIND_BOOST_MOST`, `OWNED_BOOSTS`: per-building numbers in side tables. | H | S |
| M7 | sim/fuel_market.py:107, :110; sim/ownership.py:53 | `POWER_PLANT_OUTPUT = 3000` and `POWER_PLANT_PRICE = 2_000_000`, one value for three plant kinds, far from the plants. `ELECTRICITY_PER_POINT = 10` scales every home and factory. | H | S |
| M8 | sim/ladders.py:766-768 | `_resource(chance=4000, most_nearby=1)` defaults: nine resource ladders get their base chance from a default argument. Same for `_works(base=0.0)` at :788 (every call passes it; remove the default). | H | S |
| M9 | sim/goods.py:385-387, :491 | `MARKET_TIMES = 5` × `FREIGHT_SHARE = 0.2` = 1.0, so `freight_of(g) == PRICES[g]`: two constants that cancel. Tuning one rescales all freight. Drop both, or keep one with a reason. | H | S |
| M10 | sim/goods.py:1217-1223 | `BONUS_PART` 10/60/200 must equal half the homes' FOOD needs (20/120/400 in `_NEEDS`). Derive it, or write it next to the need. | H | S |
| M11 | sim/events.py:35 | `WAR_GROWTH = 11` multiplies military chance outside the ladder. By design, but undocumented. | M | S |
| M12 | sim/growth.py:309-312 | `DIE_AFTER` (10 / 30 years) is a hidden default for every ladder without `starving_years`; ladders.py writes 250 (×10) and 50 (×4) bare. | M | S |

### 1b. Values that must stay in sync by hand

| # | Where | What | Fix | Sev | Eff |
|---|---|---|---|---|---|
| M13 | sim/growth.py:124 | `MONTHS_BEFORE_PLAY = 1000 * 12` must equal `(START_YEAR − HISTORY_START) × 12`. tools/fingerprint.py:36-39 copies it again as `YEARS = 1000`. | derive in clock.py, import | H | S |
| M14 | sim/tiles.py:150, newgame.py:41, sim/fuel_market.py:89 | Year 1804 three times (`FIRST_YEAR`, `EARLIEST_START_YEAR`, fuel `OPENS`). | one constant | H | S |
| M15 | sim/economy.py:98 vs sim/railway.py:131; used at sim/station_trade.py:437 vs :470 | Hand-out radius: `DELIVERY_RADIUS = 2` (fallback when no railway, `StationOrder` default) vs `FIRST_REACH = 1` (railway, preview). The inner ring differs by one depending on the path. | use `FIRST_REACH` in the fallback; drop `DELIVERY_RADIUS` or say what it is | H | S |
| M16 | mapgen/deposits.py:46, sim/happenings.py:33 | `GARRISON_SIZE = (10, 18)` twice; happenings.py:239 writes `(BASE, OUTPOST, OUTPOST)` instead of `STARTING_OUTPOSTS = 2`. | import | H | S |
| M17 | newgame.py:45, mapgen/lands.py:32, tutorial/place.py:32/:45, tools/build_poland_map.py:31 | `KM_PER_CELL = 2.0` four times. Seed 7 also in tools ×3 and `lands.NOISE_SEED`. | one constant | M | S |
| M18 | sim/growth.py:300, sim/novelties.py:27 | Reference land 120_000 vs 121_000. | one constant | M | S |
| M19 | sim/tiles.py:155 vs clock.py | `BENCHMARK_YEAR = 1850` equals `START_YEAR`; tiles.py:222 `PERIODS` look like `ERAS` levels but differ (8/9, 17/18, 21/22, 30/27). | say whether tied; derive or document | M | S |
| M20 | ui/help_window.py:40-42 | `lot_price()` copies the body of `Lots.price()` (sim/lots.py:210). | module-level `lot_price` in lots.py | M | S |
| M21 | ui/build_menu.py:106, :116; ui/trains_window.py:766; ui/help_window.py:87-111; ui/main_menu.py:97; ui/map_overlays.py:59; sim/game.py:215 | Help and hint prose spells out sim numbers ("twice across a river", "a quarter of its price", "half what it cost", "seven lots", "50%+", "level 4 or 5"). | format from constants | H | S |
| M22 | ui/hex_details.py:313, ui/waiting_screen.py:36 | Both assume satisfaction tops out at 200%; only `ownership.PRICE_BY_SATISFACTION`'s last point says so. | `MOST_SATISFACTION` in sim | M | S |
| M23 | ui/trains_window.py:691-694, tools/train_tonnage.py:22-24 | Re-implement the pull slowdown `1 − SLOWDOWN × min(1, t/pull)` from `Railway.pull`. | one sim function (lands with the train rewrite: Later) | M | S |
| M24 | ui/building_spots.py:35, ui/build_menu.py:61 | `DOUBLE_CLEARANCE = 0.11` is `track_art.DOUBLE_SPACING / 2` copied; build_menu has a different `DOUBLE_SPACING = 0.28` under the same name. | import; rename the other | M | S |
| M25 | ui/pixel/relief.py, ground.py, streams.py vs ground_shader.py (e.g. relief.py:179/shader:309, relief.py:453-459/shader:401-403, streams.py:49/shader:119, ground.py:992/shader:990, ground.py:1847/shader:466, snowline 140 at shader:410) | CPU and GPU ground share dozens of tuning literals written twice; a drift makes the two renderers differ. | name each and substitute `@NAME@` as the shader already does for others | H | M |
| M26 | ui/minimap.py:24-25, tools/build_world_files.py:41-42, tools/build_world_streams.py:26-27, ui/land_offers.py:54 | World latitude bounds and lon/lat→UV maths copied. | one module | M | S |
| M27 | ui/reports_window.py:88-91 | `PERIODS {1, 3, 12}` and month names re-typed instead of `clock.QUARTER`, `clock.MONTH_NAMES`. | import | L | S |
| M28 | sim/demand.py:22 | `MONTHS_REMEMBERED = 3` equals `clock.QUARTER`. | import | L | S |
| M29 | ui/trains_window.py:72 vs ui/theme.py:149-150 | `WINDOW_TOP = 96` shadows `theme.WINDOW_TOP = 56`; 96 is typed again in tutorial_panel.py:21 and summaries_panel.py:131. | one `SIDE_WINDOW_TOP` | M | S |
| M30 | ui/pixel/world.py:59, ui/app.py:74, ui/map_view.py:102, ui/menu_scene.py:21 | `CLOSE = 4` must equal `sheet.FINEST`; `OPENING_ZOOM = 3`, `VISIT_ZOOM = 4` are indices into `ZOOM_CELL_WIDTHS`; `MATRIX [[32,0],[0,16]]` restates `PIXEL_CELL_WIDTH` and `TILT`. | derive | M | S |
| M31 | ui/demand_view.py:24 | `return 3` hard-codes `COINS`. | import | M | S |
| M32 | sim/lots.py:74, mapgen/lands.py:65, tutorial/place.py:34 | `rows // 2 // 2 * 2` (lot middle on an even row) three times. | one helper | L | S |

### 1c. Bare numbers that want a name

| # | Where | What | Sev | Eff |
|---|---|---|---|---|
| M33 | 25+ sites: clock.py:74; growth.py:309-312, 755, 979-990; autoplay.py:128-130, 209-211; happenings.py:117; ownership.py:122, 175; game.py:127; ledger.py:106, 143, 161; seeding.py:25, 130; newgame.py:205; land.py:75; ui/hex_details.py:110; ui/app.py:222-223, 764; ui/reports_window.py:116, 124, 286, 380 | `12` as months per year (and `120` as months per decade). No `MONTHS_PER_YEAR` exists. | M | S |
| M34 | economy.py ×7, demand.py ×2, calls.py ×3, station_trade.py, facilities.py, finance.py, room.py, fuel_market.py, grids.py ×2; angle `+1e-6` in routes.py ×3 and constructions.py:708 | Float epsilons 1e-9 / 1e-12 / 1e-6. One `EPSILON` (and `ANGLE_EPS`). | L | S |
| M35 | sim/constructions.py:283-336 | `space=3` ×13 is `CELL_SPACE // 2`. | M | S |
| M36 | sim/ladders.py:880-936, 1168-1238 | `starving_years` 250/50, town thresholds (POP 40/30/80/25/120, IND 2/3/6), `land_cells_each` 20_000/30_000/… bare while `PRINTING_CELLS` is named; :1220 `{POP: 3}` is probably `FOOD_TOWN`. Per-building numbers may stay at the building, so only name the shared tiers. | L | S |
| M37 | sim/track_rules.py:42-43 | Bridge and tunnel `price_factor` 100/3 and 200/3 are absolute prices disguised as factors of `TRACK_PRICE`. | M | S |
| M38 | sim/fuel_market.py:88-90 | Fuel market `OPENS` 1925/1935 not tied to the engines or eras that need them. | M | S |
| M39 | sim/goods.py:364, :727 | Gas import year 1950; sawmill `5 / SAWMILL_MOST`, `3 / SAWMILL_MOST`. | L | S |
| M40 | sim/novelties.py:56-57, :67, :175-176; newgame.py:43-44, :222-230; route_map.py:89; map_view.py:106 | `FILM_TOWN`, `FAIR_TOWN`, `until_year=9999` sentinel, site weights 3.0, `MAP_WIDTH_KM = 537`, RNG offsets seed+1/+2/+3, `LEAST_MADE = 30`, `MINIMAP_TOWN_POINTS = 12`: no reason given. | L | S |
| M41 | sim/autoplay.py:22-29 | Stand-in player knobs without comments (only `STATION_LEVEL` has one). | L | S |
| M42 | ui/pixel/art/__init__.py:183-192, :505-510; ui/pixel/trains.py:53, :2904 | Sprite scales 0.7/0.59/0.585, `MOST_WIDTH` "about 25 pixels", `trains.SCALE = 0.8`, `GROWN_WIDTH = 2432`, none derived or explained. | M | S |
| M43 | ui: 0.5 s refresh (lively, route_map, trains_window), six per-frame budgets 0.004-0.03 s (map_tools, layer, train_layer, app ×2, workers), 4.0 s message time ×2, window widths 860/460/360/190 ×2-3 each, button heights 34/36, `same_line(0, N)` gaps 8/4/10/6 in 11 files, `1e6` vs `2000` max height, the 0.5 "shown amount" threshold ×5, colours copied from the theme (PROFIT=BRASS, LOSS=GONE, PANEL_COLOUR, FLAG_POLE, CAR_LOADED, station blue ×2), `TITLE_SCALE` ×3 with three values, `ornate.py` second palette | Layout and timing literals. Worth a small spacing/timing scale in theme.py; not worth chasing every pixel. | L | M |

## 2. Project structure

**Layering is clean.** `sim/` and `mapgen/` import nothing from `ui/`. There is no runtime import
cycle. Type-level cycles (through `if TYPE_CHECKING`) and lazy imports exist:

| SCC | Modules | How it's dodged |
|---|---|---|
| S1 | sim.events, sim.goods, sim.tiles | lazy imports: goods.py:369, 377, 1093, 1141, 1158, 1176, 1193; tiles.py:567 (redundant: tiles.py:35 already imports goods) |
| S2 | sim.growth + calls, forests, happenings, ladders, novelties, ownership, power_plants, railway, room, station_trade (11) | TYPE_CHECKING `Growth` in six modules; lazy happenings.py:169 |
| S3 | track_network, traffic, trains | Later |
| S4 | newgame, saves | TYPE_CHECKING saves.py:41 |
| S5 | ui.map_view, map_input, map_tools | TYPE_CHECKING |
| S6 | ui.app, goal_view, top_bar | TYPE_CHECKING |
| S7 | art/__init__ → art.stations → track_art → art.painter → art/__init__ | lazy art/stations.py:471 |

Also redundant: ui/pixel/art/built.py:632 lazy-imports `painter.chimney_top`, already imported at :18;
ui/map_tools.py:19 type-imports `game.Done`, which needs no guard.

| # | Where | Finding | Proposed change | Sev | Eff |
|---|---|---|---|---|---|
| S1 | sim/goods.py:367-380, :1090-1200 | Era checks (`good_exists`, `need_exists`) and year-adjusted recipes (`makes_of`, `needs_of`, `boosts_of`, `demands_of`, `bonus_of`) make goods.py depend on tiles and events, causing cycle S1. | Era checks → tiles.py; year-adjusted lookups → new `sim/yearly.py`. goods.py becomes pure tables. | M | M |
| S2 | sim/goods.py (1280 lines) | Enums, eras, prices, recipes, the trade `PIPELINE`, bonuses in one file. | After S1: `goods.py` (Good, Need, General, categories), `prices.py`, `recipes.py` (Recipe, RECIPES, MAKES, NEEDS, merging), `bonuses.py`; PIPELINE next to economy. | M | M |
| S3 | sim/growth.py (1444 lines) | Breeding tables, dying rules, the monthly loop and placement search mixed. Growth is also the hub of S2 (holds railway, station_trade, owners, ledger, room, economy, towns, happenings, novelties; six modules type-import it back). | Split `breeding.py` (BREEDING_TIMES, DRAWN_BY*, SHUNNED_BY, breeding_of), `dying.py`, `placement.py` (_hits, _likeliest, _open_cells, _fits, _skyscraper, _replace). Longer term: Game owns the collaborators, Growth takes narrow Protocols. | H | M (split) / L (Protocols) |
| S4 | sim/economy.py (1153 lines) | Reach geometry (98-256), trade result types, and a 633-line private solver `_Month` (520-1153) that calls `Economy._around`/`_gather` privately. | `sim/reach.py` (REACH_SHARES, rings_of, Rings, Neighbourhoods); `sim/trade_month.py` with `_Month` renamed `MonthOfTrade`. | M | M |
| S5 | sim/ladders.py (1313 lines) | Terrain queries (Land, Surroundings 228-437), ladder types, POINTS, ~800 lines of data. | `surroundings.py`, `ladder_types.py`; POINTS and PASSENGERS_PER_POINT → `points.py`; ladders.py keeps the data. | M | S |
| S6 | ui/map_view.py (1116 lines; 60 methods, 47 attributes) | Camera, panels, month stepping, ground texture caching and shaders, lenses and moods. | `ui/map_ground.py` GroundRenderer; `ui/map_lenses.py` Lenses. | H | M |
| S7 | ui/app.py (836 lines) | App, Shell, the autosave state machine (329-400) and the CLI (754-836). | `ui/autosave.py` Autosaver; `ui/shell.py`; `railroads/cli.py`. | M | S |
| S8 | ui/pixel/ground_shader.py (2429 lines) | ~1270 lines of GLSL strings, source assembly, GL helpers; dots_shader.py:21 imports four of its private names. | `ground_glsl.py`; `gl_util.py` with public `program`, `saved_state`, `restore`, `Target`. | M | S |
| S9 | ui/pixel/ground.py (1926 lines) | ~400 lines of palette/tuning constants, Ground, free tree geometry. | `ground_palette.py`, `woods.py`. | M | M |
| S10 | ui/pixel/trains.py (2964 lines) | Palette, engine models, car bodies, ENGINES look table, sprite rendering. | `train_engines.py`, `train_cars.py`. Touches only drawing, but wait until the train rewrite lands. | M | M |
| S11 | ui/pixel/art/__init__.py (589 lines) | The kind→sprite registry lives in the package `__init__`, so importing `art.painter` loads every sprite (cycle S7). | `art/registry.py`; empty `__init__`. | M | S |
| S12 | ui/pixel/art/industry.py (2354 lines) | ~30 shared industrial parts imported privately by 7 modules. | `art/industry_parts.py`, public names (see K8). | M | S |
| S13 | ui/hex_details.py:120-258, :554-619; ui/trains_window.py:712-953 | Pure view-model and text builders inside imgui windows. | `ui/hex_rows.py`, `ui/train_texts.py`: testable without imgui. | L | S |
| S14 | tools/readme_tables.py:25-26 | The README generator imports imgui modules for `building_help`, `engine_help`, `fuel_use`, which contain no imgui. | Move them to `railroads/catalogue.py`. | M | S |
| S15 | sim/finance.py:176 `dollars`; sim/trains.py:352-364; sim/game.py:144-254 | UI text built in sim ("This track costs X; you have Y."). | Formatters → `railroads/words.py` (P2); Game returns structured refusals, UI phrases them (P1). | M | M |
| S16 | ui/map_tools.py:170-195; ui/build_menu.py:94-95 | Hover note re-derives bulldoze/station/platform prices with the formulas of game.py:149, 200, 216, 237; HILLS_PRICE and BRIDGE_PRICE derived in the UI. | One price function per action in sim (P1). | M | S |
| S17 | sim/station_trade.py:48 | Imports `FIRST_REACH` from railway only for a constant while railway type-imports StationTrade. | Move station level/reach constants (railway.py:128-135) to `station_goods.py`. Touches railway.py: Later. | M | S |
| S18 | saves.py:41 ↔ newgame.py | Import each other. | `railroads/settings.py` for `Settings`. | L | S |
| S19 | ui/pixel/painter.py vs ui/pixel/art/painter.py; ui/relief.py vs ui/pixel/relief.py; ui/ground.py vs ui/pixel/ground.py | Same module names for different things. | Rename the first of each pair (`chunk_queue.py`, `flat_relief.py`, …). | L | S |

## 3. Classes and naming

| # | Where | Finding | Proposed change | Sev | Eff |
|---|---|---|---|---|---|
| K1 | sim/regrid.py:105-240 | Regrid writes straight into the private state of Game (`_reach`, `stand_in._grid`), Constructions (`_cells`, `_stations`, `_platforms`, `_track`), Growth (`_months`, `_served`, `_starving`, `_traded`, `_stack()`, `_reckon_everywhere()`, `_places`), Owners (`_shares`), Railway (`_towns`), StationNames (`_names`). | A `moved(move)` method on each class, next to its own state; regrid only orchestrates. Railway's part: Later. | H | M |
| K2 | sim/routes.py:256, :782, :797 | `constructions._way_problem` used from another module. | Make it public. | M | S |
| K3 | ui/pixel/dots_shader.py:21; ground_shader.py:52, :1571; ui/pixel/trains.py:30; ui/pixel/web_painter.py:26; tools/fingerprint.py:114 | Other cross-module private imports (`_program`, `noise._TABLE`, `cpu._GRASS`, `models._outline`, `painter._Job/_Queue`, `view._pixel_layer()`). | Make public where shared. | L | S |
| K4 | sim/game.py vs ui/station_window.py:144-219, ui/trains_window.py:158-799, ui/buy_train.py:159, ui/map_view.py:384 | Map actions go through Game and return `Done`; station, train and share actions are called straight on Railway/Owners from the UI and return bool, `str \| None` or `Done`. | Route every player action through Game with one result type (P1). | M | M |
| K5 | `Railway.reach()` (indexed `[0]/[1]` at station_trade.py:248, 437, 450; game.py:119; economy.py:797; autoplay.py:69; build_menu.py:339, 353); grids.py:167 `track_exit`; ladders.py:90 POINTS `(Points, int)`; tiles.py:309-335 level pairs | Tuples indexed by position. | `Reach(delivery, collection)`, `TrackExit(cell, side)`, `Gives(points, amount)`, `Levels(entry, full)` NamedTuples. `Reach` lives in railway.py: Later. | M | S |
| K6 | sim/game.py:162, routes.py:129/142, build_menu.py:264, map_tools.py:208; constructions.py:459-460 | `width, electric, maglev` passed as three loose parameters; `Track.electric` + `Track.maglev` allow the invalid "both" state. | `TrackPower` enum {PLAIN, ELECTRIC, MAGLEV}. Touches track_network: coordinate with the rewrite. | M | M |
| K7 | ui/pixel/world.py:71, 89, 107; saves.py:101-361; map_tools.py:208 | Same eight world-build arguments three times; save header as `dict[str, Any]`; an untyped `tuple[object, ...]` cache key. | `WorldSpec`, `SaveHeader`, a frozen key dataclass. | L | S |
| K8 | ui/pixel/art/parts.py and shared helpers in homes.py, farms.py, industry.py | The whole shared art API is underscore-named and imported across 15 modules (`_band`, `_on_wall`, `_foot`, `_smoking_chimney`, …); farms `_plinth` / `_track` shadow parts/industry names. | Drop the underscores; shared ones to parts.py / industry_parts.py. Mechanical. | L | S |
| K9 | sim/points.py `Points` enum; grids.py:167, routes.py:371, :782 `point: int`; `Point = tuple[float, float]` at trains.py:776, lots.py:26, projection.py:23, art/painter.py:11 | "point" means three things. | Hex sides become `side`; one `Point`. | M | M |
| K10 | ladders.py:39-43, forests.py:31, economy.py:82, ground_texture.py:23; RGB ×4 + `Color` in ui/buildings.py:16 | Array and colour aliases redefined (`Cells` = flat indices is not the plural of `Cell`). | Define once in grids.py / one pixel alias module. | M | S |
| K11 | 8 modules import `Points` via sim/ladders.py | Re-export instead of the home module. | Import from sim/points.py. | L | S |
| K12 | sim/game.py:36 `Done` (returned when nothing was done); economy.py:520 `_Month`; `Growth.trade` (the last trade); removal verbs bulldoze / take_down / remove / clear | Misleading names. | `Outcome`, `MonthOfTrade`, `last_trade`; a short glossary in docs/principles.md. | L | S |
| K13 | power_plants.TOWNS vs ladders.TOWNS; happenings.GUN_FACTORIES vs the ladder; towns.TOWN_REACH (1) vs tutorial/place.TOWN_REACH (3); goods.PRICES vs facilities.PRICES; three `TITLE_SCALE`; two `GROWN_WIDTH` | Name clashes with different values. | Rename one of each. | L | S |

God classes, by size: Railway (73 methods; Later), Growth (62 methods, 39 attributes; S3),
MapView (60/47; S6), Ground (37/22; S9), `_Month` (23/32; S4), App + Shell (S7). Game itself is
small (254 lines) and is the right place to gather player actions.

## 4. DRY

| # | Where | Finding | Proposed change | Sev | Eff |
|---|---|---|---|---|---|
| D1 | Full form `np.where(d>0, n/np.where(d>0,d,1), e)`: economy.py:378, 729, 763, 1062, 1112; station_trade.py:190; fuel_market.py:191. Short form `n/np.where(d>0,d,1)`: economy.py:384, 668, 678, 831, 918, 1001, 1034, 1041, 1078, 1088, 1141; station_trade.py:233, 350. Other styles: ui/pixel/relief.py:554, mapgen/area.py:170 | ~20 hand-written safe divisions. | `ratio(n, d, empty=0.0)` in sim/grids.py. | M | S |
| D2 | sim/game.py:150-154, 187-188, 203-204, 219-222, 240-243 | "X costs $Y; you have $Z." five times. | One helper now; P1 later. | L | S |
| D3 | sim/growth.py:695-699, 819-822, 1356-1359 (+ take_down 795-801) | owners.lose / constructions.remove ×n / add(−n) sequence three times. | `Growth._pull_down(cell, kind, n)`. | M | S |
| D4 | sim/power_plants.py:106-131, sim/novelties.py:87-112 | `_nearest_free` twice; "biggest town away from these cells" twice. | `nearest_free`, `biggest_towns` in sim/towns.py. | L | S |
| D5 | sim/economy.py:241-253 vs sim/autoplay.py:176-188 | Same parity-offset ring building. | `parity_offsets(grid, steps)` in grids.py. | L | M |
| D6 | ui/help_window.py:172, sim/growth.py:912 | `ladder_of(kind)` twice. | One in ladders.py. | L | S |
| D7 | 23 sites (trains_window ×15, hex_details ×3, goal_view, buy_train, station_window, save_files, route_map) | `if is_item_hovered(): set_tooltip(...)` that `theme.tip` covers; six pass `allow_when_disabled`. | `tip(text, disabled=False)`; replace all. | M | S |
| D8 | ui/station_window.py:222-234, ui/buy_train.py:149-156, ui/land_offers.py:221-228, ui/trains_window.py:782-801 | "Priced button" (disabled when unaffordable + "Not enough money") four ways. | `priced_button` in theme.py with a selectable variant. | L | S |
| D9 | ui/station_window.py:174-185, ui/station_preview.py:27-35, ui/map_overlays.py:405-410; ornate.py:92-94, build_menu.py:206-208, tiles_window.py:338-340, map_view.py:754-756 | Hover-card window block ×3; deep panel style push ×4. | `hover_card()` and `deep_panel()` context managers. | L | S |
| D10 | 40 sites; trains_window.py:225, app.py:717 | `imgui.text_colored(imgui.ImVec4(*C), …)`; `get_color_u32(ImVec4(*X))` instead of `color_u32`. | `colored(color, text)`. | L | S |
| D11 | ui/trains_window.py:829-848 vs ui/buy_train.py:240-273 | `engine_summary` and `engine_figures` hold the same ten facts. | One list of (label, value), two renderings. | M | S |
| D12 | ui/trains_window.py:626-663 | Offer and general-car chip loops hand-roll `theme.next_good_slot` (with gap 4 vs 10). | `next_good_slot(gap=)`, one loop. | L | S |
| D13 | `${x:,.2f}` ×3; plural `'' if n == 1 else 's'` ×5; "name x{n}" ×3; `tonnes_text` + " t" ×9 living in a screen module (buy_train.py:49) | Hand-rolled formatting. | P2. | L | S |
| D14 | trains_window.py:163 vs stations_window.py:35 | "Right of screen minus the cards" placement formula twice. | One theme helper. | L | S |
| D15 | ui/web_setup.py:70, 131, 157, ui/pixel/copies.py:56, 88; places.py:59-93, mapgen/streams.py:75-93, mapgen/world_files.py:114-130; mapgen/lands.py:324-330 vs tutorial/place.py:167-173 | `pickle.dumps(…, HIGHEST_PROTOCOL)` ×5; npz to/from bytes ×3; smoothstep value noise ×2. | `pack`, `npz_bytes`, `read_npz`; one noise helper. | L | S |
| D16 | tools/fingerprint.py:36-110 | Rebuilds `newgame.make_game` by hand (seed offsets, YEARS). `chunk_pixels.py` already calls `make_game`. | Reuse `make_game`. | M | S |
| D17 | sim/ladders.py:908-922 vs :950-959 | `_town_sight` and `_town_craft` build the same Ladder core. | `_town_sight` calls `_town_craft`. | L | S |

Closed since the last DRY pass: `goods.FOOD_INPUTS`, `StationTrade.rewards`, TAVERNS vs _BAKERY,
the room.py copy of `ladder_of`.

## 5. Consistency

### 5a. Caches

There are 35 hand-rolled caches in eight shapes. Key and invalidation are mostly right; the
exceptions are B1 and B3.

| Shape | Examples |
|---|---|
| (a) `(version, value)` tuple attribute | ui/route_map.py:524-543; ui/map_view.py:802-806 |
| (b) separate `_x_version` int | ui/pixel/world.py:211-223, :552-557 |
| (c) class-level default `(-1, {})` per-version dict | ui/pixel/world.py:302-312; track_network ×5 (Later) |
| (d) module `WeakKeyDictionary[Constructions]` | ui/build_menu.py:471-485; ui/pixel/scenery.py:448-475 |
| (e) version stored with each value | ui/pixel/train_layer.py:426-438; traffic, track_art (Later) |
| (f) composite keys compared element by element with `==` / `is` | ui/map_view.py:818-824, 885-903 (`_built_previewed` is never pruned) |
| (g) time-throttled dicts | ui/trains_window.py:119-146 (keyed on train name or station, never pruned when a train is sold); ui/route_map.py:381-388; ui/hex_details.py:113-131 |
| (h) shader version slots, int or identity | ui/map_view.py, ground_shader.py:1928-2144, dots_shader.py, picture_show.py |

`sim/year_caches.py` (`year_cached`, `on_forget`) is the one shared helper and works well; P3 builds
the rest on the same idea.

### 5b. Docstrings and comments

- C1 — Plain-prose docstrings everywhere, no TODO/FIXME, no stale "tech tree" or "contract"
  references. Good.
- C2 — Module docstrings missing in six files: `__main__.py`, mapgen/generate.py, mapgen/noise.py,
  sim/geo.py, ui/app.py, ui/iso_camera.py. L/S.
- C3 — Stale comments: B4-B7 above; sim/growth.py:6-15 restates `DRAWN_BY` weights in prose
  ("six times the people…"); ui/help_window.py:93-97 states measured timings ("3 seconds",
  "25 ms") that will drift; route_map.py:92 comment on `SHADE_FROM` contradicts how
  `Marks.shades` divides by `QUARTER`. L/S.
- C4 — Old-save code against the "no old saves" rule: sim/constructions.py:242-245
  `BuildingKind._missing_` ("meat factory"). More in railway.py and track_network.py (Later).
  `SAVE_FORMAT` already refuses old saves. L/S.

### 5c. Error handling

- C5 — ui/pixel/dots_shader.py:164 and ground_shader.py:1975 `except Exception: return None`
  swallow GL errors silently. Print once to stderr, as web_painter does. L/S.
- C6 — Diagnostics go to stdout in ui/web_setup.py:172, 175 and ui/app.py:460, to stderr
  elsewhere. Pick stderr. L/S.
- C7 — mapgen/world_files.py:89 is the only `except Exception` without the `# noqa: BLE001 -
  reason` note. L/S.
- C8 — 30 asserts used for type narrowing, repetitive in ui/minimap.py (×5 `panel is not None`),
  sim/autoplay.py (×3), station_trade.py (×2). Typed accessors or non-optional fields. L/S.
- "Can't" is reported as `str | None` problems, `Done(ok=False)` and `None` returns, consistently
  within each layer. Leave it until P1.

### 5d. Tests

Full run: 2607 passed in 124 s.

- C9 — `make_map` defined 18 times; several copies identical (test_military.py:17 ==
  test_novelties.py:22; test_churches.py:21 == test_autoplay.py:18; test_hover_card.py:17 is just
  `flat_map(12, 12)`). Many build layers with `np.full` instead of `builders.layer`. Add square and
  coastal builders to tests/builders.py. M/M.
- C10 — imgui context fixture copied in 14 test files; `FakeTexture` defined 5 times. One `gui`
  fixture in conftest.py, `FakeTexture` in builders.py. M/S.
- C11 — Tests import helpers from other test modules at 10 sites (`joined_stations`,
  `played_game`, `make_world`, `world_over_mountains`, `FakeTexture`). Move them to builders.py.
  M/S.
- C12 — `make_game(cash)` twice and `Game(Constructions(flat_map(...)))` + cash in 12 files;
  `grown(*placed)` in 7 variants. Into builders.py. L/S.
- C13 — 135 of 1557 test functions interleave asserts with further actions (most in
  test_railway 16, test_engine_roster 15, test_routes 13). Against Arrange-Act-Assert; split the
  multi-phase ones. No AAA comments anywhere (good). M/M.
- C14 — 227 reads of private attributes in tests (`._turn` ×16, `._pixel_layer` ×11, `._task`
  ×11…). Accept as white-box, or expose a few seams. L/M.
- C15 — Slow setups: test_land.py 5.9 s + 2.2 s, test_seeding.py 5.7 s, test_lands_map.py 4.8 s,
  two app-frame tests 11.2 s and 5.8 s. Module-scoped fixtures and smaller maps. L/M.

## 6. Pattern potential

Only where it removes real code or a class of bugs.

- **P1 — One player-action path (command objects).** sim/game.py:144-254 repeats
  problem → price → affordability message → do and spend for five actions; ui/map_tools.py:170-195
  re-derives the prices for the hover note; station, train and share buttons check affordability
  themselves (K4, S15, S16, D2, D8). An `Action` with `problem()`, `price()`, `apply()` and one
  `Game.do(action) -> Outcome` lets the UI ask the same object for the hover price and the greyed
  state. No undo (none exists, none asked for). H/M.
- **P2 — `railroads/words.py` for formatting.** `dollars`, `tonnes`, `plural`, `counted`,
  `unit_name`. Sim-safe (no imgui). L/S.
- **P3 — `railroads/kept.py`, a small cache helper.** `Kept` (recompute when the key changes,
  comparing arrays by identity), `KeptMap` (per-version dict, bounded), `kept_per(owner)` (weak
  per-owner), `Fresh(seconds)` (time-throttled, prunable). Covers shapes (a)-(g); shader slots stay.
  The gain is correctness: B1/B3-style key mistakes become visible, and per-train dicts get pruned.
  M/M.
- **P4 — A small UI kit in theme.py:** `tip(disabled=)`, `colored`, `hover_card`, `deep_panel`,
  `priced_button`, `next_good_slot(gap=)`, a spacing scale. Removes ~80 repeated idioms. M/S.
- **P5 — `ratio()`** (D1). M/S.
- **P6 — Per-building spec (`BuildingSpec`).** One kind's numbers are spread over ~25 tables in 8
  modules (goods: RECIPES, MAKES, _NEEDS, DIESEL_USERS, KIND_BOOST_MOST, OWNED_BOOSTS,
  KIND_NEED_YEARS, KIND_GOOD_YEARS, BONUS_PART, VENUE_BONUSES, KIND_BONUSES; ladders: _POINTS;
  tiles: UNLOCKS, KIND_ERAS; growth: DRAWN_BY_KIND, SHUNNED_BY; constructions: BUILDING_RULES,
  MERGED, GREW_FROM; fuel_market: PLANT_FUEL; plus UI tables). A spec per kind would fit the
  "numbers at the building" rule best, but it is a large move. Cheaper first step: a test that every
  BuildingKind appears in the tables it must be in. M/L.
- **Not worth it:** imgui table setup (10 `begin_table` calls that differ for good reasons),
  `set_next_window_pos` blocks (18, each anchored differently), a visitor for regrid (plain
  `moved()` methods are enough), undo.

## 7. Later (train-traffic files)

To pick up once the traffic rewrite is on staging.

- L1 — sim/railway.py (901 lines, 73 methods) mixes fleet, weight and pull, fuel, fares, station
  levels and facilities, and the monthly run. Split `fleet.py`, `loading.py`, `fares.py`,
  `station_levels.py` (with the constants at :128-135, fixing S17).
- L2 — `Reach` NamedTuple at railway.py:692 (K5); `Railway.pull` shared with
  trains_window/train_tonnage (M23); `Railway.moved(move)` for regrid (K1).
- L3 — track_network.py:28-41: caches as class-level mutable defaults "so games saved without them
  load", five per-version dict resets with the same four lines (:50-53, 95-100, 111-118, 129-136,
  216-221, 317-322). route, route_length and fuel_length share one key. Init in `__init__`, one
  `KeptMap`. `electric, maglev` pairs → `TrackPower` (K6).
- L4 — railway.py:170-174 `station_trade`, `facilities`, `supplied` default to None "in older
  saves", forcing `or {}` at :502, 728, 742-743, 751.
- L5 — trains.py:910-913 `path_ends`, `path_stretches` are caches pickled into saves; exclude them.
  traffic.py:79-81 compares `train.path` (a list) element by element on every call; use `is`.
- L6 — Cycle S3 {track_network, traffic, trains}: the new signals.py should depend on trains and
  track_network, not the reverse. railway.py:115-117 TYPE_CHECKING back-edges into S2.
- L7 — trains.py:62 `GENERATION_YEARS` is a year-based second copy of era timing in tiles.ERAS;
  `MAGLEV_FARE = 11` unexplained; car sizes 5/13/34/89 restate CAR_SIZES; `traffic.LINE_PATIENCE`
  derived by hand from `SECONDS_PER_MONTH`; sim/railway.py:138 `GENERAL_CARS_ERA = 3`.
- L8 — track_art.py: `DOUBLE_SPACING` export (M24), fourth railway_version cache next to world.py's
  three, the painter-thread race comment at :564; `_draw` with 8 parameters. train_layer:
  `MAX_TEXTURE` drives `trains.GROWN_WIDTH` by hand; `PAGES_TIME` / `PAGES_BUDGET` repeat layer.py's
  budgets; `trains.SCALE` not derived from the gauge.

## 8. Proposed order of work

Each step is one commit with `uv run pytest -q`, `uv run ruff check src tests -q` and
`uv run ruff format -q src tests` passing. Steps 1-11 change no game numbers; where a step could,
check with `PYTHONHASHSEED=0 uv run python tools/fingerprint.py` and `tools/chunk_pixels.py check`
before and after. None touches the train-traffic files.

1. **Bug fixes** B1-B3 (`input_stock` → `@year_cached`, "All N trains" uses the cheapest swap,
   `_consumers_of` key), plus the stale comments B4-B7 and the FURNITURE/LOGS colour (B8; pick a
   colour). S.
2. **Calendar constants:** `MONTHS_PER_YEAR`, derived `MONTHS_BEFORE_PLAY`, one first year (1804),
   `QUARTER` reused (M13, M14, M27, M28, M33). S.
3. **Duplicated constants:** `KM_PER_CELL`, `GARRISON_SIZE` / `STARTING_OUTPOSTS`, reference land,
   `lot_price`, lot middle, track spacing, side-window top, zoom indices, `COINS`, world bounds
   (M16-M18, M20, M24, M26, M29-M32). S.
4. **Hand-out radius** (M15): one radius in station_trade, no edit in railway.py. S.
5. **Help and hint text from constants** (M21, M22). S.
6. **Numeric helpers:** `ratio()` and `EPSILON` (D1, M34). S.
7. **UI kit** (P4: D7-D10, D12, D14). S/M.
8. **Formatting module** (P2: D13) and one affordability message (D2). S.
9. **Small sim DRY:** `_pull_down`, `nearest_free` / `biggest_towns`, `ladder_of`, `_town_sight`,
   `parity_offsets`, `engine_figures` as the single source, fingerprint reusing `make_game`
   (D3-D6, D11, D16, D17). S/M.
10. **Private access and names:** public `way_problem`, GL helpers, art helpers without
    underscores, `Points` imports, aliases defined once, `Outcome` / `MonthOfTrade` / `last_trade`,
    name clashes (K2, K3, K8, K10-K13). S each; can ship as two or three commits.
11. **Tests:** `gui` fixture, `FakeTexture`, shared builders, no cross-test imports, then AAA
    splits (C9-C13). M.
12. **Cache helper** `kept.py` and move the non-traffic caches onto it (P3). M.
13. **Module splits, one per commit:** art registry (S11, breaks cycle S7), ground_glsl/gl_util
    (S8), catalogue for readme_tables (S14), app → autosave/shell/cli (S7), map_view → ground and
    lenses (S6), ladders (S5), goods + yearly.py (S1-S2, breaks cycle S1), economy reach and
    trade_month (S4), growth breeding/dying/placement (S3), ground palette and woods (S9),
    hex_rows and train_texts (S13), settings.py (S18), renames (S19). M each.
14. **Regrid via `moved()` methods** for everything but Railway (K1); Railway follows the traffic
    rewrite. M.
15. **One player-action path** (P1, K4, S15, S16). M/L.
16. **Hidden scaling (needs the user's decisions first, item by item):** write big/mega numbers
    out per kind (M1), breeding weights at each ladder (M2), passengers per building (M3),
    demolition prices (M4), diesel/boost/power-plant numbers at the building (M6, M7), resource
    chances explicit (M8), freight constants (M9), `BONUS_PART` (M10), `DIE_AFTER` tiers (M12).
    Each keeps today's numbers, only written where they belong; decide whether M5 (price from
    output) and M11 (war growth) stay global. M each.
17. **Old-save code** in constructions (C4), error-handling tidy-ups (C5-C8). S.
18. **After the traffic rewrite lands:** section 7.
