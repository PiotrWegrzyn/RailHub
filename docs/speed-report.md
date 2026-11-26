# Speed report, night of 16–17 November 2026

Pure speed work: no rule, number or look changed. Every change was checked to grow the same game
(building counts per kind, and a hash of every kind's count in every cell, on Poland 1850 seeds 1
and 2), the full fingerprint of `tools/fingerprint.py` (run with `PYTHONHASHSEED=0`, since one of
its lines prints a set of strings whose order follows Python's hash seed) and pixel-identical
ground pictures.

## How it was measured

All on one MacBook (Apple M3 Pro). Scripts are not in the repo; they drive the real game:

- **Generating a new game** (desktop): `make_game` for Poland, seed 1 and 2, start 1850 (towns
  grown from 850), timed alone.
- **New game, first frame** (desktop and browser): the real `Shell` starts a new game (Poland,
  seed 1, 1850) and the time until the game's first frame is drawn. In the browser this includes
  the setup worker growing the towns and handing them over.
- **Generation inside Pyodide**: `make_game` run on the page's main thread, and in the setup
  worker on an idle page.
- **Page load**: `index.html?play` served locally, to the main menu's first frame.
- **Play scene** (desktop and browser) on two saves made by a stand-in player (56 stations, 104
  trains, 10 years simulated): Poland 1850 (play from 1860) and Poland 1950 (play from 1960).
  The scene loads the save, then times every frame's GUI: 300 frames watching at normal speed,
  400 panning at zoom 4, 240 cycling zoom levels, 600 at speed 6, 600 with a month forced every
  60 frames (month frames and the autosave frame counted apart), the windows, buying 20 trains
  and laying 10 pieces of track. Numbers are mean / 95th percentile / worst, in milliseconds.
- **A month of play** (desktop): the 1960 save, months run step by step as the app runs them;
  total work per month and the longest step (one frame).
- **Autosave**: pickling the 1960 game as an autosave does (`saves.snapshot`).
- Browser numbers come from headless Chrome (real GPU, ANGLE Metal, 1440×860 at 2× density)
  driven over the DevTools protocol.

Other sessions changed growth numbers, train pictures and the roster on staging during the night,
so "before" is staging with this work's commits reverted on top (everything else the same), and
"after" is staging with them. Both grow the same games: the hashes match on both seeds. For
reference, at the start of the night (75ed0e6) growing Poland 1850 took 24.2 s (seed 1) and
25.5 s (seed 2) on desktop and a new game took 53.6 s to show in the browser.

## Results

In short: a new game shows in the browser in 22.7 s instead of 52.1 s (desktop 12.9 s instead of
28.1 s), because growing Poland 1850's towns takes 10 s instead of 23 s on desktop and 19 s
instead of 48 s in the browser. A month of play costs a quarter less, and its worst frame and the
autosave no longer stall the game: the browser's worst frame of a month with an autosave went from
182 ms to 98 ms, the autosave frame from 182 ms to 31 ms, and zooming out from 117 ms to 35 ms on
the 1950 save. Everyday frames (watching, panning, fast speed) were already cheap and stayed so.

Not improved: one month frame on the 1950 save still reaches ~98 ms in the browser, before and
after; it was not tracked down tonight. "lay_track call" lays track straight across the map in
one call, where the game plans it over several frames first, so it is not what a player waits
for.

### Generating and starting games

| | before | after |
|---|---|---|
| Poland 1850, seed 1: towns grown (desktop) | 23.1 s | 10.0 s (identical) |
| Poland 1850, seed 2: towns grown (desktop) | 23.1 s | 9.7 s (identical) |
| New game to first frame, desktop | 28.05 s | 12.89 s |
| New game to first frame, browser | 52.11 s | 22.71 s |
| Towns grown in Pyodide, page's main thread | 48.04 s | 18.52 s |
| Towns grown in the setup worker, idle page | 48.1 s | 19.2 s |
| Page load to the menu's first frame (local server) | 2.65 s | 2.62 s |
| Loading the 1850 save, desktop | 1.92 s | 1.92 s |
| Loading the 1850 save, browser | 3.09 s | 2.92 s |
| Loading the 1950 save, desktop | 2.03 s | 1.99 s |
| Loading the 1950 save, browser | 3.24 s | 2.99 s |

### A month of play (desktop, work per month and longest frame of it)

| | before | after |
|---|---|---|
| 1850 save: whole month | 203.0 ms | 151.5 ms |
| 1850 save: longest step | 23.0 ms | 17.0 ms |
| 1950 save: whole month | 264.5 ms | 195.0 ms |
| 1950 save: longest step | 34.0 ms | 19.0 ms |
| 1950 save: autosave snapshot, whole | 64.0 ms | 44.7 ms |
| 1950 save: autosave snapshot, longest frame | 64.0 ms | 12.1 ms |

### Frames, desktop, new game, Poland 1850 (GUI ms: mean / 95th percentile / worst)

| | before | after |
|---|---|---|
| watch | 0.5 / 1.1 / 6.4 | 0.4 / 1.1 / 5.5 |
| pan | 0.6 / 0.9 / 6.8 | 0.6 / 0.9 / 5.1 |
| zoom | 1.0 / 2.7 / 61.2 | 0.7 / 2.5 / 11.4 |
| fast | 0.3 / 0.5 / 0.9 | 0.3 / 0.4 / 0.6 |
| months | 3.4 / 18.4 / 51.4 | 2.8 / 11.7 / 38.4 |
| month frame | 10.9 / 33.3 / 37.9 | 7.4 / 22.0 / 38.4 |
| autosaving frame | 3.6 / 51.4 / 51.4 | 0.9 / 9.7 / 9.7 |
| full_gc | 33.6 / 37.8 / 37.8 | 29.9 / 40.3 / 40.3 |

### Frames, desktop, 1850 save (124 trains) (GUI ms: mean / 95th percentile / worst)

| | before | after |
|---|---|---|
| watch | 1.6 / 1.7 / 43.1 | 1.5 / 1.7 / 42.9 |
| pan | 2.0 / 3.2 / 18.4 | 1.9 / 2.8 / 20.3 |
| zoom | 3.9 / 8.8 / 45.8 | 3.7 / 8.6 / 14.2 |
| fast | 1.5 / 2.1 / 2.6 | 1.3 / 1.6 / 2.2 |
| months | 5.6 / 23.8 / 87.4 | 4.8 / 15.3 / 47.6 |
| month frame | 13.4 / 38.1 / 42.9 | 10.1 / 25.1 / 47.6 |
| autosaving frame | 6.4 / 87.4 / 87.4 | 2.5 / 19.0 / 19.0 |
| windows | 2.2 / 3.8 / 20.6 | 2.0 / 3.8 / 21.2 |
| buy | 2.6 / 3.0 / 14.5 | 2.3 / 2.6 / 13.8 |
| track | 2.0 / 1.9 / 30.8 | 1.8 / 1.7 / 30.8 |
| lay_track call | 147.5 / 297.9 / 297.9 | 149.3 / 295.2 / 295.2 |
| full_gc | 21.1 / 33.0 / 33.0 | 21.3 / 30.7 / 30.7 |

### Frames, desktop, 1950 save (124 trains) (GUI ms: mean / 95th percentile / worst)

| | before | after |
|---|---|---|
| watch | 1.7 / 1.9 / 46.2 | 1.6 / 1.8 / 47.6 |
| pan | 2.3 / 8.6 / 21.9 | 2.2 / 8.7 / 19.1 |
| zoom | 5.0 / 11.8 / 55.2 | 4.9 / 12.9 / 20.2 |
| fast | 1.6 / 2.1 / 2.7 | 1.4 / 1.7 / 2.2 |
| months | 7.0 / 31.6 / 98.6 | 6.1 / 20.4 / 50.3 |
| month frame | 15.3 / 46.4 / 64.0 | 12.1 / 30.3 / 50.3 |
| autosaving frame | 7.1 / 98.6 / 98.6 | 2.7 / 21.8 / 21.8 |
| windows | 2.3 / 3.9 / 22.2 | 2.3 / 3.9 / 21.5 |
| buy | 2.7 / 3.6 / 15.2 | 2.7 / 3.4 / 15.3 |
| track | 2.4 / 2.1 / 33.2 | 2.2 / 2.0 / 34.2 |
| lay_track call | 198.8 / 318.5 / 318.5 | 193.7 / 303.7 / 303.7 |
| full_gc | 21.9 / 32.6 / 32.6 | 20.5 / 31.2 / 31.2 |

### Frames, browser, new game, Poland 1850 (GUI ms: mean / 95th percentile / worst)

| | before | after |
|---|---|---|
| watch | 1.3 / 2.1 / 7.4 | 1.5 / 2.4 / 6.5 |
| pan | 1.9 / 3.3 / 4.1 | 2.0 / 3.3 / 4.1 |
| zoom | 3.0 / 6.5 / 127.9 | 2.6 / 7.0 / 29.1 |
| fast | 1.4 / 2.3 / 2.6 | 1.6 / 2.3 / 2.8 |
| months | 7.0 / 33.2 / 117.3 | 5.8 / 20.4 / 71.2 |
| month frame | 18.6 / 63.4 / 70.8 | 12.9 / 27.8 / 71.2 |
| autosaving frame | 7.9 / 5.9 / 117.3 | 7.0 / 20.0 / 22.6 |
| full_gc | 61.3 / 61.3 / 61.3 | 63.2 / 63.2 / 63.2 |

### Frames, browser, 1850 save (124 trains) (GUI ms: mean / 95th percentile / worst)

| | before | after |
|---|---|---|
| watch | 4.2 / 4.8 / 45.3 | 4.1 / 5.0 / 44.0 |
| pan | 5.1 / 9.5 / 39.1 | 4.9 / 9.3 / 32.7 |
| zoom | 8.2 / 11.6 / 121.8 | 7.7 / 11.5 / 32.7 |
| fast | 4.2 / 4.8 / 6.2 | 4.2 / 4.9 / 6.9 |
| months | 10.6 / 38.9 / 150.1 | 9.3 / 25.8 / 81.3 |
| month frame | 21.1 / 53.5 / 82.0 | 17.0 / 39.4 / 81.3 |
| autosaving frame | 11.2 / 8.1 / 150.1 | 9.8 / 24.3 / 24.8 |
| windows | 5.5 / 7.3 / 55.1 | 5.5 / 7.6 / 54.0 |
| buy | 6.3 / 8.6 / 31.7 | 6.2 / 8.8 / 29.5 |
| track | 5.0 / 4.8 / 96.8 | 4.9 / 4.8 / 95.4 |
| lay_track call | 317.2 / 628.5 / 628.5 | 317.1 / 621.9 / 621.9 |
| full_gc | 46.5 / 46.5 / 46.5 | 46.0 / 46.0 / 46.0 |

### Frames, browser, 1950 save (124 trains) (GUI ms: mean / 95th percentile / worst)

| | before | after |
|---|---|---|
| watch | 4.4 / 5.1 / 45.7 | 4.4 / 5.0 / 45.1 |
| pan | 5.6 / 11.3 / 37.8 | 5.7 / 11.1 / 38.8 |
| zoom | 8.9 / 13.3 / 116.6 | 8.5 / 13.1 / 34.7 |
| fast | 4.5 / 5.4 / 6.2 | 4.6 / 5.2 / 5.7 |
| months | 12.7 / 48.8 / 182.0 | 11.2 / 35.5 / 98.3 |
| month frame | 23.0 / 64.6 / 98.1 | 19.3 / 47.4 / 98.3 |
| autosaving frame | 13.8 / 30.6 / 182.0 | 12.0 / 28.3 / 31.3 |
| windows | 5.9 / 7.9 / 49.8 | 6.0 / 7.8 / 48.9 |
| buy | 6.8 / 11.4 / 34.2 | 6.9 / 11.9 / 33.6 |
| track | 5.6 / 5.3 / 106.4 | 5.6 / 5.1 / 102.2 |
| lay_track call | 392.3 / 616.6 / 616.6 | 399.2 / 618.3 / 618.3 |
| full_gc | 53.1 / 53.1 / 53.1 | 49.2 / 49.2 / 49.2 |

## What changed

On staging as versions up to 0.4.6 (the report itself with 0.4.7).

Growing towns (history and every month of play):

- **Chances of a new building's ladders in one go** (`growth.py`). Since the growth formula
  started counting what draws a ladder (9c9cd9b), every building changed the chances of about
  40 ladders in its 7 cells, each with a dozen small numpy calls: 1.7 million calls for one
  Poland 1850 history, half its time. All ladders' chances, lands, nearby and served points are
  now rows of one array each (the places and dicts keep views of their rows), and
  `Growth.add` works out every touched ladder at once (`_reckon_touched`), with the same sums in
  the same order.
- **Land weights read, not recomputed.** Putting a building up recomputed the land weights of
  the 7 cells around it from the map; growth already keeps every land's weights up to date as
  forests change, so `Growth.land` reads them (checked equal to fresh weights at every call over
  whole histories with stand-in years).
- **Forest changes weigh the land once per kind of land**: what does not depend on the land
  (water, mountains, forest and open land around each cell) is worked out once in a
  `Surroundings` for all ~16 kinds of land.
- **Every ladder's chances over the whole map** (a new game, a loaded game) share the doublings
  of what draws them (12 distinct draws for 89 ladders) and work on whole arrays: 100 → 27 ms.
- New buildings count their points by flat neighbour indices; every ladder's highest chance is
  found in one pass after a trade.

The monthly trade (`economy.py`, `station_trade.py`, `demand.py`):

- Processor stockpiles are kept only for the kinds and inputs some building holds
  (`economy.Stockpiles`; asked for, a missing one is empty), instead of a map-sized array for
  each of 243 kinds and inputs that every month copied and cleared and every autosave scanned:
  about 200 MB less memory in a late game.
- Trading pairs (sellers and the buyers in reach) are found on a map padded with empty cells,
  ring by ring, instead of checking every pair against the map's edges.
- What each kind of building wants of each need is looked up once per kind and year instead of
  rebuilding recipe sets for every kind and need, also when a train unloads at a station.
- Stations skip gathering a good nobody offers or they have no room for; each trade works out
  every kind's satisfaction share once (it was worked out two or three times a month); each
  good's sold share is worked out once instead of once per kind making it.
- Paying for goods a station hands out goes unit by unit as before, with the same float
  operations, but without three method calls per unit.

Spreading work over frames (no change in what is computed):

- The month's longest steps are split where they naturally divide (before the trade, inside
  settling the trade, between paying for goods handed out and for goods passed along, between
  starving buildings and served points).
- **Autosaves are pickled in steps while the month is still being done**: one pickler pickles
  the game's biggest parts a step each (buildings, counts, economy, last trade, starving counts,
  lands), then the game, which only refers back to them; loading reads them back a part at a
  time (a new game handed over by the browser's setup worker too). The state saved is the same
  as before: the end of the month, before any train moves.
- Buildings are pickled as their kinds in each cell's list (17k objects pickled one by one
  before).
- numpy's masked arrays are imported with the game: `np.unique` imported them the first time
  forests changed, stalling the first month of every session.

Drawing:

- The zoomed-out ground picture with baked grid lines is kept and touched up only where cells
  changed colour (as the picture without lines already was), and painted while a game loads,
  instead of copied and shaded whole (3 million pixels) on the frame you zoom out and after
  every month that changed colours.

## Tried and dropped

- **Faster enum hashing.** cProfile showed `Enum.__hash__` (a Python function) as one of the
  biggest costs of a month. Serving the same hash value from C (`functools.partial` over
  `operator.attrgetter`) made no difference outside the profiler: a hash takes 46 ns either
  way, and the profiler inflates small Python calls. Not kept.
- **Counting non-zeros directly** when pickling big arrays (`np.count_nonzero` on the integer
  view instead of on a `!= 0` mask): slower, since numpy counts booleans much faster. Checking
  `max()` first for all-zero arrays: no measurable gain. Not kept.
- **One gather for all three channels of the grid-line shading**: two to three times slower
  than the three channel passes already there. Not kept.
- **Building the ground picture's grid lines eagerly** after every touch-up: correct, but it
  cost time on months when the lined picture was not shown; it is now done lazily when that
  picture is next wanted, rebuilt whole if more than 2,000 boxes changed.
- **Checking stations' room before gathering** gave no measurable gain on its own (line
  profiles overstated it); kept with the other trade changes as it costs nothing.

Not done, worth considering:

- A new game in the browser spends ~20 s growing towns in the setup worker. The main menu picks
  its seed when it opens, so the worker could start growing the game the menu shows while the
  player looks at it, and hand it over at once if the settings are unchanged when they press
  New game. That changes when work happens, so it is left for you to decide.
- Autosaves still scan a map-sized count array for each of ~190 kinds of building, most of them
  never on the map. Keeping counts only for kinds on the map, as the stockpiles now are, would
  make saves and months cheaper still, but touches much of the growth code.

## Loading and sprites (17 November, versions 0.4.20 and 0.4.21)

The question was whether drawing every sprite before a game shows is what makes loading slow.
Measured with every part of loading timed (same machine and headless Chrome as above):

| Browser, new game (Poland 1850, seed 1), click to first game frame | seconds |
|---|---|
| Setup worker's Python starting (overlaps the menu in normal play) | ~2.2 |
| Growing the towns, 850 → 1850 | ~20.2 |
| Unpacking the grown game | 0.3 |
| Map preparation: ground, relief, scenery, ground pictures | ~1.3 |
| Loading both building sprite sheets (drawn ahead into PNGs at build time) | 0.07 |

Building sprites are drawn when the site is built, not in the browser, so loading only reads
them: 0.3% of a new game's load. The one set of pictures drawn whole at load was the train sheet,
every look of every engine and car of every era (201 looks, 24 headings each), built when a game
with trains loads, or, in a new game, on the first buy.

| | before | after |
|---|---|---|
| Browser, 1950 save, load to first game frame | 3.54 s | 2.40 s |
| Browser, 1950 save, longest frame while loading | 2,688 ms | 351 ms |
| Browser, new game, longest frame after the towns arrive | 1,333 ms | ~280 ms |
| Browser, new game, click to first game frame | 24.3 s | 23.3–24.8 s (unchanged) |
| Desktop, 1950 save, load to first game frame | 2.06 s | 1.59 s |
| Desktop, 1850 save, load to first game frame | 2.07 s | 1.48 s |
| Train sheet drawn while loading a save (browser / desktop) | 0.82 s / 0.53 s | 0.07 s / 0.03 s |

What changed:

- **Train looks are drawn as they are wanted** (`trains.GrownTrainSprites`): a game loads with
  only the looks of the trains it has. A look not on the sheet yet is drawn when a train or the buy
  screen first wants it (about 2.5 ms on desktop, 5 ms in the browser), and frames that show
  trains draw the rest ahead within the time they already allow for close-up pages. The sheet goes
  to the graphics card again only when a frame needs a look missing from it. Each look's pictures
  are exactly those of the old sheet, outline included (checked pixel for pixel for all 201 looks).
- **The map is prepared a part at a time** (`MapView.prepare_steps`, `PixelWorld.built`), so the
  waiting screen keeps moving in the browser instead of freezing for one long frame.

The whole wait for a new game is still the towns growing: about 20 of the 23 seconds. The
remaining loading frames of 200–350 ms in the browser are single parts of the preparation
(relief, scenery, ground, grading the track bed, the zoomed-out picture), each one piece of work
that would have to be split inside itself.

## The monthly freeze (18 November, a new game with 3 trains)

The complaint: in the browser every month froze the game for about half a second, even in a new
game with 3 trains. Measured on a new game (Poland, seed 1, 1850, the cropped square map) with 4
stations, track between them and 3 trains, at zoom 3 over the first station, a month forced every
90 frames, 12 months; same machine and headless Chrome as above. "Freeze" is the time from the frame
a month begins to the frame it ends: the trains and the clock wait all of it.

What was found: the month itself was not dear. Its work was about 25 ms on desktop (33 ms with the
map's pictures) and about 76 ms in the browser, much as before the day's economy changes (owned
shares, worth of what is handed out, experience, taverns, bonus goods). But it came in 17 or 18
steps and the game gave each step a frame of its own, so a month took 18 frames, each drawing the
whole map while the trains stood still: about 18 × 14 ms of drawing in the browser on top of the
work. An autosave month added a frame per part of the save.

| | before | after |
|---|---|---|
| Browser: trains frozen per month | 359 ms (18 frames) | 131 ms (4 frames) |
| Browser: trains frozen, month with an autosave | 581 ms (26 frames) | 207 ms (6 frames) |
| Browser: longest frame of a month | 41 ms | 42 ms |
| Desktop: trains frozen per month | 228 ms (17 frames) | 71 ms (3 frames) |
| Desktop: trains frozen, month with an autosave | 379 ms (25 frames) | 119 ms (4 frames) |
| Desktop: longest frame of a month | 21 ms | 30 ms |
| Desktop: scenery put up again after buildings changed | 9.2 ms | 7.9 ms |
| Desktop: changes pickled for the painting workers | 3.5 ms | 1.7 ms |

A frame without a month takes about 13 ms in the browser and 6.5 ms on desktop in this scene.

What changed (no rule or number; the month computes exactly what it did):

- **A month's steps run several to a frame** (`sim/steps.budgeted`, `MONTH_FRAME_SECONDS` = 16 ms
  in `ui/app.py`): steps follow one another until 16 ms have gone on them, then the frame is drawn.
  The autosave's parts after the month go the same way. A frame can take one step longer than
  the 16 ms, which is why the longest frame grew a little on desktop.
- **The track near each cell is kept** between months for the scenery (`scenery._track_near`),
  found again only when the railway changes (`Constructions.railway_version`), and sprites are
  looked up by name in a dictionary instead of a list.
- **The painting workers hear of new buildings as kinds**, as saves keep them
  (`Constructions.__getstate__`), instead of every building pickled as an object.
- **Putting the scenery up again is split into three steps** (`PixelWorld.buildings_changed_steps`),
  so the ground, the scenery and the trees' shading can land on different frames.

Tried and dropped: enum hashing showed as the biggest cost in Pyodide's profiler, but a lookup by
an (enum, enum) key takes 39 ns with the enum's own hash and 37 ns with the object's, so it is the
profiler's overhead, as on desktop.

What is left of the 131 ms: about 75 ms of work (the trade about half of it, the scenery and the
ground pictures most of the rest) and the drawing of 4 frames. Bringing it lower would mean making
the trade itself cheaper, or putting the scenery up only around the cells that changed instead of
over the whole map.

## Lag again on 0.5.83 (18 November, the same scene)

The complaint: the browser build lagged badly again, frames as well as months. Measured with the
same scene as the section above (new game, Poland seed 1, 4 stations, 3 trains, zoom 3 over the
first station, a month forced every 90 frames), before = staging 0.5.83, after = it with the fixes.

What was found:

- **Every frame drew the chain round each lot for sale link by link** (`ui/land_offers.py`). At
  zoom 3 a lot's sides run thousands of points across the screen, so each frame drew about
  10,000 links of two ellipses each, nearly all off screen, and worked every lot's corners out
  again. This was most of a frame's work in this scene, and had been since land could be bought,
  which is why the section above measured 13 ms frames in the browser.
- **The start map is bigger**: the honeycomb of seven lots makes a new game's map 48,697 hexes
  instead of 31,872, and everything a month does over the whole map (the scenery, the ground's
  biomes and fields, the towns' growth) grew with it, about 1.5 times.
- The new ledger of what buildings take in costs about 2.6 ms a month on desktop, half of it
  shifting each kind's last three months along with `np.roll`.

| | before (0.5.83) | after |
|---|---|---|
| Browser: a frame without a month (mean / 95th percentile / worst) | 22.0 / 22.6 / 32.9 ms | 1.8 / 2.0 / 12.2 ms |
| Browser: trains frozen per month | 228 ms (5–6 frames) | 125 ms (6 frames) |
| Browser: trains frozen, month with an autosave | 329 ms | 201 ms |
| Desktop: a frame without a month (mean) | 11.1 ms | 0.9 ms |
| Desktop: trains frozen per month | 113 ms | 66 ms |
| Desktop: the ledger's month (`Ledger.record`) | 2.6 ms | 1.7 ms |

What changed (no rule, number or look):

- Only the chain links on screen are drawn (`land_offers.seen_part` clips each side to the
  screen), each where it lay before on its whole side; the chain's two colours are worked out
  once a frame; each lot's corners on the map and on the place's picture are kept until a lot is
  bought or the place's picture changes.
- The ledger shifts its last three months in place, and finds where cells stand in its books
  without merging them when it has them all already.

What is left: a month's work is about 110 ms in the browser on the bigger map (the trade, the
scenery and the ground's fields put up again over the whole map each month the most of it).
Putting the scenery and fields up only round the cells that changed would be the next step.

## Moving the map, hovering and starting a game in the browser (19 November, 0.5.110)

The complaint: moving the map should stay smooth, hovering the map should cost nothing, and a new
game should come up sooner. Measured in headless Chrome as above (1440×860 at 2× density, a 60 Hz
display) on a new game (Poland, seed 1, 1850, the seven-lot start of 48,697 hexes), before =
staging 0.5.100, after = it with this work, run one after the other. Each zoom level pans 200
frames across land not yet seen from a different spot of the map; the zoom cycle steps through
every level and back every 6 frames; hovering moves the mouse across the map every frame with no
tool, the station tool and the track tool (track from the middle of the view), at zoom 4 and 1.
Numbers are the Python time of a frame (mean / 95th percentile / worst, ms); every frame of every
scene kept up with the display (16.7 ms apart) before and after on this machine, so what changed is
how much room a slower one has.

What was found:

- **Zoomed out, every building was a call of its own.** At zoom 0–2 the map drew each of the
  ~4,400 buildings' dots with its own imgui call, each looking up its kind's colour: 4–6 ms of
  every frame, watching or panning. They are now two triangles each in one buffer on the graphics
  card, handed over when the buildings change, and drawn into a texture in one call when the view
  moves (`ui/pixel/dots_shader.py`). Where the window smooths edges (the browser's canvas does,
  though GL reports no samples), a pixel on a dot's edge takes as much of its colour as the dot
  covers of it, so the dots look as before (checked on screenshots at zoom 0, 1 and 2: no pixel
  differs by more than the smoothing's rounding, at most 37 levels on single edge pixels).
- **The chains, padlocks and price tags of all sixteen lots for sale** were drawn every frame,
  those far off screen too. A lot is now skipped unless its outline, with room for its tag, reaches
  the screen.
- **The station tool found its reach's border anew in every cell it crossed**: hundreds of
  lookups per cell. The same shape of cells moved by an even number of rows has the same border
  moved with it, so each shape's border is found once (`build_menu.reach_border`).
- **The ground shader asked GL where its uniforms are on every pass** and looked up each GL
  function in the page's module on every call; both are now kept after the first time.
- **The game's first frame compiled the shaders and drew the world map's picture** (130 ms in
  the browser). Where the game loads on the frames, the shaders are made behind the waiting screen,
  and the world picture comes from a table of 256 colours (8 ms instead of 46, identical bytes).
- **The track preview was drawn on the same frame its route was planned on.** It now waits for
  the next frame, with the last preview showing until then.
- **A new game spent almost all of its wait growing towns** (9.6 of 10.5 s after pressing Start,
  in the setup worker). In the browser the setup worker now starts growing the game the main menu
  would start as soon as its settings have stayed the same for a second (`PageSetup.grow_ahead`,
  `Shell._grow_ahead`); Start with those settings takes up what it has grown, other settings start
  it anew, and loading a save or the tutorial stops it. Nothing changes on the desktop.

| Browser, new game | before | after |
|---|---|---|
| Start pressed after 15 s in the menu, to the first game frame | 10.02 s | 0.98 s |
| Start pressed after 5 s in the menu | 10.72 s | 7.01 s |
| Game started from the address (`?args=`), to the first game frame | 12.08 s | 11.94 s |
| The game's first frame | 127.5 ms | 33.7 ms |
| Longest waiting-screen frame | 117–138 ms | 121–142 ms |

| Browser frames (GUI ms: mean / 95th percentile / worst) | before | after |
|---|---|---|
| watching at zoom 4 | 2.8 / 4.2 / 9.5 | 2.0 / 3.5 / 6.7 |
| panning at zoom 0 | 6.1 / 8.6 / 12.7 | 3.6 / 4.7 / 17.9 |
| panning at zoom 1 | 5.9 / 7.9 / 14.5 | 3.8 / 4.3 / 12.9 |
| panning at zoom 2 | 4.5 / 5.2 / 6.0 | 2.9 / 3.8 / 4.0 |
| panning at zoom 3 | 3.8 / 8.7 / 11.2 | 2.9 / 5.8 / 7.8 |
| panning at zoom 4 | 3.6 / 4.2 / 11.2 | 2.9 / 3.5 / 10.6 |
| panning at zoom 5 | 3.9 / 5.2 / 9.8 | 3.2 / 4.4 / 8.0 |
| panning at zoom 6 | 3.5 / 4.5 / 9.9 | 2.6 / 3.8 / 8.6 |
| panning at zoom 7 | 3.8 / 4.8 / 68.8 | 3.2 / 4.3 / 11.2 |
| zooming in and out | 3.7 / 8.0 / 10.9 | 2.1 / 4.9 / 9.9 |
| mouse resting, zoom 0, no tool | 7.8 / 8.3 / 9.8 | 3.4 / 3.9 / 4.2 |
| mouse resting, zoom 2, no tool | 4.3 / 5.2 / 5.5 | 1.3 / 1.9 / 2.0 |
| mouse resting, zoom 4, no tool | 2.2 / 3.0 / 7.2 | 1.8 / 2.4 / 7.8 |
| mouse moving, zoom 4, no tool | 1.8 / 2.4 / 11.5 | 2.3 / 3.1 / 14.9 |
| mouse resting, zoom 4, station tool | 3.5 / 4.2 / 8.5 | 3.0 / 3.7 / 9.9 |
| mouse moving, zoom 4, station tool | 3.3 / 8.6 / 12.4 | 2.8 / 4.8 / 8.0 |
| mouse resting, zoom 4, track tool | 1.7 / 1.9 / 15.1 | 1.8 / 2.8 / 12.6 |
| mouse moving, zoom 4, track tool | 3.2 / 13.2 / 25.2 | 3.1 / 11.4 / 33.3 |
| mouse resting, zoom 1, no tool | 7.1 / 8.2 / 11.6 | 3.0 / 3.8 / 4.1 |
| mouse moving, zoom 1, no tool | 7.2 / 7.8 / 8.5 | 3.3 / 4.0 / 5.2 |
| mouse resting, zoom 1, station tool | 7.4 / 7.8 / 10.7 | 4.1 / 4.8 / 5.6 |
| mouse moving, zoom 1, station tool | 8.8 / 10.6 / 12.5 | 3.8 / 4.7 / 5.6 |
| mouse resting, zoom 1, track tool | 7.1 / 7.5 / 12.9 | 3.6 / 4.2 / 10.7 |
| mouse moving, zoom 1, track tool | 7.0 / 7.6 / 13.6 | 3.5 / 4.2 / 10.4 |

The worst frames are single ones: chunks arriving from the painting workers and blown up on the
graphics card (panning, zooming), the first frame at a zoom level, and the track preview.

What is left:

- **The track preview** still takes one frame of 25–35 ms in the browser each time a new route
  shows (drawing the planned track's pixel art, `PixelWorld.planned_track`); it no longer shares
  that frame with planning the route. Drawing it in a painting worker, as chunks are, would take it
  off the frames, but the workers would have to hear of the planned pieces and paint a picture of
  any size; not done.
- **Growing the towns** is as dear as before (about 9.6 s in the setup worker, 5.7 s on desktop); a
  player who presses Start within a few seconds of the menu showing waits most of it. A game started
  straight from the address grows only after the page has loaded, as before.
- **The waiting screen** still has six or seven frames of 60–140 ms while the map is prepared
  after the towns arrive (the pixel world's parts, the scenery, the ground pictures), and the first
  game frame about 34 ms.

## Big games: the "slow" saves (26 November, 0.5.244 to 0.5.263)

The player's saves "slow" and "slow2" (Poland, 1955–1957, 64 stations, 139 trains, maps of
301 × 209 and 371 × 209 cells) ran at 20–25 frames a second in the browser. Measured in headless
Chrome as above; when those saves stopped loading (ports came after them), on a game built the same
size on the current code (64 stations at the biggest towns, track to the two nearest, 140 trains).
Every change was checked to leave play exactly as it was: trains, loads and cash hashed over 1,500
frames, three months, and laying track, building and clearing stations, on both saves; every route
between every two stations; every train part's place on its track; every signal block.

| Browser, GUI ms a frame (mean) | before | after |
|---|---|---|
| "slow", the save's own view (zoom 4) | 39.3 | 16.4 |
| "slow", resting at zoom 4 / zoom 1 | 35.5 / 50.5 | 12.0 / 12.7 |
| "slow", zooming in and out | 44.5 | 17.3 |
| Laying track, building or clearing a station: the call and the next frame | ~540 + 220 ms | ~70 + 86 ms |
| Big game, its own view / zoom 1 / zooming (worst) | 13.0 / 11.0 / 9.5 (62) | 9.0 / 7.0 / 5.9 (37) |
| Big game, first frame zoomed out to zoom 0 | 96 ms | 46 ms |
| Desktop: replanning every schedule after track changes ("slow2") | 2.8 s | 0.3 s |
| Desktop: every route between every two stations ("slow2") | 7.8 s | 0.7 s |

What it was, and what changed (no rule or number):

- **Trains were put on their track again every frame**, each part of each train: which track of
  double or quadruple track it rides was worked out from the track around it each time. What a
  piece of track branches off and how its tracks lie are kept until track changes
  (`TrackArt._branch_lanes`, `_side_profile`), a standing train's places are kept while it stands
  (`TrainLayer._shift`), its lanes are found once a frame, and a path is laid out in one go.
- **The tiles window worked out the eras' texts and the tiles' factors every frame**; they are kept.
- **Zoomed-out track** is worked out in one go for all pieces on screen and redrawn from the lines
  last worked out while the view stands still.
- **Laying track, a station or clearing one** worked out every piece of track's art again (10,000
  pieces) to grade the ground; now only the pieces within three cells of a change
  (`Constructions.railway_changed_since`). The grid's track steps and exits, which never change,
  are kept, and with them signal blocks are cut in 3.5 ms.
- **Finding ways** started a search through the whole network for every way asked for; a search
  from a station, or a point, is now taken up again for every way asked for from it since track
  last changed (`track_network._Search`), visiting cells in the same order, so the ways are the
  same. Route lengths for planning refuelling need no speeds.
- **Fine train pictures** are drawn by the painting workers in the browser, so no frame stops for
  the 50–110 ms a look's model takes to make.
- Smaller: loading a train weighs it again only when a car changed; passengers getting off reweigh
  only the need they met; the worth in the top bar sums the track's cost once a change; smoke puffs
  are coloured all at once.

Left, and why: signals are checked for every train every frame (about 1.4 ms of a big game's 2 ms
of simulation on desktop), which only an event-driven rewrite would cut; zoomed-out track on the
graphics card would save about 1 ms, as trains, not track, are what costs zoomed out; running the
simulation in a worker would take every frame's simulation and the months off the frames but needs
the whole interface to work on a copy of the game.

## The month of a big game (26 November, 0.5.290 to 0.5.291)

The "big" save built on current code (Poland, 64 stations, 140 trains). Trains stand still while a
month is worked out, a few frames of it at a time; the freezes are what the player sees.

| Month freezes in the browser (ms, months 2 to 8) | Total |
| --- | --- |
| Before: 234, 167, 245, 166, 247, 256, 291 (and 232, 162, 200, 162, 216, 225, 261) | 1458–1606 |
| After the enums and stations' goods: 213, 141, 198, 142, 207, 223, 268 (and 218, 134, 191, 137, 196, 216, 243) | 1335–1392 |

A month on desktop went from about 52 ms to about 47 ms.

- **Goods, needs and kinds of buildings were hashed by name in Python** (`Enum.__hash__`), 87,000
  times a month, and in the browser that was the most costly function of all. They are now hashed
  by identity in C (`railroads.sim.keyed.Keyed`), which keeps every dict and set the same.
- **Stations gathered goods one after another**, in the order they were built; now each cell's
  spare goods go to the nearest station first, stations as near sharing them by their room, for
  every good in one `distribute`. A rule change: the principles say so.
- **What stations hold and have room for** was looked up good by good and station by station; it
  is now a table of goods by stations.
- `ratio` no longer broadcasts arrays of the same shape, a station's demand subtracts what
  buildings got locally in one sum, and `distribute` drops the pairs with nothing to give or want
  once, skipping every distance left empty.

Left, and why: the rest of a month is a long tail of parts that each take a few percent, the trade
between buildings (`_sell_nearby` and `distribute`), placing the scenery again (about 10% in the
browser) and the growth's books. Only letting trains run while the month is worked out would take
the freezes away.
