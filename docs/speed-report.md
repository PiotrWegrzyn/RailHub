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
