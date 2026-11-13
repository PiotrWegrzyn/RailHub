# Running the game in a browser through Pyodide (2026-11-11)

Proof of concept: the game runs unchanged in a browser. `web/index.html` loads Pyodide, numpy,
Pillow, imgui-bundle and the game's own wheel, then starts `railroads.ui.app.main`. The map, the
pixel art with its raised mountains, the menus and the minimap all show. The only changes to the
game are stand-ins for its background threads: web workers paint the map's chunks and grow a new
game's towns, and loading a save and the autosave writer run on the frames.

## How to run it

```bash
uv run python web/build_site.py
python -m http.server -d build/site 8000
```

`build_site.py` puts the page, the game's wheel, and Pyodide with numpy, Pillow, micropip and
imgui-bundle into `build/site` (23.9 MB, about 16 MB gzipped). The page loads nothing from other
servers, so any static host can serve that folder as it is.

Open http://localhost:8000/ for the main menu. Only the Poland map can be picked there, since a
random place needs data the site doesn't carry. `?args=--poland --seed 7` passes a command line
and starts that game at once. Add `&measure` to run `web/measure.py` instead. It prints `LOAD` and `MEASURE` lines
to the console. The same script runs on desktop with `uv run python web/measure.py --poland --seed 7`.

## Hosting

The game is played at https://piotrwegrzyn.github.io/RailHub/, served by GitHub Pages from the
`main` branch of https://github.com/PiotrWegrzyn/RailHub. That repo holds only the built site, with
`.nojekyll` and a short README. Loaded there, the page downloads 16.6 MB (Pages gzips the wasm to
3.6 MB), has Pyodide and its packages ready after 3 s and shows the main menu soon after. A new game takes
about 22 s more on a fast machine.

To publish a new build, run `build_site.py`, replace the contents of a RailHub checkout with
`build/site` (keep its README), commit and push. Pushes this size need a larger
`http.postBuffer` (150 MB here), or GitHub answers with HTTP 400.

The game's wheel carries its Python source, so anyone playing can read the code.

## What it took

- **No Python 3.13 port was needed.** Pyodide 314.0.0 runs Python 3.14.0, and imgui-bundle
  publishes a matching `imgui_bundle-1.93.0-cp314-cp314-pyemscripten_2026_0_wasm32.whl` on PyPI.
  That is the exact version the project pins. The cp313 wheel on imgui-bundle.pages.dev is an older
  1.92.705 build kept only for old pages. Lazy annotations and `type` aliases work as they do on
  desktop.
- **imgui-bundle does the browser loop itself.** In Pyodide it replaces `immapp.run`, so the call
  returns at once and frames run on `requestAnimationFrame`. `App.gui` is unchanged.
- **The game's wheel already carries its data.** `uv build` packs `maps/*.npz`, the sprite sheets
  and the good icons, about 0.66 MB. micropip installs it with `deps=False`, because Pyodide ships
  numpy 2.4.3 and Pillow 12.2.0 while `pyproject.toml` asks for numpy ≥ 2.5.4 and Pillow ≥ 13. The
  older versions work.
- **No threads.** `railroads.ui.workers.worker` tests whether a thread can start. If not, it
  hands out a `FrameWorker`, which runs a job only when the caller asks. The map's chunks are
  painted by up to four web workers instead (`web/painter.js`, `railroads.ui.pixel.web_painter`),
  each a Pyodide of its own holding a copy of the pixel world (`railroads.ui.pixel.copies`): it
  is pickled whole while a game loads and kept up with what changes in it. `?painters=N` on the
  page sets how many start, and 0 paints a chunk a frame once the map and mouse stand still, as
  happens until a worker is ready. Raised chunks are never drawn again finely in the browser, as
  the desktop does, so its mountains look as they always have there. A new game's towns grow in a
  web worker of their own (`railroads.ui.web_setup`), started with the page and stopped once the
  game is handed over, compressed, then uncompressed on the frames a few megabytes at a time.
  Loading a save runs on the frames after the waiting screen first shows, a step each. An autosave is pickled on the frame after the month turns, then compressed and written
  a step at a time, at most `WRITING_SECONDS` (4 ms) of each frame, over the next two seconds or
  so, the frames kept at full speed meanwhile. A month that ends before that is done is autosaved
  once it is, and an open saves window lists the new autosave as soon as it is written.
- **Saves.** `index.html` mounts Emscripten's IndexedDB file system over the saves folder and
  copies it to IndexedDB every 2 s, so named saves and autosaves survive a reload. They stay in
  that browser; there is no way yet to download a save or load one from a file.

## Measurements

Apple M3 Pro. Desktop is the normal build. Browser is headless Chrome 156 on WebGL2 through ANGLE
on Metal. Both use a 1280×800 window at 2× pixel density. The run is the Poland map, seed 7, at zoom
5/7: 120 frames resting on the middle of the map, then 600 frames panning 6×3 screen pixels per
frame across the highest ground. Idle throttling is off.

### Download and start

| | |
|---|---|
| Over the wire, cold cache | ≈ 16.4 MB: Pyodide core 6.2 MB (wasm 3.4, stdlib 2.5, js 0.3), imgui-bundle 5.5, numpy 2.9, Pillow 1.0, game 0.66, micropip 0.1 |
| Unpacked | ≈ 24 MB |
| Pyodide ready | 1.4 s |
| numpy, Pillow, imgui-bundle and game installed | 2.5 s (local machine, fast network) |
| Python start to first frame | 23.6 s in the browser vs 10.9 s on desktop (2.2×) |
| Page open to first frame | ≈ 26.7 s |

Almost all of the start is `seed_buildings` growing towns for 1000 years before play. It runs in
one go on the browser's main thread, so the page freezes and cannot even update its loading text.

### Chunk drawing (`PixelWorld.render`, resolution 2)

| Chunk | Desktop | Browser | Ratio |
|---|---|---|---|
| Flat | 32 ms | 59 ms | 1.8× |
| Raised, quick | 82 ms | 138 ms | 1.7× |
| Raised, fine | 267 ms | 377 ms | 1.4× |

The heavy numpy work loses little in WebAssembly. Pure-Python work, like town growth, loses more.

### Panning

| | Desktop | Browser |
|---|---|---|
| Average frame rate | 82 fps | 24 fps |
| Median frame | 12.8 ms | 16.7 ms (60 Hz cap) |
| 95th percentile frame | 65 ms | 238 ms |
| Worst frame | 103 ms | 467 ms |

Most browser frames still take 16.7 ms. The average drops because of stalls. On desktop the fine
mountain redraw runs on a worker thread. In the browser it runs on the main thread, with 30 stalls
of about 400 ms during the pan, about 11 s of the 24.5 s it took. Without those stalls the browser
would pan at about 45 fps, still held back by the 140 ms quick draws of raised chunks.

## What blocks or slows a full web version

1. **New random games need data outside the package.** `start_new_game` reads
   `data/elevation.sqlite` (79 MB), the Natural Earth files in `data/raw` (52 MB) and
   `data/places.npz` (3.3 MB). Only the baked Poland map works in the browser today. Options: bake
   a set of start maps, cut the elevation data down to the start regions and load it lazily, or
   build maps on a server.
2. **A 24-second frozen start.** Town growth before play blocks the page. Options: save the grown
   start with the map, grow in slices that yield to the browser (async), or grow in a Web Worker.
   Done: the towns grow in a web worker. The page no longer freezes (its longest frame while a
   Poland game starts is about 140 ms), but the game shows after about 29.5 s instead of 23 s:
   growth runs about a fifth slower in the worker, and handing the game over adds about a second.
3. **No threads, so redraws stall.** Fine mountain redraws cost about 400 ms each on the main
   thread. Pyodide has no threads, and a Web Worker would need its own Pyodide and its own copy of
   the world. Cheaper options: skip the fine redraw in the browser, split it into strips across
   frames, or move map drawing to GPU shaders. Shaders would also remove the 60–140 ms quick
   chunk draws, the next biggest cost.
   Done: the ground shader runs in the browser on WebGL 2 (`railroads.ui.pixel.webgl`), the
   browser skips the fine redraw, chunks draw about twice as fast as they did, and web workers
   with their own Pyodide and copy of the world paint them, so no frame waits on one. In headless
   Chrome on the 1440×860 window at 2×, filling the screen after a jump onto unseen land at zoom
   4/7 went from 2.2 s to 0.3 s over flat land and from 5.1 s to 0.7 s over mountains, with frames
   at 16.7 ms throughout instead of 100–270 ms. Each worker takes about 250–350 MB.
4. **Pure Python is about 2× slower.** Monthly growth and train simulation will feel this too.
   They were not measured in play.
5. **Download.** About 16 MB before anything shows. That is acceptable for a game but needs a real
   loading screen. imgui-bundle alone is 5.5 MB.

Saving, autosaving and loading were checked in headless Chrome, across a reload. Not checked:
the ImPlot reports window, keyboard shortcuts, long play sessions,
memory use, Firefox and Safari.
