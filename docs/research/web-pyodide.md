# Running the game in a browser through Pyodide (2026-11-11)

Proof of concept: the game runs unchanged in a browser. `web/index.html` loads Pyodide, numpy,
Pillow, imgui-bundle and the game's own wheel, then starts `railroads.ui.app.main`. The map, the
pixel art with its raised mountains, the menus and the minimap all show. The only change to the game is a fallback
for the background thread that redraws mountain chunks.

## How to run it

```bash
uv build --wheel -o web/wheels
python -m http.server -d web 8000
```

Open http://localhost:8000/. It plays the baked map of Poland; `?args=--poland --seed 7` passes a
command line. Add `&measure` to run `web/measure.py` instead. It prints `LOAD` and `MEASURE` lines
to the console. The same script runs on desktop with `uv run python web/measure.py --poland --seed 7`.

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
- **No threads.** `PixelLayer` now tests whether a thread can start. If not, it hands the fine
  redraw of raised chunks to a `FrameWorker`, which runs one job on each frame that draws no new
  chunk.

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
3. **No threads, so redraws stall.** Fine mountain redraws cost about 400 ms each on the main
   thread. Pyodide has no threads, and a Web Worker would need its own Pyodide and its own copy of
   the world. Cheaper options: skip the fine redraw in the browser, split it into strips across
   frames, or move map drawing to GPU shaders. Shaders would also remove the 60–140 ms quick
   chunk draws, the next biggest cost.
4. **Pure Python is about 2× slower.** Monthly growth and train simulation will feel this too.
   They were not measured in play.
5. **Download.** About 16 MB before anything shows. That is acceptable for a game but needs a real
   loading screen. imgui-bundle alone is 5.5 MB.

Not checked: saving and loading, the ImPlot reports window, keyboard shortcuts, long play sessions,
memory use, Firefox and Safari.
