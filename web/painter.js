// A web worker painting chunks of the map for the page (railroads.ui.pixel.web_painter). It runs a
// Python of its own, with numpy, Pillow and the game, and holds a copy of the map's pixel world
// (railroads.ui.pixel.copies.WorldCopy). Messages from the page:
//     {kind: "world", world, data}   take up a new world, pickled whole
//     {kind: "update", world, data}  take up what changed in it
//     {kind: "paint", world, job, chunk: [x, y, resolution, fine relief, over ground]}
// and back to it: {kind: "ready"} once Python has started, {kind: "painted", world, job, pixels,
// shape, ms} with the pixels' buffer handed over rather than copied, {kind: "failed", world, job,
// error}, and {kind: "dead", error} if the worker can't go on. Pyodide runs only in module workers.
// A worker started to grow a new game's towns instead (railroads.ui.web_setup) is sent
//     {kind: "setup", data}               the game's settings, pickled
// and answers {kind: "progress", text} as the towns grow, then {kind: "grown", game, size} with
// the game pickled and compressed, or {kind: "failed", error}.
import { loadPyodide } from "./pyodide/pyodide.mjs";

const GAME_WHEEL = "wheels/railroads-0.1.0-py3-none-any.whl";

let pyodide = null;
let copy = null;
let dead = false;
const early = [];

self.onmessage = event => {
    if (copy === null) early.push(event.data);
    else handle(event.data);
};

function die(error) {
    dead = true;
    postMessage({ kind: "dead", error: String(error?.message || error) });
    close();
}

function handle(message) {
    if (dead) return;
    const { kind, world } = message;
    if (kind === "paint") {
        paint(message);
        return;
    }
    if (kind === "setup") {
        setup(message);
        return;
    }
    const data = pyodide.toPy(message.data);
    try {
        if (kind === "world") copy.start(world, data);
        else copy.update(world, data);
    } catch (error) {
        die(error);
    } finally {
        data.destroy();
    }
}

function paint({ world, job, chunk }) {
    const started = performance.now();
    let pixels = null;
    try {
        pixels = copy.paint(world, chunk);
        if (pixels === undefined) {
            postMessage({ kind: "failed", world, job, error: "a chunk of a world it no longer has" });
            return;
        }
        const buffer = pixels.getBuffer("u8");
        const data = buffer.data.slice();
        const shape = buffer.shape;
        buffer.release();
        const ms = performance.now() - started;
        postMessage({ kind: "painted", world, job, pixels: data.buffer, shape, ms }, [data.buffer]);
    } catch (error) {
        postMessage({ kind: "failed", world, job, error: String(error?.message || error) });
    } finally {
        pixels?.destroy();
    }
}

function setup({ data }) {
    const settings = pyodide.toPy(data);
    let grown = null;
    try {
        const report = text => postMessage({ kind: "progress", text });
        grown = pyodide.pyimport("railroads.ui.web_setup").grow(settings, report);
        // Copied by its address in Python's memory, as Pyodide's own copies go wrong past 2 GiB.
        const game = pyodide._module.HEAPU8.slice(grown.address, grown.address + grown.length);
        postMessage({ kind: "grown", game: game.buffer, size: grown.size }, [game.buffer]);
    } catch (error) {
        postMessage({ kind: "failed", error: String(error?.message || error) });
    } finally {
        settings.destroy();
        grown?.destroy();
    }
}

async function start() {
    pyodide = await loadPyodide({ indexURL: new URL("pyodide/", location.href).href });
    await pyodide.loadPackage(["numpy", "pillow", new URL(GAME_WHEEL, location.href).href]);
    copy = pyodide.pyimport("railroads.ui.pixel.copies").WorldCopy();
    postMessage({ kind: "ready" });
    for (const message of early.splice(0)) handle(message);
}

start().catch(die);
