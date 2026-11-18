// A web worker painting chunks of the map for the page (railroads.ui.pixel.web_painter). It runs a
// Python of its own, with numpy, Pillow and the game, and holds a copy of the map's pixel world
// (railroads.ui.pixel.copies.WorldCopy). Messages from the page:
//     {kind: "world", world, data}   take up a new world, pickled whole
//     {kind: "update", world, data}  take up what changed in it
//     {kind: "paint", world, job, chunk: [x, y, resolution, fine relief, over ground]}
// and back to it: {kind: "ready", version} once Python has started, with the game's version,
// {kind: "painted", world, job, pixels, shape, ms} with the pixels' buffer handed over rather than
// copied, {kind: "failed", world, job, error}, and {kind: "dead", error} if the worker can't go on.
// Pyodide runs only in module workers.
// A worker started to grow a new game's towns instead (railroads.ui.web_setup) is sent
//     {kind: "setup", data}               the game's settings and how to picture it, pickled
// and answers {kind: "progress", text}, {kind: "advance", done}, the share of the towns'
// history grown, and {kind: "picture", pixels}, a picture of the map pickled
// (railroads.ui.history_pictures) with its buffer handed over, as the towns grow, then
// {kind: "grown", game, size, version} with the game pickled and compressed and the game's version
// that grew it, or {kind: "failed", error}.
import { loadPyodide } from "./pyodide/pyodide.mjs";

const GAME_WHEEL = "wheels/railroads-__VERSION__-py3-none-any.whl";

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
        const advance = done => postMessage({ kind: "advance", done });
        const see = (address, length) => {
            const pixels = pyodide._module.HEAPU8.slice(address, address + length);
            postMessage({ kind: "picture", pixels: pixels.buffer }, [pixels.buffer]);
        };
        grown = pyodide.pyimport("railroads.ui.web_setup").grow(settings, report, advance, see);
        // Copied by its address in Python's memory, as Pyodide's own copies go wrong past 2 GiB.
        const game = pyodide._module.HEAPU8.slice(grown.address, grown.address + grown.length);
        const done = { kind: "grown", game: game.buffer, size: grown.size, version: grown.version };
        postMessage(done, [game.buffer]);
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
    postMessage({ kind: "ready", version: pyodide.pyimport("railroads.version").VERSION });
    for (const message of early.splice(0)) handle(message);
}

start().catch(die);
