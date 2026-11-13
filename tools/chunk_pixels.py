"""Saves the pixel art of a fixed set of chunks, or checks it against a saved set, to prove that a
change to the drawing code left every pixel as it was. It grows the seeded map of Poland with a few
years of stand-in play, lays single, double and electric track between its towns and into the
mountains, then draws chunks of flat land, coast, lakes and rivers, towns, fields, woods, hills and
mountains, track, bridges, tunnels and stations, the map's edge and beyond it: each at both
resolutions, plain and over the ground the graphics card draws, and with fine raised ground where
the land rises. Track being planned, the lift of points on raised ground and the ground showing at
pixels are saved too. Last, every cell round a flat chunk and a hilly one that will take them is
filled with fields and groves, crowded far more than any real farmland, and both are drawn again.

Run it on the code before a change, then on the code after; the check names every difference:
    uv run python tools/chunk_pixels.py save build/chunks.npz --game build/chunks-game.save
    uv run python tools/chunk_pixels.py check build/chunks.npz --game build/chunks-game.save
Building the game takes a while, so `--game` keeps it in a save file to start from next time. Both
files go under build/, which git leaves out.
"""

import argparse
import contextlib
import io
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from railroads.newgame import MapChoice, Settings, allowed_start_year, make_game
from railroads.saves import SavedGame, Saves
from railroads.sim.clock import HISTORY_START
from railroads.sim.constructions import Building, BuildingKind, Track, TrackWidth
from railroads.sim.grids import Cell
from railroads.sim.growth import MONTHS_BEFORE_PLAY
from railroads.sim.ladders import Points
from railroads.sim.tilemap import Relief, Terrain
from railroads.sim.track_rules import TrackBed
from railroads.ui.biomes import Biome
from railroads.ui.map_view import LOOKS
from railroads.ui.pixel.ground import CROP_ROWS, GROVES
from railroads.ui.pixel.projection import CHUNK
from railroads.ui.pixel.sheet import FINE, SpriteSheet
from railroads.ui.pixel.world import PixelWorld

SEED = 7
STAND_IN_YEARS = 6
CASH = 1e12


def build_game(cache: Path | None) -> SavedGame:
    """The seeded game of Poland after a few years of stand-in play, read from `cache` if it is
    there and written to it if not."""
    saves = None if cache is None else Saves(cache.parent)
    if saves is not None and saves.exists(cache.stem):
        return saves.load(cache.stem)
    settings = Settings(
        seed=SEED,
        map=MapChoice.POLAND,
        start_year=allowed_start_year(HISTORY_START + MONTHS_BEFORE_PLAY // 12),
        stand_in_years=STAND_IN_YEARS,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        saved = make_game(settings, report=lambda _: None)
    if saves is not None:
        saves.save(cache.stem, saved)
    return saved


def towns(saved: SavedGame, count: int) -> list[Cell]:
    """The most crowded cells, each at least 8 cells from the others."""
    people = saved.game.growth.points(Points.POPULATION)
    picked: list[Cell] = []
    for flat in np.argsort(people, axis=None, kind="stable")[::-1].tolist():
        row, col = divmod(flat, people.shape[1])
        if all(abs(row - r) + abs(col - c) >= 8 for r, c in picked):
            picked.append((row, col))
        if len(picked) == count:
            break
    return picked


def highest(saved: SavedGame) -> Cell:
    heights = saved.game.tile_map.height_m
    row, col = np.unravel_index(np.argmax(heights), heights.shape)
    return int(row), int(col)


def lay_track(saved: SavedGame) -> list[Cell]:
    """Builds stations in the biggest towns and lays track of every kind between them, from the
    town nearest the highest ground up into the mountains, and across mountains until some runs
    through a tunnel; returns the cells track runs through."""
    game = saved.game
    game.finances.cash = CASH
    big = towns(saved, 6)
    peak = highest(saved)
    near_peak = min(big, key=lambda cell: abs(cell[0] - peak[0]) + abs(cell[1] - peak[1]))
    for cell in big:
        game.build_station(cell)
    lines = [
        (big[0], big[1], TrackWidth.SINGLE, False),
        (big[1], big[2], TrackWidth.DOUBLE, False),
        (big[2], big[3], TrackWidth.SINGLE, True),
        (big[3], big[4], TrackWidth.DOUBLE, True),
        (near_peak, peak, TrackWidth.SINGLE, False),
    ]
    for start, end, width, electric in lines:
        game.lay_track(start, end, width, electric)
    tile_map = game.tile_map
    land = tile_map.terrain == Terrain.LAND
    mountains = land & (tile_map.relief == Relief.MOUNTAINS)
    for row, col in np.argwhere(mountains).tolist():
        start, end = (row, col - 3), (row, col + 3)
        if not (tile_map.grid.contains(*start) and tile_map.grid.contains(*end)):
            continue
        if mountains[start] or mountains[end] or not (land[start] and land[end]):
            continue
        game.lay_track(start, end)
        if any(piece.bed == TrackBed.TUNNEL for piece in game.constructions.track()):
            break
    return sorted({piece.cell for piece in game.constructions.track()})


def world_of(saved: SavedGame) -> PixelWorld:
    """The pixel map of the game, as the map view makes it."""
    tile_map = saved.game.tile_map
    look = LOOKS[tile_map.grid.kind]
    world = PixelWorld(
        tile_map,
        saved.game.constructions,
        SpriteSheet.load(),
        look.angle_deg,
        look.cell_units,
        fine=SpriteSheet.load_fine(),
        station_levels=saved.game.railway.levels,
    )
    world.satisfaction = saved.game.growth.satisfaction().copy()
    return world


def chunk_of(world: PixelWorld, cell: Cell) -> tuple[int, int]:
    px, py = world.projection.to_pixels(*world.tile_map.grid.center(*cell))
    return int(px) // CHUNK, int(py) // CHUNK


def pick_chunks(world: PixelWorld, saved: SavedGame, track: list[Cell]) -> dict[str, tuple]:
    """The chunks to draw, each by what it shows."""
    tile_map = world.tile_map
    biomes = world.ground.biomes
    across, down = world.projection.chunks
    chunks: dict[str, tuple[int, int]] = {}
    for index, town in enumerate(towns(saved, 3)):
        chunks[f"town {index}"] = chunk_of(world, town)
    peak = highest(saved)
    cx, cy = chunk_of(world, peak)
    for dx, dy in ((0, 0), (0, -1), (1, 0), (-1, 1), (0, 1)):
        chunks[f"mountains {dx} {dy}"] = (cx + dx, cy + dy)
    land = tile_map.terrain == Terrain.LAND
    kinds = {
        "coast": (tile_map.terrain == Terrain.SEA) & np.roll(land, 1, axis=1),
        "lake": biomes == Biome.LAKE,
        "river": biomes == Biome.RIVER,
        "forest": (biomes == Biome.FOREST) & np.roll(biomes == Biome.FOREST, 2, axis=1),
        "hills": land & (tile_map.relief == Relief.HILLS),
    }
    for name, where in kinds.items():
        cells = np.argwhere(where)
        for index, pick in enumerate((len(cells) // 3, 2 * len(cells) // 3)):
            if len(cells):
                chunks[f"{name} {index}"] = chunk_of(world, tuple(cells[pick].tolist()))
    flat = [
        (x, y)
        for y in range(down)
        for x in range(across)
        if not world.rises(x, y) and (x, y) not in chunks.values()
    ]
    for index, chunk in enumerate(flat[len(flat) // 4 :: max(1, len(flat) // 6)][:5]):
        chunks[f"flat {index}"] = chunk
    for index, cell in enumerate(track[:: max(1, len(track) // 8)]):
        chunks[f"track {index}"] = chunk_of(world, cell)
    beds: dict[TrackBed, Cell] = {}
    for piece in world.constructions.track():
        beds.setdefault(piece.bed, piece.cell)
    for bed, cell in beds.items():
        chunks[f"track {bed.name.lower()}"] = chunk_of(world, cell)
    for index, cell in enumerate(world.constructions.stations()[:4]):
        chunks[f"station {index}"] = chunk_of(world, cell)
    chunks |= {
        "edge left": (0, down // 2),
        "edge top": (across // 2, 0),
        "edge corner": (across - 1, down - 1),
        "off before": (-1, -1),
        "off after": (across, down // 2),
    }
    return chunks


def drawings(world: PixelWorld, chunks: dict[str, tuple]) -> Iterator[tuple[str, np.ndarray]]:
    """Every way each chunk is drawn, by name: plain and over ground at both resolutions where
    nothing rises, and plain at both resolutions and with fine raised ground where it does."""
    seen: set[tuple[int, int]] = set()
    for name, chunk in chunks.items():
        if chunk in seen:
            continue
        seen.add(chunk)
        label = f"{name} {chunk[0]} {chunk[1]}"
        for k in (1, FINE):
            yield f"{label} x{k}", world.render(*chunk, k)
        if world.rises(*chunk):
            yield f"{label} fine x{FINE}", world.render(*chunk, FINE, fine_relief=True)
        else:
            for k in (1, FINE):
                yield f"{label} over x{k}", world.render(*chunk, k, over_ground=True)


def extras(world: PixelWorld, saved: SavedGame) -> Iterator[tuple[str, np.ndarray]]:
    """What else the drawing code works out: track being planned, flat and lifted, the lift of
    plane points round the highest ground and the ground showing at pixels there."""
    grid = world.tile_map.grid
    peak = highest(saved)
    town = towns(saved, 1)[0]
    for name, cell in (("peak", peak), ("town", town)):
        steps = grid.track_steps()
        ends = (0, len(steps) // 2)
        pieces = [Track(cell, ends, TrackWidth.SINGLE, TrackBed.GROUND, False)]
        px, py = world.projection.to_pixels(*grid.center(*cell))
        left, top = int(px) - 40, int(py) - 30
        for k in (1, FINE):
            yield f"planned {name} x{k}", world.planned_track(pieces, left, top, 80, 60, k)
    rng = np.random.default_rng(SEED)
    cx, cy = grid.center(*peak)
    x = cx + rng.uniform(-12, 12, 4000)
    y = cy + rng.uniform(-12, 12, 4000)
    yield "lift", world.elevation.lift(x, y)
    px, py = world.projection.to_pixels(cx, cy)
    shown = [
        world.elevation.ground_at(float(px) + dx, float(py) + dy)
        for dx, dy in rng.integers(-200, 200, (60, 2))
    ]
    yield "ground at", np.array(shown)


def crowd_fields(world: PixelWorld, chunks: list[tuple[int, int]]) -> None:
    """Builds fields, crops or a grove in every cell round these chunks that will take one, a
    different kind from cell to cell, so their patches overlap more than in any real farmland."""
    grid = world.tile_map.grid
    kinds = [BuildingKind.FARM, *CROP_ROWS, *GROVES]
    px, py = world.projection.to_pixels(*grid.center(*np.indices(grid.shape)))
    built = []
    for cx, cy in chunks:
        near = (px > cx * CHUNK - 40) & (px < (cx + 1) * CHUNK + 40)
        near &= (py > cy * CHUNK - 40) & (py < (cy + 1) * CHUNK + 40)
        for row, col in np.argwhere(near).tolist():
            turn = (row + col) % len(kinds)
            for kind in kinds[turn:] + kinds[:turn]:
                building = Building((row, col), kind)
                if world.constructions.why_not(building) is None:
                    world.constructions.build(building)
                    built.append((row, col))
                    break
    world.buildings_changed(built)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("action", choices=("save", "check"))
    parser.add_argument("pixels", type=Path, help="the .npz file to save to or check against")
    parser.add_argument("--game", type=Path, help="a save file to keep the built game in")
    args = parser.parse_args()
    began = time.perf_counter()
    saved = build_game(args.game)
    track = lay_track(saved)
    world = world_of(saved)
    chunks = pick_chunks(world, saved, track)
    print(f"game ready in {time.perf_counter() - began:.1f} s; {len(chunks)} chunks", flush=True)
    began = time.perf_counter()
    drawn = dict(drawings(world, chunks)) | dict(extras(world, saved))
    crowded = {f"crowded {name}": chunks[name] for name in ("flat 0", "hills 0") if name in chunks}
    crowd_fields(world, list(crowded.values()))
    drawn |= dict(drawings(world, crowded))
    print(f"{len(drawn)} drawn in {time.perf_counter() - began:.1f} s", flush=True)
    if args.action == "save":
        np.savez_compressed(args.pixels, **drawn)
        return
    with np.load(args.pixels) as before:
        names = set(before.files) | set(drawn)
        different = []
        for name in sorted(names):
            if name not in before.files or name not in drawn:
                different.append(f"{name}: only {'now' if name in drawn else 'before'}")
            elif not np.array_equal(before[name], drawn[name]):
                old, new = before[name], drawn[name]
                if old.shape != new.shape:
                    different.append(f"{name}: shape {old.shape} became {new.shape}")
                else:
                    changed = np.abs(old.astype(np.float64) - new.astype(np.float64))
                    different.append(
                        f"{name}: {int((changed > 0).sum())} values differ, by up to "
                        f"{float(changed.max()):g}"
                    )
    for line in different:
        print(line)
    print(f"{len(names) - len(different)} of {len(names)} the same", flush=True)
    sys.exit(1 if different else 0)


if __name__ == "__main__":
    main()
