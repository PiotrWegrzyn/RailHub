"""Cut the HydroRIVERS reaches that drain at least FLOOR_KM2 into world/streams/, square by
square, for the minor rivers of every map (railroads.mapgen.streams), and list the squares in
world/index.npz. Each reach goes to the square its first point lies in, its points thinned to
about STEP_DEG apart, which is plenty for 2 km cells.

Needs HydroRIVERS_v10_shp.zip unpacked into data/raw/hydrorivers/ (download command in
README.md). Rewrites world/streams/ only; tools/build_world_files.py runs it too.

Run with: uv run python tools/build_world_streams.py [--data DIR]
"""

import argparse
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

from railroads.mapgen import world_files
from railroads.mapgen.streams import StreamLines, write_streams
from railroads.mapgen.world_files import (
    HIGHEST_LAT,
    LOWEST_LAT,
    WorldIndex,
    square_bounds,
    square_of,
    streams_file,
)
from railroads.newgame import DATA_DIR

FLOOR_KM2 = 300.0
STEP_DEG = 0.01
SHAPEFILE = Path("raw/hydrorivers/HydroRIVERS_v10_shp/HydroRIVERS_v10")
# Where UPLAND_SKM, the land a reach drains in km², lies in each .dbf record, and how wide it is.
UPLAND_FIELD = slice(56, 66)


def uplands(dbf: Path) -> np.ndarray:
    header = np.fromfile(dbf, dtype=np.uint8, count=32)
    count = int(header[4:8].view("<u4")[0])
    header_size, record_size = (int(v) for v in header[8:12].view("<u2"))
    records = np.memmap(dbf, dtype=np.uint8, mode="r", offset=header_size,
                        shape=(count, record_size))  # fmt: skip
    field = np.ascontiguousarray(records[:, UPLAND_FIELD])
    return field.view(f"S{field.shape[1]}").ravel().astype(float)


def reach_lines(shp: Path, shx: Path, wanted: np.ndarray) -> list[np.ndarray]:
    index = np.fromfile(shx, dtype=">i4", offset=100).reshape(-1, 2)
    offsets = index[wanted, 0].astype(np.int64) * 2
    data = np.memmap(shp, dtype=np.uint8, mode="r")
    lines = []
    for offset in offsets.tolist():
        content = offset + 8
        parts, points = np.frombuffer(data[content + 36 : content + 44], dtype="<i4").tolist()
        start = content + 44 + 4 * parts
        xy = np.frombuffer(data[start : start + 16 * points], dtype="<f8").reshape(-1, 2)
        lines.append(thinned(xy))
    return lines


def thinned(line: np.ndarray) -> np.ndarray:
    """The line's first and last points and one about every STEP_DEG along it between."""
    along = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(line, axis=0).T))])
    steps = np.floor(along / STEP_DEG)
    keep = np.flatnonzero(np.diff(steps, prepend=-1) > 0)
    if keep[-1] != len(line) - 1:
        keep = np.append(keep, len(line) - 1)
    return line[keep]


def write_stream_squares(data: Path, out: Path) -> set[tuple[int, int]]:
    """Writes out/streams/ afresh from the HydroRIVERS files in `data` and says which squares
    it wrote."""
    base = data / SHAPEFILE
    upland = uplands(base.with_suffix(".dbf"))
    wanted = np.flatnonzero(upland >= FLOOR_KM2)
    print(f"{len(wanted)} of {len(upland)} reaches drain at least {FLOOR_KM2:.0f} km²")
    lines = reach_lines(base.with_suffix(".shp"), base.with_suffix(".shx"), wanted)
    by_square: dict[tuple[int, int], list[int]] = defaultdict(list)
    for number, line in enumerate(lines):
        lon, lat = line[0]
        if LOWEST_LAT <= lat < HIGHEST_LAT:
            by_square[square_of(lon, lat)].append(number)
    shutil.rmtree(out / "streams", ignore_errors=True)
    (out / "streams").mkdir(parents=True)
    for (col, row), numbers in by_square.items():
        kept = [lines[n] for n in numbers]
        square = StreamLines(
            np.concatenate(kept),
            np.cumsum([len(line) for line in kept], dtype=np.int64),
            upland[wanted[numbers]],
        )
        bounds = square_bounds(col, row)
        middle = (bounds.west + bounds.east) / 2, (bounds.south + bounds.north) / 2
        (out / streams_file(col, row)).write_bytes(write_streams(square, *middle))
    size = sum(path.stat().st_size for path in (out / "streams").iterdir()) / 1e6
    print(f"{len(by_square)} squares of streams, {size:.1f} MB")
    return set(by_square)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", type=Path, default=DATA_DIR, help="the source data folder")
    out = world_files.WORLD_DIR
    squares = write_stream_squares(parser.parse_args().data, out)
    index_path = out / world_files.INDEX_FILE
    index = WorldIndex.read(index_path.read_bytes())
    index = WorldIndex(index.tiles, index.shape_squares, index.place_squares,
                       frozenset(squares))  # fmt: skip
    index_path.write_bytes(index.write())


if __name__ == "__main__":
    main()
