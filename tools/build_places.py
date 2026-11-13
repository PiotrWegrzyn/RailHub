"""Build the worldwide file of place names (data/places.npz) that tools/build_world_files.py cuts
into the squares stations are named from.

Reads GeoNames' cities500 (every place with more than 500 people, CC BY 4.0) straight from
data/raw/geonames/cities500.zip (see README.md) and keeps each populated place's name,
latitude, longitude and population. Sections of places (city districts), and abandoned,
destroyed and historical places, are left out.

Run with: uv run python tools/build_places.py
"""

import io
import zipfile

import numpy as np

from railroads.newgame import DATA_DIR

SOURCE = DATA_DIR / "raw" / "geonames" / "cities500.zip"
PLACES_FILE = DATA_DIR / "places.npz"
LEFT_OUT = {"PPLX", "PPLQ", "PPLW", "PPLH", "PPLCH"}


def main() -> None:
    names: list[str] = []
    lats: list[float] = []
    lons: list[float] = []
    people: list[int] = []
    with zipfile.ZipFile(SOURCE) as archive, archive.open("cities500.txt") as raw:
        for line in io.TextIOWrapper(raw, encoding="utf-8"):
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 15 or fields[6] != "P" or fields[7] in LEFT_OUT:
                continue
            names.append(fields[1])
            lats.append(float(fields[4]))
            lons.append(float(fields[5]))
            people.append(int(fields[14] or 0))
    encoded = [name.encode("utf-8") for name in names]
    ends = np.cumsum([len(name) for name in encoded])
    PLACES_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        PLACES_FILE,
        names=np.frombuffer(b"".join(encoded), dtype=np.uint8),
        ends=ends.astype(np.int64),
        lat=np.array(lats, dtype=np.float32),
        lon=np.array(lons, dtype=np.float32),
        population=np.array(people, dtype=np.int64),
    )
    print(f"{len(names)} places written to {PLACES_FILE}")


if __name__ == "__main__":
    main()
