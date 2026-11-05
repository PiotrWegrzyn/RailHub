# Railroads

A mechanics-first railroad tycoon game.

Run the game:

```bash
uv run railroads
```

Run the tests and linter:

```bash
uv run pytest
uv run ruff check
```

`src/railroads/sim` is the game logic and must never import the UI (ruff enforces this).
`src/railroads/ui` draws the game with Dear ImGui (imgui-bundle).

## Map data

`src/railroads/sim/maps/poland.npz` is baked by `tools/build_poland_map.py` from public-domain data:

- Elevation: NOAA ETOPO1, 1 arc-minute (via NOAA CoastWatch ERDDAP, dataset `etopo180`).
- Lakes and rivers: [Natural Earth](https://www.naturalearthdata.com/) 1:10m.

To rebuild it:

```bash
mkdir -p data/raw && cd data/raw
curl -o etopo_poland.csv "https://coastwatch.pfeg.noaa.gov/erddap/griddap/etopo180.csv?altitude%5B(48.9):1:(55.0)%5D%5B(13.6):1:(24.4)%5D"
for f in ne_10m_lakes ne_10m_lakes_europe ne_10m_rivers_lake_centerlines ne_10m_rivers_europe; do
  curl -O "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/$f.geojson"
done
cd ../.. && uv run python tools/build_poland_map.py
```
