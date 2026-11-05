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
`src/railroads/mapgen` turns map data into a map. `src/railroads/ui` draws the game with
Dear ImGui (imgui-bundle).

## Maps

A map is a grid of cells, squares or hexagons, where every cell has the same properties:
terrain (land, sea, lake, river), vegetation (barren, grass, forest), temperature (cold, normal,
hot) and height in metres. Each property is one numpy layer, saved together with the map's scale,
cell shape, bounds and data sources in one `.npz` file.

`railroads.mapgen` makes a map by laying a grid over a lon/lat box and filling in each property
with a strategy, in order, so a strategy can use the layers before it. Swap a strategy to change
where a property comes from. The UI then draws every cell as a biome picked from its properties
(`railroads.ui.biomes`): water is always water, hot and barren is desert, hot and forest is jungle.

## Map data

`src/railroads/maps/poland-square.npz` and `poland-hex.npz` are baked by
`tools/build_poland_map.py` from public-domain data:

- Elevation: NOAA ETOPO1, 1 arc-minute (via NOAA CoastWatch ERDDAP, dataset `etopo180`).
- Sea, lakes and rivers: [Natural Earth](https://www.naturalearthdata.com/) 1:10m. Rivers running
  less than 200 km on the map are left out. Without the ocean file, the sea falls back to low
  ground (0 m or below) connected to the map edge.
- Temperature: a climate rule from latitude and height.
- Forests: random placeholder patches until a land-cover source is added.

To rebuild them:

```bash
mkdir -p data/raw && cd data/raw
curl -o etopo_poland.csv "https://coastwatch.pfeg.noaa.gov/erddap/griddap/etopo180.csv?altitude%5B(48.9):1:(55.0)%5D%5B(13.6):1:(24.4)%5D"
for f in ne_10m_ocean ne_10m_lakes ne_10m_lakes_europe ne_10m_rivers_lake_centerlines ne_10m_rivers_europe; do
  curl -O "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/$f.geojson"
done
cd ../.. && uv run python tools/build_poland_map.py
```
