# Railroads

A mechanics-first railroad tycoon game.

Run the game:

```bash
uv run railroads
```

Every game starts at a random, mostly-land spot in the world, on a 600 x 600 km map. The limits
for that spot (longitude, latitude, how much land) are `START_REGION` in
`src/railroads/newgame.py`. Options:

```bash
uv run railroads --seed 42
```

replays the start location of seed 42, and

```bash
uv run railroads --poland
```

plays on the baked map of Poland.

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
where a property comes from.

The UI draws the map from those properties alone:

- `railroads.ui.biomes`: every cell is a biome picked from terrain, relief, temperature and
  vegetation. Water is always water, mountains are grey rock or snow, hot and barren is desert,
  hot and forest is jungle.
- `railroads.ui.relief`: hills are drawn lighter, and wherever the land steps up a level the
  higher cell gets a bevel lit from the top-left of the screen.
- `railroads.ui.ground`: puts both together into the ground picture, repainted when the view
  turns so the light keeps coming from the same side.

Relief (flat, hills, mountains) comes from how steeply each cell climbs to its neighbours:
hills from a 3% grade, mountains from 8%. `railroads.sim.track_rules` says what that means for
play: no track on mountains, and on hills track costs twice as much and trains run at 60% speed.

## Map data

Maps are built from these sources:

- Elevation for new games: Mapzen terrain tiles on [AWS Open Data](https://registry.opendata.aws/terrain-tiles/),
  made from SRTM, GMTED2010, ETOPO1, EU-DEM and others
  ([attribution](https://github.com/tilezen/joerd/blob/master/docs/attribution.md)). They are
  downloaded once into a local database, so starting a game needs no internet.
- Elevation for Poland: NOAA ETOPO1, 1 arc-minute (via NOAA CoastWatch ERDDAP, dataset `etopo180`).
- Sea, lakes, rivers and deserts: [Natural Earth](https://www.naturalearthdata.com/) 1:10m,
  public domain. Rivers running less than 200 km on the map are left out.
- Temperature: a climate rule from latitude and height that works anywhere in the world.
- Forests: random placeholder patches until a land-cover source is added.

Set up the data once:

```bash
mkdir -p data/raw && cd data/raw
curl -o etopo_poland.csv "https://coastwatch.pfeg.noaa.gov/erddap/griddap/etopo180.csv?altitude%5B(48.9):1:(55.0)%5D%5B(13.6):1:(24.4)%5D"
for f in ne_10m_ocean ne_10m_lakes ne_10m_lakes_europe ne_10m_rivers_lake_centerlines ne_10m_rivers_europe ne_10m_geography_regions_polys; do
  curl -O "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/$f.geojson"
done
cd ../..
```

Then build the elevation database for new games (about 70 MB download, resumes if interrupted):

```bash
uv run python tools/build_elevation_db.py
```

and, to rebake the Poland maps:

```bash
uv run python tools/build_poland_map.py
```
