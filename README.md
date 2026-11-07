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

replays the start location and buildings of seed 42, and

```bash
uv run railroads --poland
```

plays on the baked map of Poland. Towns grow for 300 years before play starts;

```bash
uv run railroads --years 100
```

grows them for 100 instead.

In the corner of the view, a minimap (M to show or hide) shows where the map lies in the world
and the whole map from above with the part on screen outlined; click or drag on it to look there.

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

The UI draws the map from those properties alone, always from the same direction: squares as
diamonds with north at the top left, hexagons with north up. The three furthest zoom levels show
the map as one smooth picture; the three closest show it as pixel art, at 1, 2 and 3 screen pixels
per art pixel.

- `railroads.ui.biomes`: every cell is a biome picked from terrain, relief, temperature and
  vegetation. Water is always water, mountains are grey rock or snow, hot and barren is desert,
  hot and forest is jungle.
- `railroads.ui.relief`: hills are drawn lighter, and wherever the land steps up a level the
  higher cell gets a bevel lit from the top-left of the screen.
- `railroads.ui.ground`: puts both together into the smooth ground picture.
- `railroads.ui.buildings`: where each building stands in its share of the cell, drawn as a
  coloured dot when zoomed out.
- `railroads.ui.pixel`: the pixel art. The ground gets a few shades per biome, dithered pixel
  by pixel, with sandy shores, rippled water and striped fields under farms, and every slope lit
  from the top left by the land's real heights. Hills, mountains, trees, rocks and buildings stand
  on it as sprites, back to front. Hills and mountain peaks are small height fields lit from the
  left, drawn in grey in the sheet, and take the shades of the ground under them, so the same hill
  fits grass, forest or desert; the tallest peaks keep white snow on top. The map is drawn in
  256-pixel chunks as the view reaches them and kept as textures.
- `railroads.ui.minimap`: the world with the map marked on it, from a quarter-degree picture of
  the world's land baked into `src/railroads/maps/world-land.npz`, next to the whole map.

The sprites are drawn in code by `railroads.ui.pixel.art` and saved to
`src/railroads/ui/pixel/sprites.png`, with `sprites.json` saying where each one sits and which pixel
stands on the ground. Repaint the PNG by hand to change a sprite; to start over from the code, run:

```bash
uv run python tools/build_sprites.py
```

Relief (flat, hills, mountains) comes from how steeply each cell climbs to its neighbours:
hills from a 3% grade, mountains from 8%. `railroads.sim.track_rules` says what that means for
play: no track on mountains except in a tunnel, and on hills track costs twice as much and trains
run at 60% speed.

## Building

`railroads.sim.constructions` holds what the player can build and the rules for where it fits.
Every cell has room for 6 units:

- Track is laid one piece per cell, from one of the cell's track points to another: the middles
  of its edges and its corners, 12 on a hexagon and 8 on a square. Through an edge, track runs
  into the neighbour; through a corner, it runs along the edge between the two cells beside it
  and into the cell beyond. A piece turns at most 60°.
- Single track takes 1 unit and double track 2, so a cell with a double track still fits four
  houses.
- Track lies on the ground, on a bridge over a river or lake (x5 cost), or in a tunnel through
  mountains (x8 cost).
- A station fills its whole cell.
- Every building takes 1 unit. Only mines and town buildings (houses, tenements, flats and
  skyscrapers) can be built in mountains, and ports need water next to them.

## Growth

`railroads.sim.growth` grows the whole map by itself, like a cellular automaton, one month at a
time: towns, farms, orchards, mines, lumber mills, ports and several kinds of factory. Every
building gives points of one kind, and each cell adds up the points of every kind in its
neighbourhood (the cell and the six around it):

| Points | Buildings |
|---|---|
| Population | house 1, tenement (kamienica) 2, block of flats 5 |
| Food | farm 1, orchard 1 |
| Ore | mine 1 |
| Timber | lumber mill 1 |
| Trade | port 1 |
| Industry | factory 1, big factory 3, mega factory 9, food plant 2, steel mill 3, shipyard 3 |
| Corporate | skyscraper 1 |

Each kind of building grows on its own ladder. Every month each cell may take a turn on each
ladder, with the ladder's base chance plus its chance per nearby point, but none until the
neighbourhood holds the points the ladder needs. The land multiplies the chance by its
vegetation, relief, temperature and the best water next to the cell. The chances below are for
content buildings; unsatisfied ones grow at half of them and thriving ones at one and a half:

| Ladder | Chance per month | Needs nearby | Land |
|---|---|---|---|
| Houses | 0.0000085 + 0.0002125 per population | | forest x0.3, mountains x0.05, barren x0.5, hills x0.8, cold x0.6, hot x0.8 |
| Farms | 0.000004 + 0.00003 per population + 0.00002 per food | | flat grass; hills x0.5, barren x0.1, forest x0.02, cold x0.2, hot x0.6; river or lake x5, sea x2; none in mountains |
| Orchards | 0.000002 + 0.000004 per population + 0.00002 per food | | hills; flat x0.5, hot x1.2, cold x0.05; river or lake x3, sea x2 |
| Mines | 0.000008 + 0.0001 per ore | | mountains; hills x0.4, flat x0.01; beside mountains x3 |
| Lumber mills | 0.00002 + 0.00005 per population | | forest with open land next to it; river x1.5 |
| Ports | 0.000004 + 0.00004 per population | 3 population | next to the sea, or a lake x0.3 |
| Factories | 0.000002 per population + 0.00002 per industry | | flat; hills x0.5, forest x0.2; river x1.5 |
| Food plants | 0.00004 per food + 0.000002 per population | 1 food, 6 population | as factories |
| Steel mills | 0.0002 per ore + 0.00002 per industry | 1 ore, 2 population | as factories |
| Shipyards | 0.0001 per trade + 0.00002 per industry | 1 trade | on the water |

Farms and orchards may crowd together; lumber mills keep to two per neighbourhood, ports and the
special factories to one, and mines to five. On its turn a ladder builds its
first kind, unless it can merge: six houses nearby let three of them make way for a tenement and
six tenements let three make way for a block of flats; three factories nearby make a big factory
and three big factories a mega factory. On a town or factory turn, a neighbourhood with 100
population and 10 industry points puts up a skyscraper instead: it pulls down three blocks of
flats and three points of factories and gives a corporate point.

### Goods and satisfaction

`railroads.sim.economy` shares goods out every month (once a year before play, to seed quickly).
Buildings make and need goods, and the land feeds people by itself: every water cell makes 2
food, every flat or forest cell 1.

| Building | Makes | Needs |
|---|---|---|
| House | | 1 food |
| Tenement | | 2 food, 1 goods |
| Block of flats | | 5 food, 2 goods |
| Farm | 6 food | |
| Orchard | 3 food | |
| Mine | 2 coal | |
| Lumber mill | 2 lumber | |
| Factory, big, mega | 3, 9, 27 goods | 2, 6, 18 lumber |
| Food plant | 3 goods | 3 food |
| Steel mill | 2 steel | 3 coal |
| Shipyard | 3 goods | 2 steel, 1 lumber |
| Port | 2 goods | |
| Skyscraper | | 3 goods |

Goods go no further than a neighbourhood: each cell serves itself first, then shares what is left
with the cells next to it, unsatisfied cells first, then content ones. Factories make only as much
as their inputs allow. A building that gets all it needs and gives away all it makes is
satisfied; at least half, content; less, unsatisfied. Each ladder's chances in a cell are scaled
by how its kinds of building fare around it: x0.5 unsatisfied, x1 content, x1.5 satisfied, and x1
where there are none yet. The ground under content and satisfied buildings has a warm glow, and
the status bar names the hovered cell's satisfaction.

Farms with more than 10 population points around them are pulled down. Stations still go up
after 100 years of growth, one per 3,000 land cells beside the bigger groups of buildings, but do
nothing until there are trains.

Chances are kept up to date as buildings come and go, and cells are drawn by thinning, so a
month costs about as much as what grows in it.

A new map grows for three centuries before play starts (`--years` changes that);
`railroads.mapgen.seeding` runs it. In play the calendar (`railroads.sim.clock`) runs from
January 1850 to January 2050, a month every two and a half minutes: a hundred hours at normal
speed. The map keeps growing each month.

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

and, to rebake the Poland maps and the world picture for the minimap:

```bash
uv run python tools/build_poland_map.py
uv run python tools/build_world_map.py
```
