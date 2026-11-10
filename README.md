# Railroads

A mechanics-first railroad tycoon game.

Run the game:

```bash
uv run railroads
```

Every game starts at a random, mostly-land spot in the world, on a 537 x 537 km map. The limits
for that spot (longitude, latitude, how much land) are `START_REGION` in
`src/railroads/newgame.py`. Options:

```bash
uv run railroads --seed 42
```

replays the start location and buildings of seed 42, and

```bash
uv run railroads --poland
```

plays on the baked map of Poland. Towns grow for 1,000 years before play starts, from the year 850,
so play begins in 1850;

```bash
uv run railroads --years 1100
```

grows them for 1,100 years instead, so play begins in 1950 with a century more growth behind it
and the clock showing 1950; play always ends in 2050. A shorter history begins play earlier.

```bash
uv run railroads --poland --seed 1 --simulate
```

lets a stand-in player play all 200 years first, printing the map every ten years, and then opens
the map in 2050 to look at the result (`--simulate 100` stops in 1950 instead, and the stand-in
goes on playing if the clock runs on). It builds 5 stations when play starts and 2 more each
decade, each on the free cell whose reach holds the most people no station serves yet, and each
raised one level so it reaches three hexes out. It lays no track and runs no trains; of the cells
a station reaches, a third thrive, half are content (half their needs met) and the rest go
unsatisfied, and they grow and glow as if trains served them. Only the hexagon map is grown.

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

`railroads.sim.game` holds one game: the buildings, the growing towns, the railway, the money and
the yearly history. It runs each month and every player action: building a station, laying
track, bulldozing and moving a station. Each action checks the price, pays it and returns what
to tell the player, so the map view only passes on clicks and shows the answer.

To check that a refactor changed nothing the game works out, run the fingerprint before and after
it; every line must match:

```bash
uv run python tools/fingerprint.py
```

It grows a seeded map of Poland and builds and plays on it through the map view's own clicks.
Then it hashes the buildings, money, trains, goods, history, messages, rendered chunks and
planned routes.

## Maps

A map is a grid of hexagon cells (the grids can also be squares, which the game no longer uses),
where every cell has the same properties:
terrain (land, sea, lake, river), vegetation (barren, grass, forest), temperature (cold, normal,
hot) and height in metres. Each property is one numpy layer, saved together with the map's scale,
cell shape, bounds and data sources in one `.npz` file.

`railroads.mapgen` makes a map by laying a grid over a lon/lat box and filling in each property
with a strategy, in order, so a strategy can use the layers before it. Swap a strategy to change
where a property comes from.

The UI draws the map from those properties alone, always from the same direction, with north
up. The three furthest zoom levels show
the map as one smooth picture; the four closest show it as pixel art, at 1, 2, 3 and 4 screen
pixels per art pixel.

- `railroads.ui.biomes`: every cell is a biome picked from terrain, relief, temperature and
  vegetation. Water is always water, mountains are grey rock or snow, hot and barren is desert,
  hot and forest is jungle.
- `railroads.ui.relief`: hills are drawn lighter, and wherever the land steps up a level the
  higher cell gets a bevel lit from the top-left of the screen.
- `railroads.ui.ground`: puts both together into the smooth ground picture.
- `railroads.ui.buildings`: where each building stands in its share of the cell, drawn as a
  coloured dot when zoomed out.
- `railroads.ui.pixel`: the pixel art. The ground gets a few shades per biome, dithered pixel
  by pixel, with sandy shores, rippled water, striped fields under farms and rows of fruit trees under orchards, and every slope lit
  from the top left by the land's real heights. Hills, mountains, trees, rocks and buildings stand
  on it as sprites, back to front. Hills and mountain peaks are small height fields lit from the
  left, drawn in grey in the sheet, and take the shades of the ground under them, so the same hill
  fits grass, forest or desert; the tallest peaks keep white snow on top. The map is drawn in
  256-pixel chunks as the view reaches them and kept as textures. Wherever an art pixel covers
  an even number of screen pixels (2 and 4 times, and every zoom on a Retina screen), chunks are
  drawn twice as finely. The ground is blown up as before, but track, trees and buildings are
  drawn at 2× from the finer sprite sheet, so their outlines and windows stay thin instead of
  growing blocky. A chunk takes about a third longer to draw and four times the memory; nothing
  changes from frame to frame.
- `railroads.ui.minimap`: the world with the map marked on it, from a quarter-degree picture of
  the world's land baked into `src/railroads/maps/world-land.npz`, next to the whole map.

The sprites are drawn in code by `railroads.ui.pixel.art` and saved to
`src/railroads/ui/pixel/sprites.png`, with `sprites.json` saying where each one sits and which pixel
stands on the ground. `sprites@2x.png` and `sprites@2x.json` hold the same sprites drawn twice as
finely, in the same order. The same drawing code makes them with every coordinate doubled and lines
kept one pixel wide, and hills and mountains are blown up. Repaint a PNG by hand to change a sprite;
to start over from the code, run:

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
  houses. Either can be electric, with wires overhead that electric engines need: `Track.electric`
  says so for a piece, and `Constructions.electrified(cell)` for a cell (every piece in it, or
  the piece between two given track points).
- Track lies on the ground, on a bridge over a river or lake (x5 cost), or in a tunnel through
  mountains (x8 cost).
- A station fills its whole cell.
- Every building takes 1 unit. Only mines and town buildings (houses, tenements, flats and
  skyscrapers) can be built in mountains, and ports need water next to them.

## Growth

`railroads.sim.growth` grows the whole map by itself, by the ladders of `railroads.sim.ladders`,
like a cellular automaton, one month at a time: towns, farms, orchards, mines, lumber mills, ports and several kinds of factory. Its
other jobs live beside it:
- `railroads.sim.room`: making room for track and stations, buildings spilling into nearby cells,
  rehousing people and putting the player's buildings up again.
- `railroads.sim.ownership`: what the player owns and is paid.
- `railroads.sim.station_trade`: settling goods with the stations and rewarding growth near them.
- `railroads.sim.forests`: the forest changes.

Every building gives points of one kind (`railroads.sim.points`), and each cell adds up the points
of every kind in its neighbourhood (the cell and the six around it):

| Points | Buildings |
|---|---|
| Population | house 1, tenement (kamienica) 6, block of flats 18 |
| Food | farm, orchard, date palms, banana plantation, orange farm, sugar field, potato farm, cattle ranch, flax field, hop garden, beekeeper, incense grove 1 |
| Ore | mine, clay pit, limestone quarry, sand pit 1 |
| Timber | lumber mill 1 |
| Trade | port 1 |
| Industry | factory 1, big factory 3, mega factory 9, cannery 2, steel mill 3, shipyard 3, machine works, plastics works, appliance factory 3, power plant 5, other works 2 |
| Corporate | tavern 1, department store 3, hotel 5, skyscraper 10 |
| Worship | church 5, basilica 20, cathedral 50, St. Paul cathedral 2137 |

Each kind of building grows on its own ladder. Every month each cell may take a turn on each
ladder, with the ladder's base chance, plus its chance for every doubling of each kind of points
nearby (log2 of one more than the points), plus the worth of the served buildings in the cell
itself, but none until the neighbourhood holds the points the ladder needs. Because points nearby
count by their doublings, growth from neighbours alone stays steady instead of compounding. A lone
house draws no more houses: population nearby only counts once there are 2 points of it, so
houses dot the countryside on their own and a town starts only where two land close together. The
buildings in a cell that stations serve count their points up to 10 times over, by the share of
what they need and make that they got and gave: a tenement that gets all its food but no salt
still counts nearly 10 times over. Each point is worth 2 doublings, so only what the player
supplies compounds.
The land multiplies the chance by its vegetation, relief, temperature and the best water next to
the cell:

| Ladder | Chance per month | Needs nearby | Land |
|---|---|---|---|
| Houses | 0.0000012 + 0.00011 per doubling of population, from 2 population nearby | | forest x0.3, mountains x0.05, barren x0.5, hills x0.8, cold x0.6, hot x0.8; river x1.5, lake or sea x1.3 |
| Farms | 0.000000112 + 0.00000672 per doubling of population + 0.00000028 per doubling of food | fewer than 8 population | flat grass x10 while empty; hills x0.25, barren x0.1, forest x0.02, cold x0.2, hot x0.6; river or lake x5, sea x2; none in mountains |
| Orchards | 0.000000216 + 0.000000108 per doubling of population + 0.00000216 per doubling of food | | hills x1.5; flat x0.4, hot x1.2, cold x0.05; river or lake x3, sea x2 |
| Date palms | 0.000000168 + 0.00000168 per doubling of population + 0.00000126 per doubling of food | | hot desert; river or lake x5, sea x2 |
| Banana plantations | as date palms | | hot jungle; river or lake x2, sea x1.5 |
| Orange farms | as date palms | | hot grassland, not desert or jungle; hills x0.7; river or lake x2, sea x1.5 |
| Sugar fields | 0.0000000168 + 0.000000042 per doubling of population + 0.000000168 per doubling of food | | open flat land, not cold; mild x0.6; river x3 |
| Mines | 0.000001 + 0.0000014 per doubling of ore | | mountains; hills x0.4, flat x0.01; beside mountains x3 |
| Lumber mills | 0.000016 + 0.000001 per doubling of population | | forest with at least 4 forest cells in its neighbourhood and flat open land next to it; river x1.5 |
| Ports | 0.0000008 + 0.000002 per doubling of population | 4 population | next to the sea, or a lake x0.3 |
| Factories | 0.00000004 + 0.000000006 per doubling of population + 0.000065 of industry + 0.0000008 of timber, each only from 5 points nearby | | flat; hills x0.5, forest x0.2; river x1.5 |
| Canneries | 0.000000015 + 0.00000026 per doubling of food + 0.00000076 of industry | 1 food | as factories |
| Steel mills | 0.00000008 + 0.00001 per doubling of ore + 0.0000006 of industry | | as factories |
| Shipyards | 0.000005 + 0.000002 per doubling of trade + 0.0000004 of industry | 1 trade | on the water |
| Potato farms | 0.000000017 + 0.000001 per doubling of population + 0.00000004 of food | fewer than 8 population | cool open plains; hills x0.4; river or lake x2 |
| Cattle ranches | 0.0000000033 + 0.00000018 per doubling of population + 0.000000009 of food | fewer than 8 population | grassland and hills; hot x0.3 |
| Flax fields | 0.000000006 + 0.0000003 per doubling of population + 0.00000002 of food | fewer than 8 population | cool wet land; river x3, lake x2, away from water x0.3 |
| Hop gardens | 0.0000000012 + 0.00000012 per doubling of population | fewer than 8 population | open plains and hills; cold x0.3, hot x0.2 |
| Clay pits | 0.0000001 + 0.0000001 per doubling of population + 0.0000002 of ore | | river lowlands; lake x0.5 of a river, away from water x0.03 |
| Limestone quarries | 0.00000015 + 0.00000015 per doubling of ore | | hills; mountains x0.4, flat x0.02 |
| Sand pits | 0.000000075 + 0.00000015 per doubling of ore | | by the sea (x4), rivers and lakes (x2), desert x3 |
| Taverns | 0.00000001 + 0.000004 per doubling of population | 3 population | as houses, one to a neighbourhood |
| Department stores | 0.00005 per doubling of population, from 1880 | 60 population | as houses, 6 hexes apart, one per 10,000 land cells |
| Hotels | as department stores, from 1900 | 60 population | as houses, 5 hexes apart, one per 8,000 land cells |

Farms and orchards may crowd together; lumber mills and steel mills keep to two per neighbourhood, canneries to
six, mines to eight, and ports and shipyards to one. Factories, mills and mines open mostly by
chance, with only a weak pull towards their own kind, so they spread out across the land while
houses gather into towns. On its turn a ladder builds its
first kind, unless it can merge: 16 houses nearby let six of them make way for a tenement and
five tenements let three make way for a block of flats; three factories nearby make a big factory
and three big factories a mega factory. On a town turn, a neighbourhood with 150 population
points and three blocks of flats puts up a skyscraper instead, one turn in 200: a business centre
worth 10 corporate points, so it sends and wants 200 passengers and 40 pieces of mail a month. It
pulls down the three blocks of flats, and their people move into 10 houses in the nearest cells
with room for them, up to 6 hexes away. On Poland with the stand-in player (below) that gives
about one by 1950 and 20 to 40 by 2050.

A building that finds no room in its cell spills over: it goes up in the nearest cell, at most 3
cells away, with room for it, land its ladder grows on and nothing the ladder's own caps forbid
there, picked among cells equally near at random by how well their land suits it, so a field
spills onto the plain beside it rather than into the forest. A cell already at its ladder's cap, such as a
church too near another, sends nothing elsewhere.

A hex has room for 6 units. Most buildings take 1; factories, canneries and mills take 3; farms,
orchards and every other field fill the whole hex and colour all of it.

### Forests

Forests come and go as the map grows. Each month a forest cell may be cleared: 0.00002 for every
industry point or 4 population points in its neighbourhood, and 0.002, a hundred times as much, for every
lumber mill there. Open grassland with nothing built on it turns to forest by 0.00004 for every
forest cell next to it. A lumber mill left with fewer than 3 forest cells in its neighbourhood
closes, so mills clear their patch of forest and move on. Wherever a forest comes or goes, the
land weighs again for every ladder.

### Goods and satisfaction

`railroads.sim.economy` lets stations hand goods out every month, by the goods, needs and recipes
of `railroads.sim.goods`. Buildings need *needs*, each
met by one or more goods: houses need food, and any food good feeds them one for one (wheat,
apples, bananas, dates, sugar, cans or bread); canneries need fruit (apples, bananas, dates or
sugar) and bakeries wheat. Towns eat a lot, so they are hard to feed in full, and every farm and
bakery a station can reach is worth having.

| Building | Makes at most | Needs |
|---|---|---|
| House | | 10 food, 1 beer, 1 newspaper, 1 heating |
| Tenement | | 60 food, 2 salt, 20 goods, 2 coffee, 3 clothes, 2 furniture, 10 tools, 3 beer, 1 vodka, 2 shoes, 5 bricks, 2 refined sugar, 3 newspapers, 3 heating |
| Block of flats | | 180 food, 4 salt, 40 goods, 4 coffee, 30 clothes, 20 furniture, 10 tools, 5 fuel, 1 car, 4 beer, 2 vodka, 5 shoes, 5 bricks, 5 cement, 5 glass, 5 newspapers, 1 appliance, 4 refined sugar, 5 heating |
| Skyscraper | | 30 goods, 10 coffee, 5 clothes, 5 furniture, 20 wine, 20 fuel, 30 electronics, 10 glass, 10 cement, 20 newspapers, 20 paper, 5 appliances, 2 porcelain |
| Farm | 5 wheat | |
| Orchard | 5 apples | |
| Date palms | 6 dates | |
| Orange farm | 6 oranges | |
| Banana plantation | 9 bananas | |
| Sugar field | 8 sugar | |
| Mine | 100 coal | |
| Lumber mill | 20 lumber | |
| Port | 20 goods, 50 oil (from 1900), 50 gas (from 1950) | |
| Fishery | 8 fish | |
| Iron, copper mine | 250 iron ore, 40 copper | |
| Gold, diamond mine | 20 gold, 10 diamonds | |
| Oil well | 150 oil, 50 gas (from 1950) | |
| Salt works | 60 salt | |
| Sheep farm | 60 wool, 10 sheep | 90 wheat |
| Cotton field | 80 cotton | |
| Vineyard | 60 grapes | |
| Beekeeper | 2 honey, 10 wax | |
| Incense grove | 30 incense | |
| Coffee, rubber plantation | 60 coffee, 60 rubber | |
| Steel mill | 200 steel | 100 iron ore, 100 coal |
| Tool works | 80 tools | 80 steel |
| Textile mill | 100 clothes | 100 wool or cotton |
| Furniture maker | 60 furniture | 60 lumber |
| Barrel maker | 40 barrels | 20 lumber |
| Winery | 80 wine | 80 grapes, 40 barrels |
| Refinery | 300 fuel | 150 oil |
| Chemical works | 100 chemicals | 50 salt, 50 boiler fuel |
| Electronics plant | 50 electronics | 50 copper, 50 chemicals |
| Car factory | 20 cars | 100 steel, 40 fuel |
| Gun factory | 100 guns | 50 steel, 25 lumber |
| Aircraft factory | 20 planes | 60 steel, 20 copper, 20 fuel, 20 electronics (from 1960) |
| Tank factory | 20 tanks | 120 steel, 40 fuel |
| Military outpost | | 60 guns, 30 cars |
| Military base | | 180 guns, 90 cars, 30 planes |
| Headquarters | | 450 guns, 180 cars, 90 planes, 90 tanks |
| Airport | | 40 fuel |
| Church | | 10 wax, 10 incense |
| Basilica | | 30 wax, 20 incense |
| Cathedral | | 60 wax, 40 incense |
| St. Paul cathedral | | 200 wax, 120 incense |
| Jeweller | | 20 gold, 10 diamonds |
| Car dealer | | 20 cars, 10 rubber |
| Cannery | 40 cans | 10 fruit, 10 steel |
| Bakery | 20 bread | 10 wheat |
| Factory | 30 goods | 10 lumber, 10 steel |
| Big factory | 90 goods | 30 steel, 30 copper (steel alone before 1880) |
| Mega factory | 270 goods | 90 steel, 45 copper, 45 machinery (each only once it exists) |
| Shipyard | 30 goods | 20 steel, 10 lumber |
| Clay pit, limestone quarry, sand pit | 60 clay, 60 limestone, 60 sand | |
| Potato farm | 12 potatoes (food) | |
| Cattle ranch | 10 cattle | 120 wheat |
| Flax field | 60 flax (a fibre, like wool and cotton) | |
| Hop garden | 40 hops | |
| Machine works (1870) | 40 machinery | 40 steel, 20 copper (from 1880) |
| Brickworks | 100 bricks | 50 clay, 25 coal |
| Cement works (1880) | 100 cement | 50 limestone, 25 coal |
| Glassworks | 60 glass | 30 sand, 15 coal |
| Porcelain works | 30 porcelain | 15 clay, 8 coal |
| Brewery | 160 beer | 40 wheat, 40 hops, 16 barrels |
| Distillery | 40 vodka | 20 potatoes, 4 glass |
| Sugar refinery | 60 refined sugar | 30 sugar, 6 coal |
| Tannery | 40 leather | 20 cattle or sheep |
| Meat factory | 60 meat (food) | 30 cattle or sheep |
| Shoe factory | 60 shoes | 30 leather |
| Paper mill | 100 paper | 50 lumber, 25 chemicals (from 1880) |
| Printing works | 3,000 newspapers | 150 paper |
| Plastics works (1950) | 100 plastics | 50 oil, 50 chemicals |
| Appliance factory (1950) | 30 appliances | 30 steel, 15 copper, 15 plastics, 8 electronics (from 1960) |
| Tavern | | 20 beer, 10 vodka |
| Department store (1880) | | 20 clothes, 20 shoes, 10 furniture, 5 porcelain, 10 appliances (from 1950) |
| Hotel (1900) | | 80 food, 10 wine, 10 beer, 10 newspapers |
| Power plant (1900) | | 200 boiler fuel |
| Farm, from 1920 | | 1 machinery |

Steel mills, canneries, factories, shipyards and every works are processors: each keeps a stockpile of up to
300 of every input for every building, takes in what reaches it even while it waits for another
input, and works from it, so 100 iron ore and 100 coal make 200 steel and 10 apples and 10 steel make
40 cans. Nothing is made that has nowhere to go: a building makes only what a station takes,
and a processor uses up only the inputs for what it made, so a steel mill with no one to take its
steel keeps its coal and lumber waiting. Needs are handed out in the order of the chain, coal and
lumber, then steel, then wheat, then fruit, then food, then goods, so bakeries take their wheat and
canneries their fruit before houses eat what is left. A bakery bakes each unit of wheat into two
of bread, which sells for $8 against wheat's $6. An input that does not exist yet is left out of
a recipe until its year: a big factory works on steel alone until copper comes in 1880, and a mega
factory needs machinery only from 1870. Breweries take their wheat with the bakeries, distilleries
their potatoes and sugar refineries their sugar before houses eat what is left. Homes burn
lumber, coal or (from 1950) gas for heating, and power plants and chemical works burn coal, oil or
gas as boiler fuel; a station hands out the dearest of them first, as it does for every need. Cattle ranches
and sheep farms eat a lot of wheat, alongside the bakeries and breweries, and their cattle and
sheep go to tanneries and meat factories; meat feeds houses like any food.

Goods move only through stations, and only through those that track joins to another station:
nothing passes between neighbours by itself, and the land feeds no one. A building that gets all
it needs and gives away all it makes is thriving; one that gets and gives nothing is unsatisfied;
anything between is content. Needs are counted by amount, so a little salt or a car matters
little next to a town's food. Processors are judged on what they actually make, not on what they
could. The ground under content and thriving buildings has a warm glow, and the status bar names
the hovered cell's satisfaction.

Factories gather into districts: industry nearby draws more of them only once at least 5
industry points stand there, more than one steel mill, and then strongly, so three meet and merge
into a big factory and three of those into a mega factory. On Poland with the stand-in player
that gives about 100 factories and 20 big factories in 1850, and some 280 factories, 40 big and 5
mega factories by 2050.

Farms with 8 or more population points around them are pulled down, and none grow there, so farms
ring towns rather than standing in them. On flat grassland farms, orchards, date palms, bananas,
orange farms and sugar fields are ten times as likely, but where anything stands already only
one turn in ten goes ahead, so they favour empty plains. Bakeries open in towns, one to a
neighbourhood, a little likelier the more people live around.

### Rare resources and eras

Every new map gets deposits from its own geography and seed (`railroads.mapgen.deposits`), shown
as coloured specks on the ground and named in the hex details: iron ore, copper, gold and diamonds
in mountains and the hills beside them; oil and salt on inland plains; fish on shores; wool on
grassy hills; grapes on mild hills; cotton on hot plains; coffee on hot hills; rubber in hot
forest; and, very rarely, incense on any land off the mountains. A map gets a number of clusters for its land, spread apart, closer together where little
land allows them; those that do not fit are left out, and a map without the land for a resource
has none. Each resource is worked only on its own deposit, at most once per neighbourhood.

Other works open anywhere by chance, a little likelier where people and industry are, whether
or not their inputs are near, so their inputs usually have to be brought from far away. No
factory, mill or works needs people nearby. Jewellers appear only in the bigger towns (20
population points nearby) and car dealers only in big cities (120), and a map holds only a few:
one jeweller per 26,000 land cells, one car dealer per 16,000. Power plants open from 1900 where
at least 40 population points live nearby, 6 hexes apart and one per 15,000 land cells, and
count as 5 industry points, so factories gather around them. A map has only a few printing
works too, one per 40,000 land cells, each printing 3,000 newspapers a month for every home
around. Jewellers also keep at least 10
hexes from one another, so they spread over the towns rather than clumping in one. Chemical works, refineries, car factories and electronics plants are capped the same way.

Things arrive with the years: tool works, machine works and the need for tools in 1870, copper,
chemical works and cement works in 1880, department stores in 1880, hotels, power plants, oil and rubber in 1900, refineries in 1905, the need for fuel in 1910, car
factories and dealers in 1930, gas, plastics works and appliance factories in 1950, electronics in
1960. Cars for a good can be added to a train once
that good exists.

### The military

Every map gets 1 to 3 hidden patches of military ground, 10 to 18 cells each and far apart
(`Deposit.GARRISON`), and outposts open from 1800 only there, faster the more military points are
around, though slowly enough that a map has only 2 or 3 headquarters by 2050. Three outposts in
a neighbourhood become a base and three bases a headquarters. They are
judged on what they get like any other building, so a garrison that is kept supplied grows far
faster than one left alone. Each military point sends and wants 20 passengers and 4 pieces of mail
(outpost 3, base 9, headquarters 27), as much as a small town. Cars are wanted from 1930, planes from 1935 by bases and
headquarters, tanks from 1940 by headquarters.

Gun factories open from 1800 where there is industry, one per 26,000 land cells. Aircraft factories open from 1935 and tank factories from 1940,
one per 40,000 land cells each. Airports open from 1920 with at least 80 population points nearby,
one per 20,000 land cells; each sends and wants 200 passengers and 40 pieces of mail.

### Novelties

Some games get one-off buildings the others never see (`railroads.sim.novelties`). When a game
starts its seed picks 2 or 3 of them on a map with Poland's land, about 121,000 cells, and more or
fewer as the land is bigger or smaller. Nothing is weighed cell by cell: once a year each picked
novelty whose era has come rolls a 15% chance, and only then looks for a free hex by its own
rules, trying again the next year if none fits. Each fills a whole hex, appears once, and wants
and sends only goods the game already has, in vast amounts.

| Novelty | From | Where | Wants each month | Sends |
|---|---|---|---|---|
| Royal palace | 1850 | by the biggest town, best on a hill or a river | 200 food, 50 wine, 20 porcelain, 20 furniture | 300 passengers |
| Salt mine cathedral | 1850 | on a salt deposit | | 200 salt, 200 passengers |
| Spa resort | 1850 | hills beside a river or lake | 100 food, 20 wine, 20 beer, 20 newspapers | 300 passengers |
| World's Fair hall | 1880 to 1940 | by the biggest town, once 100 people live there | 200 goods, 100 glass, 100 steel | 1,000 passengers |
| Film studio | 1920 | by the biggest town, once 150 people live there | 50 chemicals, 50 clothes, 30 electronics (from 1960) | 200 passengers |
| Amusement park | 1950 | by the biggest town, once 100 people live there | 200 food, 100 beer, 50 fuel | 800 passengers |
| Space center | 1960 | anywhere on flat land, 20 times likelier within 10 hexes of military ground | 300 electronics, 1,000 fuel, 300 food | 500 passengers |
| Pear Company headquarters | 1980 | once a skyscraper stands, as near as it can to the one with most people around | 500 electronics, 1,000 food | 1,000 passengers |

Olympic stadiums are apart from the pick. In every real Summer Games year (1896 to 1912, 1920 to
1936, 1948 to 2016, 2021, then every four years) a map hosts one with a 5% chance on Poland's land,
scaled by the map's land up to 50%. The stadium goes up on the free hex nearest the biggest town
that has not hosted yet, at least 10 hexes from every earlier host, and wants 500 food, 200 beer
and 200 newspapers and sends 2,000 passengers.

Some works are so big that they too fill a whole hex: car, tank and aircraft factories, mega
factories, refineries, power plants and shipyards.

### Owning buildings and tech

Any building that makes something can be bought from the hex details (Buy): farms, mines, mills,
works and factories, but no homes, shops, churches or military. One costs five years of the
player's share of what it makes at full output, and that share is half of what it sells to the
player's stations, so it pays back only if a station takes its goods. Growth never takes a
building from its owner: one it pulls down (a farm crowded out by a town, a lumber mill whose
forest is gone, anything cleared for track) goes up again as theirs in the nearest cell its
land allows, and one that merges into a bigger building (three factories into a big factory)
makes the bigger one theirs.

The tech window (R, or the flask in the top bar) has seven trees, 8 to 13 nodes each
(`railroads.sim.tech`). Every tree gains experience in dollars: food and drink, metals and
machines, fuel and minerals, craft and textiles and modern industry from what the player's own
buildings of that kind earn them; commerce from what the shops around their stations take;
passengers and mail from what passengers and mail pay as they get off. The first level needs
$50,000 and each next one 1.6 times as much as the one before; each level gives a point to spend
on a node whose requirements are taken. Nodes raise the production and the owner's share of the
player's buildings of their tree, what that tree's cargo earns, station storage and waiting rooms,
speed, cars per train and coach capacity, cut upkeep and the price of buildings, make the
player's buildings count more towards growth around them, and open engines.

A building in reach of a station rising a rank (two stars for a tenement or big factory, three
for flats or a mega factory, four for a skyscraper) gives experience as if a train full of its
goods, 250 units, had been delivered for every star: for what it makes, or the mix of what it
needs, to the trees those goods belong to.

| Engine | From | Needs | Speed | Cars | Price |
|---|---|---|---|---|---|
| Planet steam engine | 1850 | | 1x | 5 | $5,000 |
| Mogul freight engine | 1880 | Heavy boilers (metals) | 1x | 8 | $12,000 |
| Pacific express engine | 1910 | Superheaters (metals) | 1.5x | 8 | $25,000 |
| Boxcab electric | 1925 | Electric traction (modern) | 1.4x | 10 | $40,000 |
| Express electric | 1965 | Thyristor control (modern) | 2.2x | 12 | $90,000 |

Electric engines run only on electrified track. A train's engine is chosen when it is bought.

### Churches

Churches go up where a neighbourhood holds at least 10 population points and no other church of
any rank stands within 4 hexes. They do not merge: a church becomes a basilica once 120 population
points live in its neighbourhood, and a basilica a cathedral at 240. The first cathedral to reach
400 becomes the map's one St. Paul cathedral. Each point of worship sends and wants one passenger
(church 5, basilica 20, cathedral 50, St. Paul cathedral 2137), and a fifth as much mail like any
other building. Honey counts as food; wax comes only from beekeepers and incense only from groves
on its own rare deposits.

### Finding things

The search button in the top bar (or F) opens a list of every kind of building; type part of a
name, or a good it makes or wants. Picking one outlines every one on the map and marks them on the
overview, and the arrow buttons or the left and right keys step from one to the next, nearest the
view first. The help button in the top right corner (or H) lists every building with what it makes
and wants in a month, and every good with its base price and who makes and wants it.

### Reports

The chart button in the top right corner (or G) opens the reports. The first chart shows the cars
delivered each year as stacked bars, one colour per good. A car counts as delivered when it leaves
its load at a station other than the one it was loaded at. The other charts show the stations you
had at the end of each year and the population of the whole map. The population chart goes back to
the year history starts, 850. The year under way shows as it stands so far. The charts can be
zoomed and dragged, and double-clicking one fits it again (railroads.sim.stats,
railroads.ui.reports_window).

### Building and trains

Each station takes the name of the real town it stands in (`railroads.places`): of the places
whose reach covers it, the one it is deepest within, where a village of 500 reaches about a
kilometre from its centre and a city of a million about 13. Outside every town it takes the
closest place within 50 km. A second station with the same name is numbered ("Łódź [2]"), and a
station with no place that near is called Station 1, Station 2 and so on. Zoomed in, each
station's name shows on a tag above it.

A game starts with $80,000 (`railroads.sim.finance`). Track costs $300 a cell on the flat, $450
in hills, $10,000 a cell of bridge and $20,000 a cell of tunnel; electric track $450 a cell more,
and double track 1.8 times as much as single, wires included, so a cell on the flat costs $540
double, $750 electric and $1,350 double and electric. A station costs $15,000 and a steam engine $5,000, which
sells back for half. Every month each station costs $200 in upkeep and each cell of track $2,
$3 more where it is electric and 1.8 times as much where it is double, charged once a cell for
the dearest track in it (`track_upkeep`). Building needs the cash for it, but upkeep is charged
anyway and can run into debt. The top bar shows the cash, with this month's and last month's books on hover, and the
build bar shows what the track under the mouse would cost. Route planning still weighs hills,
bridges and tunnels lightly, so it stays quick; click a point in between to steer track round a
river.

Freight earns money when the buildings around a station take it, by the demand there
(`railroads.sim.demand`). Each month a station's capacity for a need is twice what the buildings
in its inner area need of it. What they use up is everything the station hands them; the demand
left is the capacity less that use averaged over the last 3 months. Each unit taken earns its
price times the share of demand left and uses up demand itself, so a town nobody feeds pays the
full price, one supplied in full every month a little over half, and flooding one drives the price
down for up to three months. Goods a station gathered itself and passes along to the buildings
around it use up demand the same way and earn 30% of what a train delivery would have. The
Stations window shows the share of demand left for each need.

| Price | Goods |
|---|---|
| $4-8 | sand 4, clay 5, limestone 5, potatoes 6, wheat, apples, bananas, dates, sugar, fish, bread, coal, lumber, salt, bricks 8 |
| $9-15 | iron ore, grapes, wool, gas 10, hops 10, newspapers 10, oil, cotton, flax 12, cement 12, copper, cans, refined sugar 14, beer 15, paper 15 |
| $15-35 | sheep 15, meat 16, steel, cattle 20, glass 20, rubber, fuel, coffee, chemicals, leather 25, plastics 25, furniture, clothes, tools, goods, vodka 30, shoes 35 |
| $40-150 | wine, machinery 60, porcelain 60, electronics 120, appliances 150 |
| $800, $2,000, $5,000 | gold, diamonds, cars |

Passengers pay $10 and mail $25, as they get off the train. A good is worth nothing where no
building wants it.

Stations also send and want passengers and mail. Every month each building in a station's inner
ring sends passengers by its points, the same number it wants to receive: 1 per population point,
5 per food or timber point, 10 per ore, industry or trade point and 20 per corporate point. A fifth
as many pieces of mail go with them, rounded down for the station as a whole. Up to 100
passengers and 100 pieces of mail wait at a station, apart from the freight stockpile. They ride
in passenger coaches and mail vans of 50 each and earn money where they get off just like cargo,
by the demand there, which only trains use up; they then go on their way rather than waiting to
be handed out.

The build bar along the bottom of the map chooses what a left click builds (keys 1 and 2): a station (what stands in the cell moves to a
cell next to it, or comes down when it cannot move; built on a line, it takes the place of the
track in its cell, which joins it on every side), track, or the bulldozer (key 3). Clearing costs
money (`railroads.sim.demolition`): every building that has to leave a cell for a station or
track, smallest first, is paid for as if taken down, and the price shows before the click. A
building costs $500 a population point (a house $500, a block of flats $9,000), $1,000 a field,
$15,000 an industry point (a factory $15,000, a mega factory $135,000), $20,000 a mine or a point
of a shop or business, $30,000 a port, and a skyscraper $1,000,000; one-off buildings, the
novelties and the St. Paul cathedral, cannot be taken down, and neither station nor track goes
where one would have to. The bulldozer clears a whole cell at those prices, track for a quarter
of what it cost to lay and a station for $5,000, buildings the player owns too. A station can be
moved from its window: it costs a new station and clearing the new cell, its trains, name and
goods go with it, and it starts again at level 1 with no extra storage. For track,
click where it starts and then where it goes; while choosing where it goes, the planned track shows
as the real track would look, greyed out, once the pointer rests on a cell for a moment. `railroads.sim.routes` plans the cheapest way that
turns at most 60 degrees per cell, through edges or corners, bridging rivers and lakes and
tunnelling through mountains, and makes room the same way. While the track tool is on, the bar
chooses single or double track and whether it is electric. Track lays only along the way it was
planned: started on track already laid, its first piece branches off where that track runs on,
turning at most 60 degrees, so trains can run from one onto the other; ended on track, its last
piece joins it the same way, facing the way the new track was heading. A dead end there turns
to where the new track goes rather than staying as a stub, for no more than the difference in
price. Where the way crosses track it crosses it in one cell, and running along track already
laid costs the planner half as much as new track, so it follows a line rather than laying a
second one beside it. The track it runs along is widened or electrified to match, for the
difference in price, so laying electric track along a line electrifies it. An open end with no
track runs straight on across its cell. While placing a station, and when the
mouse is over one, the cells it hands goods out to (2 steps away) are shaded orange, the areas of
other stations pink, cells two stations share grey and the stations themselves blue, with a blue
line round the cells it gathers from (4 steps). A right click cancels whatever is being built;
with nothing being built it opens the details of a cell: its land, points, and every building
with what it makes and needs and what it got and gave last month. Clicking a station, left or
right, opens the station instead: its level and reach, its stockpile, waiting passengers and mail, demand and
the trains that call there, with its reach shaded on the map. There it can be raised a level,
$100,000 to level 2 and $500,000 to level 3, each level handing goods out and gathering them one
cell further (3 and 5 cells, then 4 and 6), and given more storage, $10,000 for each 500 more of
freight, passengers and mail.

The bar also opens the Trains window (T) and the Stations window (S). The Trains window lists
the trains on the left and shows the chosen one on the right: what it pulls, each car filled by
its load, and its route as a table of stops. It buys a steam engine and sets its stops from the
stations. Each stop has its own
cars, up to 5, which the train pulls when it leaves that stop for the next, so the way out and the
way back can carry different goods; each car is for one good and holds 50 of it, so a food car
carries only food. Every car costs $20 each time the train leaves a stop with it. A new train
starts at its first stop. The window shows what the train pulls
now. The Stations window shows every station's stockpile and the goods waiting to be handed out
there, with a button to look at the station; clicking a name opens the station. Trains
(`railroads.sim.railway`) run along the track at 60 cells a month on the flat, slower on hills,
and each car slows them: an empty car weighs 1, a full one 2, and every unit of weight costs 8%
of the engine's speed, so five full cars still run at a little over half of it. On the pixel map
(`railroads.ui.pixel.trains`) a steam engine with its tender leads the cars, each drawn as pixel
art from a small model of boxes and cylinders for 24 headings, so they turn with the track, at
twice the map's art resolution: a car of its own for every good: red boxcars for goods and
cans (with an orange stripe), grain and sugar hoppers, low wagons heaped with apples or dates,
yellow banana vans, hoppers for coal,
flatcars with logs or steel beams, and puffs of smoke from the chimney. Closed cars show an open
door when empty, open ones their bare deck. Track is drawn into the ground as pixel art too:
gravel, sleepers and rails, timber decks over rivers and lakes, and portals where it enters
mountains, rounding each turn at the middle of its cell. All the beds go down first, then all
the sleepers, then all the rails, so pieces meeting in a cell make one junction, where a branch
curves away from the line, or a crossing, with the rails of both running through. Double track
is two tracks side by side that close up where they run into single track, and single track
branching off double leaves from its outer track on that side. Electric track has a wire over
each track and a mast beside it every cell, on its far side, reaching over to the wires. Zoomed
out, trains and track are plain shapes: two rails for double track and a dark wire beside
electric track.

At each stop a train unloads its cars, takes on that stop's cars and fills them from the station's
stockpile. Unless the stop says not to, it waits there up to a month for its cars to fill, topping
them up as the station gathers more. Trains run through stations on the way that are not among their
stops, and can turn round in any station, but not on open track. Each time a train sets off it takes
the shortest way to its next stop over all the track there is. Once a month each station that track joins to another station hands what trains brought
to buildings within 2 cells, unsatisfied first, then content, nearest first, then what it gathered
before, for 30% of the price, and takes every spare good that buildings made within 4 cells into
its stockpile, which counts as selling them. So a station is a market for its own area: a mine in
its reach feeds a steel mill in its reach a month later, without a train. A station on its own,
with no track to another, does nothing. What trains brought and what it gathered share one limit, 1,000 goods in all; a
train unloads only what fits and keeps the rest aboard if its next leg pulls a car for that good,
otherwise the rest is lost. Goods a train brought beyond what the area can use in a month can be
loaded onto another train, so hubs can pass freight along. Freight remembers where it came from:
it pays only when buildings at another station take it, and back at its own station it just
returns to the stockpile, so shuttling goods around earns nothing.

Buildings grow as each month turns over. Whenever that changes something within reach of a
station (4 cells), a card for the month appears along the right edge of the map
(`railroads.sim.summaries`): how many buildings went up and came down near how many stations.
Clicking a card opens what changed, station by station, and outlines those cells on the map;
a right click dismisses it. The newest five cards show.

The top bar shows the date and runs the game at x1, x2, x3 or x6 speed; x12 and x24 are there
for testing. It also zooms, and toggles the map overview
(M) and the world map (W). What is under the mouse shows in a strip at the top of the map, and
below it a small card lists the buildings in that cell, one to a line, each with how satisfied it
is and the icons of what it makes and needs.

Chances are kept up to date as buildings come and go, and cells are drawn by thinning, so a
month costs about as much as what grows in it.

A new map grows for a thousand years before play starts, from 850 (`--years` changes that);
`railroads.mapgen.seeding` runs it. In play the calendar (`railroads.sim.clock`) runs from
January of the year the history reached, 1850 after a thousand years, to January 2050, a month
every two and a half minutes: a hundred hours at normal speed from 1850. The map keeps growing each month.

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
- Station names: [GeoNames](https://www.geonames.org/) cities500, every place with more than 500
  people, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Without it stations are
  numbered.

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

Build the place names stations are named from (about 14 MB download):

```bash
mkdir -p data/raw/geonames
curl -o data/raw/geonames/cities500.zip https://download.geonames.org/export/dump/cities500.zip
uv run python tools/build_places.py
```

and, to rebake the Poland maps and the world picture for the minimap:

```bash
uv run python tools/build_poland_map.py
uv run python tools/build_world_map.py
```
