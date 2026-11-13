"""The game's sprites, drawn pixel by pixel in code and packed into one sheet.

`tools/build_sprites.py` saves the sheet as a PNG with a JSON index next to it, so any sprite can
be repainted by hand afterwards; the game only ever reads the PNG. Sprites are drawn for a cell 32
pixels wide, seen from the fixed view angle, with light from the top left. Each one stands on an
anchor: the point that sits on the ground where the thing is.

Each family of sprites has its own module; the Painter they draw with is in painter, and pieces
several families share are in parts. This module lists them all (SPRITES) and packs the sheet.
"""

from collections.abc import Callable

from PIL import Image

from railroads.ui.pixel.art.churches import basilica, cathedral, church, st_pauls
from railroads.ui.pixel.art.commerce import airport, department_store, hotel, port, shop, tavern
from railroads.ui.pixel.art.farms import (
    bananas,
    beekeeper,
    cattle_ranch,
    chicken_farm,
    date_palms,
    farm,
    fishery,
    flax_field,
    hop_garden,
    incense_grove,
    orchard,
    pasture,
    pig_farm,
    plantation,
    potato_farm,
    sugar,
)
from railroads.ui.pixel.art.homes import flats, house, skyscraper, tenement
from railroads.ui.pixel.art.industry import (
    appliance_factory,
    bakery,
    barrel_maker,
    big_factory,
    brewery,
    brickworks,
    car_factory,
    cement_works,
    chemical_works,
    distillery,
    electronics_plant,
    factory,
    food_plant,
    furniture_maker,
    gasworks,
    glassworks,
    machine_works,
    meat_factory,
    mega_factory,
    paper_mill,
    plane_factory,
    plastics_works,
    porcelain_works,
    power_plant,
    printing_works,
    railway_works,
    refinery,
    shipyard,
    shoe_factory,
    steel_mill,
    sugar_refinery,
    tank_factory,
    tannery,
    textile_mill,
    tool_works,
    winery,
    works,
)
from railroads.ui.pixel.art.landmarks import (
    amusement_park,
    film_studio,
    olympic_stadium,
    pear_hq,
    royal_palace,
    salt_cathedral,
    spa_resort,
    space_center,
    worlds_fair,
)
from railroads.ui.pixel.art.military import base, headquarters, outpost
from railroads.ui.pixel.art.mines import (
    clay_pit,
    limestone_quarry,
    lumber_mill,
    mine,
    oil_well,
    salt_works,
    sand_pit,
    shaft_mine,
)
from railroads.ui.pixel.art.nature import leafy, pine, rocks
from railroads.ui.pixel.art.painter import (
    FLAT_WALLS,
    LEAVES,
    PINES,
    ROOFS,
    TENEMENT_ROOFS,
    TENEMENT_WALLS,
    WALLS,
    Painter,
    Sprite,
    drawn_at,
    shade,
)
from railroads.ui.pixel.art.stations import TURNED_STATIONS, station, station_2, station_3

__all__ = [
    "SPRITES",
    "Painter",
    "Sprite",
    "draw_sheet",
    "drawn_at",
    "fine_sprite",
    "shade",
    "shrink",
]

BUILDING_SCALE = 0.7

HOMES = ("house", "tenement", "flats")

HOME_SCALE = 0.59

# A farm's barn, an orchard's cottage and a bakery stand a touch bigger than a house.
FARMSTEADS = ("farm", "orchard", "orange_farm", "bakery")

FARMSTEAD_SCALE = 0.585

BUILDINGS = (
    "house",
    "tenement",
    "flats",
    "farm",
    "orchard",
    "date_palms",
    "bananas",
    "sugar",
    "iron_mine",
    "copper_mine",
    "gold_mine",
    "diamond_mine",
    "oil_well",
    "salt_works",
    "fishery",
    "sheep_farm",
    "cotton_field",
    "vineyard",
    "coffee_plantation",
    "rubber_plantation",
    "textile_mill",
    "furniture_maker",
    "tool_works",
    "winery",
    "refinery",
    "chemical_works",
    "electronics_plant",
    "car_factory",
    "car_dealer",
    "jeweller",
    "orange_farm",
    "outpost",
    "base",
    "headquarters",
    "gun_factory",
    "tank_factory",
    "plane_factory",
    "airport",
    "barrel_maker",
    "beekeeper",
    "incense_grove",
    "church",
    "basilica",
    "cathedral",
    "st_pauls",
    "factory",
    "big_factory",
    "mega_factory",
    "skyscraper",
    "food_plant",
    "bakery",
    "gasworks",
    "railway_works",
    "steel_mill",
    "shipyard",
    "mine",
    "lumber_mill",
    "port",
    "clay_pit",
    "limestone_quarry",
    "sand_pit",
    "potato_farm",
    "cattle_ranch",
    "pig_farm",
    "chicken_farm",
    "flax_field",
    "hop_garden",
    "machine_works",
    "brickworks",
    "cement_works",
    "glassworks",
    "brewery",
    "distillery",
    "tannery",
    "shoe_factory",
    "plastics_works",
    "appliance_factory",
    "paper_mill",
    "printing_works",
    "sugar_refinery",
    "porcelain_works",
    "tavern",
    "department_store",
    "hotel",
    "meat_factory",
    "power_plant",
    "space_center",
    "pear_hq",
    "royal_palace",
    "salt_cathedral",
    "spa_resort",
    "worlds_fair",
    "film_studio",
    "amusement_park",
    "olympic_stadium",
)

SPRITES: dict[str, Callable[[], Sprite]] = {
    **{f"pine_{i}": (lambda c=c, i=i: pine(c, i)) for i, c in enumerate(PINES)},
    **{f"leafy_{i}": (lambda c=c, i=i: leafy(c, i)) for i, c in enumerate(LEAVES)},
    "rocks": rocks,
    **{f"house_{i}": (lambda w=WALLS[i % 3], r=ROOFS[i % 4]: house(w, r)) for i in range(12)},
    **{
        f"tenement_{i}": (lambda w=w, r=TENEMENT_ROOFS[i % 2]: tenement(w, r))
        for i, w in enumerate(TENEMENT_WALLS)
    },
    **{f"flats_{i}": (lambda w=w: flats(w)) for i, w in enumerate(FLAT_WALLS)},
    "farm": farm,
    "factory": factory,
    "big_factory": big_factory,
    "mega_factory": mega_factory,
    "skyscraper": skyscraper,
    "mine": mine,
    "lumber_mill": lumber_mill,
    "port": port,
    "orchard": orchard,
    "iron_mine": lambda: shaft_mine((104, 72, 62), (176, 98, 70)),
    "copper_mine": lambda: shaft_mine((96, 84, 66), (206, 130, 76)),
    "gold_mine": lambda: shaft_mine((88, 78, 62), (255, 214, 60)),
    "diamond_mine": lambda: shaft_mine((58, 60, 72), (190, 238, 255)),
    "oil_well": oil_well,
    "salt_works": salt_works,
    "fishery": fishery,
    "sheep_farm": pasture,
    "cotton_field": lambda: plantation((196, 176, 136), (130, 92, 62)),
    "vineyard": lambda: plantation((220, 204, 170), (160, 84, 56)),
    "coffee_plantation": lambda: plantation((186, 150, 108), (96, 66, 44)),
    "rubber_plantation": lambda: plantation((150, 136, 104), (72, 92, 62)),
    "textile_mill": textile_mill,
    "furniture_maker": furniture_maker,
    "tool_works": tool_works,
    "winery": winery,
    "refinery": refinery,
    "chemical_works": chemical_works,
    "electronics_plant": electronics_plant,
    "car_factory": car_factory,
    "car_dealer": lambda: shop((226, 224, 218), (236, 80, 132), (60, 90, 170)),
    "jeweller": lambda: shop((70, 62, 84), (232, 190, 70), (190, 238, 255)),
    "orange_farm": lambda: orchard([(244, 142, 36)]),
    "outpost": outpost,
    "base": base,
    "headquarters": headquarters,
    "airport": airport,
    "barrel_maker": barrel_maker,
    "beekeeper": beekeeper,
    "incense_grove": incense_grove,
    "church": church,
    "basilica": basilica,
    "cathedral": cathedral,
    "st_pauls": st_pauls,
    "gun_factory": lambda: works((120, 104, 78), (84, 74, 58), (190, 56, 48), 1),
    "tank_factory": tank_factory,
    "plane_factory": plane_factory,
    "date_palms": date_palms,
    "bananas": bananas,
    "sugar": sugar,
    "station": station,
    "station_2": station_2,
    "station_3": station_3,
    **TURNED_STATIONS,
    "food_plant": food_plant,
    "bakery": bakery,
    "gasworks": gasworks,
    "railway_works": railway_works,
    "steel_mill": steel_mill,
    "shipyard": shipyard,
    "clay_pit": clay_pit,
    "limestone_quarry": limestone_quarry,
    "sand_pit": sand_pit,
    "potato_farm": potato_farm,
    "cattle_ranch": cattle_ranch,
    "pig_farm": pig_farm,
    "chicken_farm": chicken_farm,
    "flax_field": flax_field,
    "hop_garden": hop_garden,
    "machine_works": machine_works,
    "brickworks": brickworks,
    "cement_works": cement_works,
    "glassworks": glassworks,
    "brewery": brewery,
    "distillery": distillery,
    "tannery": tannery,
    "shoe_factory": shoe_factory,
    "plastics_works": plastics_works,
    "appliance_factory": appliance_factory,
    "paper_mill": paper_mill,
    "printing_works": printing_works,
    "sugar_refinery": sugar_refinery,
    "porcelain_works": porcelain_works,
    "tavern": tavern,
    "department_store": department_store,
    "hotel": hotel,
    "meat_factory": meat_factory,
    "power_plant": power_plant,
    "space_center": space_center,
    "pear_hq": pear_hq,
    "royal_palace": royal_palace,
    "salt_cathedral": salt_cathedral,
    "spa_resort": spa_resort,
    "worlds_fair": worlds_fair,
    "film_studio": film_studio,
    "amusement_park": amusement_park,
    "olympic_stadium": olympic_stadium,
}


def shrink(sprite: Sprite, factor: float) -> Sprite:
    """A smaller copy of a sprite: each new pixel takes the colour of the old pixel under its
    middle, so edges stay hard and colours stay the sprite's own."""
    w, h = sprite.image.size
    small = sprite.image.resize(
        (max(1, round(w * factor)), max(1, round(h * factor))), Image.Resampling.NEAREST
    )
    ax, ay = sprite.anchor
    return Sprite(small, (round(ax * factor), round(ay * factor)))


def _scale_of(name: str) -> float:
    if name.startswith(HOMES):
        return HOME_SCALE
    if name in FARMSTEADS:
        return FARMSTEAD_SCALE
    return BUILDING_SCALE if name.startswith(BUILDINGS) else 1.0


def fine_sprite(name: str, resolution: int) -> Sprite:
    """A sprite drawn at `resolution` pixels for every art pixel of the map, at its size on the
    map."""
    with drawn_at(_scale_of(name) * resolution):
        return SPRITES[name]()


def draw_sheet(padding: int = 1, resolution: int = 1) -> tuple[Image.Image, dict[str, list[int]]]:
    """Every sprite packed in rows on one sheet, and where each sits: x, y, width, height and its
    anchor inside it. Buildings other than stations are drawn at BUILDING_SCALE, houses,
    tenements and flats at HOME_SCALE, and farms and orchards at FARMSTEAD_SCALE. At a
    resolution above 1 each sprite is drawn that many times finer (fine_sprite) rather than
    shrunk from its full size."""
    if resolution == 1:
        sprites = {name: shrink(draw(), _scale_of(name)) for name, draw in SPRITES.items()}
    else:
        sprites = {name: fine_sprite(name, resolution) for name in SPRITES}
    width = 128 * resolution
    x = y = row_height = 0
    places: dict[str, list[int]] = {}
    for name, sprite in sprites.items():
        w, h = sprite.image.size
        if x + w > width:
            x, y, row_height = 0, y + row_height + padding, 0
        places[name] = [x, y, w, h, *sprite.anchor]
        x += w + padding
        row_height = max(row_height, h)
    sheet = Image.new("RGBA", (width, y + row_height), (0, 0, 0, 0))
    for name, sprite in sprites.items():
        sheet.paste(sprite.image, tuple(places[name][:2]))
    return sheet, places
