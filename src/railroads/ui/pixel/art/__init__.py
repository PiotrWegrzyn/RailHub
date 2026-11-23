"""The game's sprites, drawn pixel by pixel in code and packed into one sheet.

`tools/build_sprites.py` saves the sheet as a PNG with a JSON index next to it, so any sprite can
be repainted by hand afterwards; the game only ever reads the PNG. Sprites are drawn for a cell 32
pixels wide, seen from the fixed view angle, with light from the top left. Each one stands on an
anchor: the point that sits on the ground where the thing is.

Each family of sprites has its own module; the Painter they draw with is in painter, and pieces
several families share are in parts. This module lists them all (SPRITES) and packs the sheet.
"""

from collections.abc import Callable
from functools import cache, partial

from PIL import Image

from railroads.sim.constructions import BUILDING_RULES, MERGED, BuildingKind
from railroads.ui.pixel.art import (
    built_works,
    grand_crafts,
    grand_fire,
    grand_food,
    grand_modern,
)
from railroads.ui.pixel.art.built import modelled
from railroads.ui.pixel.art.churches import basilica, cathedral, church, st_peters
from railroads.ui.pixel.art.commerce import (
    airport,
    department_store,
    hotel,
    port,
    shop,
    sushi_hut,
    tavern,
)
from railroads.ui.pixel.art.crafts import (
    carpet_weaver,
    clock_maker,
    fish_dryer,
    foundry,
    hospital,
    manor,
    mirror_maker,
    miso_maker,
    potter,
    sausage_maker,
    soy_sauce_brewery,
    theatre,
    tile_works,
    tofu_maker,
)
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
    rice_paddy,
    soy_field,
    sugar,
    tea_plantation,
)
from railroads.ui.pixel.art.grown import grown
from railroads.ui.pixel.art.homes import flats, house, skyscraper, tenement
from railroads.ui.pixel.art.industry import (
    appliance_factory,
    barrel_maker,
    big_factory,
    brewery,
    brickworks,
    car_factory,
    cement_works,
    chemical_works,
    diesel_power_plant,
    distillery,
    electronics_plant,
    factory,
    food_plant,
    furniture_maker,
    gas_power_plant,
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
    sake_brewery,
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
    gigafactory,
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
    logging_camp,
    mine,
    oil_well,
    salt_works,
    sand_pit,
    sawmill,
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
from railroads.ui.pixel.art.provisions import (
    bakery,
    big_bakery,
    big_dairy,
    dairy,
    mega_bakery,
    mega_dairy,
)
from railroads.ui.pixel.art.showpieces import (
    big_clock_maker,
    big_mirror_maker,
    big_steel_mill,
    mega_clock_maker,
    mega_mirror_maker,
    mega_steel_mill,
)
from railroads.ui.pixel.art.stations import SQUARE_STATIONS, TURNED_STATIONS

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

# A farm's barn and an orchard's cottage stand a touch bigger than a house.
FARMSTEADS = ("farm", "orchard", "orange_farm")

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
    "st_peters",
    "factory",
    "big_factory",
    "mega_factory",
    "skyscraper",
    "food_plant",
    "bakery",
    "big_bakery",
    "mega_bakery",
    "gasworks",
    "potter",
    "tile_works",
    "foundry",
    "dairy",
    "big_dairy",
    "mega_dairy",
    "sausage_maker",
    "carpet_weaver",
    "clock_maker",
    "big_clock_maker",
    "mega_clock_maker",
    "mirror_maker",
    "big_mirror_maker",
    "mega_mirror_maker",
    "manor",
    "theatre",
    "hospital",
    "railway_works",
    "steel_mill",
    "big_steel_mill",
    "mega_steel_mill",
    "shipyard",
    "mine",
    "logging_camp",
    "sawmill",
    "port",
    "clay_pit",
    "limestone_quarry",
    "sand_pit",
    "potato_farm",
    "rice_paddy",
    "soy_field",
    "tea_plantation",
    "soy_sauce_brewery",
    "tofu_maker",
    "miso_maker",
    "fish_dryer",
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
    "sake_brewery",
    "tannery",
    "shoe_factory",
    "plastics_works",
    "appliance_factory",
    "paper_mill",
    "printing_works",
    "sugar_refinery",
    "porcelain_works",
    "tavern",
    "sushi_hut",
    "department_store",
    "hotel",
    "meat_factory",
    "power_plant",
    "diesel_power_plant",
    "gas_power_plant",
    "space_center",
    "pear_hq",
    "gigafactory",
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
    "logging_camp": logging_camp,
    "sawmill": sawmill,
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
    "st_peters": st_peters,
    "gun_factory": lambda: works((120, 104, 78), (84, 74, 58), (190, 56, 48), 1),
    "tank_factory": tank_factory,
    "plane_factory": plane_factory,
    "date_palms": date_palms,
    "bananas": bananas,
    "sugar": sugar,
    **SQUARE_STATIONS,
    **TURNED_STATIONS,
    "food_plant": food_plant,
    "bakery": bakery,
    "big_bakery": big_bakery,
    "mega_bakery": mega_bakery,
    "gasworks": gasworks,
    "potter": potter,
    "tile_works": tile_works,
    "foundry": foundry,
    "dairy": dairy,
    "big_dairy": big_dairy,
    "mega_dairy": mega_dairy,
    "sausage_maker": sausage_maker,
    "carpet_weaver": carpet_weaver,
    "clock_maker": clock_maker,
    "big_clock_maker": big_clock_maker,
    "mega_clock_maker": mega_clock_maker,
    "mirror_maker": mirror_maker,
    "big_mirror_maker": big_mirror_maker,
    "mega_mirror_maker": mega_mirror_maker,
    "manor": manor,
    "theatre": theatre,
    "hospital": hospital,
    "railway_works": railway_works,
    "steel_mill": steel_mill,
    "big_steel_mill": big_steel_mill,
    "mega_steel_mill": mega_steel_mill,
    "shipyard": shipyard,
    "clay_pit": clay_pit,
    "limestone_quarry": limestone_quarry,
    "sand_pit": sand_pit,
    "potato_farm": potato_farm,
    "rice_paddy": rice_paddy,
    "soy_field": soy_field,
    "tea_plantation": tea_plantation,
    "soy_sauce_brewery": soy_sauce_brewery,
    "tofu_maker": tofu_maker,
    "miso_maker": miso_maker,
    "fish_dryer": fish_dryer,
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
    "sake_brewery": sake_brewery,
    "tannery": tannery,
    "shoe_factory": shoe_factory,
    "plastics_works": plastics_works,
    "appliance_factory": appliance_factory,
    "paper_mill": paper_mill,
    "printing_works": printing_works,
    "sugar_refinery": sugar_refinery,
    "porcelain_works": porcelain_works,
    "tavern": tavern,
    "sushi_hut": sushi_hut,
    "department_store": department_store,
    "hotel": hotel,
    "meat_factory": meat_factory,
    "power_plant": power_plant,
    "diesel_power_plant": diesel_power_plant,
    "gas_power_plant": gas_power_plant,
    "space_center": space_center,
    "pear_hq": pear_hq,
    "gigafactory": gigafactory,
    "royal_palace": royal_palace,
    "salt_cathedral": salt_cathedral,
    "spa_resort": spa_resort,
    "worlds_fair": worlds_fair,
    "film_studio": film_studio,
    "amusement_park": amusement_park,
    "olympic_stadium": olympic_stadium,
}

# The buildings drawn from 3D models (railroads.ui.pixel.art.built) instead of painted: the
# gasworks, whose brick rotunda took the place of its painted gasholder, and the stations.
MODELS: dict[str, Callable[[], Sprite]] = {"gasworks": modelled(built_works.gasworks)}
DRAWN_FROM_MODELS = (*MODELS, *SQUARE_STATIONS, *TURNED_STATIONS)
SPRITES |= MODELS

# Big and mega works drawn as buildings of their own, by trade.
for _module in (grand_fire, grand_food, grand_crafts, grand_modern):
    SPRITES |= {
        name: draw
        for name, draw in vars(_module).items()
        if name.startswith(("big_", "mega_")) and callable(draw)
    }

# Works merge into big and mega ones (railroads.sim.constructions.MERGED): unless they have
# sprites of their own, a big works is two of the buildings of the works it grew from and a mega
# works three (railroads.ui.pixel.art.grown), no wider on the map than GROWN_WIDTH pixels.
GROWN_WIDTH = {"big": 38, "mega": 46}

for _kind in MERGED:
    _base = _kind.name.lower()
    if _base in SPRITES:
        for _size, _most in GROWN_WIDTH.items():
            SPRITES.setdefault(
                f"{_size}_{_base}",
                partial(grown, SPRITES[_base], _size, _most / BUILDING_SCALE),
            )


def shrink(sprite: Sprite, factor: float) -> Sprite:
    """A smaller copy of a sprite: each new pixel takes the colour of the old pixel under its
    middle, so edges stay hard and colours stay the sprite's own."""
    w, h = sprite.image.size
    small = sprite.image.resize(
        (max(1, round(w * factor)), max(1, round(h * factor))), Image.Resampling.NEAREST
    )
    ax, ay = sprite.anchor
    return Sprite(small, (round(ax * factor), round(ay * factor)))


# How wide a works may be drawn at most, in pixels, so it stands inside its cell, which is about
# 25 pixels across: by how much of the cell it takes (railroads.sim.constructions.BUILDING_RULES),
# and for big and mega works by their size, a mega one as wide as the cell.
MOST_WIDTH = {1: 20, 2: 20, 3: 22, 4: 23, 5: 23, 6: 23}
GROWN_MOST_WIDTH = {"big": 25, "mega": 26}
# How much smaller a works is tried again when shrinking it pixel by pixel left it too wide.
SHRINK_STEP = 0.98


def _base_scale(name: str) -> float:
    size, _, base = name.partition("_")
    if size in GROWN_WIDTH and name not in BUILDINGS and base in SPRITES:
        return _base_scale(base)
    if name.startswith(HOMES):
        return HOME_SCALE
    if name in FARMSTEADS:
        return FARMSTEAD_SCALE
    return BUILDING_SCALE if name.startswith(BUILDINGS) else 1.0


@cache
def _full_size(name: str) -> Sprite:
    with drawn_at(1):
        return SPRITES[name]()


def _width(sprite: Sprite) -> int:
    """How many pixels wide a sprite is, from its leftmost painted pixel to its rightmost."""
    box = sprite.image.getchannel("A").getbbox()
    return 1 if box is None else box[2] - box[0]


def _scale_of(name: str) -> float:
    """How big a sprite is drawn on the map: its kind's scale, and smaller still if that would
    make a works wider than MOST_WIDTH lets it be."""
    scale = _base_scale(name)
    kind = BuildingKind.__members__.get(name.upper())
    if kind is None or name in FARMSTEADS:
        return scale
    most = GROWN_MOST_WIDTH.get(name.partition("_")[0], MOST_WIDTH[BUILDING_RULES[kind].space])
    full = _full_size(name)
    scale = min(scale, most / _width(full))
    while _width(shrink(full, scale)) > most:
        scale *= SHRINK_STEP
    return scale


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
    shrunk from its full size; one drawn from a model is shrunk from its drawing at 2, so both
    sheets hold it alike."""
    if resolution == 1:
        sprites = {
            name: shrink(fine_sprite(name, 2), 0.5)
            if name in DRAWN_FROM_MODELS
            else shrink(draw(), _scale_of(name))
            for name, draw in SPRITES.items()
        }
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
