"""A small picture of every good and of the needs goods meet, drawn on one sheet of equal squares
that windows show in place of the goods' names."""

from collections.abc import Callable
from pathlib import Path

from PIL import Image

from railroads.ui.pixel.art import Painter, Sprite, shade

ICON_DIR = Path(__file__).parent
ICONS_PNG = ICON_DIR / "good_icons.png"
ICONS_INDEX = ICON_DIR / "good_icons.json"
ICON_PIXELS = 16
COLUMNS = 8

RGB = tuple[int, int, int]
LEAF: RGB = (86, 156, 62)
STEM: RGB = (120, 84, 50)
WOOD: RGB = (160, 110, 62)
DARK_EDGE: RGB = (150, 150, 162)
METAL: RGB = (168, 176, 188)
CREAM: RGB = (240, 232, 208)


def _canvas() -> Painter:
    return Painter(ICON_PIXELS, ICON_PIXELS, (8, 8))


def wheat() -> Sprite:
    p = _canvas()
    p.line([(0, 7), (0, -7)], (176, 138, 64))
    for y in (-6, -3, 0, 3):
        p.ellipse(-5, y - 1, -1, y + 2, (232, 190, 86))
        p.ellipse(1, y - 1, 5, y + 2, (232, 190, 86))
    p.ellipse(-2, -8, 2, -4, (244, 208, 112))
    return p.sprite()


def _round_fruit(body: RGB) -> Sprite:
    p = _canvas()
    p.ellipse(-6, -4, 6, 7, body)
    p.blob(-4, -2, -1, 1, shade(body, 1.35))
    p.line([(0, -4), (1, -7)], STEM)
    p.polygon([(1, -6), (5, -8), (6, -5), (2, -4)], LEAF)
    return p.sprite()


def apples() -> Sprite:
    return _round_fruit((206, 48, 48))


def oranges() -> Sprite:
    return _round_fruit((240, 140, 36))


def bananas() -> Sprite:
    p = _canvas()
    yellow: RGB = (244, 214, 74)
    p.polygon(
        [(-6, -7), (-3, -7), (-2, -1), (2, 3), (7, 3), (6, 6), (0, 7), (-5, 3), (-7, -3)],
        yellow,
        shade(yellow, 0.6),
    )
    p.line([(-4, -5), (-3, 1), (1, 5)], shade(yellow, 0.72))
    p.rect(-6, -8, -4, -7, STEM)
    return p.sprite()


def dates() -> Sprite:
    p = _canvas()
    p.line([(0, -7), (0, -3), (-5, 0)], STEM)
    p.line([(0, -3), (5, 0)], STEM)
    fruit: RGB = (130, 76, 44)
    for left, top in ((-7, 0), (-2, 1), (3, 0)):
        p.ellipse(left, top, left + 4, top + 7, fruit)
        p.point(left + 1, top + 2, shade(fruit, 1.5))
    return p.sprite()


def sugar() -> Sprite:
    p = _canvas()
    p.box(0, 4, 6, 7, (240, 238, 232), (255, 255, 252))
    return p.sprite()


def honey() -> Sprite:
    p = _canvas()
    amber: RGB = (236, 168, 40)
    p.rect(-5, -3, 5, 7, amber)
    p.rect(-4, -7, 4, -4, CREAM)
    p.rect(-3, 0, 3, 4, (250, 232, 170))
    p.line([(-3, 2), (3, 2)], amber)
    return p.sprite()


def fish() -> Sprite:
    p = _canvas()
    body: RGB = (112, 176, 212)
    p.polygon([(3, 0), (8, -4), (7, 0), (8, 4)], shade(body, 0.75))
    p.ellipse(-8, -4, 5, 4, body)
    p.point(-5, -1, (24, 30, 40))
    p.line([(-2, -3), (-2, 3)], shade(body, 0.72))
    return p.sprite()


def cans() -> Sprite:
    p = _canvas()
    p.rect(-5, -6, 5, 7, METAL)
    p.rect(-5, -3, 5, 4, (238, 138, 50))
    p.ellipse(-5, -8, 5, -4, (206, 212, 220))
    return p.sprite()


def bread() -> Sprite:
    p = _canvas()
    crust: RGB = (196, 128, 58)
    p.ellipse(-7, -4, 7, 6, crust)
    p.ellipse(-6, -5, 6, 2, shade(crust, 1.18))
    for x in (-3, 0, 3):
        p.line([(x - 1, -2), (x + 1, -4)], (238, 200, 140))
    return p.sprite()


def lumber() -> Sprite:
    p = _canvas()
    for y in (-6, -1, 4):
        p.rect(-7, y, 5, y + 3, WOOD)
        p.ellipse(3, y, 7, y + 3, (214, 170, 112))
    return p.sprite()


def _lump(fill: RGB, edge: RGB) -> Painter:
    p = _canvas()
    p.polygon([(-7, 6), (-6, -2), (-2, -6), (3, -5), (7, 0), (6, 6)], fill, edge)
    return p


def coal() -> Sprite:
    p = _lump((46, 46, 54), DARK_EDGE)
    p.polygon([(-3, -3), (1, -4), (0, 0)], (96, 96, 110))
    return p.sprite()


def iron_ore() -> Sprite:
    p = _lump((150, 92, 72), (94, 58, 46))
    for x, y in ((-3, -1), (2, -3), (3, 3), (-2, 3)):
        p.rect(x, y, x + 1, y + 1, (86, 54, 46))
    return p.sprite()


def _ingot(fill: RGB) -> Sprite:
    p = _canvas()
    p.polygon([(-8, 5), (-5, -3), (5, -3), (8, 5)], fill, shade(fill, 0.6))
    p.polygon([(-5, -3), (5, -3), (3, -6), (-3, -6)], shade(fill, 1.25), shade(fill, 0.7))
    return p.sprite()


def copper() -> Sprite:
    return _ingot((208, 128, 74))


def gold() -> Sprite:
    return _ingot((250, 208, 62))


def diamonds() -> Sprite:
    p = _canvas()
    ice: RGB = (176, 232, 255)
    p.polygon([(-7, -3), (-3, -7), (3, -7), (7, -3), (0, 7)], ice, shade(ice, 0.65))
    p.polygon([(-3, -3), (3, -3), (0, 7)], (226, 248, 255))
    p.line([(-7, -3), (7, -3)], shade(ice, 0.7))
    return p.sprite()


def _drop(fill: RGB, edge: RGB) -> Sprite:
    p = _canvas()
    p.polygon([(0, -8), (-5, 0), (5, 0)], fill, edge)
    p.ellipse(-6, -3, 6, 7, fill)
    p.line([(-6, 1), (-5, 6), (0, 7), (5, 6), (6, 1)], edge)
    p.point(-3, 1, (130, 130, 150))
    return p.sprite()


def oil() -> Sprite:
    return _drop((34, 32, 42), DARK_EDGE)


def salt() -> Sprite:
    p = _canvas()
    white: RGB = (240, 242, 248)
    p.polygon([(-8, 7), (0, -6), (8, 7)], white, (168, 172, 186))
    for x, y in ((-2, 3), (2, 1), (0, -2), (3, 5), (-4, 6)):
        p.point(x, y, (176, 182, 198))
    return p.sprite()


def wool() -> Sprite:
    p = _canvas()
    p.ellipse(-7, -7, 7, 7, CREAM)
    edge = shade(CREAM, 0.68)
    p.line([(-5, -4), (2, 6)], edge)
    p.line([(-2, -6), (5, 3)], edge)
    p.line([(-6, 2), (5, -4)], edge)
    return p.sprite()


def cotton() -> Sprite:
    p = _canvas()
    p.line([(0, 7), (0, 2)], STEM)
    p.polygon([(-4, 2), (4, 2), (0, 6)], (116, 88, 56))
    white: RGB = (252, 252, 248)
    for left, top in ((-7, -4), (1, -4), (-3, -8), (-3, -2)):
        p.ellipse(left, top, left + 6, top + 6, white)
    return p.sprite()


def grapes() -> Sprite:
    p = _canvas()
    purple: RGB = (126, 62, 156)
    p.polygon([(0, -8), (5, -8), (4, -5), (1, -5)], LEAF)
    for (
        x,
        y,
    ) in ((-5, -5), (-1, -5), (3, -5), (-3, -2), (1, -2), (-1, 1)):
        p.ellipse(x - 1, y, x + 3, y + 4, purple)
        p.point(x, y + 1, shade(purple, 1.6))
    return p.sprite()


def coffee() -> Sprite:
    p = _canvas()
    bean: RGB = (120, 76, 48)
    p.ellipse(-7, -7, 1, 3, bean)
    p.line([(-5, -5), (-2, 1)], shade(bean, 0.55))
    p.ellipse(-1, -2, 7, 7, shade(bean, 1.15))
    p.line([(1, 0), (5, 6)], shade(bean, 0.6))
    return p.sprite()


def rubber() -> Sprite:
    p = _canvas()
    p.ellipse(-7, -7, 7, 7, (44, 44, 50))
    p.line([(-5, -5), (-3, -6)], DARK_EDGE)
    p.ellipse(-3, -3, 3, 3, (150, 152, 158))
    p.blob(-1, -1, 1, 1, (44, 44, 50))
    return p.sprite()


def wax() -> Sprite:
    p = _canvas()
    p.rect(-3, -3, 3, 7, (246, 234, 176))
    p.line([(0, -3), (0, -5)], (64, 56, 50))
    p.polygon([(0, -8), (-2, -5), (2, -5)], (252, 176, 52))
    return p.sprite()


def incense() -> Sprite:
    p = _canvas()
    p.line([(-2, -1), (-4, -4), (-2, -6), (-3, -8)], (206, 178, 226))
    p.line([(2, -1), (4, -4), (2, -6), (3, -8)], (206, 178, 226))
    gold: RGB = (216, 170, 76)
    p.polygon([(-6, 0), (6, 0), (4, 5), (-4, 5)], gold, shade(gold, 0.6))
    p.rect(-2, 5, 2, 7, shade(gold, 0.8))
    return p.sprite()


def steel() -> Sprite:
    p = _canvas()
    p.rect(-7, -6, 7, -3, METAL)
    p.rect(-2, -3, 2, 3, shade(METAL, 0.82))
    p.rect(-7, 3, 7, 6, METAL)
    return p.sprite()


def tools() -> Sprite:
    p = _canvas()
    p.polygon([(-4, 7), (-7, 4), (2, -4), (5, -1)], WOOD, shade(WOOD, 0.6))
    p.polygon([(-1, -5), (3, -8), (8, -2), (4, 1)], METAL, shade(METAL, 0.55))
    return p.sprite()


def fuel() -> Sprite:
    p = _canvas()
    can: RGB = (214, 186, 60)
    p.polygon([(-6, -4), (-2, -7), (6, -7), (6, 7), (-6, 7)], can, shade(can, 0.58))
    p.line([(-3, -1), (3, 5)], shade(can, 0.55))
    p.line([(3, -1), (-3, 5)], shade(can, 0.55))
    p.rect(1, -8, 4, -6, shade(can, 0.7))
    return p.sprite()


def chemicals() -> Sprite:
    p = _canvas()
    glass: RGB = (214, 232, 226)
    p.polygon([(-2, -7), (2, -7), (2, -2), (7, 7), (-7, 7), (-2, -2)], glass, shade(glass, 0.55))
    p.polygon([(-4, 2), (4, 2), (6, 6), (-6, 6)], (130, 220, 96))
    p.point(-1, 4, (214, 250, 186))
    return p.sprite()


def furniture() -> Sprite:
    p = _canvas()
    wood: RGB = (168, 118, 68)
    p.rect(-5, -8, -3, 7, wood)
    p.rect(-5, 0, 6, 2, shade(wood, 1.2))
    p.rect(4, 2, 6, 7, wood)
    p.rect(-5, -7, 2, -5, shade(wood, 1.2))
    return p.sprite()


def clothes() -> Sprite:
    p = _canvas()
    cloth: RGB = (222, 122, 178)
    p.polygon(
        [
            (-3, -7),
            (0, -5),
            (3, -7),
            (8, -3),
            (5, 0),
            (4, -1),
            (4, 7),
            (-4, 7),
            (-4, -1),
            (-5, 0),
            (-8, -3),
        ],
        cloth,
        shade(cloth, 0.55),
    )
    return p.sprite()


def barrels() -> Sprite:
    p = _canvas()
    oak: RGB = (162, 108, 58)
    p.polygon(
        [(-4, -7), (4, -7), (6, -2), (6, 2), (4, 7), (-4, 7), (-6, 2), (-6, -2)],
        oak,
        shade(oak, 0.55),
    )
    for y in (-4, 3):
        p.line([(-5, y), (5, y)], (84, 80, 80))
    p.line([(-1, -7), (-1, 7)], shade(oak, 0.72))
    return p.sprite()


def wine() -> Sprite:
    p = _canvas()
    bottle: RGB = (120, 22, 56)
    p.rect(-1, -8, 1, -3, bottle)
    p.polygon([(-1, -4), (1, -4), (4, -1), (4, 7), (-4, 7), (-4, -1)], bottle, shade(bottle, 1.5))
    p.rect(-3, 1, 3, 5, CREAM)
    return p.sprite()


def electronics() -> Sprite:
    p = _canvas()
    for offset in (-4, 0, 4):
        p.line([(offset, -8), (offset, 7)], METAL)
        p.line([(-8, offset), (7, offset)], METAL)
    p.rect(-6, -6, 6, 6, (40, 112, 124))
    p.rect(-3, -3, 3, 3, (92, 226, 232))
    return p.sprite()


def cars() -> Sprite:
    p = _canvas()
    paint: RGB = (222, 64, 92)
    p.polygon([(-4, -1), (-2, -6), (4, -6), (6, -1)], (172, 214, 234), shade(paint, 0.6))
    p.polygon([(-8, -1), (8, -1), (8, 4), (-8, 4)], paint, shade(paint, 0.6))
    for x in (-5, 5):
        p.ellipse(x - 2, 2, x + 2, 6, (32, 32, 38))
    return p.sprite()


def guns() -> Sprite:
    p = _canvas()
    metal: RGB = (168, 176, 166)
    p.rect(-7, -6, 7, -2, metal)
    p.rect(5, -7, 7, -6, shade(metal, 0.7))
    p.polygon([(-7, -2), (-1, -2), (-3, 7), (-8, 7)], WOOD, shade(WOOD, 0.6))
    p.line([(-1, -2), (2, -2), (2, 2), (-2, 2)], metal)
    return p.sprite()


def tanks() -> Sprite:
    p = _canvas()
    hull: RGB = (106, 124, 72)
    p.rect(-1, -5, 8, -3, shade(hull, 0.8))
    p.polygon([(-5, -7), (2, -7), (3, -1), (-6, -1)], hull, shade(hull, 0.58))
    p.polygon([(-8, -1), (8, -1), (6, 3), (-6, 3)], shade(hull, 1.15), shade(hull, 0.6))
    p.ellipse(-8, 2, 8, 7, (54, 54, 56))
    return p.sprite()


def planes() -> Sprite:
    p = _canvas()
    body: RGB = (204, 214, 226)
    edge = shade(body, 0.55)
    p.polygon([(-2, -8), (2, -8), (2, 7), (-2, 7)], body, edge)
    p.polygon([(-8, -1), (8, -1), (8, 2), (-8, 2)], body, edge)
    p.polygon([(-4, 5), (4, 5), (4, 7), (-4, 7)], body, edge)
    p.rect(-1, -6, 1, -4, (88, 148, 216))
    return p.sprite()


def goods() -> Sprite:
    p = _canvas()
    blue: RGB = (84, 124, 214)
    p.box(0, 4, 7, 8, blue, shade(blue, 1.3))
    p.line([(0, 2), (0, 8)], (232, 214, 150))
    return p.sprite()


def passengers() -> Sprite:
    p = _canvas()
    far: RGB = (70, 150, 124)
    p.ellipse(1, -7, 7, -1, far)
    p.polygon([(0, 7), (1, 0), (7, 0), (8, 7)], far)
    near: RGB = (118, 214, 174)
    p.ellipse(-7, -6, -1, 0, near)
    p.polygon([(-8, 7), (-7, 1), (-1, 1), (0, 7)], near)
    return p.sprite()


def mail() -> Sprite:
    p = _canvas()
    paper: RGB = (232, 222, 248)
    edge: RGB = (120, 88, 168)
    p.polygon([(-8, -5), (7, -5), (7, 6), (-8, 6)], paper, edge)
    p.line([(-8, -5), (0, 2), (7, -5)], edge)
    return p.sprite()


def food() -> Sprite:
    p = _canvas()
    crust: RGB = (212, 150, 76)
    p.ellipse(-8, -4, 8, 6, crust)
    for x in (-4, 0, 4):
        p.line([(x, -3), (x + 2, 0)], (246, 214, 148))
    return p.sprite()


def fruit() -> Sprite:
    p = _canvas()
    p.line([(-3, -1), (0, -8), (4, 0)], STEM)
    p.ellipse(-8, -2, -1, 5, (212, 42, 58))
    p.ellipse(1, 0, 8, 7, (226, 56, 66))
    p.polygon([(0, -8), (5, -8), (3, -5)], LEAF)
    return p.sprite()


def fibre() -> Sprite:
    p = _canvas()
    p.rect(-6, -7, 6, -5, WOOD)
    p.rect(-6, 5, 6, 7, WOOD)
    thread: RGB = (236, 226, 196)
    p.rect(-4, -5, 4, 5, thread)
    for y in (-3, 0, 3):
        p.line([(-4, y), (4, y + 1)], shade(thread, 0.65))
    return p.sprite()


def clay() -> Sprite:
    p = _canvas()
    wet: RGB = (190, 104, 62)
    p.line([(4, -8), (1, -1)], WOOD)
    p.line([(2, -8), (6, -8)], WOOD)
    p.polygon([(-8, 7), (-7, 2), (-3, -2), (3, -2), (7, 2), (7, 7)], wet, shade(wet, 0.58))
    p.polygon([(-5, 2), (-3, 0), (1, 0), (-2, 3)], shade(wet, 1.25))
    p.polygon([(-1, -3), (3, -2), (2, 1), (-1, 0)], METAL, shade(METAL, 0.55))
    return p.sprite()


def limestone() -> Sprite:
    p = _canvas()
    pale: RGB = (224, 220, 200)
    p = _lump(pale, shade(pale, 0.6))
    p.polygon([(-6, -2), (-2, -6), (3, -5), (-3, -1)], shade(pale, 1.1))
    for y in (1, 4):
        p.line([(-6, y), (6, y - 1)], shade(pale, 0.8))
    p.ellipse(1, -3, 4, 0, shade(pale, 0.86))
    p.point(2, -2, shade(pale, 0.62))
    return p.sprite()


def sand() -> Sprite:
    p = _canvas()
    grain: RGB = (234, 208, 136)
    p.ellipse(-8, -2, 7, 16, grain)
    p.line([(-8, 7), (7, 7)], shade(grain, 0.6))
    p.blob(-5, 0, 0, 4, shade(grain, 1.12))
    for x, y in ((-3, 5), (1, 2), (4, 4), (-1, 6), (5, 1)):
        p.point(x, y, shade(grain, 0.72))
    return p.sprite()


def potatoes() -> Sprite:
    p = _canvas()
    skin: RGB = (190, 146, 92)
    for left, top, right, bottom in ((-8, -2, 0, 4), (-1, -6, 7, 0), (-4, 1, 5, 7)):
        p.ellipse(left, top, right, bottom, skin)
        p.blob(left + 2, top + 1, left + 4, top + 2, shade(skin, 1.2))
        p.point((left + right) // 2 + 1, (top + bottom) // 2 + 1, shade(skin, 0.6))
    return p.sprite()


def cattle() -> Sprite:
    p = _canvas()
    white: RGB = (240, 238, 232)
    black: RGB = (40, 38, 42)
    for x in (-4, -2, 3, 5):
        p.line([(x, 2), (x, 6)], black)
    p.ellipse(-5, -4, 7, 3, white)
    p.blob(-2, -3, 2, 1, black)
    p.blob(4, -3, 6, -1, black)
    p.line([(7, -2), (8, 2)], black)
    p.rect(-8, -6, -4, -1, white)
    p.rect(-8, -3, -6, -1, (236, 170, 170))
    p.point(-6, -5, black)
    p.line([(-8, -8), (-7, -6)], CREAM)
    p.line([(-4, -8), (-5, -6)], CREAM)
    return p.sprite()


def pigs() -> Sprite:
    p = _canvas()
    skin: RGB = (232, 166, 160)
    for x in (-4, -1, 3, 5):
        p.line([(x, 2), (x, 5)], shade(skin, 0.6))
    p.ellipse(-6, -4, 7, 3, skin)
    p.rect(-8, -3, -5, 1, shade(skin, 0.92))
    p.point(-8, -2, shade(skin, 0.55))
    p.point(-8, 0, shade(skin, 0.55))
    p.point(-5, -3, (40, 30, 30))
    p.line([(-5, -5), (-4, -3)], shade(skin, 0.8))
    p.line([(7, -2), (8, -3), (8, -1)], shade(skin, 0.7))
    return p.sprite()


def chickens() -> Sprite:
    p = _canvas()
    white: RGB = (244, 240, 230)
    p.line([(-1, 4), (-1, 7)], (222, 160, 60))
    p.line([(2, 4), (2, 7)], (222, 160, 60))
    p.ellipse(-5, -2, 6, 5, white)
    p.line([(5, -4), (7, -1), (6, 2)], shade(white, 0.85))
    p.ellipse(-7, -7, -2, -2, white)
    p.point(-8, -5, (222, 160, 60))
    p.point(-5, -5, (40, 30, 30))
    p.line([(-6, -8), (-4, -8)], (200, 50, 44))
    p.point(-6, -2, (200, 50, 44))
    return p.sprite()


def eggs() -> Sprite:
    p = _canvas()
    straw: RGB = (196, 156, 84)
    p.ellipse(-7, 1, 7, 7, straw)
    p.line([(-6, 3), (6, 3)], shade(straw, 0.8))
    for x, shell in ((-4, (244, 236, 220)), (0, (226, 190, 150)), (4, (244, 236, 220))):
        p.ellipse(x - 2, -4, x + 2, 2, shell)
        p.point(x - 1, -3, shade(shell, 1.1))
    return p.sprite()


def animals() -> Sprite:
    p = _canvas()
    skin: RGB = (232, 166, 160)
    p.ellipse(-1, -1, 8, 5, skin)
    p.rect(-3, 0, -1, 3, shade(skin, 0.92))
    p.line([(1, 5), (1, 7)], shade(skin, 0.6))
    p.line([(6, 5), (6, 7)], shade(skin, 0.6))
    white: RGB = (244, 240, 230)
    p.ellipse(-8, -6, -2, -1, white)
    p.ellipse(-9, -8, -6, -5, white)
    p.point(-10, -7, (222, 160, 60))
    p.point(-8, -9, (200, 50, 44))
    return p.sprite()


def coke() -> Sprite:
    p = _canvas()
    for x, y, size in ((-4, 2, 4), (2, 3, 5), (-1, -3, 4), (4, -2, 3)):
        p.blob(x - size / 2, y - size / 2, x + size / 2, y + size / 2, (70, 70, 76))
        p.point(x - 1, y - 1, (130, 130, 138))
    return p.sprite()


def milk() -> Sprite:
    p = _canvas()
    tin: RGB = (196, 200, 206)
    p.rect(-4, -3, 4, 7, tin)
    p.rect(-2, -6, 2, -3, tin)
    p.line([(-4, -3), (4, -3)], shade(tin, 0.7))
    p.line([(-3, 2), (3, 2)], shade(tin, 0.8))
    p.line([(-2, -7), (2, -7)], shade(tin, 0.6))
    p.point(-2, 0, (240, 240, 236))
    return p.sprite()


def dairy() -> Sprite:
    p = _canvas()
    cheese: RGB = (238, 200, 90)
    p.polygon([(-7, 4), (5, 4), (5, -1), (-7, 1)], cheese, shade(cheese, 0.7))
    p.polygon([(-7, 1), (5, -1), (1, -4)], shade(cheese, 1.1))
    for x, y in ((-3, 2), (2, 2)):
        p.point(x, y, shade(cheese, 0.7))
    p.rect(1, 1, 7, 6, (246, 232, 170))
    return p.sprite()


def pots() -> Sprite:
    p = _canvas()
    clay: RGB = (176, 104, 70)
    p.ellipse(-6, -3, 6, 7, clay)
    p.rect(-3, -6, 3, -3, shade(clay, 0.9))
    p.line([(-4, -6), (4, -6)], shade(clay, 1.2))
    p.line([(-5, 1), (5, 1)], shade(clay, 0.7))
    return p.sprite()


def stove_tiles() -> Sprite:
    p = _canvas()
    glaze: RGB = (70, 120, 170)
    for x in (-6, 1):
        for y in (-6, 1):
            p.rect(x, y, x + 5, y + 5, glaze)
            p.point(x + 2, y + 2, (236, 236, 226))
    return p.sprite()


def castings() -> Sprite:
    p = _canvas()
    iron: RGB = (74, 76, 82)
    p.ellipse(-7, -2, 5, 6, iron)
    p.line([(5, 0), (8, -2)], iron)
    p.line([(-6, -2), (4, -2)], shade(iron, 1.5))
    p.point(-3, 1, shade(iron, 1.4))
    return p.sprite()


def sausages() -> Sprite:
    p = _canvas()
    skin: RGB = (170, 74, 54)
    for y in (-4, 1):
        p.ellipse(-7, y, 7, y + 4, skin)
        p.line([(-4, y + 1), (4, y + 1)], shade(skin, 1.3))
    p.line([(-8, -2), (-8, 3)], (236, 220, 180))
    return p.sprite()


def luxuries() -> Sprite:
    p = _canvas()
    gold: RGB = (232, 190, 70)
    p.rect(-6, -2, 6, 7, (110, 50, 110))
    p.line([(-6, 2), (6, 2)], gold)
    p.line([(0, -2), (0, 7)], gold)
    p.ellipse(-3, -7, 0, -2, gold)
    p.ellipse(0, -7, 3, -2, gold)
    return p.sprite()


def flax() -> Sprite:
    p = _canvas()
    stalk: RGB = (170, 170, 92)
    for bottom, top in ((-4, -6), (-2, -3), (0, 0), (2, 3), (4, 6)):
        p.line([(bottom, 7), (bottom * 0.2, 2), (top, -5)], stalk)
    p.rect(-2, 1, 2, 3, (156, 112, 64))
    for x, y in ((-6, -6), (-3, -7), (0, -6), (3, -7), (6, -6)):
        p.blob(x - 1, y - 1, x + 1, y + 1, (110, 140, 230))
    return p.sprite()


def hops() -> Sprite:
    p = _canvas()
    cone: RGB = (150, 198, 86)
    p.line([(0, -5), (0, -8)], STEM)
    p.polygon([(1, -7), (6, -8), (5, -4), (2, -5)], LEAF, shade(LEAF, 0.6))
    p.polygon([(0, -5), (5, -1), (4, 3), (0, 7), (-4, 3), (-5, -1)], cone, shade(cone, 0.55))
    for y in (-2, 1, 4):
        p.line([(-3, y), (0, y + 2), (3, y)], shade(cone, 0.7))
    p.point(-2, -1, shade(cone, 1.25))
    return p.sprite()


def machinery() -> Sprite:
    p = _canvas()
    for x, y in ((-1, -8), (-1, 5), (-8, -1), (5, -1), (-6, -6), (3, -6), (-6, 3), (3, 3)):
        p.rect(x, y, x + 2, y + 2, shade(METAL, 0.85))
    p.ellipse(-6, -6, 6, 6, METAL)
    p.ellipse(-3, -3, 3, 3, shade(METAL, 1.12))
    p.blob(-1, -1, 1, 1, (54, 56, 64))
    return p.sprite()


def bricks() -> Sprite:
    p = _canvas()
    red: RGB = (184, 78, 54)
    p.rect(-8, -6, 7, 7, (214, 204, 188))
    for row, y in enumerate((-6, -2, 2)):
        lefts = (-8, 0) if row % 2 == 0 else (-8, -4, 4)
        for i, left in enumerate(lefts):
            right = (lefts[i + 1] if i + 1 < len(lefts) else 8) - 1
            p.rect(left, y, min(right, 7), y + 3, red if (row + i) % 2 else shade(red, 1.1))
    return p.sprite()


def cement() -> Sprite:
    p = _canvas()
    sack: RGB = (190, 190, 186)
    for dx, top, tone in ((-1, 1, 0.9), (1, -6, 1.05)):
        fill = shade(sack, tone)
        p.polygon(
            [
                (-8 + dx, top + 2),
                (-6 + dx, top),
                (6 + dx, top),
                (8 + dx, top + 2),
                (8 + dx, top + 4),
                (6 + dx, top + 6),
                (-6 + dx, top + 6),
                (-8 + dx, top + 4),
            ],
            fill,
            shade(fill, 0.55),
        )
        p.rect(-3 + dx, top + 2, 3 + dx, top + 4, (110, 114, 124))
    p.blob(-7, 5, -3, 8, (222, 222, 218))
    return p.sprite()


def glass() -> Sprite:
    p = _canvas()
    pane: RGB = (170, 220, 236)
    edge: RGB = (92, 146, 176)
    p.polygon([(-4, -8), (7, -8), (7, 4), (-4, 4)], shade(pane, 0.85), edge)
    p.polygon([(-8, -4), (3, -4), (3, 7), (-8, 7)], pane, edge)
    p.line([(-6, 2), (-2, -2)], (246, 252, 255))
    p.line([(-5, 4), (0, -1)], (246, 252, 255))
    return p.sprite()


def beer() -> Sprite:
    p = _canvas()
    amber: RGB = (232, 166, 44)
    p.line([(4, -2), (7, -2), (7, 4), (4, 4)], shade(amber, 0.6))
    p.line([(4, -1), (6, -1), (6, 3), (4, 3)], shade(amber, 0.6))
    p.rect(-6, -4, 4, 7, amber)
    for x in (-4, -1, 2):
        p.line([(x, -2), (x, 5)], shade(amber, 1.18))
    foam: RGB = (250, 248, 238)
    for left, top in ((-7, -7), (-4, -8), (-1, -7)):
        p.ellipse(left, top, left + 5, top + 4, foam)
    return p.sprite()


def vodka() -> Sprite:
    p = _canvas()
    clear: RGB = (222, 236, 244)
    p.rect(-1, -6, 1, -3, clear)
    p.rect(-1, -8, 1, -6, (206, 50, 50))
    p.polygon([(-1, -3), (1, -3), (3, -1), (3, 7), (-3, 7), (-3, -1)], clear, (130, 160, 184))
    p.line([(-2, 0), (-2, 6)], (252, 254, 255))
    p.rect(-3, 1, 3, 4, (250, 250, 248))
    p.line([(-2, 2), (2, 2)], (206, 50, 50))
    return p.sprite()


def leather() -> Sprite:
    p = _canvas()
    hide: RGB = (176, 110, 62)
    p.polygon(
        [
            (-6, -7),
            (-2, -5),
            (2, -5),
            (6, -7),
            (5, -2),
            (7, 1),
            (5, 4),
            (4, 7),
            (0, 5),
            (-4, 7),
            (-5, 4),
            (-7, 1),
            (-5, -2),
        ],
        hide,
        shade(hide, 0.55),
    )
    p.polygon([(-3, -3), (3, -3), (4, 1), (2, 4), (-2, 4), (-4, 1)], shade(hide, 1.15))
    return p.sprite()


def shoes() -> Sprite:
    p = _canvas()
    boot: RGB = (124, 72, 44)
    p.polygon([(-5, -8), (1, -8), (1, 1), (5, 2), (7, 4), (7, 6), (-5, 6)], boot, shade(boot, 0.5))
    p.rect(-6, 6, 7, 7, (44, 36, 34))
    p.line([(-4, -7), (-4, 4)], shade(boot, 1.35))
    for y in (-6, -3, 0):
        p.point(0, y, (226, 206, 160))
    return p.sprite()


def plastics() -> Sprite:
    p = _canvas()
    blue: RGB = (58, 130, 220)
    p.line([(-6, -3), (-4, -8), (4, -8), (6, -3)], (226, 62, 62))
    p.polygon([(-6, -3), (6, -3), (4, 7), (-4, 7)], blue, shade(blue, 0.55))
    p.rect(-7, -4, 7, -2, shade(blue, 1.3))
    p.line([(-3, 0), (-2, 5)], shade(blue, 1.5))
    return p.sprite()


def appliances() -> Sprite:
    p = _canvas()
    white: RGB = (236, 238, 240)
    p.rect(-6, -8, 6, 7, white)
    p.line([(-5, -5), (5, -5)], shade(white, 0.7))
    p.point(3, -7, (90, 200, 120))
    p.point(-4, -7, shade(white, 0.6))
    p.ellipse(-4, -3, 4, 5, (90, 104, 120))
    p.blob(-2, -1, 2, 3, (150, 196, 226))
    return p.sprite()


def paper() -> Sprite:
    p = _canvas()
    sheet: RGB = (244, 242, 234)
    p.polygon([(-7, 4), (2, 4), (4, 7), (-5, 7)], shade(sheet, 0.92), shade(sheet, 0.6))
    p.rect(-7, -5, 3, 4, sheet)
    p.line([(-6, -3), (2, -3)], shade(sheet, 0.86))
    p.ellipse(0, -5, 6, 4, shade(sheet, 0.96))
    p.ellipse(2, -2, 4, 1, shade(sheet, 0.8))
    p.point(3, -1, (120, 100, 80))
    return p.sprite()


def newspapers() -> Sprite:
    p = _canvas()
    page: RGB = (232, 230, 220)
    p.rect(-7, -7, 7, 7, page)
    p.rect(-5, -5, 5, -4, (40, 40, 48))
    p.rect(-5, -2, -1, 2, (150, 150, 162))
    for y in (-2, 0, 2):
        p.line([(1, y), (5, y)], (150, 150, 162))
    p.line([(-5, 4), (5, 4)], (150, 150, 162))
    return p.sprite()


def refined_sugar() -> Sprite:
    p = _canvas()
    white: RGB = (250, 250, 252)
    blue: RGB = (64, 96, 186)
    p.polygon([(-1, -8), (1, -8), (6, 7), (-6, 7)], white, (170, 176, 196))
    p.polygon([(-3.5, 0), (3.5, 0), (6, 7), (-6, 7)], blue, shade(blue, 0.6))
    p.line([(-4, 0), (4, 0)], (236, 214, 150))
    p.line([(-2, -4), (-3, -1)], (216, 222, 236))
    return p.sprite()


def porcelain() -> Sprite:
    p = _canvas()
    china: RGB = (246, 246, 250)
    blue: RGB = (54, 86, 176)
    p.ellipse(-8, 3, 8, 7, china)
    p.line([(5, -3), (7, -3), (7, 1), (5, 1)], shade(china, 0.6))
    p.polygon([(-5, -5), (5, -5), (4, 3), (-4, 3)], china, shade(china, 0.6))
    p.line([(-4, -2), (4, -2)], blue)
    for x in (-3, 0, 3):
        p.point(x, 0, blue)
    p.line([(-5, -5), (5, -5)], blue)
    return p.sprite()


def sheep() -> Sprite:
    p = _canvas()
    fleece: RGB = (244, 242, 234)
    face: RGB = (44, 40, 42)
    for x in (-3, -1, 3, 5):
        p.line([(x, 3), (x, 7)], face)
    for left, top in ((-4, -5), (0, -6), (3, -4), (-4, -1), (0, -2), (3, 0)):
        p.ellipse(left, top, left + 5, top + 5, fleece)
    p.ellipse(-8, -6, -4, 0, face)
    p.line([(-5, -6), (-3, -7)], face)
    p.point(-7, -4, (210, 210, 214))
    return p.sprite()


def meat() -> Sprite:
    p = _canvas()
    red: RGB = (192, 58, 54)
    bone: RGB = (244, 238, 222)
    p.line([(1, -1), (5, -5)], bone)
    p.line([(1, 0), (5, -4)], shade(bone, 0.8))
    p.blob(3, -8, 6, -5, bone)
    p.blob(5, -6, 8, -3, bone)
    p.ellipse(-8, -3, 3, 7, red)
    p.ellipse(-6, -2, 2, 5, shade(red, 1.18))
    p.line([(-5, 1), (-2, 3)], (236, 196, 180))
    p.line([(-4, -1), (-1, 0)], (236, 196, 180))
    return p.sprite()


def livestock() -> Sprite:
    p = _canvas()
    hide: RGB = (150, 92, 56)
    p.line([(0, -6), (0, -8)], CREAM)
    p.line([(7, -6), (7, -8)], CREAM)
    p.point(-1, -3, hide)
    p.ellipse(0, -6, 7, 3, hide)
    p.line([(3, -5), (4, -2)], CREAM)
    p.ellipse(1, 0, 6, 5, (232, 168, 160))
    p.point(2, -3, (30, 28, 30))
    p.point(5, -3, (30, 28, 30))
    face: RGB = (44, 40, 42)
    for left, top in ((-8, -4), (-5, -5), (-2, -4)):
        p.ellipse(left, top, left + 4, top + 4, (244, 242, 234))
    p.line([(-8, 0), (-6, 1)], face)
    p.line([(0, 0), (-2, 1)], face)
    p.ellipse(-6, -1, -2, 7, face)
    p.point(-5, 1, (210, 210, 214))
    p.point(-3, 1, (210, 210, 214))
    return p.sprite()


def _flame(p: Painter, x: float, y: float, size: float, outer: RGB, inner: RGB) -> None:
    """A flame standing on (x, y), `size` pixels tall."""
    s = size / 8
    p.polygon(
        [
            (x, y - 8 * s),
            (x + 3 * s, y - 4 * s),
            (x + 4 * s, y - 1 * s),
            (x + 3 * s, y),
            (x - 3 * s, y),
            (x - 4 * s, y - 1 * s),
            (x - 3 * s, y - 4 * s),
            (x - 1 * s, y - 5 * s),
        ],
        outer,
        shade(outer, 0.62),
    )
    p.polygon([(x, y - 4.5 * s), (x + 2 * s, y - 1.5 * s), (x, y), (x - 2 * s, y - 1.5 * s)], inner)


def gas() -> Sprite:
    p = _canvas()
    _flame(p, 0, 4, 12, (64, 124, 232), (186, 220, 255))
    p.rect(-6, 4, 6, 6, (120, 124, 134))
    for x in (-4, 0, 4):
        p.point(x, 4, (40, 42, 48))
    return p.sprite()


def heating() -> Sprite:
    p = _canvas()
    iron: RGB = (70, 70, 78)
    p.rect(2, -8, 4, -3, shade(iron, 1.2))
    p.rect(-6, -3, 6, 5, iron)
    p.line([(-5, 6), (-5, 7)], iron)
    p.line([(5, 6), (5, 7)], iron)
    p.rect(-4, -1, 2, 4, (36, 30, 30))
    _flame(p, -1, 4, 5, (240, 124, 40), (255, 214, 96))
    return p.sprite()


def boiler_fuel() -> Sprite:
    p = _canvas()
    p.polygon([(-8, 7), (-8, 3), (-6, 1), (-3, 1), (-1, 3), (-1, 7)], (46, 46, 54), DARK_EDGE)
    p.point(-6, 3, (96, 96, 110))
    oil: RGB = (34, 32, 42)
    p.polygon([(4, 0), (1, 4), (7, 4)], oil, DARK_EDGE)
    p.ellipse(1, 2, 7, 7, oil)
    p.point(3, 4, (130, 130, 150))
    _flame(p, 0, 0, 8, (240, 124, 40), (255, 214, 96))
    return p.sprite()


ICONS: dict[str, Callable[[], Sprite]] = {
    "wheat": wheat,
    "apples": apples,
    "bananas": bananas,
    "dates": dates,
    "oranges": oranges,
    "sugar": sugar,
    "honey": honey,
    "fish": fish,
    "cans": cans,
    "bread": bread,
    "lumber": lumber,
    "coal": coal,
    "iron ore": iron_ore,
    "copper": copper,
    "gold": gold,
    "diamonds": diamonds,
    "oil": oil,
    "salt": salt,
    "wool": wool,
    "cotton": cotton,
    "grapes": grapes,
    "coffee": coffee,
    "rubber": rubber,
    "wax": wax,
    "incense": incense,
    "steel": steel,
    "tools": tools,
    "fuel": fuel,
    "chemicals": chemicals,
    "furniture": furniture,
    "clothes": clothes,
    "barrels": barrels,
    "wine": wine,
    "electronics": electronics,
    "cars": cars,
    "guns": guns,
    "tanks": tanks,
    "planes": planes,
    "goods": goods,
    "passengers": passengers,
    "mail": mail,
    "food": food,
    "fruit": fruit,
    "fibre": fibre,
    "clay": clay,
    "limestone": limestone,
    "sand": sand,
    "potatoes": potatoes,
    "cattle": cattle,
    "coke": coke,
    "milk": milk,
    "dairy products": dairy,
    "pots": pots,
    "stove tiles": stove_tiles,
    "castings": castings,
    "sausages": sausages,
    "luxury goods": luxuries,
    "pigs": pigs,
    "chickens": chickens,
    "eggs": eggs,
    "animals": animals,
    "flax": flax,
    "hops": hops,
    "machinery": machinery,
    "bricks": bricks,
    "cement": cement,
    "glass": glass,
    "beer": beer,
    "vodka": vodka,
    "leather": leather,
    "shoes": shoes,
    "plastics": plastics,
    "appliances": appliances,
    "paper": paper,
    "newspapers": newspapers,
    "refined sugar": refined_sugar,
    "porcelain": porcelain,
    "sheep": sheep,
    "meat": meat,
    "livestock": livestock,
    "gas": gas,
    "heating": heating,
    "boiler fuel": boiler_fuel,
}


def draw_icon_sheet() -> tuple[Image.Image, dict[str, list[int]]]:
    """Every icon in rows of COLUMNS squares, and each one's top-left corner on the sheet."""
    rows = (len(ICONS) + COLUMNS - 1) // COLUMNS
    sheet = Image.new("RGBA", (COLUMNS * ICON_PIXELS, rows * ICON_PIXELS), (0, 0, 0, 0))
    places: dict[str, list[int]] = {}
    for index, (name, draw) in enumerate(ICONS.items()):
        row, col = divmod(index, COLUMNS)
        places[name] = [col * ICON_PIXELS, row * ICON_PIXELS]
        sheet.paste(draw().image, tuple(places[name]))
    return sheet, places
