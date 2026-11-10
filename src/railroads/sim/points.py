"""The kinds of points buildings give (railroads.sim.ladders says which give what)."""

from enum import Enum


class Points(Enum):
    POPULATION = "population"
    FOOD = "food"
    ORE = "ore"
    TIMBER = "timber"
    TRADE = "trade"
    INDUSTRY = "industry"
    CORPORATE = "corporate"
    MILITARY = "military"
    AIR = "air travel"
    WORSHIP = "worship"
