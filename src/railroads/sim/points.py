"""The kinds of points buildings give (railroads.sim.ladders says which give what)."""

from railroads.sim.keyed import Keyed


class Points(Keyed):
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
