# The map is a grid of square tiles. Everything is built in whole tiles (track is 1 tile wide,
# a house 1 tile, an industry a few tiles square), and is drawn larger than life on purpose.
# KM_PER_TILE is the one number that turns tile distances into kilometres for the economy:
# cargo payment, vehicle speeds and running costs.
KM_PER_TILE = 2.0


def tiles_to_km(tiles: float) -> float:
    return tiles * KM_PER_TILE


def km_to_tiles(km: float) -> float:
    return km / KM_PER_TILE
