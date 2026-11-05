"""Make a TileMap from any map data: lay a grid over an area, then fill in one property at a time.

Each property comes from a strategy, so changing where a property comes from (real elevation, a
generator's output, a simple rule) means swapping one strategy, not changing the rest.
"""
