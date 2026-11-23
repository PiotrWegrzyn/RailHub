"""What a new game is set up with: its map (MapChoice), seed, start year, cash, stand-in years and
difficulty."""

from dataclasses import dataclass
from enum import StrEnum

from railroads import maps
from railroads.sim.clock import (
    START_YEAR,
)
from railroads.sim.company import Difficulty
from railroads.sim.finance import START_CASH


class MapChoice(StrEnum):
    """Each baked map by its title in railroads.maps.START_MAPS, and a random place. Saves keep
    the value, so a title once used must not change."""

    POLAND = "Poland"
    GERMANY = "Germany"
    FRANCE = "France: Paris and the Loire"
    BRITAIN = "Great Britain"
    ITALY = "Italy: Bologna and Florence"
    MOSCOW = "Moscow"
    CONSTANTINOPLE = "Constantinople"
    ALPS = "The Alps"
    NORWAY = "Norwegian fjords"
    US_NORTHEAST = "US Northeast"
    CALIFORNIA = "California"
    COLORADO = "Colorado Rockies"
    CHILE = "Central Chile"
    JAPAN = "Japan: Osaka to Nagoya"
    EAST_CHINA = "China: the Yangtze delta"
    SOUTHEAST_AUSTRALIA = "Southeast Australia"
    LANDS = "Every land (test map)"
    WORLD = "A random place in the world"
    TUTORIAL = "Tutorial"

    @property
    def start_map(self) -> maps.StartMap | None:
        """The baked map this choice plays on, or None for a random place."""
        return maps.start_map(self.value)

    @property
    def offered(self) -> bool:
        """Whether the new game page lists it among its maps: the baked maps only."""
        return self.start_map is not None


@dataclass(frozen=True)
class Settings:
    """What a new game is set up with."""

    seed: int
    map: MapChoice = MapChoice.POLAND
    # Towns grow from HISTORY_START until this year, when play begins (see allowed_start_year).
    start_year: int = START_YEAR
    cash: float = START_CASH
    # Years a stand-in player builds and runs the railway before the player takes over.
    stand_in_years: int = 0
    # Which target the company is set for 1 January of END_YEAR (railroads.sim.company).
    difficulty: Difficulty = Difficulty.NORMAL
    # Whether the waiting screen shows the map as its towns grow (railroads.ui.history_pictures),
    # which makes setting it up take longer.
    show_map_growing: bool = False
