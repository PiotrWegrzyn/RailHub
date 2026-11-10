"""What happened each year, for the reports: the cars trains delivered of every good, the stations
the player had and the population of the whole map.

Growth closes each year as it ends, from HISTORY_START on, with the stations standing and the
population then; the railway counts a car as delivered whenever it leaves cargo at a station other
than the one it was loaded at.
"""

from collections import Counter
from dataclasses import dataclass, field

from railroads.sim.goods import Good


@dataclass
class History:
    delivered: dict[int, Counter[Good]] = field(default_factory=dict)
    stations: dict[int, int] = field(default_factory=dict)
    population: dict[int, int] = field(default_factory=dict)

    def deliver(self, year: int, good: Good, cars: int = 1) -> None:
        self.delivered.setdefault(year, Counter())[good] += cars

    def close_year(self, year: int, stations: int, population: int) -> None:
        self.stations[year] = stations
        self.population[year] = population
