"""Game time: the calendar runs a month every SECONDS_PER_MONTH of play, from January of the year
the map's history left off, and on for ever. A map's history begins in HISTORY_START, so the
usual thousand years of it leave off in START_YEAR; play can begin as late as the year before
END_YEAR, and from START_YEAR to END_YEAR is a hundred hours at normal speed. Play can run faster,
at any of SPEEDS.
"""

from dataclasses import dataclass

SECONDS_PER_MONTH = 150.0
START_YEAR = 1850
# The year every map's history begins: by default a thousand years before play.
HISTORY_START = 850
END_YEAR = 2050
SPEEDS = (1, 2, 3, 6)
MONTH_NAMES = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)


@dataclass
class GameClock:
    months: int = 0
    seconds: float = 0.0
    seconds_per_month: float = SECONDS_PER_MONTH
    # How many times faster than normal the game runs.
    speed: int = 1
    # The year play began, which is where the map's history left off.
    first_year: int = START_YEAR

    def months_in(self, seconds: float) -> float:
        """How much of a month `seconds` of play make at the current speed."""
        return seconds * self.speed / self.seconds_per_month

    def advance(self, seconds: float) -> int:
        """Lets `seconds` of play pass at the current speed; returns how many months turned
        over."""
        self.seconds += seconds * self.speed
        turned = int(self.seconds // self.seconds_per_month)
        self.seconds -= turned * self.seconds_per_month
        self.months += turned
        return turned

    @property
    def date(self) -> str:
        return self.date_of(self.months)

    def date_of(self, months: int) -> str:
        """The date this many months after play began."""
        year, month = divmod(months, 12)
        return f"{MONTH_NAMES[month]} {self.first_year + year}"
