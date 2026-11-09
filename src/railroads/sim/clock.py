"""Game time: the calendar runs a month every SECONDS_PER_MONTH of play, from January of
START_YEAR to January of END_YEAR, where it stops: 2,400 months, a hundred hours at normal speed.
Play can run faster, at any of SPEEDS; those above MAX_PLAY_SPEED are for testing.
"""

from dataclasses import dataclass

SECONDS_PER_MONTH = 150.0
START_YEAR = 1850
END_YEAR = 2050
SPEEDS = (1, 2, 3, 6, 12, 24)
MAX_PLAY_SPEED = 6
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

    def months_in(self, seconds: float) -> float:
        """How much of a month `seconds` of play make at the current speed."""
        return 0.0 if self.finished else seconds * self.speed / self.seconds_per_month

    def advance(self, seconds: float) -> int:
        """Lets `seconds` of play pass at the current speed; returns how many months turned
        over."""
        if self.finished:
            return 0
        self.seconds += seconds * self.speed
        turned = min(int(self.seconds // self.seconds_per_month), self.months_left)
        self.seconds -= turned * self.seconds_per_month
        self.months += turned
        if self.finished:
            self.seconds = 0.0
        return turned

    @property
    def months_left(self) -> int:
        return (END_YEAR - START_YEAR) * 12 - self.months

    @property
    def finished(self) -> bool:
        return self.months_left <= 0

    @property
    def date(self) -> str:
        return self.date_of(self.months)

    @staticmethod
    def date_of(months: int) -> str:
        """The date this many months after the start."""
        year, month = divmod(months, 12)
        return f"{MONTH_NAMES[month]} {START_YEAR + year}"
