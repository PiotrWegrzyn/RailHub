"""Game time: the calendar runs a month every SECONDS_PER_MONTH of play, from January of
START_YEAR to January of END_YEAR, where it stops: 2,400 months, a hundred hours at normal speed.
"""

from dataclasses import dataclass

SECONDS_PER_MONTH = 150.0
START_YEAR = 1850
END_YEAR = 2050
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

    def advance(self, seconds: float) -> int:
        """Lets `seconds` of play pass; returns how many months turned over."""
        if self.finished:
            return 0
        self.seconds += seconds
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
        year, month = divmod(self.months, 12)
        return f"{MONTH_NAMES[month]} {START_YEAR + year}"
