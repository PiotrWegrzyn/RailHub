"""Numbers the game works out, written as its text writes them, so the text follows the numbers
when they change."""

from collections.abc import Iterable
from fractions import Fraction

_NUMBERS = (
    "no",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
)
_PARTS = {2: "half", 3: "third", 4: "quarter", 5: "fifth", 6: "sixth", 7: "seventh"}
_TIMES = {2: "twice", 3: "three times"}


def dollars(amount: float, cents: bool = False) -> str:
    """An amount of money as the game shows it: $15,000, or -$1,200 in debt; with cents if
    `cents`."""
    if cents:
        return f"${amount:,.2f}"
    sign = "-" if amount < -0.5 else ""
    return f"{sign}${abs(amount):,.0f}"


def tonnes_text(tonnes: float) -> str:
    """Tonnes to a tenth, so cars that look as if they fit an engine's pull do."""
    return f"{tonnes:,.1f}".removesuffix(".0")


def plural(word: str, n: int) -> str:
    """A word for `n` of something: "month" for one, "months" for any other number."""
    return word if n == 1 else f"{word}s"


def times_over(name: str, n: int) -> str:
    """A name with how many of it there are, past one: "farm", "farm x3"."""
    return name if n == 1 else f"{name} x{n}"


def counted(n: int) -> str:
    """A count in words up to twelve, in digits past it."""
    return _NUMBERS[n] if 0 <= n < len(_NUMBERS) else f"{n:,}"


def share_of(share: float) -> str:
    """A share as words where it is a simple part ("half", "a quarter", "three quarters"), as a
    percentage otherwise."""
    part = Fraction(share).limit_denominator(max(_PARTS))
    if abs(float(part) - share) < 1e-9 and part.denominator in _PARTS and 0 < part < 1:
        name = _PARTS[part.denominator]
        if part.denominator == 2:
            return name
        if part.numerator == 1:
            return f"a {name}"
        return f"{counted(part.numerator)} {name}s"
    return f"{share:.0%}"


def times(factor: float) -> str:
    """How many times as much: "twice", "three times", "2.5 times"."""
    if factor == int(factor) and int(factor) in _TIMES:
        return _TIMES[int(factor)]
    return f"{factor:g} times"


def either(items: Iterable[object]) -> str:
    """Items as a choice: "4 or 5", "3, 4 or 5"."""
    words = [str(item) for item in items]
    return words[0] if len(words) == 1 else f"{', '.join(words[:-1])} or {words[-1]}"
