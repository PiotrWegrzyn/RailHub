"""How long buildings last starving before they die out: homes HOMES_DIE_AFTER_MONTHS, the rest
DIE_AFTER_MONTHS unless their ladder says, and those that never die (UNDYING)."""

from railroads.sim.clock import MONTHS_PER_YEAR
from railroads.sim.constructions import (
    ONE_OFFS,
    BuildingKind,
)
from railroads.sim.goods import HOMES, MAKES, RECIPES
from railroads.sim.ladders import (
    LADDERS,
    MILITARY,
)

# How long buildings last starving: homes, and everything else unless its ladder says
# (Ladder.starving_years).
HOMES_DIE_AFTER_MONTHS = 10 * MONTHS_PER_YEAR


DIE_AFTER_MONTHS = 30 * MONTHS_PER_YEAR


_STARVING_MONTHS = {
    kind: ladder.starving_years * MONTHS_PER_YEAR
    for ladder in LADDERS
    if ladder.starving_years is not None
    for kind in ladder.kinds
}


def dies_after(kind: BuildingKind) -> int:
    """How many months buildings of a kind last starving."""
    if kind in HOMES:
        return HOMES_DIE_AFTER_MONTHS
    return _STARVING_MONTHS.get(kind, DIE_AFTER_MONTHS)


# Producers of what the land gives, which stand idle with nobody to sell to rather than dying out:
# the land decides where they are, and so how many people and what industry it can keep; and the
# one-off buildings and the soldiers, which wait for what they need however long it takes.
UNDYING = (
    frozenset(kind for kind in MAKES if kind not in RECIPES) | ONE_OFFS | frozenset(MILITARY.kinds)
)
