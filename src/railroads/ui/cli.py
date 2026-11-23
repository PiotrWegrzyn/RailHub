"""The game started from the command line: its options, and the settings of a new game they ask for."""

import argparse

from imgui_bundle import immapp

from railroads import maps
from railroads.newgame import EARLIEST_START_YEAR, allowed_start_year
from railroads.saves import (
    Saves,
)
from railroads.settings import MapChoice, Settings
from railroads.sim.clock import (
    END_YEAR,
    HISTORY_START,
    MONTHS_PER_YEAR,
    START_YEAR,
)
from railroads.sim.growth import MONTHS_BEFORE_PLAY
from railroads.ui.app import WINDOW_SIZE, Shell
from railroads.ui.main_menu import (
    random_seed,
)


def settings_from(args: argparse.Namespace) -> Settings | None:
    """The settings the command line asks for, or None to open the main menu."""
    if (
        args.seed is None
        and not args.poland
        and args.map is None
        and args.years is None
        and args.simulate is None
    ):
        return None
    years = MONTHS_BEFORE_PLAY // MONTHS_PER_YEAR if args.years is None else args.years
    baked = maps.start_map("poland" if args.poland else args.map or "")
    return Settings(
        seed=random_seed() if args.seed is None else args.seed,
        map=MapChoice.WORLD if baked is None else MapChoice(baked.title),
        start_year=allowed_start_year(HISTORY_START + years),
        stand_in_years=0 if args.simulate is None else args.simulate,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """The command line's options, refusing a history that begins play before
    EARLIEST_START_YEAR."""
    parser = argparse.ArgumentParser(
        prog="railroads",
        description="A railroad tycoon game. Without options it opens the main menu; with any, "
        "it starts that game straight away.",
    )
    parser.add_argument(
        "--seed", type=int, help="replay the start location and buildings of this seed"
    )
    parser.add_argument("--poland", action="store_true", help="play on the baked map of Poland")
    parser.add_argument(
        "--map",
        choices=[start.key for start in maps.START_MAPS],
        help="play on this baked map of a real place",
    )
    parser.add_argument(
        "--years",
        type=int,
        help=f"how many years towns grow, from {HISTORY_START}, before play starts in the year "
        f"they reach: at least {EARLIEST_START_YEAR - HISTORY_START}, so play begins in "
        f"{EARLIEST_START_YEAR} or later ({START_YEAR - HISTORY_START} is suggested and the "
        "default)",
    )
    parser.add_argument(
        "--simulate",
        type=int,
        nargs="?",
        const=END_YEAR - START_YEAR,
        metavar="YEARS",
        help="let a stand-in player build stations and serve the towns for this many years "
        "(all of play if left out), then open the map where it stopped",
    )
    args = parser.parse_args(argv)
    if args.years is not None and HISTORY_START + args.years < EARLIEST_START_YEAR:
        parser.error(
            f"--years must be at least {EARLIEST_START_YEAR - HISTORY_START}, so play begins in "
            f"{EARLIEST_START_YEAR} or later; {START_YEAR - HISTORY_START} (play beginning in "
            f"{START_YEAR}) is suggested"
        )
    return args


def main() -> None:
    args = parse_args()
    shell = Shell(Saves(), settings_from(args))
    immapp.run(
        gui_function=shell.gui,
        window_title="RailHub",
        window_size=(int(WINDOW_SIZE[0]), int(WINDOW_SIZE[1])),
        with_implot=True,
    )
