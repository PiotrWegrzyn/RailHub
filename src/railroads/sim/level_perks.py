"""What is special about a tile's level on a board (railroads.sim.tiles), for the marks above its
card: an era its wagons come with, once every supply tile reaches it; an engine it opens for sale;
a kind of bridge it opens for building; the buyers a demand level opens. A level brings nothing
special when it only adds its share."""

from dataclasses import dataclass
from enum import Enum
from functools import cache

from railroads.sim.tiles import (
    BRIDGES_OPENED_BY,
    ERAS,
    KIND_ERAS,
    Board,
    Tile,
    era_news,
    unlocks_of,
)
from railroads.sim.track_rules import BRIDGES, Bridge
from railroads.sim.trains import ENGINES, OPENED_BY


class Perk(Enum):
    ERA = "era"
    ENGINE = "engine"
    BRIDGE = "bridge"
    BUYERS = "buyers"


@dataclass(frozen=True, slots=True)
class LevelPerk:
    kind: Perk
    text: str


def _kinds(names: list[str]) -> str:
    """Kinds of building in words, big and mega ones counted with the plain one."""
    plain = sorted({name.removeprefix("big ").removeprefix("mega ") for name in names})
    named = [f"{name[:-1]}ies" if name.endswith("y") else f"{name}s" for name in plain]
    if len(named) <= 2:
        return " and ".join(named)
    return f"{named[0]}, {named[1]} and {len(named) - 2} more"


def opened_at(tile: Tile, at: int) -> str:
    """Who starts taking what at a demand tile's level, in words; "" for no one."""
    opened = [(kind, need) for lv, kind, need in unlocks_of(tile) if lv == at]
    if not opened:
        return ""
    needs = sorted({need.value for _, need in opened})
    who = _kinds([kind.value for kind, _ in opened])
    return f"{who[0].upper()}{who[1:]} take {', '.join(needs)}"


@cache
def era_text(number: int) -> str:
    """What an era brings and what it waits for, in words; worked out once, as the tiles window
    shows it for every tile every frame."""
    era = ERAS[number - 1]
    text = f"Era {number} comes once every supply tile is at level {era.level}. "
    news = era_news(era).removeprefix("A new era: ")
    text += news[0].upper() + news[1:]
    built = sorted(kind.value for kind, at in KIND_ERAS.items() if at == number)
    if built:
        text += f" New buildings: {', '.join(built)}."
    return text


def perks(board: Board, tile: Tile, level: int) -> list[LevelPerk]:
    """What a tile's level brings on a board beyond its share, the biggest first."""
    return list(_perks(board, tile, level))


@cache
def _perks(board: Board, tile: Tile, level: int) -> tuple[LevelPerk, ...]:
    """perks worked out once for each tile and level: the tiles window marks every level of
    every tile every frame."""
    return tuple(_found_perks(board, tile, level))


def bridge_text(bridge: Bridge) -> str:
    """What a new kind of bridge brings, in words."""
    kind = BRIDGES[bridge]
    most = "one track" if kind.tracks == 1 else f"{kind.tracks} tracks"
    text = f"New bridge: {kind.name}, {most}, trains at {kind.speed_factor:.0%}"
    return text + (", and flyovers over track" if kind.over_track else "")


def _found_perks(board: Board, tile: Tile, level: int) -> list[LevelPerk]:
    bridges = [
        LevelPerk(Perk.BRIDGE, bridge_text(bridge))
        for bridge, opened in BRIDGES_OPENED_BY.items()
        if opened == (board, tile, level)
    ]
    if board == Board.DEMAND:
        buyers = opened_at(tile, level)
        return bridges + ([LevelPerk(Perk.BUYERS, buyers)] if buyers else [])
    found = [
        LevelPerk(Perk.ERA, era_text(number))
        for number, era in enumerate(ERAS, 1)
        if number > 1 and era.level == level
    ]
    found += [
        LevelPerk(Perk.ENGINE, f"New engine for sale: {ENGINES[name].title}")
        for name, (of, at) in OPENED_BY.items()
        if of == tile and at == level
    ]
    return found + bridges
