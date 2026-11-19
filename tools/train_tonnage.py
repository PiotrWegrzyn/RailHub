"""Prints, for every engine, the most it pulls in tonnes, what full cars weigh behind it and how
fast it runs with a few typical trains, in cells a month on flat track, next to what the old
drag by car count gave, so the weights in railroads.sim.trains can be tuned.

    uv run python tools/train_tonnage.py"""

from railroads.sim.goods import Good
from railroads.sim.railway import CELLS_PER_MONTH
from railroads.sim.trains import (
    ENGINES,
    SLOWDOWN,
    Engine,
    Kind,
    bigger,
    car_size,
    full_tonnes,
    tare,
)

OLD_DRAG_PER_CAR = 0.08
OLD_LOAD_DRAG = {Kind.LIGHT: 1.5, Kind.HEAVY: 0.5}
OLD_BULK = frozenset({Good.WHEAT, Good.COAL})


def capacity(engine: Engine, good: Good) -> float:
    return bigger(car_size(good), engine.car_levels)


def cells(engine: Engine, tonnes: float) -> float:
    pull = 1.0 - SLOWDOWN[engine.kind] * min(1.0, tonnes / engine.tonnes)
    return CELLS_PER_MONTH * engine.speed * pull


def old_cells(engine: Engine, cars: int, good: Good | None) -> float:
    heft = 144.0 / 55.0 if good in OLD_BULK else 1.0
    weight = cars * (1.0 + (0.0 if good is None else heft))
    if engine.express:
        weight += 1.0
    drag = OLD_DRAG_PER_CAR * 5 / engine.hauls * OLD_LOAD_DRAG.get(engine.kind, 1.0)
    return CELLS_PER_MONTH * engine.speed / (1.0 + drag * weight)


def train(engine: Engine, good: Good, full: bool) -> tuple[int, float, float, float]:
    """As many cars of a good as the engine takes and can pull full: how many, their weight, and
    the speed now and with the old drag."""
    size = capacity(engine, good)
    car = full_tonnes(good, size) if full else tare(size)
    dining = tare(car_size(Good.PASSENGERS)) if engine.express else 0.0
    fit = int((engine.tonnes - dining + 1e-9) // car) if car else engine.cars
    count = min(engine.cars, fit)
    tonnes = dining + count * car
    return count, tonnes, cells(engine, tonnes), old_cells(engine, count, good if full else None)


def column(engine: Engine, good: Good, full: bool = True) -> str:
    count, tonnes, now, old = train(engine, good, full)
    if count == 0:
        return "—"
    return f"{count}× {tonnes:.0f} t: **{now:.1f}** ({old:.1f})"


def main() -> None:
    print("Full car weights behind an engine without bigger cars, in tonnes:")
    for good in (Good.COAL, Good.WHEAT, Good.STEEL, Good.OIL, Good.GOODS, Good.PASSENGERS):
        size = car_size(good)
        print(
            f"  {good.value}: {size:.0f} a car, {tare(size):.1f} t empty, "
            f"{full_tonnes(good, size):.1f} t full"
        )
    print()
    print(
        "| Engine | Year | Kind | Cars | Max pull | Alone | Empty cars | Full general goods "
        "| Full wheat | Full coal |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")
    for engine in ENGINES.values():
        alone = CELLS_PER_MONTH * engine.speed
        if engine.express:
            empty = column(engine, Good.PASSENGERS, full=False)
            rows = [column(engine, Good.PASSENGERS), "—", "—"]
        else:
            empty = column(engine, Good.GOODS, full=False)
            rows = [column(engine, good) for good in (Good.GOODS, Good.WHEAT, Good.COAL)]
        print(
            f"| {engine.title} | {engine.year} | {engine.kind.value} | {engine.hauls} "
            f"| {engine.tonnes:.0f} t | {alone:.1f} | {empty} | " + " | ".join(rows) + " |"
        )


if __name__ == "__main__":
    main()
