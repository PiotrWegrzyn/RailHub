"""Prints what a unit of every good weighs, how many a car of each generation holds and what it
weighs full, and, for every engine, the most it pulls in tonnes and how fast it runs with a few
typical trains of its newest cars, in cells a month on flat track, so the weights in
railroads.sim.trains can be tuned.

    uv run python tools/train_tonnage.py"""

from railroads.sim.goods import Good
from railroads.sim.railway import CELLS_PER_MONTH
from railroads.sim.trains import (
    BASE_GENERATION,
    ENGINES,
    GENERATIONS,
    SLOWDOWN,
    Engine,
    car_size,
    full_tonnes,
    tare_of,
    unit_tonnes,
)


def cells(engine: Engine, tonnes: float, generation: int) -> float:
    pull = 1.0 - SLOWDOWN[engine.kind] * min(1.0, tonnes / engine.pull_at(generation))
    return CELLS_PER_MONTH * engine.speed * pull


def column(engine: Engine, good: Good, generation: int, full: bool = True) -> str:
    """As many cars of a good of a generation as the engine takes and can pull full: how many,
    their weight, and the speed."""
    size = car_size(good, generation)
    if not size:
        return "—"
    car = full_tonnes(good, size, generation) if full else tare_of(good, generation)
    dining = tare_of(Good.PASSENGERS, generation) if engine.express else 0.0
    most = engine.pull_at(generation)
    count = min(engine.cars, int((most - dining + 1e-9) // car))
    if count == 0:
        return "—"
    tonnes = dining + count * car
    return f"{count}× {tonnes:.0f} t: **{cells(engine, tonnes, generation):.1f}**"


def goods_table() -> None:
    generations = range(1, GENERATIONS + 1)
    print("| Good | Unit | " + " | ".join(f"G{g}" for g in generations) + " | Full G2 car |")
    print("|---|---|" + "---|" * GENERATIONS + "---|")
    for good in sorted(Good, key=lambda good: -full_tonnes(good, car_size(good))):
        sizes = " | ".join(
            f"{car_size(good, g):.0f}" if car_size(good, g) else "—" for g in generations
        )
        full = full_tonnes(good, car_size(good, BASE_GENERATION))
        full_text = f"{full:.1f} t" if car_size(good) else "—"
        print(f"| {good.value} | {unit_tonnes(good) * 1000:g} kg | {sizes} | {full_text} |")


def engines_table() -> None:
    print(
        "| Engine | Year | Kind | Cars | Newest cars | Pulls with G2 | Pulls with its newest "
        "| Alone | Empty cars | Full general goods | Full wheat | Full coal |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for engine in ENGINES.values():
        top = engine.top_generation
        alone = CELLS_PER_MONTH * engine.speed
        if engine.express:
            rows = [
                column(engine, Good.PASSENGERS, top, full=False),
                column(engine, Good.PASSENGERS, top),
                "—",
                "—",
            ]
        else:
            rows = [column(engine, Good.GOODS, top, full=False)]
            rows += [column(engine, good, top) for good in (Good.GOODS, Good.WHEAT, Good.COAL)]
        print(
            f"| {engine.title} | {engine.year} | {engine.kind.value} | {engine.hauls} | G{top} "
            f"| {engine.pull_at(BASE_GENERATION):.0f} t | {engine.tonnes:.0f} t | {alone:.1f} | "
            + " | ".join(rows)
            + " |"
        )


def main() -> None:
    goods_table()
    print()
    engines_table()


if __name__ == "__main__":
    main()
