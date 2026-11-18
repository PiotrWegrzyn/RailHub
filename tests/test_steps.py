from railroads.sim.steps import Steps, budgeted, finish


def counted(steps: int, done: list[int]) -> Steps[str]:
    for step in range(steps):
        done.append(step)
        yield
    return "end"


def pauses(steps: Steps[str]) -> int:
    count = 0
    while True:
        try:
            next(steps)
        except StopIteration:
            return count
        count += 1


def test_steps_within_the_budget_run_without_a_pause() -> None:
    done: list[int] = []

    paused = pauses(budgeted(counted(5, done), 60.0))

    assert paused == 0
    assert done == [0, 1, 2, 3, 4]


def test_without_a_budget_every_step_pauses() -> None:
    done: list[int] = []

    paused = pauses(budgeted(counted(5, done), 0.0))

    assert paused == 5


def test_budgeted_steps_return_what_the_steps_return() -> None:
    steps = counted(3, [])

    result = finish(budgeted(steps, 0.0))

    assert result == "end"
