import importlib.util
import tomllib
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_build() -> ModuleType:
    spec = importlib.util.spec_from_file_location("desktop_build", ROOT / "desktop" / "build.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build = load_build()


def test_the_release_tag_is_the_version_in_pyproject() -> None:
    version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]

    tag = build.release_tag()

    assert tag == f"v{version}"


def test_a_clean_checkout_of_the_released_commit_may_be_uploaded() -> None:
    problem = build.upload_problem("v0.3.0", "abc1234567", "abc1234567", "")

    assert problem is None


@pytest.mark.parametrize(
    ("released", "head", "changes", "said"),
    [
        (None, "abc1234567", "", "no release v0.3.0"),
        ("abc1234567", "def7654321", "", "git checkout v0.3.0"),
        ("abc1234567", "abc1234567", " M src/railroads/ui/app.py", "uncommitted"),
    ],
)
def test_an_app_that_is_not_the_release_is_not_uploaded(
    released: str | None, head: str, changes: str, said: str
) -> None:
    problem = build.upload_problem("v0.3.0", released, head, changes)

    assert problem is not None and said in problem
