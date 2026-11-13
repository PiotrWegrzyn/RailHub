"""The entry point of the desktop apps PyInstaller builds (desktop/build.py).

With RAILROADS_SMOKE_TEST set, it imports every module of the game instead of opening a window and,
if the variable names a save file, loads it, so a build can be checked where there is no screen."""

import importlib
import os
import pkgutil
import sys
from pathlib import Path


def smoke_test(save: str) -> None:
    import railroads

    for module in pkgutil.walk_packages(railroads.__path__, "railroads."):
        if not module.name.endswith("__main__"):
            importlib.import_module(module.name)
    if save != "1":
        from railroads.saves import Saves

        path = Path(save)
        Saves(path.parent).load(path.stem)
    print("smoke test passed")


if __name__ == "__main__":
    if save := os.environ.get("RAILROADS_SMOKE_TEST"):
        smoke_test(save)
        sys.exit()
    from railroads.ui.app import main

    main()
