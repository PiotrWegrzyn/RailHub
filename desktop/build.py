"""Builds the desktop app for the system it runs on with PyInstaller, and packs it for download:

    uv run --with pyinstaller python desktop/build.py

macOS gives build/desktop/RailHub-macOS.zip (RailHub.app), Windows RailHub-Windows.zip (a folder
with RailHub.exe) and Linux RailHub-Linux.tar.gz (a folder with RailHub). The release workflow
(.github/workflows/release.yml) runs this on each system and attaches the archives to a release."""

import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "build" / "desktop"
DIST = OUT / "dist"
NAME = "RailHub"


def system() -> str:
    return {"Darwin": "macOS", "Windows": "Windows"}.get(platform.system(), "Linux")


def freeze() -> None:
    subprocess.run(
        [
            sys.executable, "-m", "PyInstaller",
            "--noconfirm", "--log-level", "WARN",
            "--name", NAME,
            "--windowed",
            "--distpath", str(DIST),
            "--workpath", str(OUT / "work"),
            "--specpath", str(OUT),
            "--collect-submodules", "railroads",
            "--collect-data", "railroads",
            "--collect-data", "imgui_bundle",
            "--copy-metadata", "railroads",
            str(ROOT / "desktop" / "railroads_app.py"),
        ],
        check=True,
    )  # fmt: skip


def pack(name: str) -> Path:
    archive = OUT / f"{NAME}-{name}"
    if name == "macOS":
        target = archive.with_suffix(".zip")
        target.unlink(missing_ok=True)
        app = DIST / f"{NAME}.app"
        subprocess.run(["ditto", "-c", "-k", "--keepParent", str(app), str(target)], check=True)
        return target
    form = "zip" if name == "Windows" else "gztar"
    return Path(shutil.make_archive(str(archive), form, DIST, NAME))


def executable(name: str) -> Path:
    if name == "macOS":
        return DIST / f"{NAME}.app" / "Contents" / "MacOS" / NAME
    return DIST / NAME / (f"{NAME}.exe" if name == "Windows" else NAME)


if __name__ == "__main__":
    name = system()
    freeze()
    print(executable(name))
    print(pack(name))
