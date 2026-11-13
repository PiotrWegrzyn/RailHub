"""Builds the desktop app for the system it runs on with PyInstaller, and packs it for download:

    uv run --with pyinstaller python desktop/build.py

macOS gives build/desktop/RailHub-macOS.zip (RailHub.app), Windows RailHub-Windows.zip (a folder
with RailHub.exe) and Linux RailHub-Linux.tar.gz (a folder with RailHub). The release workflow
(.github/workflows/release.yml) runs this on Windows and Linux and attaches the archives to a
release. GitHub's Macs are too often all busy, so the Mac app is added by hand once the release
is out, from a clean checkout of its tag:

    git checkout vX.Y.Z
    uv run --with pyinstaller python desktop/build.py --upload

which checks the app starts and adds its archive to that release, replacing one already there."""

import argparse
import os
import platform
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "build" / "desktop"
DIST = OUT / "dist"
NAME = "RailHub"
REPO = "PiotrWegrzyn/RailHub"


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
            "--add-data", f"{ROOT / 'world'}:world",
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


def release_tag() -> str:
    with (ROOT / "pyproject.toml").open("rb") as file:
        return f"v{tomllib.load(file)['project']['version']}"


def upload_problem(tag: str, released: str | None, head: str, changes: str) -> str | None:
    """Why the app built from this checkout must not go into the release `tag`, if it must not:
    `released` is the commit the release was made from, `head` this checkout's and `changes` what
    git status lists."""
    if released is None:
        return f"There is no release {tag} yet: it comes out when master gets that version."
    if released != head:
        return f"{tag} was released from {released[:7]}, not {head[:7]}: git checkout {tag} first."
    if changes:
        return "This checkout has uncommitted changes, which the release does not have."
    return None


def output(*command: str) -> str | None:
    done = subprocess.run(command, check=False, capture_output=True, text=True, cwd=ROOT)
    return done.stdout.strip() if done.returncode == 0 else None


def check_release(tag: str) -> None:
    released = output("gh", "api", f"repos/{REPO}/commits/{tag}", "--jq", ".sha")
    head = output("git", "rev-parse", "HEAD") or ""
    changes = output("git", "status", "--porcelain") or ""
    if problem := upload_problem(tag, released, head, changes):
        sys.exit(problem)


def smoke_test(name: str) -> None:
    environment = {**os.environ, "RAILROADS_SMOKE_TEST": "1"}
    subprocess.run([str(executable(name))], env=environment, check=True)


def upload(tag: str, archive: Path) -> None:
    subprocess.run(
        ["gh", "release", "upload", tag, str(archive), "--clobber", "--repo", REPO], check=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--upload", action="store_true", help="add the app to its version's release"
    )
    upload_it = parser.parse_args().upload
    name = system()
    tag = release_tag()
    if upload_it:
        check_release(tag)
    freeze()
    print(executable(name))
    archive = pack(name)
    print(archive)
    if upload_it:
        smoke_test(name)
        upload(tag, archive)
        print(f"Added {archive.name} to {tag}: https://github.com/{REPO}/releases/tag/{tag}")
