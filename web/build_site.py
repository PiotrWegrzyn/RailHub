"""Builds the game as a static web site that needs no other server: the page, the web worker that
paints the map (painter.js), the game's wheel, Pyodide with numpy, Pillow, micropip and
imgui-bundle beside it, and the world's map data (world/), of which a game downloads only the few
files its map needs. Downloads are kept in a cache, so only the first build fetches them.

    uv run python web/build_site.py
    python -m http.server -d build/site 8000

Any static host can serve build/site as it is."""

import hashlib
import shutil
import subprocess
import tomllib
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
SITE = ROOT / "build" / "site"
CACHE = ROOT / "build" / "web-cache"

PYODIDE_URL = "https://cdn.jsdelivr.net/pyodide/v314.0.0/full/"
PYODIDE_FILES = (
    "pyodide.js",
    "pyodide.mjs",
    "pyodide.asm.mjs",
    "pyodide.asm.wasm",
    "python_stdlib.zip",
    "pyodide-lock.json",
    "micropip-0.11.1-py3-none-any.whl",
    "numpy-2.4.3-cp314-cp314-pyemscripten_2026_0_wasm32.whl",
    "pillow-12.2.0-cp314-cp314-pyemscripten_2026_0_wasm32.whl",
)
IMGUI_BUNDLE_URL = (
    "https://files.pythonhosted.org/packages/8e/53/"
    "ba4f4b5b0f03cb23b18d9f8e947e24e48e33c8426ab003e3e4c95031cd3f/"
    "imgui_bundle-1.93.0-cp314-cp314-pyemscripten_2026_0_wasm32.whl"
)
IMGUI_BUNDLE_SHA256 = "9bba2f050057752a52b9073a0abe5cfe5e27e5ed53fec5ace07ea3166be4c28c"


def cached(url: str, sha256: str | None = None) -> Path:
    path = CACHE / url.rsplit("/", 1)[1]
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {path.name}")
        with urllib.request.urlopen(url) as response:
            data = response.read()
        if sha256 is not None and hashlib.sha256(data).hexdigest() != sha256:
            raise RuntimeError(f"{path.name} does not match its known checksum")
        path.write_bytes(data)
    return path


def build() -> None:
    shutil.rmtree(SITE, ignore_errors=True)
    (SITE / "pyodide").mkdir(parents=True)
    (SITE / "wheels").mkdir()
    for name in PYODIDE_FILES:
        shutil.copy(cached(PYODIDE_URL + name), SITE / "pyodide" / name)
    shutil.copy(cached(IMGUI_BUNDLE_URL, IMGUI_BUNDLE_SHA256), SITE / "wheels")
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(SITE / "wheels")], cwd=ROOT, check=True
    )
    (SITE / "wheels" / ".gitignore").unlink(missing_ok=True)
    version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    for name in ("index.html", "painter.js", "measure.py"):
        text = (WEB / name).read_text().replace("__VERSION__", version)
        (SITE / name).write_text(text)
    shutil.copytree(ROOT / "world", SITE / "world")
    (SITE / ".nojekyll").touch()
    size = sum(path.stat().st_size for path in SITE.rglob("*") if path.is_file())
    print(f"Built {SITE} ({size / 1e6:.1f} MB)")


if __name__ == "__main__":
    build()
