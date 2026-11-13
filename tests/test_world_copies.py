import subprocess
import sys


def loads_imgui(module: str) -> bool:
    script = f"import sys, {module}; print(any(m.startswith('imgui_bundle') for m in sys.modules))"
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    return result.stdout.strip() == "True"


def test_the_pixel_world_paints_without_imgui() -> None:
    loaded = loads_imgui("railroads.ui.pixel.world")

    assert not loaded
