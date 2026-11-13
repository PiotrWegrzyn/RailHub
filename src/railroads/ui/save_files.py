"""Save files carried between the game in a browser and on a computer: Export file hands a save's
file to the player, Import file brings one in (railroads.saves.Saves.export and bring_in).

In a browser the file is downloaded, and picked with the page's own file chooser; on a computer
the system's save and open dialogs ask where (imgui-bundle's portable file dialogs). Both answer
later, so SaveFiles.poll is asked each frame for what has come back."""

import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from imgui_bundle import icons_fontawesome_4 as fa
from imgui_bundle import imgui

from railroads.saves import SUFFIX, SaveError, Saves

IN_BROWSER = sys.platform == "emscripten"


@dataclass
class Picked:
    """A file the player picked to bring in."""

    name: str
    data: bytes


class BrowserFiles:
    """Downloads and the file chooser of the page the game runs in. Bytes cross between Python's
    memory and the page's by their address in the one memory both share, as Pyodide's own
    conversions break once that memory passes 2 GiB."""

    def __init__(self) -> None:
        self.picked: list[Picked] = []
        self._kept: list[Any] = []

    @staticmethod
    def _memory() -> Any:
        import js

        return js.pyodide._module.HEAPU8

    def save_as(self, name: str, data: bytes) -> None:
        import js
        from pyodide.ffi import create_once_callable

        array = np.frombuffer(data, np.uint8)
        address = int(array.__array_interface__["data"][0])
        copy = self._memory().slice(address, address + len(array))
        blob = js.Blob.new([copy])
        link = js.document.createElement("a")
        link.href = js.URL.createObjectURL(blob)
        link.download = f"{name}{SUFFIX}"
        link.click()
        url = link.href
        js.setTimeout(create_once_callable(lambda: js.URL.revokeObjectURL(url)), 60_000)

    def pick(self) -> None:
        import js
        from pyodide.ffi import create_once_callable, create_proxy

        chooser = js.document.createElement("input")
        chooser.type = "file"
        chooser.accept = SUFFIX

        def chosen(_: Any) -> None:
            if chooser.files.length == 0:
                return
            file = chooser.files.item(0)
            file.arrayBuffer().then(create_once_callable(lambda buffer: self._read(file, buffer)))

        handler = create_proxy(chosen)
        self._kept.append(handler)
        chooser.addEventListener("change", handler)
        chooser.click()

    def _read(self, file: Any, buffer: Any) -> None:
        import js

        array = np.empty(buffer.byteLength, np.uint8)
        address = int(array.__array_interface__["data"][0])
        self._memory().set(js.Uint8Array.new(buffer), address)
        self.picked.append(Picked(str(file.name), array.tobytes()))

    def poll(self) -> list[Picked]:
        picked, self.picked = self.picked, []
        return picked


@dataclass
class DesktopFiles:
    """The system's save and open dialogs, which run beside the game's window."""

    saving: list[tuple[Any, bytes]] = field(default_factory=list)
    opening: list[Any] = field(default_factory=list)

    def save_as(self, name: str, data: bytes) -> None:
        from imgui_bundle import portable_file_dialogs as pfd

        start = str(Path.home() / f"{name}{SUFFIX}")
        dialog = pfd.save_file("Export save", start, ["RailHub saves", f"*{SUFFIX}"])
        self.saving.append((dialog, data))

    def pick(self) -> None:
        from imgui_bundle import portable_file_dialogs as pfd

        dialog = pfd.open_file("Import save", str(Path.home()), ["RailHub saves", f"*{SUFFIX}"])
        self.opening.append(dialog)

    def poll(self) -> list[Picked]:
        picked = []
        for dialog, data in list(self.saving):
            if dialog.ready(0):
                self.saving.remove((dialog, data))
                if target := dialog.result():
                    path = Path(target)
                    if path.suffix != SUFFIX:
                        path = path.with_name(path.name + SUFFIX)
                    path.write_bytes(data)
        for dialog in list(self.opening):
            if dialog.ready(0):
                self.opening.remove(dialog)
                for chosen in dialog.result():
                    path = Path(chosen)
                    picked.append(Picked(path.name, path.read_bytes()))
        return picked


def save_files() -> BrowserFiles | DesktopFiles:
    return BrowserFiles() if IN_BROWSER else DesktopFiles()


def draw_file_buttons(
    files: BrowserFiles | DesktopFiles,
    saves: Saves,
    selected: str | None,
    width: float,
    brought_in: Callable[[str], None],
) -> str | None:
    """Export file and Import file side by side; returns a message about what happened, if
    anything did, and tells `brought_in` the name a save came in under."""
    message = None
    half = (width - imgui.get_style().item_spacing.x) / 2
    imgui.begin_disabled(selected is None)
    if imgui.button(f"{fa.ICON_FA_DOWNLOAD}  Export file", (half, 0)) and selected:
        try:
            files.save_as(selected, saves.export(selected))
        except (SaveError, OSError) as error:
            message = str(error)
    imgui.end_disabled()
    if imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled.value):
        imgui.set_tooltip("Keep this save as a file, to load it in the game online or offline.")
    imgui.same_line()
    if imgui.button(f"{fa.ICON_FA_UPLOAD}  Import file", (half, 0)):
        files.pick()
    if imgui.is_item_hovered():
        imgui.set_tooltip("Add a save file exported from the game online or offline.")
    try:
        for picked in files.poll():
            name = saves.bring_in(picked.name, picked.data)
            brought_in(name)
            message = f"Imported '{name}'."
    except (SaveError, OSError) as error:
        message = str(error)
    return message
