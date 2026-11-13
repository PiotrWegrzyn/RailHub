from dataclasses import dataclass
from pathlib import Path

from railroads.ui.save_files import DesktopFiles, Picked


@dataclass
class Dialog:
    answer: object
    done: bool = True

    def ready(self, timeout: int = 20) -> bool:
        return self.done

    def result(self) -> object:
        return self.answer


def test_an_export_is_written_where_the_save_dialog_says_with_the_save_ending(
    tmp_path: Path,
) -> None:
    files = DesktopFiles(saving=[(Dialog(str(tmp_path / "my line")), b"save bytes")])

    files.poll()

    assert (tmp_path / "my line.save").read_bytes() == b"save bytes"
    assert files.saving == []


def test_a_file_picked_in_the_open_dialog_comes_back_once(tmp_path: Path) -> None:
    (tmp_path / "from the web.save").write_bytes(b"save bytes")
    files = DesktopFiles(opening=[Dialog([str(tmp_path / "from the web.save")])])

    first = files.poll()
    second = files.poll()

    assert first == [Picked("from the web.save", b"save bytes")]
    assert second == []


def test_dialogs_still_open_are_asked_again_on_the_next_frame(tmp_path: Path) -> None:
    waiting = Dialog([str(tmp_path / "later.save")], done=False)
    files = DesktopFiles(opening=[waiting])

    picked = files.poll()

    assert picked == []
    assert files.opening == [waiting]


def test_cancelling_the_save_dialog_writes_nothing(tmp_path: Path) -> None:
    files = DesktopFiles(saving=[(Dialog(""), b"save bytes")])

    files.poll()

    assert list(tmp_path.iterdir()) == []
    assert files.saving == []
