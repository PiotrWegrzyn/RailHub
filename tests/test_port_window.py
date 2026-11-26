from collections.abc import Iterator

import pytest
from imgui_bundle import imgui  # noqa: TID251
from screens import FakeTexture, gui_context

from builders import flat_map, layer
from railroads.sim.constructions import Building, BuildingKind, Constructions
from railroads.sim.game import Game
from railroads.sim.goods import Good
from railroads.sim.growth import Growth
from railroads.sim.ownership import SHARES
from railroads.sim.ports import Port, Slot, Tier
from railroads.sim.tilemap import Terrain
from railroads.ui import theme
from railroads.ui.port_window import PortWindow, port_name

HOME = (4, 1)
AWAY = (12, 1)


@pytest.fixture
def gui(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    with gui_context():
        monkeypatch.setattr(
            theme.hello_imgui, "create_texture_gpu_from_rgba_data", lambda _: FakeTexture()
        )
        yield


def two_ports() -> Growth:
    terrain = layer(16, 16, Terrain.LAND)
    terrain[:, 0] = Terrain.SEA
    game = Game(Constructions(flat_map(16, 16, terrain=terrain)))
    growth = game.growth
    for cell, kind in ((HOME, BuildingKind.MEGA_PORT), (AWAY, BuildingKind.PORT)):
        growth.build(Building(cell, kind))
        growth.owners.own(cell, kind, SHARES)
    growth.ports.sync()
    growth.ports.at[HOME] = Port([Slot(Good.COAL, Tier.COMMON, True)])
    return growth


def draw_frames(window: PortWindow, growth: Growth, frames: int = 2) -> list[object]:
    opened: list[object] = []
    for _ in range(frames):
        imgui.get_io().delta_time = 1 / 60
        imgui.new_frame()
        window.draw(growth, lambda _: None, lambda cell, kind: opened.append((cell, kind)))
        imgui.render()
    return opened


def test_the_port_window_draws_a_controlled_mega_port_with_its_pick_and_ships(gui: None) -> None:
    growth = two_ports()
    growth.ports.pick(HOME, Good.CARS, Tier.LEGENDARY)
    growth.ports.ship(HOME, Good.WOOL, AWAY)
    window = PortWindow(cell=HOME)

    draw_frames(window, growth)

    assert window.cell == HOME


def test_the_port_window_closes_once_its_port_is_gone(gui: None) -> None:
    growth = two_ports()
    growth.take_down(Building(AWAY, BuildingKind.PORT))
    growth.ports.sync()
    window = PortWindow(cell=AWAY)

    draw_frames(window, growth, 1)

    assert window.cell is None


def test_a_port_is_named_by_its_size_and_cell() -> None:
    growth = two_ports()

    name = port_name(growth, HOME)

    assert name == "Mega port at 4, 1"
