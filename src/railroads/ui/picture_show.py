"""The waiting screen's view of a new map growing: the newest picture of its history
(railroads.ui.history_pictures) over the whole window, its ground drawn by the graphics card
(railroads.ui.pixel.ground_shader) where the picture leaves it out, as the game draws its own."""

import numpy as np
from imgui_bundle import hello_imgui, imgui

from railroads.ui.history_pictures import GroundLook, Picture
from railroads.ui.pixel.ground_shader import GroundShader, cell_info, screen_of, view_of
from railroads.ui.pixel.layer import blown_up_texture


class PictureShow:
    """Shows the pictures a new map's history sends, from whichever thread sends them, on a
    screen `density` device pixels to a point."""

    def __init__(self, density: int = 1) -> None:
        self.density = density
        self._ground: GroundLook | None = None
        self._newest: Picture | None = None
        self._shown: tuple[Picture, hello_imgui.TextureGpu] | None = None
        self._shader: GroundShader | None = None
        self._shader_tried = False
        # The textures last drawn, the ground's first, for the screen's breaking up.
        self.textures: list[imgui.ImTextureRef] = []

    def see(self, picture: Picture) -> None:
        if picture.ground is not None:
            self._ground = picture.ground
        self._newest = picture

    @property
    def year(self) -> int | None:
        """The year of the newest picture, None before the first."""
        return None if self._newest is None else self._newest.year

    def draw(
        self, draw_list: imgui.ImDrawList, corner: imgui.ImVec2, size: tuple[float, float]
    ) -> None:
        """Draws the newest picture, if there is one, over the window at `corner` of `size`
        points."""
        picture = self._newest
        self.textures = []
        if picture is None:
            return
        if self._shown is None or self._shown[0] is not picture:
            self._shown = (picture, blown_up_texture(picture.pixels, self.density))
        bottom_right = (corner.x + size[0], corner.y + size[1])
        ground = self._ground_texture(picture, size)
        if ground is not None:
            self.textures.append(imgui.ImTextureRef(ground))
        self.textures.append(imgui.ImTextureRef(self._shown[1].texture_id()))
        for texture in self.textures:
            draw_list.add_image(texture, corner, bottom_right)

    def _ground_texture(self, picture: Picture, size: tuple[float, float]) -> int | None:
        """The ground under a picture that leaves it out, drawn by the graphics card for a window
        of `size` points; None where the picture brings its own or the card can't draw it."""
        ground = self._ground
        if picture.biomes is None or ground is None:
            return None
        if not self._shader_tried:
            self._shader_tried = True
            self._shader = GroundShader.make(
                ground.projection, ground.shape, ground.shading, ground.heights
            )
        shader = self._shader
        if shader is None:
            return None
        if shader.cells_version != picture.year:
            glow = np.zeros(ground.shape, dtype=np.float64)
            shader.set_cells(
                cell_info(picture.biomes, ground.deposits, ground.hills, glow, ground.mountains),
                picture.year,
            )
        view = view_of(picture.origin, 1, (0.0, 0.0), size)
        screen = screen_of(picture.origin, 1, (0.0, 0.0), size, self.density)
        return shader.picture(view, screen, blend=True)
