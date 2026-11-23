"""The few OpenGL calls every shader of the map makes the same way: compiling programs, setting
uniforms, textures and framebuffers to draw into, and saving and restoring the GL state imgui
draws with. On the desktop GL is PyOpenGL; in the browser build, the WebGL 2 imgui draws with
(railroads.ui.pixel.webgl); None without either."""

import sys
from dataclasses import dataclass

import numpy as np

if sys.platform == "emscripten":
    from railroads.ui.pixel import webgl as GL

    HEADER = "#version 300 es\nprecision highp float;\nprecision highp int;\n"
else:
    try:
        from OpenGL import GL
    except ImportError:
        GL = None
    HEADER = "#version 330 core\n"


@dataclass(frozen=True)
class FrameBuffer:
    """A framebuffer the shaders draw into, `size` device pixels."""

    framebuffer: int
    size: tuple[int, int]


# Textures of chunks let go of since GL was last reached, deleted then: they can be let go of by
# whichever thread drops the last hold on them, and GL is reached from the frames only.
_UNWANTED: list[int] = []


class ChunkTexture:
    """A chunk's texture made by the graphics card (StandShader.stand), shown like any
    texture imgui made and deleted once nothing holds it."""

    def __init__(self, texture: int) -> None:
        self._texture = texture

    def texture_id(self) -> int:
        return self._texture

    def __del__(self) -> None:
        _UNWANTED.append(self._texture)


def free_textures() -> None:
    """Deletes the chunks' textures nothing holds any more; called from the frames."""
    if _UNWANTED:
        unwanted = [_UNWANTED.pop() for _ in range(len(_UNWANTED))]
        GL.glDeleteTextures(len(unwanted), unwanted)


class RenderTarget:
    """A texture the shaders draw into, with its framebuffer: of bytes, or of floats where
    `floats`, for what is worked on before it is shown."""

    def __init__(self, floats: bool = False) -> None:
        self.texture = 0
        self.framebuffer = GL.glGenFramebuffers(1)
        self.size = (0, 0)
        self._form = (GL.GL_RGBA32F, GL.GL_FLOAT) if floats else (GL.GL_RGBA8, GL.GL_UNSIGNED_BYTE)

    def fit(self, width: int, height: int) -> bool:
        """Makes the texture this size; returns whether it had to be made anew."""
        if self.size == (width, height):
            return False
        if self.texture:
            GL.glDeleteTextures(1, [self.texture])
        self.texture = new_texture()
        internal, kind = self._form
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, internal, width, height, 0, GL.GL_RGBA, kind, None)
        saved = GL.glGetIntegerv(GL.GL_FRAMEBUFFER_BINDING)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.framebuffer)
        GL.glFramebufferTexture2D(
            GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, self.texture, 0
        )
        complete = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER) == GL.GL_FRAMEBUFFER_COMPLETE
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, int(saved))
        if not complete:
            raise RuntimeError("The graphics card can't draw into this texture")
        self.size = (width, height)
        return True


def use(program: int, uniforms: dict[str, tuple[str, tuple]], inputs: tuple[int, ...]) -> None:
    """Draws with `program` from now on, with these uniforms, reading `inputs` from texture
    units 0 on."""
    GL.glUseProgram(program)
    for name, (kind, values) in uniforms.items():
        at = uniform(program, name)
        match kind:
            case "1i":
                GL.glUniform1i(at, *values)
            case "2i":
                GL.glUniform2i(at, *values)
            case "1f":
                GL.glUniform1f(at, *values)
            case "2f":
                GL.glUniform2f(at, *values)
            case "4f":
                GL.glUniform4f(at, *values)
            case "m2":
                matrix = np.array(values, np.float32)
                GL.glUniformMatrix2fv(at, 1, GL.GL_FALSE, matrix)
    for unit, texture in enumerate(inputs):
        GL.glActiveTexture(GL.GL_TEXTURE0 + unit)
        GL.glBindTexture(GL.GL_TEXTURE_2D, texture)


# Where each program's uniforms are, by program and name, found once: a program's uniforms stay
# where they are, and asking GL each time cost a call per uniform every frame.
_UNIFORMS: dict[tuple[int, str], int] = {}


def uniform(program: int, name: str) -> int:
    """Where the program's uniform of this name is."""
    key = (program, name)
    at = _UNIFORMS.get(key)
    if at is None:
        at = _UNIFORMS[key] = int(GL.glGetUniformLocation(program, name))
    return at


def new_texture() -> int:
    texture = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, texture)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
    return texture


def gl_program(vertex: str, fragment: str) -> int:
    program = GL.glCreateProgram()
    for kind, source in ((GL.GL_VERTEX_SHADER, vertex), (GL.GL_FRAGMENT_SHADER, fragment)):
        shader = GL.glCreateShader(kind)
        GL.glShaderSource(shader, source)
        GL.glCompileShader(shader)
        if not GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS):
            raise RuntimeError(GL.glGetShaderInfoLog(shader))
        GL.glAttachShader(program, shader)
    GL.glLinkProgram(program)
    if not GL.glGetProgramiv(program, GL.GL_LINK_STATUS):
        raise RuntimeError(GL.glGetProgramInfoLog(program))
    return program


def saved_gl_state() -> tuple:
    return (
        int(GL.glGetIntegerv(GL.GL_FRAMEBUFFER_BINDING)),
        tuple(int(v) for v in GL.glGetIntegerv(GL.GL_VIEWPORT)),
        int(GL.glGetIntegerv(GL.GL_CURRENT_PROGRAM)),
        int(GL.glGetIntegerv(GL.GL_VERTEX_ARRAY_BINDING)),
        int(GL.glGetIntegerv(GL.GL_ACTIVE_TEXTURE)),
        bool(GL.glIsEnabled(GL.GL_BLEND)),
        bool(GL.glIsEnabled(GL.GL_SCISSOR_TEST)),
    )


def restore_gl_state(saved: tuple) -> None:
    framebuffer, viewport, program, vao, active, blend, scissor = saved
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, framebuffer)
    GL.glViewport(*viewport)
    GL.glUseProgram(program)
    GL.glBindVertexArray(vao)
    for unit in range(7, 0, -1):
        GL.glActiveTexture(GL.GL_TEXTURE0 + unit)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
    GL.glActiveTexture(active)
    if blend:
        GL.glEnable(GL.GL_BLEND)
    if scissor:
        GL.glEnable(GL.GL_SCISSOR_TEST)
