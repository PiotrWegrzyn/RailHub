"""The few OpenGL calls the ground shader makes, for the browser build, where PyOpenGL can't run.

Each call goes to the C function of the same name that Emscripten exports from Pyodide's main
module, the same GL the browser build of imgui draws with, so textures made here are known to
imgui by the same numbers. Python and that GL share one memory, so arrays and strings are handed
over by their address in it, as C would."""

import ctypes
import functools
from typing import Any

import numpy as np

GL_FALSE = 0
GL_TRUE = 1
GL_TRIANGLE_STRIP = 0x0005
GL_BLEND = 0x0BE2
GL_SCISSOR_TEST = 0x0C11
GL_VIEWPORT = 0x0BA2
GL_TEXTURE_2D = 0x0DE1
GL_UNSIGNED_BYTE = 0x1401
GL_FLOAT = 0x1406
GL_RED = 0x1903
GL_RGBA = 0x1908
GL_NEAREST = 0x2600
GL_LINEAR = 0x2601
GL_TEXTURE_MAG_FILTER = 0x2800
GL_TEXTURE_MIN_FILTER = 0x2801
GL_TEXTURE_WRAP_S = 0x2802
GL_TEXTURE_WRAP_T = 0x2803
GL_CLAMP_TO_EDGE = 0x812F
GL_TEXTURE0 = 0x84C0
GL_TEXTURE1 = 0x84C1
GL_ACTIVE_TEXTURE = 0x84E0
GL_ARRAY_BUFFER = 0x8892
GL_STATIC_DRAW = 0x88E4
GL_FRAGMENT_SHADER = 0x8B30
GL_VERTEX_SHADER = 0x8B31
GL_COMPILE_STATUS = 0x8B81
GL_LINK_STATUS = 0x8B82
GL_CURRENT_PROGRAM = 0x8B8D
GL_RGBA8 = 0x8058
GL_R32F = 0x822E
GL_VERTEX_ARRAY_BINDING = 0x85B5
GL_FRAMEBUFFER_BINDING = 0x8CA6
GL_COLOR_ATTACHMENT0 = 0x8CE0
GL_FRAMEBUFFER = 0x8D40

LOG_BYTES = 4096


@functools.cache
def module() -> Any:
    """Emscripten's module object, which holds the exported GL functions."""
    import pyodide_js

    return pyodide_js._module


def _call(name: str, *args: int) -> Any:
    return getattr(module(), f"_{name}")(*args)


def address(array: np.ndarray | None) -> int:
    """Where an array's data starts in the memory GL reads, or 0 for no data."""
    if array is None:
        return 0
    return int(array.__array_interface__["data"][0])


def _ids(name: str, count: int) -> int:
    ids = np.zeros(count, np.uint32)
    _call(name, count, address(ids))
    return int(ids[0])


def _text(source: str) -> np.ndarray:
    return np.frombuffer(source.encode() + b"\0", np.uint8)


def glGenVertexArrays(count: int) -> int:
    return _ids("glGenVertexArrays", count)


def glGenBuffers(count: int) -> int:
    return _ids("glGenBuffers", count)


def glGenTextures(count: int) -> int:
    return _ids("glGenTextures", count)


def glGenFramebuffers(count: int) -> int:
    return _ids("glGenFramebuffers", count)


def glDeleteTextures(count: int, textures: list[int]) -> None:
    ids = np.array(textures, np.uint32)
    _call("glDeleteTextures", count, address(ids))


def glBufferData(target: int, size: int, data: np.ndarray, usage: int) -> None:
    _call("glBufferData", target, size, address(data), usage)


def glVertexAttribPointer(
    index: int, size: int, kind: int, normalized: int, stride: int, offset: ctypes.c_void_p
) -> None:
    _call("glVertexAttribPointer", index, size, kind, normalized, stride, offset.value or 0)


def glTexImage2D(
    target: int,
    level: int,
    internal: int,
    width: int,
    height: int,
    border: int,
    form: int,
    kind: int,
    pixels: np.ndarray | None,
) -> None:
    _call(
        "glTexImage2D",
        target,
        level,
        internal,
        width,
        height,
        border,
        form,
        kind,
        address(pixels),
    )


def glGetUniformLocation(program: int, name: str) -> int:
    text = _text(name)
    return int(_call("glGetUniformLocation", program, address(text)))


def glUniformMatrix2fv(location: int, count: int, transpose: int, values: np.ndarray) -> None:
    values = np.ascontiguousarray(values, np.float32)
    _call("glUniformMatrix2fv", location, count, transpose, address(values))


def glShaderSource(shader: int, source: str) -> None:
    text = _text(source)
    pointers = np.array([address(text)], np.uint32)
    _call("glShaderSource", shader, 1, address(pointers), 0)


def glGetIntegerv(name: int) -> np.ndarray | int:
    values = np.zeros(4, np.int32)
    _call("glGetIntegerv", name, address(values))
    return values if name == GL_VIEWPORT else int(values[0])


def _status(name: str, handle: int, which: int) -> int:
    value = np.zeros(1, np.int32)
    _call(name, handle, which, address(value))
    return int(value[0])


def glGetShaderiv(shader: int, which: int) -> int:
    return _status("glGetShaderiv", shader, which)


def glGetProgramiv(program: int, which: int) -> int:
    return _status("glGetProgramiv", program, which)


def _log(name: str, handle: int) -> bytes:
    length = np.zeros(1, np.int32)
    log = np.zeros(LOG_BYTES, np.uint8)
    _call(name, handle, LOG_BYTES, address(length), address(log))
    return log[: int(length[0])].tobytes()


def glGetShaderInfoLog(shader: int) -> bytes:
    return _log("glGetShaderInfoLog", shader)


def glGetProgramInfoLog(program: int) -> bytes:
    return _log("glGetProgramInfoLog", program)


def glIsEnabled(capability: int) -> bool:
    return bool(_call("glIsEnabled", capability))


def _forward(name: str):
    def call(*args: int) -> Any:
        return _call(name, *(int(arg) for arg in args))

    call.__name__ = name
    return call


glBindVertexArray = _forward("glBindVertexArray")
glBindBuffer = _forward("glBindBuffer")
glEnableVertexAttribArray = _forward("glEnableVertexAttribArray")
glBindTexture = _forward("glBindTexture")
glTexParameteri = _forward("glTexParameteri")
glBindFramebuffer = _forward("glBindFramebuffer")
glFramebufferTexture2D = _forward("glFramebufferTexture2D")
glViewport = _forward("glViewport")
glEnable = _forward("glEnable")
glDisable = _forward("glDisable")
glUseProgram = _forward("glUseProgram")
glActiveTexture = _forward("glActiveTexture")
glDrawArrays = _forward("glDrawArrays")
glCreateProgram = _forward("glCreateProgram")
glCreateShader = _forward("glCreateShader")
glCompileShader = _forward("glCompileShader")
glAttachShader = _forward("glAttachShader")
glLinkProgram = _forward("glLinkProgram")
glUniform1f = _forward("glUniform1f")
glUniform2f = _forward("glUniform2f")
glUniform1i = _forward("glUniform1i")
glUniform2i = _forward("glUniform2i")
