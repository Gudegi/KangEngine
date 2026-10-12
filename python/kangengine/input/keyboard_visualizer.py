"""Texture-backed keyboard overlay and reusable key regions."""

from __future__ import annotations

import json
from pathlib import Path

from .._core import _ke
from ..utils.color import preset_rgba


class KeyboardVisualizer:
    """Display physical key presses without capturing keyboard/mouse input.

    ``draw()`` places WASD, Shift, Space and R at bottom center by default.
    ``anchor`` and ``offset`` follow GamepadVisualizer's placement convention.
    Set ``full_keyboard=True`` to inspect the entire keyboard texture.
    ``draw_key("W")`` samples a single key from the same texture in the current
    ImGui window. Names match ``ke.keys``; FN has no GLFW key event.
    """

    # Three rows: W / ASD + Shift / Space + R.
    _LAYOUT_SIZE = (448, 256)
    _KEYS = (
        ("W", 88, 0, 80, 80),
        ("A", 0, 88, 80, 80),
        ("S", 88, 88, 80, 80),
        ("D", 176, 88, 80, 80),
        ("SHIFT", 304, 88, 144, 80),
        ("SPACE", 0, 176, 160, 80),
        ("R", 304, 176, 80, 80),
    )

    def __init__(
        self,
        app,
        *,
        width: float | None = None,
        anchor=None,
        offset=(0.0, 20.0),
        full_keyboard: bool = False,
    ):
        self._app = app
        self.width = width
        self.offset = offset
        self.anchor = _ke.ScreenAnchor.BOTTOM_CENTER if anchor is None else anchor
        self.full_keyboard = full_keyboard
        asset = Path(__file__).resolve().parents[1] / "assets" / "keyboard"
        self._regions = json.loads((asset / "regions.json").read_text())
        self._texture = app.load_texture(asset / "keyboard.png", flip=True)
        self._active = _ke.Vec4(*preset_rgba(_ke.ColorType.SKY_BLUE, 0.45))
        self._window_name = f"Keyboard Visualizer##{id(self)}"
        self._window_flags = (
            _ke.imgui.WindowFlags_NoTitleBar
            | _ke.imgui.WindowFlags_NoBackground
            | _ke.imgui.WindowFlags_NoResize
            | _ke.imgui.WindowFlags_NoMove
            | _ke.imgui.WindowFlags_NoScrollbar
            | _ke.imgui.WindowFlags_NoInputs
        )

    def _down(self, name):
        if name == "SHIFT":
            return self._down("LEFT_SHIFT") or self._down("RIGHT_SHIFT")
        key = getattr(_ke.keys, name, None)
        return key is not None and self._app.input.keyboard.down(key)

    def _image(self, rect, width, height=None, *, rounding=0.0):
        x0, y0, x1, y1 = rect
        image_width, image_height = self._regions["image_size"]
        if height is None:
            height = width * (y1 - y0) / (x1 - x0)
        # App.load_texture(flip=True) stores the top of the PNG at texture v=1.
        _ke.imgui.image(
            self._texture,
            width,
            height,
            uv0=(x0 / image_width, 1 - y0 / image_height),
            uv1=(x1 / image_width, 1 - y1 / image_height),
            rounding=rounding,
        )
        return height

    def draw_key(self, key: str, *, width: float = 48.0, height: float | None = None):
        """Draw one cropped key with its pressed highlight in the current window."""
        rect = self._regions["keys"]["LEFT_SHIFT" if key == "SHIFT" else key]
        x, y = _ke.imgui.cursor_screen_pos()
        if height is None:
            height = width * (rect[3] - rect[1]) / (rect[2] - rect[0])
        # Crop away the opaque keyboard backing outside each rounded key.
        rounding = min(width, height) * 0.1
        self._image(rect, width, height, rounding=rounding)
        if self._down(key):
            _ke.imgui.draw_rect_filled(
                x, y, x + width, y + height, self._active, rounding=rounding
            )
        if key == "SPACE":
            # The source texture's spacebar has no printed legend.
            cursor = _ke.imgui.cursor_screen_pos()
            tw, th = _ke.imgui.calc_text_size("Space")
            _ke.imgui.set_cursor_screen_pos(x + (width - tw) / 2, y + (height - th) / 2)
            _ke.imgui.text("Space")
            _ke.imgui.set_cursor_screen_pos(*cursor)

    def _window_position(self, viewport, window_width, window_height):
        x, y, width, height = viewport
        offset_x, offset_y = self.offset
        anchor = self.anchor
        if anchor in (
            _ke.ScreenAnchor.TOP_LEFT,
            _ke.ScreenAnchor.CENTER_LEFT,
            _ke.ScreenAnchor.BOTTOM_LEFT,
        ):
            px = x + offset_x
        elif anchor in (
            _ke.ScreenAnchor.TOP_CENTER,
            _ke.ScreenAnchor.CENTER,
            _ke.ScreenAnchor.BOTTOM_CENTER,
        ):
            px = x + (width - window_width) / 2 + offset_x
        else:
            px = x + width - window_width - offset_x
        if anchor in (
            _ke.ScreenAnchor.TOP_LEFT,
            _ke.ScreenAnchor.TOP_CENTER,
            _ke.ScreenAnchor.TOP_RIGHT,
        ):
            py = y + offset_y
        elif anchor in (
            _ke.ScreenAnchor.CENTER_LEFT,
            _ke.ScreenAnchor.CENTER,
            _ke.ScreenAnchor.CENTER_RIGHT,
        ):
            py = y + (height - window_height) / 2 + offset_y
        else:
            py = y + height - window_height - offset_y
        return px, py

    def draw(self):
        """Draw once during the application's ImGui render callback."""
        viewport = _ke.imgui.main_viewport_work_rect()
        default_width = 600.0 if self.full_keyboard else self._LAYOUT_SIZE[0]
        viewport_fraction = 0.38 if self.full_keyboard else 0.75
        width = (
            min(default_width, viewport[2] * viewport_fraction)
            if self.width is None
            else self.width
        )
        x0, y0, x1, y1 = self._regions["board"]
        scale = width / (x1 - x0)
        height = (
            (y1 - y0) * scale
            if self.full_keyboard
            else self._LAYOUT_SIZE[1] * width / self._LAYOUT_SIZE[0]
        )
        _ke.imgui.set_next_window_pos(
            *self._window_position(viewport, width + 16, height + 16)
        )
        _ke.imgui.set_next_window_size(width + 16, height + 16)
        if not _ke.imgui.begin(self._window_name, self._window_flags):
            _ke.imgui.end()
            return
        x, y = _ke.imgui.cursor_screen_pos()
        if not self.full_keyboard:
            scale = width / self._LAYOUT_SIZE[0]
            for key, left, top, key_width, key_height in self._KEYS:
                _ke.imgui.set_cursor_screen_pos(x + left * scale, y + top * scale)
                self.draw_key(key, width=key_width * scale, height=key_height * scale)
            _ke.imgui.end()
            return
        self._image(self._regions["board"], width)
        for name, (left, top, right, bottom) in self._regions["keys"].items():
            if self._down(name):
                _ke.imgui.draw_rect_filled(
                    x + (left - x0) * scale,
                    y + (top - y0) * scale,
                    x + (right - x0) * scale,
                    y + (bottom - y0) * scale,
                    self._active,
                    rounding=min(right - left, bottom - top) * scale * 0.1,
                )
        _ke.imgui.end()


__all__ = ["KeyboardVisualizer"]
