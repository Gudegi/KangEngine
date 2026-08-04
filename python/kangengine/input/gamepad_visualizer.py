"""Reusable ImGui gamepad input visualizer."""

from __future__ import annotations

from pathlib import Path

from .._core import _ke
from ..utils.color import preset_rgba


class GamepadVisualizer:
    """Draw a live Xbox-style gamepad overlay in an application viewport."""

    _ASPECT_RATIO = 0.75
    _PADDING = 16.0
    _VIEWPORT_WIDTH_RATIO = 0.16

    _ROUND_CUTOUTS = {
        "left_thumb": ((0.325, 0.400), 0.05625),
        "right_thumb": ((0.625, 0.533), 0.05625),
        "a": ((0.725, 0.450), 0.0225),
        "b": ((0.775, 0.383), 0.0225),
        "x": ((0.675, 0.383), 0.0225),
        "y": ((0.725, 0.317), 0.0225),
        "back": ((0.450, 0.367), 0.015),
        "start": ((0.550, 0.367), 0.015),
        "guide": ((0.500, 0.300), 0.0275),
    }
    _DPAD_AREAS = {
        "dpad_up": (0.38125, 0.48333, 0.41875, 0.55833),
        "dpad_right": (0.40000, 0.53333, 0.45625, 0.58333),
        "dpad_down": (0.38125, 0.55833, 0.41875, 0.63333),
        "dpad_left": (0.34375, 0.53333, 0.40000, 0.58333),
    }
    _DPAD_CROSS = (
        (0.34375, 0.53333, 0.45625, 0.58333),
        (0.38125, 0.48333, 0.41875, 0.63333),
    )
    _TRIGGERS = (
        (0.2425, 0.1675, "left_trigger"),
        (0.5875, 0.1700, "right_trigger"),
    )
    _BUMPERS = {
        "left_bumper": (
            (0.2425, 0.2183),
            (0.2750, 0.2000),
            (0.3125, 0.1883),
            (0.3500, 0.1833),
            (0.3875, 0.1867),
            (0.4113, 0.1917),
            (0.4063, 0.2250),
            (0.3750, 0.2183),
            (0.3375, 0.2167),
            (0.3000, 0.2167),
            (0.2625, 0.2267),
            (0.2425, 0.2333),
        ),
        "right_bumper": (
            (0.5888, 0.1917),
            (0.6125, 0.1867),
            (0.6500, 0.1833),
            (0.6875, 0.1883),
            (0.7250, 0.2000),
            (0.7575, 0.2183),
            (0.7575, 0.2333),
            (0.7375, 0.2267),
            (0.7000, 0.2167),
            (0.6625, 0.2167),
            (0.6250, 0.2183),
            (0.5938, 0.2250),
        ),
    }

    def __init__(
        self,
        app,
        *,
        gamepad_index: int = 0,
        width: float | None = None,
        anchor=None,
        offset=(20.0, 20.0),
        opacity: float = 1.0,
    ):
        self.gamepad = app.input.gamepad(gamepad_index)
        self.width = 0.0 if width is None else float(width)
        self._responsive = width is None
        self.opacity = min(1.0, max(0.0, float(opacity)))
        self._active_glow = _ke.Vec4(
            *preset_rgba(_ke.ColorType.SKY_BLUE, 0.18 * self.opacity)
        )
        self._active = _ke.Vec4(
            *preset_rgba(_ke.ColorType.SKY_BLUE, 0.98 * self.opacity)
        )
        self._stick = _ke.Vec4(
            *preset_rgba(_ke.ColorType.PASTEL_SKY, 0.98 * self.opacity)
        )
        self._inactive = _ke.Vec4(*preset_rgba(_ke.ColorType.SLATE_GRAY, self.opacity))
        self._stick_cap = _ke.Vec4(*preset_rgba(_ke.ColorType.DARK_GRAY, self.opacity))
        self.anchor = _ke.ScreenAnchor.BOTTOM_LEFT if anchor is None else anchor
        self.offset = (float(offset[0]), float(offset[1]))
        asset = (
            Path(__file__).resolve().parents[1]
            / "assets"
            / "gamepad"
            / "gemini-xbox_gamepad.png"
        )
        self._texture = app.load_texture(asset, flip=True)
        self._window_name = f"Gamepad Visualizer##{id(self)}"
        self._window_flags = (
            _ke.imgui.WindowFlags_NoTitleBar
            | _ke.imgui.WindowFlags_NoBackground
            | _ke.imgui.WindowFlags_NoResize
            | _ke.imgui.WindowFlags_NoMove
            | _ke.imgui.WindowFlags_NoScrollbar
        )

    @property
    def height(self) -> float:
        return self.width * self._ASPECT_RATIO

    def _point(self, image_x, image_y, uv, radius, color=None):
        x = image_x + uv[0] * self.width
        y = image_y + uv[1] * self.height
        if color is None:
            _ke.imgui.draw_circle_filled(x, y, radius * 2.0, self._active_glow)
        _ke.imgui.draw_circle_filled(
            x, y, radius, self._active if color is None else color
        )

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
            px = x + (width - window_width) * 0.5 + offset_x
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
            py = y + (height - window_height) * 0.5 + offset_y
        else:
            py = y + height - window_height - offset_y
        return px, py

    def _bumper(self, image_x, image_y, polygon, active=False):
        _ke.imgui.draw_convex_polygon_filled(
            [(image_x + x * self.width, image_y + y * self.height) for x, y in polygon],
            self._active if active else self._inactive,
        )

    def draw(self, state=None):
        """Draw the visualizer once in the current ImGui frame."""
        if state is None:
            state = self.gamepad.state()
        viewport = _ke.imgui.main_viewport_work_rect()
        if self._responsive:
            self.width = viewport[2] * self._VIEWPORT_WIDTH_RATIO
        window_width = self.width + self._PADDING
        window_height = self.height + self._PADDING
        window_x, window_y = self._window_position(
            viewport,
            window_width,
            window_height,
        )
        _ke.imgui.set_next_window_pos(window_x, window_y)
        _ke.imgui.set_next_window_size(window_width, window_height)
        if not _ke.imgui.begin(self._window_name, self._window_flags):
            _ke.imgui.end()
            return

        image_x, image_y = _ke.imgui.cursor_screen_pos()
        _ke.imgui.image(self._texture, self.width, self.height, self.opacity)
        scale = self.width / 400.0

        for uv, radius in self._ROUND_CUTOUTS.values():
            self._point(
                image_x,
                image_y,
                uv,
                radius * self.width,
                self._inactive,
            )

        for x1, y1, x2, y2 in self._DPAD_CROSS:
            _ke.imgui.draw_rect_filled(
                image_x + x1 * self.width,
                image_y + y1 * self.height,
                image_x + x2 * self.width,
                image_y + y2 * self.height,
                self._inactive,
            )

        for polygon in self._BUMPERS.values():
            self._bumper(image_x, image_y, polygon)

        for name in ("left_thumb", "right_thumb"):
            self._point(
                image_x,
                image_y,
                self._ROUND_CUTOUTS[name][0],
                14.0 * scale,
                self._stick_cap,
            )

        if state.connected:
            for name, (uv, radius_ratio) in self._ROUND_CUTOUTS.items():
                if getattr(state, name):
                    self._point(
                        image_x,
                        image_y,
                        uv,
                        radius_ratio * self.width,
                    )

            for name, (x1, y1, x2, y2) in self._DPAD_AREAS.items():
                if getattr(state, name):
                    _ke.imgui.draw_rect_filled(
                        image_x + x1 * self.width,
                        image_y + y1 * self.height,
                        image_x + x2 * self.width,
                        image_y + y2 * self.height,
                        self._active,
                    )

            for name, polygon in self._BUMPERS.items():
                if getattr(state, name):
                    self._bumper(image_x, image_y, polygon, active=True)

            stick_range = 20.0 * scale
            self._point(
                image_x,
                image_y,
                (
                    0.325 + state.left_x * stick_range / self.width,
                    0.400 + state.left_y * stick_range / self.height,
                ),
                7.0 * scale,
                self._stick,
            )
            self._point(
                image_x,
                image_y,
                (
                    0.625 + state.right_x * stick_range / self.width,
                    0.533 + state.right_y * stick_range / self.height,
                ),
                7.0 * scale,
                self._stick,
            )

            for start, length, name in self._TRIGGERS:
                value = (getattr(state, name) + 1.0) * 0.5
                _ke.imgui.draw_line(
                    image_x + start * self.width,
                    image_y + 0.16 * self.height,
                    image_x + (start + length * value) * self.width,
                    image_y + 0.16 * self.height,
                    self._active,
                    7.0 * scale,
                )

        _ke.imgui.end()


__all__ = ["GamepadVisualizer"]
