"""Shared font layout cached for OVRTX screen and world text."""

import numpy as np

from .._core import _ke


class OvrtxTextDraw:
    def __init__(self, debug_draw):
        self._debug = debug_draw
        self._entries = {"world": {}, "screen": {}}
        self._glyphs = {}
        self._resolution = None
        self._atlas = None
        self.font = np.array([1, 1, 0], dtype=np.uint32)

    def set_world(self, path, desc):
        self._set("world", path, desc)

    def set_screen(self, path, desc):
        self._set("screen", path, desc)

    @staticmethod
    def _signature(kind, desc):
        placement = desc.depth_mode if kind == "world" else desc.anchor
        return (desc.text, desc.position, desc.color, desc.pixel_size,
                desc.alignment, desc.hidden, placement)

    def _set(self, kind, path, desc):
        if not path:
            return
        old = self._entries[kind].get(path)
        if old is not None and self._signature(kind, old) == self._signature(kind, desc):
            return
        self._entries[kind][path] = desc
        self._glyphs.pop((kind, path), None)
        self._debug._dirty = True

    def update(self, kind, path, field, value):
        desc = self._entries[kind].get(path)
        if desc is None or getattr(desc, field) == value:
            return
        setattr(desc, field, value)
        self._glyphs.pop((kind, path), None)
        self._debug._dirty = True

    def remove(self, kind, path):
        if self._entries[kind].pop(path, None) is not None:
            self._glyphs.pop((kind, path), None)
            self._debug._dirty = True

    def clear(self, kind):
        for path in list(self._entries[kind]):
            self.remove(kind, path)

    @staticmethod
    def _visible(desc):
        return not desc.hidden and bool(desc.text) and desc.pixel_size > 0

    @property
    def active(self):
        return any(self._visible(desc) for entries in self._entries.values()
                   for desc in entries.values())

    def batches(self, width, height):
        if not self.active:
            return
        if self._atlas is None:
            self._atlas = _ke._FontAtlasData()
            pixels = self._atlas.pixels
            # Pack four R8 texels per word; upload once per stage.
            self.font = np.concatenate((
                np.array([pixels.shape[1], pixels.shape[0]], dtype=np.uint32),
                pixels.reshape(-1).view(np.uint32),
            ))
        if self._resolution != (width, height):
            for path in self._entries["screen"]:
                self._glyphs.pop(("screen", path), None)
            self._resolution = (width, height)

        # Match Native: depth-tested world text, overlay world text, screen text.
        for kind, overlay in (("world", False), ("world", True), ("screen", True)):
            for path, desc in sorted(self._entries[kind].items()):
                if not self._visible(desc):
                    continue
                if kind == "world" and (desc.depth_mode == _ke.TextDepthMode.OVERLAY) != overlay:
                    continue
                key = (kind, path)
                if key not in self._glyphs:
                    self._glyphs[key] = (self._atlas.world(desc) if kind == "world"
                                         else self._atlas.screen(desc, width, height))
                yield kind, overlay, self._glyphs[key]

    def close(self):
        self._entries = {"world": {}, "screen": {}}
        self._glyphs.clear()
        self._atlas = None
        self.font = np.array([1, 1, 0], dtype=np.uint32)
        self._debug = None
