"""Viewer-side scene graph records for simulation visuals."""

from __future__ import annotations

from dataclasses import dataclass

from ..._core import _ke
from .utils import normalize_color


@dataclass(slots=True)
class VisualBodyPick:
    """High-level body pick result resolved from a renderer selection."""

    env_id: int | None
    obj_id: int
    body_id: int
    visual: object


@dataclass(slots=True)
class VisualArticulationSceneGraph:
    """Viewer-side visual wrapper for one articulation."""

    env_id: int
    obj_id: int
    articulation_visual: object
    body_prims: list[object]
    render_prims: list[object]
    collision_prims: list[object]
    body_handles: list[int]
    handle_body_ids: dict[int, int] | None = None

    @property
    def key(self):
        return (self.env_id, self.obj_id)

    @property
    def prims(self):
        return tuple(self.render_prims)

    @property
    def visual_prims(self):
        return tuple(self.render_prims)

    @property
    def num_bodies(self) -> int:
        return len(self.body_prims)

    @property
    def collision_visuals(self):
        return tuple(self.collision_prims)

    def body_id_from_render_handle(self, handle) -> int | None:
        handle = int(handle)
        if self.handle_body_ids is not None:
            return self.handle_body_ids.get(handle)
        for body_id, body_handle in enumerate(self.body_handles):
            if int(body_handle) == handle:
                return body_id
        return None

    def pick_body(self, selection) -> VisualBodyPick | None:
        if selection is None or not getattr(selection, "hit", False):
            return None
        body_id = self.body_id_from_render_handle(getattr(selection, "handle", -1))
        if body_id is None:
            return None
        return VisualBodyPick(self.env_id, self.obj_id, int(body_id), self)

    def set_color(self, color):
        rgba_vec = normalize_color(color)
        if rgba_vec is not None:
            for prim in self.render_prims:
                prim.set_display_color_alpha(rgba_vec)
        return self

    def set_alpha(self, alpha: float):
        alpha = float(alpha)
        for prim in self.render_prims:
            color = prim.get_display_color_alpha()
            if color is None:
                prim.set_display_color_alpha(_ke.vec4(1.0, 1.0, 1.0, alpha))
            else:
                prim.set_display_color_alpha(
                    _ke.vec4(float(color.x), float(color.y), float(color.z), alpha)
                )
        return self

    def set_visible(self, visible: bool):
        for prim in self.render_prims:
            prim.set_visible(bool(visible))
        return self

    def set_collision_visible(self, visible: bool):
        for prim in self.collision_prims:
            prim.set_visible(bool(visible))
        return self


@dataclass(slots=True)
class VisualRigidSceneGraph:
    """Viewer-side visual wrapper for one rigid object."""

    env_id: int
    obj_id: int
    rigid: object
    rigid_visual: object
    body_prims: list[object]
    body_handles: list[int]

    @property
    def key(self):
        return (self.env_id, self.obj_id)

    @property
    def prims(self):
        return tuple(self.body_prims)

    @property
    def num_bodies(self) -> int:
        return len(self.body_prims)

    def body_id_from_render_handle(self, handle) -> int | None:
        handle = int(handle)
        for body_id, body_handle in enumerate(self.body_handles):
            if int(body_handle) == handle:
                return body_id
        return None

    def pick_body(self, selection) -> VisualBodyPick | None:
        if selection is None or not getattr(selection, "hit", False):
            return None
        body_id = self.body_id_from_render_handle(getattr(selection, "handle", -1))
        if body_id is None:
            return None
        return VisualBodyPick(self.env_id, self.obj_id, int(body_id), self)

    def set_color(self, color):
        rgba = normalize_color(color)
        if rgba is not None:
            for prim in self.body_prims:
                prim.set_display_color_alpha(rgba)
        return self

    def set_visible(self, visible: bool):
        for prim in self.body_prims:
            prim.set_visible(bool(visible))
        return self


__all__ = [
    "VisualArticulationSceneGraph",
    "VisualBodyPick",
    "VisualRigidSceneGraph",
]
