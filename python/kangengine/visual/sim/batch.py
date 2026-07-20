"""Public visual batch handle for simulation visual sync."""

from __future__ import annotations

from .records import VisualBodyPick


class VisualBatch:
    """Handle returned by ``SimWorldVisualizer.add(...)``.

    The handle owns no simulation state. It delegates visual operations to a CPU
    or GPU ExternalBuffer backend. The low-level C++ transform batch remains
    exposed as ``kangengine.physics.SimVisualBatch``.
    """

    def __init__(self, obj_id, env_ids, *, backend):
        self.obj_id = int(obj_id)
        self.env_ids = tuple(int(env_id) for env_id in env_ids)
        self._backend = backend
        self._released = False

    @property
    def key(self):
        self._require_valid()
        return (self.env_ids, self.obj_id)

    @property
    def num_envs(self) -> int:
        self._require_valid()
        return len(self.env_ids)

    @property
    def num_bodies(self) -> int:
        self._require_valid()
        return self._backend.num_bodies

    @property
    def prims(self):
        self._require_valid()
        return self._backend.prims

    @property
    def body_handles(self):
        self._require_valid()
        return self._backend.body_handles

    @property
    def collision_visuals(self):
        self._require_valid()
        return tuple(getattr(self._backend, "collision_prims", ()))

    def __len__(self) -> int:
        self._require_valid()
        return len(self._backend)

    def sync(self):
        self._require_valid()
        if hasattr(self._backend, "sync"):
            self._backend.sync()
        return self

    def body_id_from_render_handle(self, handle):
        self._require_valid()
        return self._backend.body_id_from_render_handle(handle)

    def pick_body(self, selection) -> VisualBodyPick | None:
        self._require_valid()
        if selection is None or not getattr(selection, "hit", False):
            return None
        body_id = self.body_id_from_render_handle(getattr(selection, "handle", -1))
        if body_id is None:
            return None
        instance_index = int(getattr(selection, "instance_index", -1))
        env_id = None
        if 0 <= instance_index < len(self.env_ids):
            env_id = self.env_ids[instance_index]
        elif len(self.env_ids) == 1:
            env_id = self.env_ids[0]
        return VisualBodyPick(env_id, self.obj_id, int(body_id), self)

    def set_visible(self, visible: bool):
        self._require_valid()
        self._backend.set_visible(visible)
        return self

    def set_color(self, color):
        self._require_valid()
        self._backend.set_color(color)
        return self

    def set_collision_visible(self, visible: bool):
        self._require_valid()
        if hasattr(self._backend, "set_collision_visible"):
            self._backend.set_collision_visible(visible)
        return self

    @property
    def is_valid(self) -> bool:
        return not self._released

    def _require_valid(self):
        if self._released:
            raise RuntimeError("VisualBatch has been released")

    def release(self):
        if self._released:
            return self
        if hasattr(self._backend, "release"):
            self._backend.release()
        self._backend = None
        self._released = True
        return self


__all__ = ["VisualBatch"]
