"""Batch contact arrows and state disks for Native and external renderers."""

import torch

from .._core import _ke
from ..render import TransformSource
from ..utils import to_external_transform_desc


CONTACT_COLORS = {
    "normal": _ke.ColorType.PASTEL_YELLOW,
    "force": _ke.ColorType.PASTEL_PURPLE,
    "open": _ke.ColorType.LIGHT_GRAY,
    "stick": _ke.ColorType.PASTEL_CYAN,
    "slip": _ke.ColorType.PASTEL_ORANGE,
}


def contact_modes(normals, forces, velocities):
    """Estimate open/stick/slip (0/1/2), not the solver's friction state."""
    tangent = velocities - (velocities * normals).sum(dim=1, keepdim=True) * normals
    loaded = torch.linalg.vector_norm(forces, dim=1) >= 1e-4
    sliding = torch.linalg.vector_norm(tangent, dim=1) >= 1e-3
    return torch.where(loaded, 1 + sliding.to(torch.int64), 0)


def glyph_matrices(output, points, directions, lengths, radius, active=None):
    """Write GLM column-major matrices for unit +Z glyphs; collapse unused slots."""
    output.zero_()
    output[:, 3, 3] = 1
    size = torch.linalg.vector_norm(directions, dim=1, keepdim=True)
    z = directions / size.clamp_min(1e-12)
    reference = torch.nn.functional.one_hot(z.abs().argmin(dim=1), 3).to(z.dtype)
    x = torch.cross(reference, z, dim=1)
    x /= torch.linalg.vector_norm(x, dim=1, keepdim=True).clamp_min(1e-12)
    y = torch.cross(z, x, dim=1)
    target = output[: len(points)]
    target[:, 0, :3] = x * radius
    target[:, 1, :3] = y * radius
    target[:, 2, :3] = z * lengths[:, None]
    target[:, 3, :3] = points
    valid = (size[:, 0] > 1e-12) & (lengths > 0)
    if active is not None:
        valid &= active
    target[:, :3, :3] *= valid[:, None, None]


class ContactDisplay:
    """Reuse source meshes and CPU/CUDA instance storage; no physics ownership."""

    def __init__(self, app, path):
        self.app = app
        self.path = path
        self.batches = {}
        self.meshes = {}
        self.materials = {}
        self.normal = True
        self.mode = True
        self.force = True
        self.scale = 1.0
        self.force_scale = 1.0
        self.thickness = 1.0
        self.length = 0.2
        self.force_gain = 0.5
        self.count = 0
        self.revision = None
        self.version = 0
        self.stream = None
        self.ready = None

    def refresh(self, source):
        self.length, self.force_gain = source.contact_display_settings()
        self.revision = None

    def _batch(self, name, count, device):
        if name not in self.batches:
            disk = name in ("open", "stick", "slip")
            mesh_name = "disk" if disk else "arrow"
            if mesh_name not in self.meshes:
                self.meshes[mesh_name] = (
                    _ke.geometry.create_cylinder_data(1.0, 1.0, _ke.UpAxis.Z, 12)
                    if disk
                    else _ke.geometry.create_arrow_data(
                        1.0,
                        0.8,
                        _ke.UpAxis.Z,
                        2.0,
                        0.2,
                        8,
                    )
                )
            if name not in self.materials:
                self.materials[name] = self.app.create_color_material()
                self.materials[name].unlit = True
            view = self.app.scene.add_mesh(
                f"{self.path}/{name}",
                self.meshes[mesh_name],
                self.materials[name],
                transform_source=TransformSource.EXTERNAL_BUFFER,
            )
            view.set_casts_shadow(False)
            view.prim.set_pickable(False)
            self.batches[name] = {"view": view, "matrices": None, "active": False}
        batch = self.batches[name]
        matrices = batch["matrices"]
        if matrices is None or len(matrices) < count or matrices.device != device:
            capacity = max(16, 1 << (count - 1).bit_length())
            batch["matrices"] = torch.zeros(
                (capacity, 4, 4), dtype=torch.float32, device=device
            )
            if device.type == "cuda":
                batch["descriptor"], batch["owner"] = to_external_transform_desc(
                    batch["matrices"]
                )
            batch["set_color"] = True
        return batch

    def sync(self, source):
        options = (
            self.normal,
            self.mode,
            self.force,
            self.scale,
            self.force_scale,
            self.thickness,
        )
        key = (source.contact_revision, options)
        if key == self.revision:
            return
        if not (self.normal or self.mode or self.force):
            self.hide()
            self.revision = key
            return

        values, dt = source.contact_data(velocities=self.mode)
        data = torch.as_tensor(values, dtype=torch.float32)
        self.count = len(data)
        if not self.count:
            self.hide()
            self.revision = (source.contact_revision, options)
            return

        caller = None
        if data.is_cuda:
            caller = torch.cuda.current_stream(data.device)
            if self.stream is not None and self.stream != caller:
                caller.wait_stream(self.stream)
            self.stream = caller
            if self.ready is None:
                self.ready = torch.cuda.Event()

        points, normals = data[:, :3], data[:, 3:6]
        forces = data[:, 6:9] / dt if dt > 0 else torch.zeros_like(points)
        force_sizes = torch.linalg.vector_norm(forces, dim=1)
        # Hide separated, unloaded candidates without compacting CUDA buffers.
        visible = (data[:, 9] <= 0) | (force_sizes >= 1e-4)
        base = self.length * self.scale
        radius = base * 0.025 * self.thickness
        active_batches = []

        def draw(name, direction, lengths, width, active=None):
            batch = self._batch(name, len(data), data.device)
            active = visible if active is None else active & visible
            glyph_matrices(batch["matrices"], points, direction, lengths, width, active)
            batch["active"] = True
            active_batches.append((name, batch))

        if self.normal:
            draw("normal", normals, data.new_full((len(data),), base), radius)
        if self.force:
            lengths = force_sizes * base * self.force_gain * self.force_scale
            draw("force", forces, lengths, radius * 1.25)
        if self.mode:
            modes = contact_modes(normals, forces, data[:, 10:13])
            for mode, name in enumerate(("open", "stick", "slip")):
                draw(
                    name,
                    normals,
                    data.new_full((len(data),), base * 0.015),
                    base * 0.2,
                    modes == mode,
                )

        active_names = {name for name, _ in active_batches}
        for name, batch in self.batches.items():
            if name not in active_names and batch["active"]:
                self._collapse(batch)
                active_batches.append((name, batch))
        self._submit(active_batches)
        self.revision = (source.contact_revision, options)

    def _submit(self, batches):
        if not batches:
            return
        self.version += 1
        if self.stream is not None:
            self.ready.record(self.stream)
        system = self.app.get_scene_render_system()
        for name, batch in batches:
            component = batch["view"].component
            if batch["matrices"].is_cuda:
                desc = batch["descriptor"]
                view = desc.view
                view.version = self.version
                view.ready_event_handle = self.ready.cuda_event
                desc.view = view
                system.set_external_buffer(component, desc)
            else:
                system.update_instances(component, batch["matrices"].numpy())
            if batch.pop("set_color", False):
                color = _ke.ColorLibrary.get(CONTACT_COLORS[name])
                system.set_instance_colors(
                    component, [[color.r, color.g, color.b, 1.0]]
                )

    @staticmethod
    def _collapse(batch):
        # Keep topology stable when contacts disappear or a glyph category is disabled.
        batch["matrices"].zero_()
        batch["matrices"][:, 3, 3] = 1
        batch["active"] = False

    def hide(self):
        if self.stream is not None:
            current = torch.cuda.current_stream(self.stream.device)
            if current != self.stream:
                current.wait_stream(self.stream)
            self.stream = current
        changed = []
        for name, batch in self.batches.items():
            if batch["active"]:
                self._collapse(batch)
                changed.append((name, batch))
        self._submit(changed)

    def release(self):
        if self.stream is not None:
            self.stream.synchronize()
        for batch in self.batches.values():
            batch["view"].remove()
        self.batches.clear()
        self.revision = None
        self.count = 0
        self.stream = None
        self.ready = None
