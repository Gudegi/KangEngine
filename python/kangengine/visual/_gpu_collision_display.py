"""Draw cached collider edges using CUDA body transforms on either renderer."""

import numpy as np
import torch

from .._core import _ke
from ..render import TransformSource
from ..utils import to_external_transform_desc
from ..utils.batched_rotations import quat_xyzw_to_matrix


def _edge_mesh(points):
    # Thin square tubes keep edges visible without a renderer-specific wireframe pass.
    starts, ends = points[:, 0].astype(np.float64), points[:, 1].astype(np.float64)
    direction = ends - starts
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    axes = np.eye(3)[np.argmin(np.abs(direction), axis=1)]
    side = np.cross(direction, axes)
    side /= np.linalg.norm(side, axis=1)[:, None]
    up = np.cross(direction, side)
    radius = max(float(np.ptp(points.reshape(-1, 3), axis=0).max()) * 0.0015, 1e-5)
    offsets = np.stack((side + up, -side + up, -side - up, side - up), axis=1) * radius
    vertices = np.concatenate(
        (starts[:, None] + offsets, ends[:, None] + offsets), axis=1
    )
    faces = np.array(
        [
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ]
    )
    indices = faces[None] + np.arange(len(points))[:, None, None] * 8
    return _ke.scene.MeshData.from_arrays(
        vertices.reshape(-1, 3).astype(np.float32),
        indices.reshape(-1).astype(np.int32),
    )


class GPUCollisionDisplay:
    """Group identical collider meshes; upload only CUDA instance matrices per frame."""

    def __init__(self, app, path, lines, device, resources):
        self.app = app
        self.groups = []
        self.stream = torch.cuda.current_stream(device)
        self.ready = torch.cuda.Event()
        self.version = 0
        self.selected = None
        groups = {}
        for body_idx, points in lines.items():
            points = np.asarray(points, dtype=np.float32)
            # Casting seam vertices to float32 can collapse a previously nonzero edge.
            points = points[np.any(points[:, 0] != points[:, 1], axis=1)]
            if not len(points):
                continue
            groups.setdefault(points.tobytes(), (points, []))[1].append(body_idx)

        if "material" not in resources:
            resources["material"] = app.create_color_material()
            resources["material"].unlit = True
        material = resources["material"]
        try:
            for group_idx, (key, (points, body_ids)) in enumerate(groups.items()):
                if key not in resources:
                    resources[key] = _edge_mesh(points)
                view = app.scene.add_mesh(
                    f"{path}/{group_idx}",
                    resources[key],
                    material,
                    transform_source=TransformSource.EXTERNAL_BUFFER,
                )
                view.set_casts_shadow(False)
                view.prim.set_pickable(False)
                ids = torch.tensor(body_ids, device=device, dtype=torch.long)
                matrices = torch.zeros(
                    (len(body_ids), 4, 4), dtype=torch.float32, device=device
                )
                matrices[:, 3, 3] = 1
                desc, owner = to_external_transform_desc(matrices)
                self.groups.append((view, ids, body_ids, matrices, desc, owner))
        except Exception:
            self.release()
            raise

    def select(self, body_idx):
        if body_idx != self.selected:
            self.selected = body_idx
            self._update_colors()

    def _update_colors(self):
        green = _ke.ColorLibrary.get(_ke.ColorType.GREEN)
        yellow = _ke.ColorLibrary.get(_ke.ColorType.YELLOW)
        for view, ids, body_ids, matrices, desc, owner in self.groups:
            colors = np.tile([green.r, green.g, green.b, 1], (len(body_ids), 1)).astype(
                np.float32
            )
            if self.selected in body_ids:
                colors[body_ids.index(self.selected)] = [
                    yellow.r,
                    yellow.g,
                    yellow.b,
                    1,
                ]
            self.app.get_scene_render_system().set_instance_colors(
                view.component, colors
            )

    def sync(self, poses, valid):
        self.version += 1
        caller_stream = torch.cuda.current_stream(poses.device)
        if self.stream != caller_stream:
            self.stream.wait_stream(caller_stream)
        with torch.cuda.stream(self.stream):
            for view, ids, body_ids, matrices, desc, owner in self.groups:
                selected = poses[ids]
                matrices[:, :3, :3] = quat_xyzw_to_matrix(selected[:, 3:]).transpose(
                    1, 2
                )
                matrices[:, 3, :3] = selected[:, :3]
                # Removed bodies collapse without reading their stale GPU rows.
                if not valid[body_ids].all():
                    dead = torch.as_tensor(
                        np.flatnonzero(~valid[body_ids]), device=poses.device
                    )
                    matrices[dead, :3, :3] = 0
            self.ready.record(self.stream)
        # The caller may reuse its pose buffer on another stream next frame.
        if caller_stream != self.stream:
            caller_stream.wait_event(self.ready)

        for view, ids, body_ids, matrices, desc, owner in self.groups:
            data = desc.view
            data.version = self.version
            data.ready_event_handle = self.ready.cuda_event
            desc.view = data
            self.app.get_scene_render_system().set_external_buffer(view.component, desc)
        if self.version == 1:
            self._update_colors()

    def release(self):
        self.stream.synchronize()
        for view, *_ in self.groups:
            view.remove()
        self.groups.clear()
