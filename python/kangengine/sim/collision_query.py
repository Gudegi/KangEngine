"""Collision shape metadata and world poses for rigid and articulation views."""

from __future__ import annotations

import numpy as np
import torch

from ..utils.batched_rotations import quat_xyzw_multiply, quat_xyzw_rotate
from ..utils.env_utils import EnvIdLike, env_id_list


def _shape_pose(view, env_ids):
    if isinstance(env_ids, torch.Tensor) and env_ids.device.type != "cpu":
        raise ValueError("collision query env_ids must be CPU metadata")
    selected = env_id_list(view._selected_env_ids(env_ids), view.world.num_envs)
    records = [view.world.get_object(env_id, view.obj_id) for env_id in selected]
    first = view.first if isinstance(view, CollisionShapeBatchMixin) else view
    if not records:
        device = view.world.device
        if view.world.sim_device.type == "cuda" and any(
            (env_id, view.obj_id) not in view.world.static_rigids for env_id in view.env_ids
        ):
            device = view.world.sim_device
        return torch.empty((0, first.num_collision_shapes, 7), device=device)

    positions, rotations = first._collision_body_pose(selected)
    body_ids = torch.tensor(
        [record._collision_layout[1] for record in records],
        dtype=torch.long, device=positions.device,
    )
    local = torch.as_tensor(
        np.stack([record._collision_layout[3] for record in records]),
        device=positions.device,
    )

    # Each environment may have a different shape-to-body mapping and local pose.
    positions = positions.gather(1, body_ids[..., None].expand(-1, -1, 3))
    rotations = rotations.gather(1, body_ids[..., None].expand(-1, -1, 4))
    return torch.cat((
        positions + quat_xyzw_rotate(rotations, local[..., :3]),
        quat_xyzw_multiply(rotations, local[..., 3:]),
    ), dim=-1)


class CollisionShapeMixin:
    __slots__ = ()

    @property
    def num_collision_shapes(self) -> int:
        return len(self._collision_layout[0])

    @property
    def collision_shape_names(self) -> list[str]:
        """Shape display names in runtime slot order."""
        return list(self._collision_layout[0])

    @property
    def collision_shape_body_ids(self) -> torch.Tensor:
        """CPU logical body indices, shape (S,)."""
        return torch.tensor(self._collision_layout[1], dtype=torch.long)

    @property
    def collision_shape_geom_ids(self) -> torch.Tensor:
        """CPU body-local authored geom indices (S,); -1 means generated geometry."""
        return torch.tensor(self._collision_layout[2], dtype=torch.long)

    def get_collision_shape_pose(self, env_ids: EnvIdLike = None) -> torch.Tensor:
        """Return current world poses (S, 7), or (N, S, 7) with explicit env_ids.

        Poses contain metres XYZ and quaternion XYZW. CPU simulation uses the
        state device; GPU dynamic queries fetch poses and return CUDA tensors.
        Static-only queries use the state device. Apply queued resets first.
        Environment IDs are CPU metadata.
        """
        poses = _shape_pose(self, env_ids)
        return poses[0] if env_ids is None else poses


class CollisionShapeBatchMixin:
    __slots__ = ()

    @property
    def num_collision_shapes(self) -> int:
        return self.first.num_collision_shapes

    @property
    def collision_shape_names(self) -> list[list[str]]:
        """Per-environment names in this view's row order; shape (N, S)."""
        return [record.collision_shape_names for record in self.records]

    @property
    def collision_shape_body_ids(self) -> torch.Tensor:
        """CPU logical body indices, shape (N, S), in this view's row order."""
        return torch.stack([record.collision_shape_body_ids for record in self.records])

    @property
    def collision_shape_geom_ids(self) -> torch.Tensor:
        """CPU body-local geom indices (N, S); -1 means generated geometry."""
        return torch.stack([record.collision_shape_geom_ids for record in self.records])

    def get_collision_shape_pose(self, env_ids: EnvIdLike = None) -> torch.Tensor:
        """Return current world poses (N, S, 7): metres XYZ, quaternion XYZW.

        None selects this view; CPU env_ids select and order rows. CPU simulation
        uses the state device; GPU dynamic queries fetch current CUDA poses
        without a host download. Static-only queries use the state device.
        Apply queued resets before querying.
        """
        return _shape_pose(self, env_ids)
