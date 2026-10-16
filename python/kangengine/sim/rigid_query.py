"""Current rigid poses and queries against their authored surface geometry."""

from __future__ import annotations

from typing import Self

import numpy as np
import torch

from ..utils.env_utils import EnvIdLike, env_id_list


class RigidQueryMixin:
    __slots__ = ()

    def _query_env_ids(self, env_ids):
        if isinstance(env_ids, torch.Tensor) and env_ids.is_cuda:
            raise ValueError(
                "query env_ids must be CPU metadata; CUDA IDs are not downloaded"
            )
        selected = self._selected_env_ids(env_ids)
        return env_id_list(selected, self.world.num_envs)

    def get_root_pose(
        self, env_ids: EnvIdLike = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return current world positions [N, 3] and XYZW rotations [N, 4].

        Includes static actors. CPU reads actor poses; GPU explicitly fetches
        canonical rigid state. Apply queued resets before calling. Environment
        IDs are CPU metadata in the returned row order; None selects this view.
        """
        return self._read_root_pose(self._query_env_ids(env_ids))

    def _read_root_pose(self, selected):
        dynamic_rows = [
            row
            for row, eid in enumerate(selected)
            if (eid, self.obj_id) in self.world.rigids
        ]
        if self.world.sim_device.type == "cuda" and dynamic_rows:
            view = self.world.state.gpu.rigid_data()
            device = torch.device("cuda", view.device_id)
            current = torch.cuda.current_stream(device)
            producer = torch.cuda.ExternalStream(view.stream_handle, device=device)

            # Copy borrowed GPU poses before the next fetch can overwrite them.
            current.wait_stream(producer)
            indices = torch.tensor(
                [
                    self.world.rigid_gpu_row(selected[row], self.obj_id)
                    for row in dynamic_rows
                ],
                dtype=torch.long,
                device=device,
            )
            state = view.torch().index_select(0, indices)
            producer.wait_stream(current)
            if len(dynamic_rows) == len(selected):
                return state[:, :3], state[:, 3:7]

            positions = torch.empty((len(selected), 3), device=device)
            quaternions = torch.empty((len(selected), 4), device=device)
            positions[dynamic_rows] = state[:, :3]
            quaternions[dynamic_rows] = state[:, 3:7]
            host_rows = [row for row in range(len(selected)) if row not in dynamic_rows]
        else:
            positions = torch.empty((len(selected), 3), device=self.world.device)
            quaternions = torch.empty((len(selected), 4), device=self.world.device)
            host_rows = list(range(len(selected)))

        # Static poses stay on the host even in a CUDA simulation.
        host_positions = np.empty((len(host_rows), 3), dtype=np.float32)
        host_quaternions = np.empty((len(host_rows), 4), dtype=np.float32)
        for index, row in enumerate(host_rows):
            rigid = self.world.rigid(selected[row], self.obj_id)
            host_positions[index] = rigid.get_root_position()
            host_quaternions[index] = rigid.get_root_quat_xyzw()
        positions[host_rows] = torch.as_tensor(host_positions, device=positions.device)
        quaternions[host_rows] = torch.as_tensor(
            host_quaternions, device=quaternions.device
        )
        return positions, quaternions

    def _get_mesh_query(self, device):
        from ..asset.rigid_asset import _visual_surface
        from ..geometry._raycast import MeshBatchRaycaster

        key = str(device)
        if key not in self._mesh_queries:
            records = self.world.rigids | self.world.static_rigids
            surfaces = [
                _visual_surface(records[(eid, self.obj_id)].source_data)
                for eid in self.env_ids
            ]
            meshes = list(dict.fromkeys(surfaces))
            asset_ids = [meshes.index(surface) for surface in surfaces]
            self._mesh_queries[key] = MeshBatchRaycaster(meshes, asset_ids, device)

        query = self._mesh_queries[key]
        if query.ready is not None:
            torch.cuda.current_stream(device).wait_event(query.ready)
        return query

    def _query_pose(self, selected, device):
        positions, quaternions = self._read_root_pose(selected)
        if positions.is_cuda and device.type == "cpu":
            raise ValueError("CUDA rigid poses require CUDA ray tensors")
        return positions.to(device), quaternions.to(device)

    def raycast(
        self,
        ray_starts: torch.Tensor,
        ray_directions: torch.Tensor,
        *,
        env_ids: EnvIdLike = None,
        max_distance: float = 1.0e6,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Cast world rays against each instance's authored visual surface.

        Float32 inputs broadcast to [N, ..., 3] in CPU env_ids order. Directions
        are normalized. Returns world hits [N, ..., 3] and distances [N, ...]
        in metres; misses and zero directions return +inf. Surface geometry may
        differ from the physics collider. CUDA inputs use the current Torch stream.
        """
        selected = self._query_env_ids(env_ids)
        positions, quaternions = self._query_pose(selected, ray_starts.device)
        query = self._get_mesh_query(ray_starts.device)
        rows = [self.env_ids.index(eid) for eid in selected]
        return query.raycast(
            query.asset_ids[rows],
            positions,
            quaternions,
            ray_starts,
            ray_directions,
            max_distance,
        )

    def sample_height(
        self,
        positions: torch.Tensor,
        *,
        env_ids: EnvIdLike = None,
        ray_start_height: float | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return surface world Z at float32 XY positions [N, ..., 3].

        Rows follow CPU env_ids. Misses are -inf. Rays start above the rotated
        surface by default; explicit world start heights broadcast to [N, ...].
        Inputs and returned heights share a device; CUDA uses the current stream.
        """
        if (
            isinstance(ray_start_height, torch.Tensor)
            and ray_start_height.device != positions.device
        ):
            raise ValueError("ray_start_height must be on the positions device")
        selected = self._query_env_ids(env_ids)
        root_positions, quaternions = self._query_pose(selected, positions.device)
        query = self._get_mesh_query(positions.device)
        rows = [self.env_ids.index(eid) for eid in selected]

        # A local bounding sphere remains valid under every rigid-body rotation.
        if ray_start_height is None:
            ray_start_height = root_positions[:, 2] + query.radii[rows] + 1.0
            ray_start_height = ray_start_height.view(
                len(selected), *((1,) * (positions.ndim - 2))
            )
        starts = positions.clone()
        starts[..., 2] = ray_start_height
        hits, distances = query.raycast(
            query.asset_ids[rows],
            root_positions,
            quaternions,
            starts,
            positions.new_tensor((0.0, 0.0, -1.0)),
            1.0e6,
        )
        return torch.where(torch.isfinite(distances), hits[..., 2], -torch.inf)

    def _set_rigid_root_state(
        self, env_ids, pos, quat_xyzw, linear_velocity, angular_velocity, immediate
    ) -> Self:
        cuda_ids = isinstance(env_ids, torch.Tensor) and env_ids.is_cuda
        cuda_pose = isinstance(pos, torch.Tensor) and pos.is_cuda
        selected = env_ids if cuda_ids else self._query_env_ids(env_ids)
        if cuda_ids and not cuda_pose:
            raise ValueError("CUDA environment IDs require CUDA root poses")
        if cuda_pose:
            if immediate:
                raise ValueError("CUDA rigid resets require immediate=False")
            count = len(selected)
            zeros = pos.new_zeros((count, 3))
            state = torch.cat(
                (
                    pos.expand(count, 3),
                    quat_xyzw.expand(count, 4),
                    zeros
                    if linear_velocity is None
                    else linear_velocity.expand(count, 3),
                    zeros
                    if angular_velocity is None
                    else angular_velocity.expand(count, 3),
                ),
                dim=-1,
            )
            self.world.set_gpu_root_state_batch(selected, self.obj_id, state)
        else:
            self.world.set_root_state(
                selected,
                self.obj_id,
                pos,
                quat_xyzw,
                linear_velocity,
                angular_velocity,
                immediate=immediate,
            )
        return self
