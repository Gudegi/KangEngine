"""Optional Warp implementation; imported only when mesh queries are used."""

import math

import numpy as np
import torch
import warp as wp


@wp.kernel
def _raycast(
    mesh: wp.uint64,
    starts: wp.array(dtype=wp.vec3),
    directions: wp.array(dtype=wp.vec3),
    max_distance: float,
    positions: wp.array(dtype=wp.vec3),
    distances: wp.array(dtype=float),
):
    i = wp.tid()
    direction = directions[i]
    length = wp.length(direction)
    position = wp.vec3(wp.inf, wp.inf, wp.inf)
    distance = wp.inf

    if length > 0.0:
        direction = direction / length
        hit = wp.mesh_query_ray(mesh, starts[i], direction, max_distance)
        if hit.result:
            distance = hit.t
            position = starts[i] + distance * direction

    positions[i] = position
    distances[i] = distance


@wp.kernel
def _sample(
    mesh: wp.uint64,
    positions: wp.array(dtype=wp.vec3),
    starts: wp.array(dtype=float),
    output: wp.array(dtype=float),
):
    i = wp.tid()
    p = positions[i]
    origin = wp.vec3(p[0], p[1], starts[i])
    hit = wp.mesh_query_ray(mesh, origin, wp.vec3(0.0, 0.0, -1.0), 1.0e6)
    height = -wp.inf
    if hit.result:
        height = origin[2] - hit.t
    output[i] = height


@wp.kernel
def _raycast_instances(
    meshes: wp.array(dtype=wp.uint64),
    asset_ids: wp.array(dtype=wp.int32),
    root_positions: wp.array(dtype=wp.vec3),
    root_quaternions: wp.array(dtype=wp.quat),
    starts: wp.array(dtype=wp.vec3, ndim=2),
    directions: wp.array(dtype=wp.vec3, ndim=2),
    max_distance: float,
    positions: wp.array(dtype=wp.vec3, ndim=2),
    distances: wp.array(dtype=float, ndim=2),
):
    env, ray = wp.tid()
    direction = directions[env, ray]
    length = wp.length(direction)
    position = wp.vec3(wp.inf, wp.inf, wp.inf)
    distance = wp.inf

    if length > 0.0:
        # Move the ray into the shared mesh frame without rebuilding its BVH.
        direction = direction / length
        rotation = root_quaternions[env]
        local_start = wp.quat_rotate_inv(
            rotation, starts[env, ray] - root_positions[env]
        )
        local_direction = wp.quat_rotate_inv(rotation, direction)
        hit = wp.mesh_query_ray(
            meshes[asset_ids[env]], local_start, local_direction, max_distance
        )
        if hit.result:
            distance = hit.t
            position = starts[env, ray] + distance * direction

    positions[env, ray] = position
    distances[env, ray] = distance


class MeshRaycaster:
    def __init__(self, vertices, indices, device):
        self.device = torch.device(device)
        wp.init()
        self.vertices = torch.tensor(
            np.array(vertices), device=self.device, dtype=torch.float32
        )
        self.indices = torch.tensor(
            np.array(indices).reshape(-1), device=self.device, dtype=torch.int32
        )
        stream = (
            wp.stream_from_torch(torch.cuda.current_stream(self.device))
            if self.device.type == "cuda"
            else None
        )
        with wp.ScopedStream(stream) if stream is not None else wp.ScopedDevice("cpu"):
            self.mesh = wp.Mesh(
                points=wp.from_torch(self.vertices, dtype=wp.vec3),
                indices=wp.from_torch(self.indices),
                support_winding_number=False,
            )
        self.ready = None
        if stream is not None:
            self.ready = torch.cuda.Event()
            self.ready.record(torch.cuda.current_stream(self.device))

    def raycast(self, ray_starts, ray_directions, max_distance):
        if ray_starts.shape[-1] != 3 or ray_directions.shape[-1] != 3:
            raise ValueError("ray starts and directions must have shape [..., 3]")
        if ray_starts.device != ray_directions.device:
            raise ValueError("ray starts and directions must be on the same device")

        # Broadcast ray batches without changing their coordinate dimension.
        starts, directions = torch.broadcast_tensors(ray_starts, ray_directions)
        shape = starts.shape[:-1]
        starts = starts.contiguous().view(-1, 3)
        directions = directions.contiguous().view(-1, 3)
        positions = torch.empty(starts.shape, dtype=torch.float32, device=self.device)
        distances = torch.empty(
            starts.shape[0], dtype=torch.float32, device=self.device
        )

        if starts.shape[0]:
            # Keep borrowed tensor storage alive until the query stream finishes.
            stream = None
            if self.device.type == "cuda":
                torch_stream = torch.cuda.current_stream(self.device)
                torch_stream.wait_event(self.ready)
                starts.record_stream(torch_stream)
                directions.record_stream(torch_stream)
                stream = wp.stream_from_torch(torch_stream)

            wp.launch(
                _raycast,
                dim=starts.shape[0],
                inputs=[
                    self.mesh.id,
                    wp.from_torch(starts, dtype=wp.vec3),
                    wp.from_torch(directions, dtype=wp.vec3),
                    max_distance,
                    wp.from_torch(positions, dtype=wp.vec3),
                    wp.from_torch(distances),
                ],
                device=str(self.device),
                stream=stream,
            )

        return positions.view(*shape, 3), distances.view(shape)

    def sample_height(self, positions, ray_start_height):
        if self.ready is not None:
            torch.cuda.current_stream(self.device).wait_event(self.ready)
        if positions.shape[-1] != 3 or positions.dtype != torch.float32:
            raise ValueError("positions must be a float32 tensor [..., 3]")
        shape = positions.shape[:-1]
        points = positions.contiguous().view(-1, 3)
        starts = torch.as_tensor(
            ray_start_height, device=self.device, dtype=torch.float32
        )
        starts = torch.broadcast_to(starts, shape).contiguous().view(-1)
        result = torch.empty(points.shape[0], device=self.device)
        if points.shape[0]:
            stream = (
                wp.stream_from_torch(torch.cuda.current_stream(self.device))
                if self.device.type == "cuda"
                else None
            )
            wp.launch(
                _sample,
                dim=points.shape[0],
                inputs=[
                    self.mesh.id,
                    wp.from_torch(points, dtype=wp.vec3),
                    wp.from_torch(starts),
                    wp.from_torch(result),
                ],
                device=str(self.device),
                stream=stream,
            )
        return result.view(shape)


class MeshBatchRaycaster:
    """Reuse asset BVHs for one independently posed mesh per environment."""

    def __init__(self, meshes, asset_ids, device):
        self.device = torch.device(device)
        self.queries = tuple(mesh._get_query(self.device) for mesh in meshes)
        self.asset_ids = torch.tensor(asset_ids, dtype=torch.int32, device=self.device)
        radii = [float(np.linalg.norm(mesh.vertices, axis=1).max()) for mesh in meshes]
        self.radii = torch.tensor(
            [radii[index] for index in asset_ids],
            dtype=torch.float32,
            device=self.device,
        )
        self.ready = None
        stream = None
        if self.device.type == "cuda":
            torch_stream = torch.cuda.current_stream(self.device)
            for query in self.queries:
                torch_stream.wait_event(query.ready)
            stream = wp.stream_from_torch(torch_stream)

        with wp.ScopedStream(stream) if stream is not None else wp.ScopedDevice("cpu"):
            self.meshes = wp.array(
                [query.mesh.id for query in self.queries], dtype=wp.uint64
            )
        if stream is not None:
            self.ready = torch.cuda.Event()
            self.ready.record(torch_stream)

    def raycast(
        self,
        asset_ids,
        root_positions,
        root_quaternions,
        ray_starts,
        ray_directions,
        max_distance,
    ):
        if ray_starts.shape[-1] != 3 or ray_directions.shape[-1] != 3:
            raise ValueError("ray starts and directions must have shape [N, ..., 3]")
        if ray_starts.device != ray_directions.device:
            raise ValueError("ray starts and directions must be on the same device")

        # Preserve the environment dimension while flattening each ray pattern.
        starts, directions = torch.broadcast_tensors(ray_starts, ray_directions)
        shape = starts.shape[:-1]
        if starts.ndim < 2 or shape[0] != len(asset_ids):
            raise ValueError(
                "the first ray dimension must match the selected environments"
            )
        count = math.prod(shape[1:])
        starts = starts.contiguous().view(shape[0], count, 3)
        directions = directions.contiguous().view(shape[0], count, 3)
        root_positions = root_positions.contiguous()
        root_quaternions = root_quaternions.contiguous()
        positions = torch.empty_like(starts)
        distances = torch.empty(
            shape[0], count, dtype=torch.float32, device=self.device
        )

        if shape[0] and count:
            stream = None
            if self.ready is not None:
                torch_stream = torch.cuda.current_stream(self.device)
                for tensor in (
                    asset_ids,
                    root_positions,
                    root_quaternions,
                    starts,
                    directions,
                ):
                    tensor.record_stream(torch_stream)
                stream = wp.stream_from_torch(torch_stream)

            wp.launch(
                _raycast_instances,
                dim=(shape[0], count),
                inputs=[
                    self.meshes,
                    wp.from_torch(asset_ids, dtype=wp.int32),
                    wp.from_torch(root_positions, dtype=wp.vec3),
                    wp.from_torch(root_quaternions, dtype=wp.quat),
                    wp.from_torch(starts, dtype=wp.vec3),
                    wp.from_torch(directions, dtype=wp.vec3),
                    max_distance,
                    wp.from_torch(positions, dtype=wp.vec3),
                    wp.from_torch(distances),
                ],
                device=str(self.device),
                stream=stream,
            )
        return positions.view(*shape, 3), distances.view(shape)
