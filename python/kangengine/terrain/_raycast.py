"""Optional Warp implementation; imported only when mesh queries are used."""

import numpy as np
import torch
import warp as wp


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


class TerrainRaycaster:
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
