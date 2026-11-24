"""Check CUDA instance handoff, updates, and CPU/GPU path changes."""

from unittest.mock import patch
import numpy as np
import torch
import kangengine as ke
from kangengine.render._ovrtx_scene import OvrtxAppAdapter
from kangengine.utils.sim_buffer import SimBuffer, to_external_transform_desc


def main():
    app = ke.App()
    adapter = OvrtxAppAdapter(256, 192)
    try:
        app._initialize_scene_renderer(256, 192, ke.UpAxis.Y, ke.scene.BackendType.NATIVE,
                                       lambda width, height: adapter)
        app.set_camera_view(position=[0, 0, 5], target=[0, 0, 0])
        app.set_light_ambient([0, 0, 0])
        app.set_light_direction([0, 0, -1])
        material = app.create_pbr_material(base_color=[0, 1, 0, 1])
        view = app.scene.add_mesh("/batch", ke.geometry.create_cube_data(1), material,
                                 transform_source=ke.render.TransformSource.EXTERNAL_BUFFER)
        system = app.get_scene_render_system()
        stream = torch.cuda.Stream()
        ready = torch.cuda.Event()
        with torch.cuda.stream(stream):
            matrices = torch.eye(4, device="cuda")[None].repeat(2, 1, 1)
            matrices[:, 3, 0] = torch.tensor([-0.8, 0.8], device="cuda")
            ready.record()

        def submit(version):
            buffer = SimBuffer(matrices, sim_device="cuda", memory_type="cuda_device", device_id=0,
                               stream_handle=stream.cuda_stream,
                               ready_event_handle=ready.cuda_event, version=version)
            desc, retained = to_external_transform_desc(buffer)
            system.set_external_buffer(view.component, desc)
            return retained

        retained = submit(1)

        def capture():
            for _ in range(12):
                app.draw_frame()
            return app.read_rgb_pixels().astype(float)

        # The renderer may download its final image; transforms must stay on CUDA.
        with patch.object(torch.Tensor, "cpu", side_effect=AssertionError("CPU transform copy")), \
             patch.object(torch.Tensor, "numpy", side_effect=AssertionError("NumPy transform copy")):
            before = capture()
            with patch.object(adapter._stage, "write_attribute", wraps=adapter._stage.write_attribute) as writes:
                retained = submit(1)
                capture()
                assert not any(call.args[1] == "omni:xform" for call in writes.call_args_list)
            with torch.cuda.stream(stream):
                matrices[:, 3, 1] = 0.8
                ready.record()
            retained = submit(2)
            after = capture()
        assert np.abs(before - after).mean() > 1
        system.update_instances(view.component, np.eye(4, dtype=np.float32)[None])
        capture()
        retained = submit(3)
        capture()
        view.remove()
        capture()
        assert not adapter._batches
        print("PASS: CUDA placement, producer event, idle reuse, update, CPU switch, and removal")
    finally:
        app.close()


if __name__ == "__main__":
    main()
