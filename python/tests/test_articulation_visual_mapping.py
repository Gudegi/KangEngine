"""ExternalBuffer visuals preserve physical link IDs when meshes are absent."""

from types import SimpleNamespace

import numpy as np
import pytest

from kangengine.visual.sim.world_visualizer import ArticulationCPUExternalBackend


@pytest.mark.parametrize("render_body_ids", [None, (0, 2), (2, 0, 2)])
def test_cpu_renderables_follow_their_physical_links(render_body_ids):
    # Four links, with distinct poses; the last link can have no visual mesh.
    positions = np.array([
        [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.]],
        [[21., 22., 23.], [24., 25., 26.], [27., 28., 29.], [30., 31., 32.]],
    ], dtype=np.float32)
    rotations = np.tile([0., 0., 2**-.5, 2**-.5], (4, 1))
    robots = [SimpleNamespace(
        get_link_positions=lambda env_id=env_id: positions[env_id],
        get_link_rotations=lambda: rotations,
    ) for env_id in range(2)]
    world = SimpleNamespace(articulation=lambda env_id, obj_id: robots[env_id])
    buffers = {}
    renderer = SimpleNamespace(set_renderable_external_buffer=buffers.__setitem__)
    app = SimpleNamespace(get_renderer=lambda: renderer)
    expected_ids = tuple(range(4)) if render_body_ids is None else render_body_ids
    handles = tuple(range(101, 101 + len(expected_ids)))
    backend = ArticulationCPUExternalBackend(
        app, world, 0, (1, 0), [object() for _ in range(4)], handles,
        render_body_ids=render_body_ids,
    )
    try:
        assert backend.num_bodies == 4
        for _ in range(2):
            backend.sync()
            assert set(buffers) == set(handles)
            for shape_id, (handle, body_id) in enumerate(zip(handles, expected_ids)):
                assert backend.body_id_from_render_handle(handle) == body_id
                assert buffers[handle].count == 2
                matrices = backend._batch.transforms(shape_id)
                for row, env_id in enumerate((1, 0)):
                    expected = np.eye(4)
                    expected[:3, :3] = [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]
                    expected[:3, 3] = positions[env_id, body_id]
                    actual = np.asarray(matrices[row]).reshape(4, 4, order="F")
                    np.testing.assert_allclose(actual, expected, atol=1e-6)
            positions += 1.
        assert backend.body_id_from_render_handle(999) is None
    finally:
        backend.release()


def test_cpu_meshless_articulation_skips_pose_and_renderer_access():
    # No world or renderer is needed when every link has no visual mesh.
    backend = ArticulationCPUExternalBackend(
        None, None, 0, (0, 1), [object(), object()], (), render_body_ids=(),
    )
    try:
        assert backend.num_bodies == 2
        assert backend.sync() is backend
        assert backend.body_id_from_render_handle(101) is None
    finally:
        backend.release()
