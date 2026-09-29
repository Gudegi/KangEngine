"""D6 constraints, Python batches, and CUDA wrench readback.

Run normally for CPU coverage. Set KANGENGINE_TEST_D6_GPU=1 to include
Direct GPU simulation and Torch CUDA wrench tests.
"""
import gc
import os

import numpy as np
import pytest
import torch
import kangengine as ke

p = ke.physics
IDENTITY = [0., 0., 0., 0., 0., 0., 1.]


@pytest.fixture(autouse=True)
def collect_world_handles():
    # PhysX owns process-wide state; collect handles between worlds.
    gc.collect()
    yield
    gc.collect()


@pytest.fixture(params=["pgs", "tgs", "gpu"])
def backend(request):
    if request.param == "gpu" and os.environ.get("KANGENGINE_TEST_D6_GPU") != "1":
        pytest.skip("set KANGENGINE_TEST_D6_GPU=1 for Direct GPU checks")
    return request.param


def world_for(backend, dt=1/120):
    config = p.PhysicsConfig.z_up()
    config.solver_type = 0 if backend == "pgs" else 1
    config.enable_gpu = backend == "gpu"
    config.dt = dt
    return p.PhysicsWorld(config)


def fixed_config():
    config = p.D6JointConfig()
    config.locked_axes = [True] * 6
    config.frame0 = [0., 0., 3., 0., 0., 0., 1.]
    config.enabled = True
    return config


def steps(world, count=10):
    for _ in range(count):
        world.step()


# Python batch creation and lifecycle.

@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("frame_device", [None, "cpu", "cuda"])
def test_body_batch(native, frame_device):
    def frames(values):
        if frame_device is None:
            return values
        torch = pytest.importorskip("torch")
        if frame_device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA unavailable")
        return torch.tensor(values, device=frame_device, dtype=torch.float64, requires_grad=True)

    # Resolve optional CUDA availability before constructing PhysX resources.
    frames([[0.]])
    p = ke.physics
    world = p.NativePhysicsWorld(p.PhysicsConfig()) if native else p.PhysicsWorld()
    builder = p.ArticulationBuilder()
    builder.add_link("hand", inertia_diagonal=(1., 1., 1.))
    robot = p.Articulation.build(world, builder.build())
    sphere = world.create_dynamic_sphere(.1, [0., 0., 1.])
    other = world.create_dynamic_sphere(.1, [2., 0., 1.])
    identity = [0., 0., 0., 0., 0., 0., 1.]
    frames0 = [[0., 0., 1., 0., 0., 0., 1.], [2., 0., 1., 0., 0., 0., 1.]]
    config = p.D6JointConfig()
    config.enabled = True
    batch = p.D6Batch.create(world, [robot.link("hand"), None], [sphere, other],
                             frames0=frames(frames0), frames1=frames([identity]*2), config=config)
    try:
        for _ in range(60):
            world.step()
        np.testing.assert_allclose(sphere.get_root_position(), [0., 0., 1.], atol=.01)
        np.testing.assert_allclose(other.get_root_position(), [2., 0., 1.], atol=.01)
        batch.set_enabled([0, 1], False)
        batch.set_frames0([0], frames([[0., 0., 2., 0., 0., 0., 1.]]))
        batch.set_frames1([0], frames([[0., 0., .1, 0., 0., 0., 1.]]))
        np.testing.assert_allclose(batch.joints[0].frame0[:3], [0., 0., 2.])
        np.testing.assert_allclose(batch.joints[0].frame1[:3], [0., 0., .1])
        np.testing.assert_allclose(batch.joints[1].frame0, frames0[1])
        np.testing.assert_allclose(batch.joints[1].frame1, identity)
        before = batch.joints[0].frame0
        with pytest.raises(ValueError):
            batch.set_frames0([0, 1], frames([identity, [float("nan")]*7]))
        np.testing.assert_array_equal(batch.joints[0].frame0, before)
        with pytest.raises(ValueError):
            p.D6Batch.create(world, [None], [], frames0=[identity], frames1=[])
        # A failed later row must release earlier joints created by this call.
        class RecordingWorld:
            def __init__(self):
                self.created = []
            def create_d6_joint(self, *args, **kwargs):
                joint = world.create_d6_joint(*args, **kwargs)
                self.created.append(joint)
                return joint
        recording = RecordingWorld()
        with pytest.raises(ValueError):
            p.D6Batch.create(recording, [None, None], [world.joint_body(sphere), None],
                             frames0=[identity]*2, frames1=[identity]*2)
        assert len(recording.created) == 1 and not recording.created[0].valid
        robot.release()
        assert not batch.joints[0].valid and batch.joints[1].valid
    finally:
        batch.release()
        robot.release()
        sphere.release()
        other.release()


# Limits, drives, and scalar readback across solver backends.

@pytest.mark.parametrize("reverse", [False, True])
def test_support_force_and_breakage(backend, reverse):
    world = world_for(backend)
    body = world.create_dynamic_box([.1]*3, [0., 0., 3.], density=1000.)
    config = fixed_config()
    if reverse:
        config.frame0, config.frame1 = config.frame1, config.frame0
    joint = world.create_d6_joint(body if reverse else None,
                                  None if reverse else body, config=config)
    try:
        with pytest.raises(RuntimeError, match="completed physics step"):
            joint.get_wrench()
        steps(world)
        sign = -1 if reverse else 1
        np.testing.assert_allclose(joint.get_wrench(), [0, 0, sign*78.48, 0, 0, 0], atol=.02)
        joint.set_enabled(False)
        np.testing.assert_array_equal(joint.get_wrench(), np.zeros(6))
        joint.set_enabled(True)
        steps(world, 240)  # Let CPU actors sleep before lowering the threshold.
        joint.set_break_force(force=10, torque=1e6)
        steps(world)
        assert joint.valid and joint.broken and not joint.enabled
        np.testing.assert_array_equal(joint.get_wrench(), np.zeros(6))
        with pytest.raises(RuntimeError, match="recreated"):
            joint.set_enabled(True)
    finally:
        joint.release()
        body.release()


@pytest.mark.parametrize("dt", [1/60, 1/240])
def test_drive_cap_and_motor_off(backend, dt):
    world = world_for(backend, dt=dt)
    body = world.create_dynamic_box([.1]*3, [0., 0., 3.], density=1000.)
    config = fixed_config()
    config.motions = [p.D6Motion.FREE] + [p.D6Motion.LOCKED]*5
    config.drives = {p.D6DriveAxis.X: p.D6DriveConfig(stiffness=1000, damping=10, force_limit=5)}
    config.drive_target = [10., 0., 0., 0., 0., 0., 1.]
    joint = world.create_d6_joint(None, body, config=config)
    try:
        steps(world, 5)
        np.testing.assert_allclose(joint.get_wrench()[:3], [5, 0, 78.48], atol=.02)
        # Zero cap turns off only the motor, leaving gravity support active.
        joint.set_drive(p.D6DriveAxis.X, p.D6DriveConfig(stiffness=1000, force_limit=0))
        steps(world, 5)
        np.testing.assert_allclose(joint.get_wrench()[:3], [0, 0, 78.48], atol=.02)
        joint.set_drive(p.D6DriveAxis.X, p.D6DriveConfig(damping=40, force_limit=10))
        joint.set_drive_velocity(linear=[1, 0, 0], angular=[0, 0, 0])
        steps(world, 1)
        assert joint.get_wrench()[0] == pytest.approx(10, abs=.02)
        # The motor's output also participates in force-based breakage.
        joint.set_break_force(force=79, torque=1e6)
        steps(world)
        assert joint.broken
    finally:
        joint.release()
        body.release()


@pytest.mark.parametrize("drive_axis, axis", [(p.D6DriveAxis.TWIST, 0),
                                               (p.D6DriveAxis.SWING, 1),
                                               (p.D6DriveAxis.SLERP, 2)])
def test_angular_drive_cap_and_torque_break(backend, drive_axis, axis):
    world = world_for(backend)
    body = world.create_dynamic_box([.1]*3, [0., 0., 0.], density=1000.)
    config = p.D6JointConfig()
    config.enabled = True
    config.drives = {drive_axis: p.D6DriveConfig(stiffness=1000, damping=10, force_limit=2)}
    target = list(IDENTITY)
    target[3+axis], target[6] = np.sin(.5), np.cos(.5)
    config.drive_target = target
    joint = world.create_d6_joint(None, body, config=config)
    try:
        steps(world, 1)
        expected = np.array([0., 0., 78.48, 0., 0., 0.])
        expected[3+axis] = 2
        wrench = joint.get_wrench()
        if drive_axis == p.D6DriveAxis.SLERP:
            # SLERP uses quaternion Jacobians; a capped solver row need not
            # produce exactly force_limit on a world torque component.
            assert 0 < wrench[3+axis] <= 2.02
            expected[3+axis] = wrench[3+axis]
        np.testing.assert_allclose(wrench, expected, atol=.02)
        joint.set_break_force(force=1e6, torque=.1)
        steps(world, 5)
        assert joint.broken
    finally:
        joint.release()
        body.release()


@pytest.mark.parametrize("solver", ["pgs", "tgs"])
def test_linear_and_angular_limits(solver):
    world = world_for(solver)
    body = world.create_dynamic_box([.1]*3, [0., 0., 0.], density=1000.)
    config = p.D6JointConfig()
    config.enabled = True
    config.motions = [p.D6Motion.LIMITED, p.D6Motion.LOCKED, p.D6Motion.LOCKED,
                      p.D6Motion.LIMITED, p.D6Motion.LOCKED, p.D6Motion.LOCKED]
    config.linear_limits = [[-.05, .05], [-1, 1], [-1, 1]]
    config.twist_limits = [-.2, .2]
    config.drives = {p.D6DriveAxis.X: p.D6DriveConfig(stiffness=1000, damping=20, force_limit=10),
                     p.D6DriveAxis.TWIST: p.D6DriveConfig(stiffness=100, damping=5, force_limit=2)}
    config.drive_target = [1, 0, 0, np.sin(.5), 0, 0, np.cos(.5)]
    joint = world.create_d6_joint(None, body, config=config)
    try:
        steps(world, 240)
        assert body.get_root_position()[0] == pytest.approx(.05, abs=.003)
        q = body.get_root_rotation()
        assert 2*np.arctan2(q[0], q[3]) == pytest.approx(.2, abs=.02)
        joint.set_linear_limit(p.D6Axis.X, -.1, .1)
        joint.set_twist_limit(-.4, .4)
        steps(world, 240)
        assert body.get_root_position()[0] == pytest.approx(.1, abs=.003)
        q = body.get_root_rotation()
        assert 2*np.arctan2(q[0], q[3]) == pytest.approx(.4, abs=.02)
    finally:
        joint.release()
        body.release()


def test_pgs_torque_is_about_attachment():
    world = world_for("pgs")
    body = world.create_dynamic_box([.1]*3, [-.2, 0., 3.], density=1000.)
    config = fixed_config()
    config.frame1 = [.2, 0, 0, 0, 0, 0, 1]
    joint = world.create_d6_joint(None, body, config=config)
    try:
        steps(world)
        np.testing.assert_allclose(joint.get_wrench(), [0, 0, 78.48, 0, .2*78.48, 0], atol=.02)
    finally:
        joint.release()
        body.release()


def test_batch_preserves_configuration_and_prevalidates():
    world = world_for("pgs")
    bodies = [world.create_dynamic_box([.1]*3, [x, 0, 3.], density=1000.) for x in (0., 2.)]
    config = fixed_config()
    config.enabled = False
    config.motions = [p.D6Motion.LIMITED] + [p.D6Motion.LOCKED]*5
    config.linear_limits = [[-.1, .1], [-1, 1], [-1, 1]]
    config.drives = {p.D6DriveAxis.X: p.D6DriveConfig(stiffness=200, damping=30, force_limit=5)}
    config.drive_target = [.05, 0, 0, 0, 0, 0, 1]
    config.drive_linear_velocity = [.1, 0, 0]
    config.break_force, config.break_torque = 1000, 100
    batch = p.D6Batch.create(world, [None]*2, bodies,
                            frames0=[[x, 0, 3, 0, 0, 0, 1] for x in (0, 2)],
                            frames1=[IDENTITY]*2, config=config)
    try:
        for joint in batch.joints:
            snapshot = joint.config
            assert snapshot.motions == config.motions
            np.testing.assert_allclose(snapshot.linear_limits, config.linear_limits)
            np.testing.assert_allclose(snapshot.drive_target, config.drive_target)
            np.testing.assert_allclose(snapshot.drive_linear_velocity, config.drive_linear_velocity)
            assert snapshot.drives[p.D6DriveAxis.X].force_limit == 5
            assert snapshot.break_force == 1000 and snapshot.break_torque == 100
            snapshot.break_force = 1
            assert joint.config.break_force == 1000
        np.testing.assert_array_equal(batch.get_wrenches(), np.zeros((2, 6)))
        assert batch.get_wrenches([]).shape == (0, 6)
        for axis in (p.D6Axis.TWIST, p.D6Axis.SWING1, p.D6Axis.SWING2):
            batch.set_motion([0], axis, p.D6Motion.FREE)
        # Row 0 permits SLERP, row 1 does not: neither must be changed.
        with pytest.raises(ValueError, match="unlocked"):
            batch.set_drive([0, 1], p.D6DriveAxis.SLERP, p.D6DriveConfig(stiffness=10))
        assert p.D6DriveAxis.SLERP not in batch.joints[0].config.drives
        batch.set_twist_limit([0, 1], -.3, .4)
        batch.set_swing_limit([0, 1], .5, .6)
        batch.set_linear_limit([0, 1], p.D6Axis.X, -.2, .2)
        batch.set_drive_targets([0, 1], [IDENTITY]*2)
        batch.set_drive_velocity([0, 1], [0, 0, 0], [0, 0, 0])
        with pytest.raises(ValueError):
            batch.set_break_force([0, 1], -1, 100)
        assert batch.joints[0].config.break_force == 1000
        batch.set_break_force([1], 1, 100)
        batch.set_enabled([1], True)
        steps(world)
        assert batch.broken == [False, True]
        with pytest.raises(RuntimeError, match="recreated"):
            batch.set_enabled([0, 1], True)
        assert batch.enabled == [False, False]
    finally:
        batch.release()
        for body in bodies:
            body.release()


@pytest.mark.parametrize("field,value", [
    ("linear_limits", [[1, -1]]*3), ("twist_limits", [-7, 7]),
    ("swing_limits", [0, 1]), ("break_force", float("inf")),
    ("drive_target", [0]*7), ("drive_linear_velocity", [float("nan"), 0, 0]),
    ("drives", {p.D6DriveAxis.X: p.D6DriveConfig(force_limit=-1)}),
    ("drives", {p.D6DriveAxis.SLERP: p.D6DriveConfig(stiffness=1),
                p.D6DriveAxis.TWIST: p.D6DriveConfig(stiffness=1)}),
])
def test_invalid_configuration(field, value):
    config = p.D6JointConfig()
    setattr(config, field, value)
    with pytest.raises(ValueError):
        config.validate()


@pytest.mark.parametrize("axis", [1, 2])
def test_swing_cone_limit_and_runtime_drive_switch(axis):
    world = world_for("pgs")
    body = world.create_dynamic_box([.1]*3, [0., 0., 0.], density=1000.)
    config = p.D6JointConfig()
    config.enabled = True
    config.motions = [p.D6Motion.LOCKED]*3 + [p.D6Motion.FREE, p.D6Motion.LIMITED, p.D6Motion.LIMITED]
    config.swing_limits = [.2, .3]
    config.drives = {p.D6DriveAxis.SWING: p.D6DriveConfig(stiffness=100, damping=10, force_limit=2)}
    target = list(IDENTITY)
    target[3+axis], target[6] = np.sin(.5), np.cos(.5)
    config.drive_target = target
    joint = world.create_d6_joint(None, body, config=config)
    try:
        steps(world, 240)
        q = body.get_root_rotation()
        assert 2*np.arctan2(q[axis], q[3]) == pytest.approx(config.swing_limits[axis-1], abs=.02)
        # Switch angular motor model with a zero-capped prior motor. Limits remain.
        joint.set_drive(p.D6DriveAxis.SWING, p.D6DriveConfig(stiffness=100, force_limit=0))
        joint.set_drive(p.D6DriveAxis.SLERP, p.D6DriveConfig(stiffness=100, damping=10, force_limit=2))
        joint.set_drive_target(IDENTITY)
        steps(world, 240)
        q = body.get_root_rotation()
        assert abs(2*np.arctan2(q[axis], q[3])) < .02
    finally:
        joint.release()
        body.release()


# Batched CUDA readback and Torch buffer lifetime.

@pytest.fixture
def gpu_scene():
    if os.environ.get("KANGENGINE_TEST_D6_GPU") != "1":
        pytest.skip("set KANGENGINE_TEST_D6_GPU=1 for Direct GPU checks")
    world = world_for("gpu")
    bodies = [world.create_dynamic_box([.1]*3, [2.*i, 0., 3.], density=1000.) for i in range(16)]
    config = fixed_config()
    batch = p.D6Batch.create(world, [None]*len(bodies), bodies,
        frames0=[[2.*i, 0, 3, 0, 0, 0, 1] for i in range(len(bodies))],
        frames1=[[0, 0, 0, 0, 0, 0, 1]]*len(bodies), config=config)
    gpu = p.PhysicsGpuSystem(world, p.GpuPhysicsConfig(max_contact_pairs=64, max_contact_points=128))
    gpu.init()
    steps(world, 5)
    yield world, bodies, batch, gpu
    batch.release()
    gpu.invalidate()
    for body in bodies:
        body.release()


@pytest.mark.parametrize("other_stream", [False, True])
def test_cuda_batch_values_reuse_and_lifetime(gpu_scene, monkeypatch, other_stream):
    world, bodies, batch, gpu = gpu_scene
    reference = torch.tensor(batch.get_wrenches(), device="cuda")
    stream = torch.cuda.Stream() if other_stream else torch.cuda.current_stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        # The batched path must never fall back to scalar host readback.
        def forbidden(*args, **kwargs):
            raise AssertionError("scalar/host readback was called")
        monkeypatch.setattr(p.D6Joint, "get_wrench", forbidden)
        monkeypatch.setattr(p.D6Batch, "get_wrenches", forbidden)
        batch.fetch_wrenches(gpu_system=gpu)
        tensor = batch.wrenches
        assert tensor.is_cuda and tensor.dtype == torch.float32 and tensor.shape == (16, 6)
        assert tensor.is_contiguous()
        torch.testing.assert_close(tensor, reference, atol=.02, rtol=1e-4)
        for _ in range(20):
            # Poisoning the tensor must be ordered before the next native write.
            tensor.fill_(12345)
            batch.fetch_wrenches()
            assert batch.wrenches.data_ptr() == tensor.data_ptr()
            torch.testing.assert_close(tensor, reference, atol=.02, rtol=1e-4)
        snapshot = tensor.clone()
        batch.set_enabled([1, 5, 12], False)
        batch.fetch_wrenches()  # Zero immediately; do not return old disabled forces.
        torch.testing.assert_close(tensor[[1, 5, 12]], torch.zeros((3, 6), device="cuda"))
        batch.set_enabled([1, 5, 12], True)
        world.step()
        batch.fetch_wrenches()
        torch.testing.assert_close(tensor[:, 2], torch.full((16,), 78.48, device="cuda"), atol=.02, rtol=1e-4)
        # Torch owns the allocation, including after explicit physics teardown.
        batch.release()
        gpu.invalidate()
        torch.testing.assert_close(tensor, snapshot, atol=.02, rtol=1e-4)
    torch.cuda.current_stream().wait_stream(stream)


def test_multiple_batches_streams_and_released_joint(gpu_scene):
    world, bodies, batch, gpu = gpu_scene
    first = p.D6Batch(batch.joints[:4])
    second = p.D6Batch(batch.joints[4:])
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    with torch.cuda.stream(streams[0]):
        first.fetch_wrenches(gpu_system=gpu)
        saved = first.wrenches.clone()
    with torch.cuda.stream(streams[1]):
        second.fetch_wrenches(gpu_system=gpu)
        torch.testing.assert_close(second.wrenches[:, 2], torch.full((12,), 78.48, device="cuda"), atol=.02, rtol=1e-4)
    with torch.cuda.stream(streams[0]):
        first.fetch_wrenches()
        torch.testing.assert_close(first.wrenches, saved, atol=.02, rtol=1e-4)
    for stream in streams:
        torch.cuda.current_stream().wait_stream(stream)
    first.joints[0].release()
    before = first.wrenches.clone()
    with pytest.raises(RuntimeError, match="released"):
        first.fetch_wrenches()
    torch.testing.assert_close(first.wrenches, before)
    # Other batches remain readable after unrelated deletion.
    second.fetch_wrenches()
    gpu.invalidate()
    with pytest.raises(RuntimeError, match="initialized"):
        second.fetch_wrenches()
    gpu.init()
    second.fetch_wrenches()


def test_native_output_validation_and_empty_batch(gpu_scene):
    _, _, batch, gpu = gpu_scene
    from kangengine.utils import to_gpu_array_view
    with pytest.raises(RuntimeError, match="first"):
        batch.fetch_wrenches()
    with pytest.raises(RuntimeError, match="fetch_wrenches"):
        _ = batch.wrenches
    for tensor in (torch.empty((16, 6)),
                   torch.empty((16, 6), device="cuda", dtype=torch.float64),
                   torch.empty((16, 12), device="cuda")[:, ::2],
                   torch.empty((1, 6), device="cuda")):
        view = to_gpu_array_view(tensor, dtype=tensor.dtype)
        with pytest.raises(ValueError, match="contiguous CUDA"):
            gpu.fetch_d6_wrenches(batch.joints, view)
    empty = p.D6Batch([])
    empty.fetch_wrenches(gpu_system=gpu)
    assert empty.wrenches.shape == (0, 6) and empty.wrenches.is_cuda
    batch.set_enabled(range(16), False)
    batch.fetch_wrenches(gpu_system=gpu)
    assert torch.count_nonzero(batch.wrenches).item() == 0


def test_breakage_and_new_joint_registration(gpu_scene):
    world, bodies, batch, gpu = gpu_scene
    batch.joints[2].set_break_force(10, 1e6)
    world.step()
    assert batch.joints[2].broken
    batch.fetch_wrenches(gpu_system=gpu)
    torch.testing.assert_close(batch.wrenches[2], torch.zeros(6, device="cuda"))
    config = batch.joints[0].config
    batch.joints[0].release()
    replacement = world.create_d6_joint(None, bodies[0], config=config)
    subset = p.D6Batch([replacement])
    try:
        with pytest.raises(RuntimeError, match="completed physics step"):
            subset.fetch_wrenches(gpu_system=gpu)
        world.step()
        subset.fetch_wrenches(gpu_system=gpu)
        torch.testing.assert_close(subset.wrenches[:, 2], torch.tensor([78.48], device="cuda"), atol=.02, rtol=1e-4)
    finally:
        subset.release()


def test_steady_fetch_has_no_host_force_transfers(gpu_scene):
    _, _, batch, gpu = gpu_scene
    batch.fetch_wrenches(gpu_system=gpu)
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as profile:
        for _ in range(5):
            batch.fetch_wrenches()
            _ = batch.wrenches[:, :3].norm(dim=-1)
    names = [event.name for event in profile.events()]
    assert any("scatterD6WrenchesKernel" in name for name in names)
    assert not any("DtoH" in name or "HtoD" in name or "cudaEventSynchronize" in name for name in names), names
