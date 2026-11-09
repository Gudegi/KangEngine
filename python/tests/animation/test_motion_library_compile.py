"""Compiled motion queries preserve state values and buffer lifecycle."""

from dataclasses import fields, is_dataclass

import numpy as np
import pytest
import torch

import kangengine as ke


@pytest.fixture
def motion_data(tmp_path):
    torch._dynamo.reset()
    path = tmp_path / "robot.xml"
    path.write_text(
        '<mujoco><worldbody><body name="root"><geom type="sphere" size="0.1"/>'
        '<body name="child" pos="0 0 1"><joint name="hinge" axis="0 1 0"/>'
        '<geom type="sphere" size="0.1"/></body></body></worldbody></mujoco>'
    )
    data = ke.asset.MJCFLoader.load(str(path), order="DFS")
    layout = ke.animation.ArticulationCoordinateLayout.from_data(
        data=data, free_root=True
    )
    q = np.zeros((3, layout.nq), dtype=np.float32)
    q[:, 0] = [0, 1, 3]
    q[:, 3] = np.cos(np.array([0, 0.2, 0.4]))
    q[:, 6] = np.sin(np.array([0, 0.2, 0.4]))
    q[:, 7] = [0.1, 0.3, -0.2]
    motion = ke.animation.ArticulationMotion.from_arrays(
        layout=layout, q=q, qd=np.ones((3, layout.nv), dtype=np.float32), fps=2.0
    )
    yield motion, layout
    torch._dynamo.reset()


def assert_state_close(actual, expected):
    assert type(actual) is type(expected)
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    elif is_dataclass(expected):
        for field in fields(expected):
            assert_state_close(getattr(actual, field.name), getattr(expected, field.name))
    elif isinstance(expected, tuple):
        for a, b in zip(actual, expected, strict=True):
            assert_state_close(a, b)
    else:
        assert actual == expected


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("backend", ["physx", "newton", "mujoco"])
@torch.inference_mode()
def test_compiled_robot_query_matches_eager(motion_data, device, backend):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    motion, layout = motion_data
    adapter_type = {
        "physx": ke.adapters.physx.PhysXMotionAdapter,
        "newton": ke.adapters.newton.NewtonMotionAdapter,
        "mujoco": ke.adapters.mujoco.MuJoCoMotionAdapter,
    }[backend]
    eager = ke.animation.MotionLibrary(
        [motion], adapter=adapter_type(layout), device=device
    )
    compiled = ke.animation.MotionLibrary(
        [motion], adapter=adapter_type(layout), device=device
    )
    compiled.compile()
    compiled.compile()
    for batch in (4, 9, 1):
        ids = torch.zeros(batch, dtype=torch.long, device=device)
        times = torch.linspace(-0.1, 1.1, batch, device=device)
        assert_state_close(
            compiled.sample_robot_state(ids, times, loop=False),
            eager.sample_robot_state(ids, times, loop=False),
        )
    frames = torch.tensor([0, 1, 2], device=device)
    assert_state_close(
        compiled.sample_robot_state(0, frame_ids=frames),
        eager.sample_robot_state(0, frame_ids=frames),
    )
    assert_state_close(
        compiled.sample_robot_state(0, times=torch.tensor([0.15, 0.8], device=device)),
        eager.sample_robot_state(0, times=torch.tensor([0.15, 0.8], device=device)),
    )


@pytest.mark.parametrize("precompute", [False, True])
@torch.inference_mode()
def test_compiled_individual_queries_and_fk(motion_data, precompute):
    motion, layout = motion_data
    libraries = [
        ke.animation.MotionLibrary(
            [motion],
            adapter=ke.adapters.physx.PhysXMotionAdapter(layout),
            precompute_kinematics=precompute,
        )
        for _ in range(2)
    ]
    eager, compiled = libraries
    compiled.compile()
    times = torch.tensor([[0.1, 0.3], [0.4, 0.9]])
    for method in ("sample_skel_state", "sample_arti_state", "sample_body_state"):
        actual = getattr(compiled, method)(0, times)
        expected = getattr(eager, method)(0, times)
        assert_state_close(actual, expected)
        if method != "sample_body_state":
            assert_state_close(
                compiled.forward_kinematics(actual), eager.forward_kinematics(expected)
            )

    # A newly uploaded clip must be visible to already-compiled queries.
    for library in libraries:
        library.add_motion(motion, name="second")
    assert_state_close(
        compiled.sample_robot_state(torch.tensor([0, 1]), times[0]),
        eager.sample_robot_state(torch.tensor([0, 1]), times[0]),
    )
    if torch.cuda.is_available():
        for library in libraries:
            library.to("cuda:0")
        ids = torch.tensor([0, 1], device="cuda:0")
        cuda_times = times[0].cuda()
        actual, _ = compiled.sample_robot_state(ids, cuda_times)
        expected, _ = eager.sample_robot_state(ids, cuda_times)
        assert_state_close(actual, expected)
        assert_state_close(
            compiled.forward_kinematics(actual), eager.forward_kinematics(expected)
        )
