from types import SimpleNamespace
from unittest.mock import Mock

from kangengine.sim.world import (
    BatchCommandBuffer,
    CommandBuffer,
    ControlMode,
    KangSimWorld,
)


def _world():
    return SimpleNamespace(
        gpu_system=SimpleNamespace(
            fetch_articulation_joint_positions=Mock(),
            fetch_articulation_joint_velocities=Mock(),
        )
    )


def test_gpu_explicit_pd_fetches_joint_state_once_for_mixed_commands():
    world = _world()
    full_batch = [
        BatchCommandBuffer(ControlMode.TORQUE, (0,), 0, object()),
        BatchCommandBuffer(ControlMode.PD_EXPLICIT, (0,), 1, object()),
    ]
    per_env = [
        ((0, 2), CommandBuffer(ControlMode.PD_EXPLICIT, object())),
    ]

    KangSimWorld._fetch_gpu_explicit_pd_state(world, full_batch, per_env)

    world.gpu_system.fetch_articulation_joint_positions.assert_called_once_with()
    world.gpu_system.fetch_articulation_joint_velocities.assert_called_once_with()


def test_gpu_non_pd_commands_do_not_fetch_joint_state():
    world = _world()
    full_batch = [
        BatchCommandBuffer(ControlMode.POS, (0,), 0, object()),
    ]
    per_env = [
        ((0, 1), CommandBuffer(ControlMode.TORQUE, object())),
    ]

    KangSimWorld._fetch_gpu_explicit_pd_state(world, full_batch, per_env)

    world.gpu_system.fetch_articulation_joint_positions.assert_not_called()
    world.gpu_system.fetch_articulation_joint_velocities.assert_not_called()
