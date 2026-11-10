"""State-machine precedence, independent clocks, and device-resident updates."""

import pytest
import torch

import kangengine as ke


@pytest.fixture(params=["cpu", "cuda:0"])
def device(request):
    if request.param.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    return request.param


def test_batch_priority_dwell_and_single_transition(device):
    edge = ke.utils.FSMTransition
    fsm = ke.utils.BatchFSM(
        {"idle": "Idle", "moving": "Moving", "stopping": "Stopping"},
        [
            edge("start", "idle", "moving", "move"),
            edge("stop", "moving", "stopping", "stop"),
            edge("stand", "stopping", "idle", "stopped", min_duration=0.5),
            edge("restart", "stopping", "moving", "move", priority=10),
        ],
        initial_state="idle",
        num_envs=3,
        device=device,
    )
    state_view = fsm.state_ids
    clock_view = fsm.elapsed
    yes = torch.ones(3, dtype=torch.bool, device=device)
    no = ~yes
    fsm.step(0.1, {"move": yes, "stop": yes, "stopped": yes})
    assert fsm.state_ids.tolist() == [1, 1, 1]
    assert fsm.transition_ids.tolist() == [0, 0, 0]
    fsm.step(0.1, {"move": no, "stop": yes, "stopped": yes})
    assert fsm.state_ids.tolist() == [2, 2, 2]
    changed = fsm.step(0.2, {"move": no, "stop": yes, "stopped": yes})
    assert not changed.any()
    assert fsm.transition_ids.tolist() == [-1, -1, -1]
    move = torch.tensor([True, False, False], device=device)
    fsm.step(0.3, {"move": move, "stop": yes, "stopped": yes})
    assert fsm.state_ids.tolist() == [1, 0, 0]
    assert fsm.transition_ids.tolist() == [3, 2, 2]
    assert fsm.state_ids is state_view and fsm.elapsed is clock_view
    assert torch.count_nonzero(fsm.elapsed) == 0

    fsm.step(0.1, {"move": no, "stop": no, "stopped": no})
    fsm.reset(torch.tensor([0, 2], device=device), state="stopping")
    assert fsm.state_ids.tolist() == [2, 0, 2]
    torch.testing.assert_close(fsm.elapsed, fsm.elapsed.new_tensor([0, 0.1, 0]))


def test_self_transition_and_definition_order(device):
    edge = ke.utils.FSMTransition
    fsm = ke.utils.BatchFSM(
        {"idle": "Idle", "other": "Other"},
        [
            edge("repeat", "idle", "idle", "trigger"),
            edge("leave", "idle", "other", "trigger"),
        ],
        initial_state="idle",
        device=device,
    )
    changed = fsm.step(0.5, {"trigger": torch.tensor([True], device=device)})
    assert changed.tolist() == [True]
    assert fsm.transition_ids.tolist() == [0]
    assert fsm.elapsed.tolist() == [0]


def test_missing_signal_does_not_partially_update():
    fsm = ke.utils.BatchFSM(
        {"a": "A", "b": "B"},
        [ke.utils.FSMTransition("next", "a", "b", "ready")],
        initial_state="a",
    )
    with pytest.raises(KeyError):
        fsm.step(0.1, {})
    assert fsm.elapsed.tolist() == [0]
    assert fsm.state_ids.tolist() == [0]


def test_update_never_reads_device_values_on_host(device, monkeypatch):
    fsm = ke.utils.BatchFSM(
        {"a": "A", "b": "B"},
        [ke.utils.FSMTransition("next", "a", "b", "ready")],
        initial_state="a",
        num_envs=4096,
        device=device,
    )
    ready = torch.ones(4096, dtype=torch.bool, device=device)

    def host_read(*args, **kwargs):
        raise AssertionError("FSM attempted a host read or dynamic compaction")

    with monkeypatch.context() as patch:
        for name in ("item", "cpu", "tolist", "numpy", "nonzero", "__bool__"):
            patch.setattr(torch.Tensor, name, host_read)
        fsm.step(1 / 60, {"ready": ready})
    assert torch.all(fsm.state_ids == fsm.state_idx("b"))
