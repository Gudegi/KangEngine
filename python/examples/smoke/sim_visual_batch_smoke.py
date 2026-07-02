"""Smoke test for low-level C++ SimModel/SimState/SimVisualBatch bindings."""

from __future__ import annotations


import torch

from kangengine import physics
from kangengine.utils import to_external_transform_desc


def _assert_close(actual, expected, eps=1.0e-5):
    if len(actual) != len(expected):
        raise AssertionError(f"length mismatch: {len(actual)} != {len(expected)}")
    for i, (a, b) in enumerate(zip(actual, expected)):
        if abs(float(a) - float(b)) > eps:
            raise AssertionError(f"value {i} mismatch: {a} != {b}")


def main():
    model = physics.SimModel()
    model.set_body_renderables([101, 202])
    if model.body_count != 2 or model.shape_count != 2 or not model.is_valid():
        raise AssertionError("SimModel did not initialize expected topology")
    model.set_body_renderables([101, 202])
    if model.body_count != 2 or model.shape_count != 2 or not model.is_valid():
        raise AssertionError("SimModel repeated set_body_renderables left stale data")
    try:
        model.add_shape(-1, 303)
    except ValueError:
        pass
    else:
        raise AssertionError("SimModel.add_shape accepted a negative body_id")
    try:
        model.add_object_boundary(10, 1)
    except IndexError:
        pass
    else:
        raise AssertionError("SimModel.add_object_boundary accepted an invalid range")

    state = physics.SimState()
    state.resize(3, 2)
    for env_id in range(state.num_envs):
        for body_id in range(state.num_bodies):
            state.set_body_transform(
                env_id,
                body_id,
                [float(env_id), float(body_id), 1.0 + float(env_id + body_id)],
                [0.0, 0.0, 0.0, 1.0],
            )

    _assert_close(state.get_body_pos(2, 1), [2.0, 1.0, 4.0])
    _assert_close(state.get_body_rot(2, 1), [0.0, 0.0, 0.0, 1.0])

    batch = physics.SimVisualBatch()
    batch.set_model(model)
    batch.prepare_from_state(state)

    if batch.renderable_count != 2:
        raise AssertionError(f"unexpected renderable_count {batch.renderable_count}")
    if batch.renderable(0) != 101 or batch.renderable(1) != 202:
        raise AssertionError("renderable handle mapping mismatch")

    try:
        state.body_index(99, 0)
    except IndexError:
        pass
    else:
        raise AssertionError("SimState.body_index accepted an invalid env_id")

    transforms = batch.transforms(1)
    if len(transforms) != 3:
        raise AssertionError(f"unexpected transform count {len(transforms)}")
    if len(transforms[0]) != 16:
        raise AssertionError("mat4 should be returned as 16 column-major floats")

    # Column-major GLM matrix: translation lives at indices 12, 13, 14.
    _assert_close(
        [transforms[2][12], transforms[2][13], transforms[2][14]],
        [2.0, 1.0, 4.0],
    )

    desc = batch.external_transform_desc(1, 9, "smoke_transforms")
    if desc.count != 3:
        raise AssertionError(f"unexpected external desc count {desc.count}")
    if list(desc.view.shape) != [3, 4, 4]:
        raise AssertionError(f"unexpected external view shape {desc.view.shape}")
    if list(desc.view.strides) != [16, 4, 1]:
        raise AssertionError(f"unexpected external view strides {desc.view.strides}")
    if desc.view.name != "smoke_transforms" or desc.view.version != 9:
        raise AssertionError("external view metadata mismatch")

    batch.clear()
    if batch.renderable_count != 0:
        raise AssertionError("SimVisualBatch.clear() left stale transforms")
    try:
        batch.renderable(0)
    except IndexError:
        pass
    else:
        raise AssertionError("SimVisualBatch.renderable accepted an invalid shape_id")

    helper_desc, helper_buffer = to_external_transform_desc(
        torch.eye(4, dtype=torch.float32).reshape(1, 4, 4),
        name="helper_transforms",
        version=11,
    )
    if helper_buffer.shape != (1, 4, 4):
        raise AssertionError("helper SimBuffer shape mismatch")
    if helper_desc.count != 1 or list(helper_desc.view.shape) != [1, 4, 4]:
        raise AssertionError("helper ExternalBufferDesc metadata mismatch")
    if helper_desc.view.name != "helper_transforms" or helper_desc.view.version != 11:
        raise AssertionError("helper ExternalBufferDesc view metadata mismatch")

    print("PASS: SimVisualBatch C++ binding smoke completed")


if __name__ == "__main__":
    main()
