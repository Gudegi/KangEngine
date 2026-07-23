"""Validate the public quaternion/array ordering boundary."""

from __future__ import annotations

import numpy as np

import kangengine as ke


def _close(actual, expected):
    if not np.allclose(np.asarray(actual), np.asarray(expected), atol=1.0e-6):
        raise AssertionError(f"quaternion mismatch: {actual} != {expected}")


def main():
    identity = ke.quat(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    _close(identity.to_wxyz(), [1.0, 0.0, 0.0, 0.0])
    _close(np.asarray(identity), [1.0, 0.0, 0.0, 0.0])

    identity_xyzw = ke.quat.from_xyzw([0.0, 0.0, 0.0, 1.0])
    _close(identity_xyzw.to_wxyz(), [1.0, 0.0, 0.0, 0.0])
    _close(identity_xyzw.to_xyzw(), [0.0, 0.0, 0.0, 1.0])

    scene = ke.scene.create_backend(ke.scene.BackendType.Native)
    prim = scene.define_prim("/World/Test", ke.scene.PrimType.Xform)
    prim.set_local_rotation(np.array([1.0, 0.0, 0.0, 0.0]))
    _close(prim.get_local_rotation(), [1.0, 0.0, 0.0, 0.0])

    print("PASS: quaternion bindings use wxyz; xyzw conversion is explicit")


if __name__ == "__main__":
    main()
