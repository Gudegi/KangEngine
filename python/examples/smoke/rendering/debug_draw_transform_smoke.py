"""Validate Python-facing debug line and arrow transform helpers."""

from __future__ import annotations

import kangengine as ke


def _translation(matrix):
    value = matrix[3]
    return (float(value.x), float(value.y), float(value.z))


def _close(actual, expected, eps=1.0e-6):
    if any(abs(a - b) > eps for a, b in zip(actual, expected)):
        raise AssertionError(f"transform mismatch: {actual} != {expected}")


def main():
    start = ke.Vec3(1.0, 2.0, 3.0)
    end = ke.Vec3(1.0, 4.0, 3.0)

    arrow = ke.scene.DebugDraw.make_arrow_transform(start, end)
    line = ke.scene.DebugDraw.make_line_transform(start, end)
    if arrow is None or line is None:
        raise AssertionError("non-zero debug primitive returned no transform")
    _close(_translation(arrow), (1.0, 2.0, 3.0))
    _close(_translation(line), (1.0, 3.0, 3.0))

    if ke.scene.DebugDraw.make_arrow_transform(start, start) is not None:
        raise AssertionError("zero-length arrow should return None")
    if ke.scene.DebugDraw.make_line_transform(start, start) is not None:
        raise AssertionError("zero-length line should return None")

    print("PASS: debug draw transform helpers")


if __name__ == "__main__":
    main()
