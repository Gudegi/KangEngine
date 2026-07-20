"""Validate MJCF visual mesh rgba reaches scene body prim colors."""

from __future__ import annotations

import tempfile
from pathlib import Path

import kangengine as ke


def _close(actual, expected, eps=1.0e-5):
    if abs(float(actual) - float(expected)) > eps:
        raise AssertionError(f"{actual} != {expected}")


def _assert_rgba(actual, expected):
    _close(actual.x, expected[0])
    _close(actual.y, expected[1])
    _close(actual.z, expected[2])
    _close(actual.w, expected[3])


def main():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        mesh_dir = root / "meshes"
        mesh_dir.mkdir()
        (mesh_dir / "body.stl").write_text(
            "\n".join(
                [
                    "solid body",
                    "facet normal 0 0 1",
                    "outer loop",
                    "vertex 0 0 0",
                    "vertex 1 0 0",
                    "vertex 0 1 0",
                    "endloop",
                    "endfacet",
                    "endsolid body",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        mjcf = root / "robot.xml"
        mjcf.write_text(
            "\n".join(
                [
                    '<mujoco model="rgba_smoke">',
                    '  <compiler angle="radian" meshdir="meshes/"/>',
                    "  <asset>",
                    '    <mesh name="body_mesh" file="body.stl"/>',
                    "  </asset>",
                    "  <worldbody>",
                    '    <body name="body">',
                    '      <joint name="root" type="free"/>',
                    '      <geom type="mesh" mesh="body_mesh" rgba="0.4 0.5 0.6 0.7"/>',
                    '      <geom type="box" size="0.1 0.2 0.3" friction="1.7 0.4 0.2"/>',
                    "    </body>",
                    "  </worldbody>",
                    "</mujoco>",
                    "",
                ]
            ),
            encoding="utf-8",
        )

        data = ke.asset.MJCFLoader.load(str(mjcf))
        if len(data.mesh_infos) != 1:
            raise AssertionError("expected one visual mesh info")
        _assert_rgba(data.mesh_infos[0].rgba, (0.4, 0.5, 0.6, 0.7))
        collision_geoms = data.collision_geoms
        if 0 not in collision_geoms or len(collision_geoms[0]) != 1:
            raise AssertionError("expected one body collision geom")
        geom = collision_geoms[0][0]
        mapped_material = ke.physics.mjcf_friction_to_physx([1.7, 0.4, 0.2])
        _close(geom.friction, 1.7)
        _close(mapped_material.static_friction, 1.7)
        _close(mapped_material.dynamic_friction, 1.7)
        _close(mapped_material.restitution, 0.0)
        _close(geom.physics_material.static_friction, 1.7)
        _close(geom.physics_material.dynamic_friction, 1.7)
        _close(geom.physics_material.restitution, 0.0)

        scene = ke.scene.create_backend(ke.scene.BackendType.Native)
        bridge = ke.visual.ArticulationVisual.from_mjcf(str(mjcf), scene, "/robot")
        body = bridge.body_prim(0)
        if body is None:
            raise AssertionError("body prim was not created")
        _assert_rgba(body.get_display_color_alpha(), (0.4, 0.5, 0.6, 0.7))

    print("PASS: MJCF visual rgba smoke completed")


if __name__ == "__main__":
    main()
