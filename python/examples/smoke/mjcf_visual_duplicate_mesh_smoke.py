"""Validate MJCF visual-only mesh geoms do not duplicate collidable mesh geoms."""

from __future__ import annotations

import tempfile
from pathlib import Path

import kangengine as ke


def _write_triangle_stl(path: Path) -> None:
    path.write_text(
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


def main():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        mesh_dir = root / "meshes"
        mesh_dir.mkdir()
        _write_triangle_stl(mesh_dir / "body.stl")

        mjcf = root / "robot.xml"
        mjcf.write_text(
            "\n".join(
                [
                    '<mujoco model="duplicate_visual_smoke">',
                    '  <compiler angle="radian" meshdir="meshes/"/>',
                    "  <asset>",
                    '    <mesh name="body_mesh" file="body.stl"/>',
                    "  </asset>",
                    "  <worldbody>",
                    '    <body name="body">',
                    '      <joint name="root" type="free"/>',
                    (
                        '      <geom type="mesh" mesh="body_mesh" '
                        'contype="0" conaffinity="0" group="1" '
                        'density="0" rgba="0.7 0.7 0.7 1"/>'
                    ),
                    (
                        '      <geom type="mesh" mesh="body_mesh" '
                        'rgba="0.2 0.2 0.2 1"/>'
                    ),
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
            raise AssertionError(
                f"expected one visual mesh info, got {len(data.mesh_infos)}"
            )

        scene = ke.scene.create_backend(ke.scene.BackendType.Native)
        asset = ke.animation.SkeletonBridgeAsset.from_mjcf(str(mjcf))
        bridge = asset.instantiate(scene, "/robot", "", True)
        render_prims = list(bridge.render_prims())
        if len(render_prims) != 1:
            raise AssertionError(
                f"expected one split visual render prim, got {len(render_prims)}"
            )
        if not render_prims[0].get_path().endswith("/visual_0"):
            raise AssertionError(f"unexpected render prim path: {render_prims[0].get_path()}")

    print("PASS: MJCF duplicate visual mesh smoke completed")


if __name__ == "__main__":
    main()
