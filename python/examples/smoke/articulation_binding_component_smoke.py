"""Validate ArticulationVisual annotates body/render prims with articulation binding."""

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


def _write_mjcf(path: Path, mesh_dir: Path) -> None:
    _write_triangle_stl(mesh_dir / "body.stl")
    path.write_text(
        "\n".join(
            [
                '<mujoco model="binding_smoke">',
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
                    'rgba="0.4 0.5 0.6 1"/>'
                ),
                '      <geom type="box" size="0.1 0.1 0.1" rgba="0.2 0.2 0.2 1"/>',
                "    </body>",
                "  </worldbody>",
                "</mujoco>",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _check_binding(component, role, body_index, body_name, root_path):
    if component is None:
        raise AssertionError("missing ArticulationBindingComponent")
    if component.role != role:
        raise AssertionError(f"role mismatch: {component.role} != {role}")
    if component.body_index != body_index:
        raise AssertionError(
            f"body index mismatch: {component.body_index} != {body_index}"
        )
    if component.body_name != body_name:
        raise AssertionError(f"body name mismatch: {component.body_name}")
    if component.articulation_root_path != root_path:
        raise AssertionError(f"root path mismatch: {component.articulation_root_path}")


def _check_articulation_root(
    component,
    root_path,
    asset_path,
    body_count,
    render_prim_count,
    split_visual_geoms,
):
    if component is None:
        raise AssertionError("missing ArticulationComponent")
    if component.root_path != root_path:
        raise AssertionError(f"root path mismatch: {component.root_path}")
    if component.asset_path != asset_path:
        raise AssertionError(f"asset path mismatch: {component.asset_path}")
    if component.body_count != body_count:
        raise AssertionError(f"body count mismatch: {component.body_count}")
    if component.render_prim_count != render_prim_count:
        raise AssertionError(
            f"render prim count mismatch: {component.render_prim_count}"
        )
    if component.split_visual_geoms != split_visual_geoms:
        raise AssertionError(
            f"split visual geoms mismatch: {component.split_visual_geoms}"
        )


def _check_collision_shape(component):
    if component is None:
        raise AssertionError("missing CollisionShapeComponent")
    if component.shape_type != ke.scene.CollisionShapeType.Box:
        raise AssertionError(f"shape type mismatch: {component.shape_type}")
    if component.source_geom_index != 0:
        raise AssertionError(
            f"source geom index mismatch: {component.source_geom_index}"
        )
    if component.has_from_to:
        raise AssertionError("box collision shape should not use from/to")
    size = component.size
    expected = (0.1, 0.1, 0.1)
    for actual, target in zip((size.x, size.y, size.z), expected):
        if abs(actual - target) > 1e-5:
            raise AssertionError(f"size mismatch: {size}")
    if abs(component.static_friction - 1.0) > 1e-5:
        raise AssertionError(f"static friction mismatch: {component.static_friction}")
    if abs(component.dynamic_friction - 1.0) > 1e-5:
        raise AssertionError(f"dynamic friction mismatch: {component.dynamic_friction}")
    if abs(component.restitution - 0.0) > 1e-5:
        raise AssertionError(f"restitution mismatch: {component.restitution}")


def main():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        mesh_dir = root / "meshes"
        mesh_dir.mkdir()
        mjcf = root / "robot.xml"
        _write_mjcf(mjcf, mesh_dir)

        scene = ke.scene.create_backend(ke.scene.BackendType.Native)
        asset = ke.visual.ArticulationVisualAsset.from_mjcf(str(mjcf))

        merged = asset.instantiate(scene, "/merged_robot", "", False)
        _check_articulation_root(
            scene.get_prim_at_path("/merged_robot").get_articulation_component(),
            "/merged_robot",
            str(mjcf),
            1,
            1,
            False,
        )
        body = merged.body_prim(0)
        _check_binding(
            body.get_articulation_binding_component(),
            ke.scene.ArticulationPrimRole.BodyFrame,
            0,
            "body",
            "/merged_robot",
        )
        render = list(merged.render_prims())[0]
        if render.get_path() != body.get_path():
            raise AssertionError("merged render prim should be the body frame")

        split = asset.instantiate(scene, "/split_robot", "", True)
        _check_articulation_root(
            scene.get_prim_at_path("/split_robot").get_articulation_component(),
            "/split_robot",
            str(mjcf),
            1,
            1,
            True,
        )
        split_body = split.body_prim(0)
        _check_binding(
            split_body.get_articulation_binding_component(),
            ke.scene.ArticulationPrimRole.BodyFrame,
            0,
            "body",
            "/split_robot",
        )
        split_render = list(split.render_prims())[0]
        _check_binding(
            split_render.get_articulation_binding_component(),
            ke.scene.ArticulationPrimRole.VisualGeom,
            0,
            "body",
            "/split_robot",
        )

        physics = ke.physics.PhysicsWorld(ke.physics.PhysicsConfig.y_up())
        articulation = ke.physics.Articulation.build(
            physics,
            ke.asset.MJCFLoader.load(str(mjcf)),
            ke.physics.ArticulationConfig.fixed_base(),
        )
        physics_bridge = ke.physics.PhysicsBridge()
        physics_bridge.add(articulation, split)
        collision_prims = physics_bridge.add_collision_visuals(
            articulation, scene, "/split_robot/collision", False
        )
        if len(collision_prims) != 1:
            raise AssertionError(
                f"expected one collision visual, got {len(collision_prims)}"
            )
        _check_binding(
            collision_prims[0].get_articulation_binding_component(),
            ke.scene.ArticulationPrimRole.CollisionGeom,
            0,
            "body",
            "/split_robot",
        )
        _check_collision_shape(collision_prims[0].get_collision_shape_component())

    print("PASS: ArticulationBindingComponent smoke completed")


if __name__ == "__main__":
    main()
