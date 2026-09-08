from pathlib import Path

import numpy as np
import pytest

import kangengine as ke


_ROOT = Path(__file__).resolve().parents[3]
_G1_URDF = _ROOT / "assets" / "characters" / "g1" / "g1_29dof.urdf"
_KW5_MJCF = _ROOT / "assets" / "characters" / "kw" / "kw5.xml"


def _write_synthetic_urdf(path: Path) -> None:
    path.write_text(
        """<?xml version="1.0"?>
<robot name="test_robot">
  <link name="root">
    <inertial>
      <origin xyz="1 2 3" rpy="0 0 0"/>
      <mass value="2"/>
      <inertia ixx="2" ixy="0.5" ixz="0" iyy="2" iyz="0" izz="4"/>
    </inertial>
  </link>
  <link name="fixed_child"/>
  <link name="branch_child"/>
  <link name="moving_child"/>
  <joint name="fixed_joint" type="fixed">
    <parent link="root"/><child link="fixed_child"/>
    <origin xyz="0 0 1" rpy="0 0 0"/>
  </joint>
  <joint name="branch_joint" type="revolute">
    <parent link="root"/><child link="branch_child"/>
    <origin xyz="0 2 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/><limit lower="-0.5" upper="0.75" effort="12"/>
  </joint>
  <joint name="moving_joint" type="revolute">
    <parent link="fixed_child"/><child link="moving_child"/>
    <origin xyz="1 0 0" rpy="0 0 1.57079632679"/>
    <axis xyz="0 0 1"/><limit lower="-1" upper="1" effort="5" velocity="3"/>
    <dynamics damping="0.25"/>
  </joint>
</robot>
""",
        encoding="utf-8",
    )


def _vec3(value) -> np.ndarray:
    return np.array([value.x, value.y, value.z], dtype=np.float32)


def _write_triangle_stl(path: Path) -> None:
    path.write_text(
        """solid triangle
facet normal 0 0 1
  outer loop
    vertex 0 0 0
    vertex 1 0 0
    vertex 0 1 0
  endloop
endfacet
endsolid triangle
""",
        encoding="utf-8",
    )


def test_urdf_graph_order_fixed_joint_and_scale(tmp_path: Path):
    path = tmp_path / "robot.urdf"
    _write_synthetic_urdf(path)

    dfs = ke.asset.URDFLoader.load(str(path), scale=2.0, order="DFS")
    bfs = ke.asset.URDFLoader.load(str(path), scale=2.0, order="BFS")

    assert dfs.skeleton_tree.node_names() == [
        "root",
        "fixed_child",
        "moving_child",
        "branch_child",
    ]
    assert bfs.skeleton_tree.node_names() == [
        "root",
        "fixed_child",
        "branch_child",
        "moving_child",
    ]
    assert dfs.skeleton_tree.parent_index(2) == 1
    assert bfs.skeleton_tree.parent_index(3) == 1
    assert np.allclose(_vec3(dfs.skeleton_tree.local_translation(2)), [2, 0, 0])

    assert 1 not in dfs.joints
    moving = dfs.joints[2][0]
    assert moving.name == "moving_joint"
    assert np.allclose(_vec3(moving.axis), [0, 0, 1])
    assert moving.lo_limit == pytest.approx(-1.0)
    assert moving.hi_limit == pytest.approx(1.0)
    assert moving.effort_limit == pytest.approx(5.0)
    assert moving.velocity_limit == pytest.approx(3.0)
    assert moving.kd == pytest.approx(0.25)


def test_urdf_inertia_tensor_is_diagonalized(tmp_path: Path):
    path = tmp_path / "robot.urdf"
    _write_synthetic_urdf(path)

    data = ke.asset.URDFLoader.load(str(path), scale=0.5)
    inertial = data.inertials[0]
    assert inertial.mass == pytest.approx(2.0)
    assert np.allclose(_vec3(inertial.com), [0.5, 1.0, 1.5])
    assert np.allclose(
        np.sort(_vec3(inertial.diag_inertia)),
        np.linalg.eigvalsh(np.array([[2, 0.5, 0], [0.5, 2, 0], [0, 0, 4]])),
    )


def test_urdf_prismatic_joint_scales_linear_limits(tmp_path: Path):
    path = tmp_path / "slider.urdf"
    path.write_text(
        """<robot name="slider">
  <link name="root"/>
  <link name="cart"/>
  <joint name="slide" type="prismatic">
    <parent link="root"/><child link="cart"/>
    <axis xyz="1 0 0"/><limit lower="-1" upper="2" effort="10" velocity="3"/>
  </joint>
</robot>
""",
        encoding="utf-8",
    )

    data = ke.asset.URDFLoader.load(str(path), scale=2.0)
    joint = data.joints[1][0]
    layout = ke.animation.ArticulationCoordinateLayout.from_data(data)

    assert joint.lo_limit == pytest.approx(-2.0)
    assert joint.hi_limit == pytest.approx(4.0)
    assert layout.blocks[0].type == ke.animation.ArticulationCoordinateType.PRISMATIC


def test_urdf_resolves_package_mesh_from_its_package_directory(tmp_path: Path):
    package = tmp_path / "test_robot_description"
    urdf_directory = package / "urdf"
    mesh_path = package / "meshes" / "link.stl"
    urdf_directory.mkdir(parents=True)
    mesh_path.parent.mkdir(parents=True)
    _write_triangle_stl(mesh_path)
    urdf_path = urdf_directory / "robot.urdf"
    uri = "package://test_robot_description/meshes/link.stl"
    urdf_path.write_text(
        f"""<robot name="package_mesh">
  <link name="root">
    <visual><geometry><mesh filename="{uri}"/></geometry></visual>
    <collision><geometry><mesh filename="{uri}"/></geometry></collision>
  </link>
</robot>
""",
        encoding="utf-8",
    )

    result = ke.asset.URDFLoader.parse(str(urdf_path))

    assert result.diagnostics.warnings == []
    assert result.articulation.visual_geoms[0].mesh_file == str(mesh_path)
    assert len(result.articulation.collision_geoms[0]) == 1


@pytest.mark.parametrize("order", ["DFS", "BFS"])
def test_g1_urdf_import_regression(order: str):
    result = ke.asset.URDFLoader.parse(str(_G1_URDF), order=order)
    data = result.articulation

    assert result.diagnostics.warnings == []
    assert data.traversal_order == order
    assert data.skeleton_tree.num_joints() == 41
    assert sum(len(joints) for joints in data.joints.values()) == 29
    assert len(data.visual_geoms) == 36
    assert len(data.inertials) == 36
    assert data.skeleton_tree.node_name(0) == "pelvis"


def test_g1_urdf_builds_visual_bridge_asset():
    visual_asset = ke.visual.ArticulationVisualAsset.from_urdf(
        urdf_path=str(_G1_URDF),
        order="DFS",
    )

    assert visual_asset.num_bodies() == 41


def test_g1_urdf_builds_high_level_articulated_surface_asset():
    visual_asset = ke.visual.ArticulatedSurfaceAsset.from_urdf(_G1_URDF)

    assert visual_asset.num_bodies == 41


def test_world_uses_one_articulation_asset_cache_for_urdf_and_mjcf():
    world = ke.sim.KangSimWorld(num_envs=1, add_ground=False)
    try:
        first = world.load_urdf(str(_G1_URDF), scale=1.0, order="DFS")
        repeated = world.load_urdf(str(_G1_URDF), scale=1.0, order="DFS")
        bfs = world.load_urdf(str(_G1_URDF), scale=1.0, order="BFS")
        mjcf = world.load_mjcf(str(_KW5_MJCF), scale=1.0, order="DFS")

        assert repeated is first
        assert bfs is not first
        assert mjcf is world.load_mjcf(str(_KW5_MJCF), scale=1.0, order="DFS")
        assert world.get_articulation_asset_cache_size() == 3
        assert world.get_articulation_asset_load_count() == 3
    finally:
        world.release()
