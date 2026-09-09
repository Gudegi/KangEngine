"""USD physics import without a Python USD dependency or GPU."""
from pathlib import Path

import numpy as np
import pytest

from kangengine import asset


USD = '''#usda 1.0
(
    defaultPrim = "Robot"
    metersPerUnit = 1
    upAxis = "Z"
)
def Xform "Prototype" {
    def Mesh "mesh" (prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]) {
        point3f[] points = [(0,0,0), (1,0,0), (0,1,0), (0,0,1)]
        int[] faceVertexCounts = [3,3,3,3]
        int[] faceVertexIndices = [0,2,1, 0,1,3, 0,3,2, 1,2,3]
        token physics:approximation = "convexHull"
        bool physics:collisionEnabled = true
    }
}
def Xform "Robot" {
    def Xform "base" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]) {
        float physics:mass = 2
        float3 physics:diagonalInertia = (1,2,3)
        point3f physics:centerOfMass = (0.1,0.2,0.3)
        quatf physics:principalAxes = (1,0,0,0)
    }
    def Xform "tip" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]) {
        float physics:mass = 1
        point3f physics:centerOfMass = (0,0,0)
        quatf physics:principalAxes = (1,0,0,0)
        float3 physics:diagonalInertia = (0.1,0.2,0.3)
        double3 xformOp:translate = (10,20,30)
        uniform token[] xformOpOrder = ["xformOp:translate"]
        def Xform "geometry" (
            prepend references = </Prototype>
            instanceable = true
        ) {
            double3 xformOp:translate = (2,0,0)
            uniform token[] xformOpOrder = ["xformOp:translate"]
        }
    }
    def PhysicsRevoluteJoint "hinge" {
        rel physics:body0 = </Robot/base>
        rel physics:body1 = </Robot/tip>
        point3f physics:localPos0 = (0,0,3)
        point3f physics:localPos1 = (0,0,0)
        quatf physics:localRot0 = (1,0,0,0)
        quatf physics:localRot1 = (0.70710678,0.70710678,0,0)
        uniform token physics:axis = "Y"
        float physics:lowerLimit = -90
        float physics:upperLimit = 45
        float drive:angular:physics:stiffness = 20
        float drive:angular:physics:damping = 2
        float drive:angular:physics:maxForce = 30
    }
}
'''


def load(tmp_path: Path, text=USD):
    path = tmp_path / "robot.usda"
    path.write_text(text)
    try:
        return asset.USDLoader.parse_articulation(str(path))
    except RuntimeError as error:
        if "USD support not compiled" in str(error):
            pytest.skip(str(error))
        raise


def xyz(v):
    return np.array([v.x, v.y, v.z])


def test_joint_frames_mass_units_and_instance_mesh(tmp_path):
    result = load(tmp_path)
    data = result.articulation
    assert data.skeleton_tree.node_names() == ["base", "tip"]
    np.testing.assert_allclose(xyz(data.skeleton_tree.local_translation(1)), [0, 0, 3])
    joint = data.joints[1][0]
    np.testing.assert_allclose(xyz(joint.axis), [0, 0, 1], atol=1e-6)
    assert joint.lo_limit == pytest.approx(-np.pi / 2)
    assert joint.hi_limit == pytest.approx(np.pi / 4)
    assert (joint.kp, joint.kd, joint.effort_limit) == (20, 2, 30)
    assert data.inertials[0].mass == 2
    np.testing.assert_allclose(xyz(data.inertials[0].com), [.1, .2, .3])
    assert len(data.visual_geoms) == 1
    hull = data.collision_geoms[1][0]
    vertices = np.asarray(hull.mesh_data.vertices)
    np.testing.assert_allclose(vertices.min(axis=0), [2, 0, 0])
    np.testing.assert_allclose(vertices.max(axis=0), [3, 1, 1])
    assert result.diagnostics.warnings


@pytest.mark.parametrize("old,new,error", [
    ('metersPerUnit = 1', 'metersPerUnit = 0.01', 'meter units'),
    ('point3f physics:localPos1 = (0,0,0)', 'point3f physics:localPos1 = (1,0,0)', 'child joint anchor'),
    ('PhysicsRevoluteJoint', 'PhysicsSphericalJoint', 'Unsupported USD joint'),
    ('"convexHull"', '"convexDecomposition"', 'convexHull'),
    ('float physics:mass = 2', 'float physics:mass = 0', 'positive USD mass'),
])
def test_unsupported_physics_fails_explicitly(tmp_path, old, new, error):
    with pytest.raises(RuntimeError, match=error):
        load(tmp_path, USD.replace(old, new))


def test_prismatic_uses_meters_and_fixed_joint_has_no_dof(tmp_path):
    result = load(tmp_path, USD.replace("PhysicsRevoluteJoint", "PhysicsPrismaticJoint"))
    joint = result.articulation.joints[1][0]
    assert joint.lo_limit == -90
    assert joint.hi_limit == 45
    fixed = load(tmp_path, USD.replace("PhysicsRevoluteJoint", "PhysicsFixedJoint"))
    assert not fixed.articulation.joints
    assert fixed.articulation.skeleton_tree.parent_index(1) == 0


def test_world_cache_and_visual_bridge_share_usd_descriptor(tmp_path):
    from kangengine import sim, visual

    load(tmp_path)
    path = str(tmp_path / "robot.usda")
    world = sim.KangSimWorld(num_envs=1, add_ground=False)
    try:
        data = world.load_usd(path)
        assert world.load_usd(path) is data
        selected = world.load_usd(path, prim_path="/Robot")
        assert selected is not data
        assert world.get_articulation_asset_cache_size() == 2
        bridge = visual.ArticulationVisualAsset.from_data(data)
        assert bridge.num_bodies() == 2
    finally:
        world.release()
