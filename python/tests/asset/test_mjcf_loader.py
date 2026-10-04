from pathlib import Path

import numpy as np
import pytest

import kangengine as ke


def _vec3(value) -> np.ndarray:
    return np.array([value.x, value.y, value.z], dtype=np.float32)


@pytest.mark.parametrize("loader", ["parse", "load"])
@pytest.mark.parametrize("joint_type", ["slide", "unknown_joint"])
@pytest.mark.parametrize("declaration", ["direct", "class", "childclass", "main"])
def test_mjcf_rejects_unsupported_joint_type(
    tmp_path, loader, joint_type, declaration
):
    defaults, body_attrs, joint_attrs = "", "", f'type="{joint_type}"'
    if declaration in ("class", "childclass"):
        defaults = (
            f'<default><default class="outer"><joint type="{joint_type}"/>'
            '<default class="inner"/></default></default>'
        )
        body_attrs = 'childclass="inner"' if declaration == "childclass" else ""
        joint_attrs = 'class="inner"' if declaration == "class" else ""
    elif declaration == "main":
        defaults = f'<default><joint type="{joint_type}"/></default>'
        joint_attrs = ""
    path = tmp_path / "unsupported_joint.xml"
    path.write_text(f'''<mujoco>{defaults}<worldbody>
      <body name="root" {body_attrs}><body name="slider">
        <joint name="bad_joint" {joint_attrs}/>
      </body></body>
    </worldbody></mujoco>''', encoding="utf-8")

    with pytest.raises(RuntimeError) as error:
        getattr(ke.asset.MJCFLoader, loader)(str(path))
    message = str(error.value)
    assert f"Unsupported MJCF joint type '{joint_type}'" in message
    assert "joint 'bad_joint'" in message
    assert "body 'slider'" in message


@pytest.mark.parametrize("defaults,attrs", [
    ("", ""),
    ("", 'type="hinge"'),
    ('<default><joint type="slide"/></default>', 'type="hinge"'),
])
def test_mjcf_preserves_default_and_explicit_hinge(tmp_path, defaults, attrs):
    path = tmp_path / "hinge.xml"
    path.write_text(f'''<mujoco>{defaults}<worldbody><body name="root">
      <body name="link"><joint name="hinge" {attrs}/></body>
    </body></worldbody></mujoco>''', encoding="utf-8")

    result = ke.asset.MJCFLoader.parse(str(path))
    joints = result.articulation.joints[1]
    assert len(joints) == 1
    assert joints[0].type == ke.asset.JointDescType.REVOLUTE
    assert not result.diagnostics.warnings


@pytest.mark.parametrize("joint", ['<freejoint/>', '<joint name="root" type="free"/>'])
def test_mjcf_preserves_free_root(tmp_path, joint):
    path = tmp_path / "free_root.xml"
    path.write_text(f'''<mujoco><worldbody><body name="root">
      {joint}<body name="link"><joint name="hinge"/></body>
    </body></worldbody></mujoco>''', encoding="utf-8")

    result = ke.asset.MJCFLoader.parse(str(path))
    assert 0 not in result.articulation.joints
    assert len(result.articulation.joints[1]) == 1
    assert not result.diagnostics.warnings


def test_mjcf_ignores_later_collinear_joint(tmp_path: Path):
    path = tmp_path / "backlash.xml"
    path.write_text(
        """<mujoco model="backlash">
  <compiler angle="radian"/>
  <worldbody>
    <body name="root">
      <body name="link">
        <joint name="hip_pitch" type="hinge" axis="0 1 0" range="-1 1"/>
        <joint name="mechanical_slack" type="hinge" axis="0 -1 0"
               range="-0.01 0.01"/>
        <geom type="sphere" size="0.1" mass="1"/>
      </body>
    </body>
  </worldbody>
</mujoco>
""",
        encoding="utf-8",
    )

    result = ke.asset.MJCFLoader.parse(str(path))
    joints = result.articulation.joints

    assert len(joints) == 1
    assert len(next(iter(joints.values()))) == 1
    joint = next(iter(joints.values()))[0]
    assert joint.name == "hip_pitch"
    assert np.allclose(_vec3(joint.axis), [0, 1, 0])
    assert any("mechanical_slack" in warning for warning in result.diagnostics.warnings)


def test_mjcf_omitted_geom_type_is_sphere(tmp_path: Path):
    path = tmp_path / "foot.xml"
    path.write_text(
        '<mujoco><worldbody><body name="foot">\n        <geom size="0.005" pos="0.12 0.03 -0.03"/>\n        <geom type="sphere" size="0.005" pos="0.12 -0.03 -0.03"/>\n        </body></worldbody></mujoco>'
    )
    data = ke.asset.MJCFLoader.load(str(path))
    geoms = data.collision_geoms[0]
    assert len(geoms) == 2
    assert geoms[0].type == geoms[1].type
    assert np.allclose(geoms[0].size, [0.005, 0, 0])
    assert np.allclose(_vec3(geoms[0].pos), [0.12, 0.03, -0.03])


@pytest.mark.parametrize("radius", [0.005, 0.02, 0.1])
def test_mjcf_geom_inertia_preserves_sphere_mass_properties(tmp_path: Path, radius):
    path = tmp_path / "sphere_inertia.xml"
    path.write_text(f'''<mujoco><worldbody><body name="sphere">
      <geom type="sphere" size="{radius}" density="1000"/>
    </body></worldbody></mujoco>''')

    data = ke.asset.MJCFLoader.load(str(path))
    inertial = data.inertials[0]
    mass = 1000.0 * (4.0 / 3.0) * np.pi * radius**3
    inertia = (2.0 / 5.0) * mass * radius**2
    np.testing.assert_allclose(inertial.mass, mass, rtol=1e-6, atol=0)
    np.testing.assert_allclose(_vec3(inertial.diag_inertia), [inertia] * 3,
                               rtol=1e-6, atol=0)


@pytest.mark.parametrize("geom,expected", [
    ('type="sphere" size="0.001" density="1000"', [1.0001e-6] * 3),
    ('type="box" size="0.00001 0.00001 1" density="1250000000"',
     [0.3333346666333333, 0.3333346666333333, 1.3333333333666667e-6]),
])
def test_mjcf_geom_inertia_regularizes_tiny_components(tmp_path: Path, geom, expected):
    path = tmp_path / "tiny_inertia.xml"
    path.write_text(f'''<mujoco><worldbody><body name="tiny">
      <geom {geom}/>
    </body></worldbody></mujoco>''')
    result = ke.asset.MJCFLoader.parse(str(path))
    np.testing.assert_allclose(
        _vec3(result.articulation.inertials[0].diag_inertia), expected,
        rtol=1e-6, atol=0,
    )
    assert not result.diagnostics.warnings


def test_visual_bridge_does_not_render_empty_root_body(tmp_path: Path):
    mesh_path = tmp_path / "body.stl"
    mesh_path.write_text(
        """solid body
facet normal 0 0 1
outer loop
vertex 0 0 0
vertex 1 0 0
vertex 0 1 0
endloop
endfacet
endsolid body
""",
        encoding="utf-8",
    )
    path = tmp_path / "empty_root.xml"
    path.write_text(
        """<mujoco model="empty_root">
  <asset><mesh name="body_mesh" file="body.stl"/></asset>
  <worldbody>
    <body name="base">
      <freejoint/>
      <body name="body">
        <geom type="mesh" mesh="body_mesh"/>
      </body>
    </body>
  </worldbody>
</mujoco>
""",
        encoding="utf-8",
    )

    asset = ke.visual.ArticulationVisualAsset.from_mjcf(str(path))
    scene = ke.scene.create_backend(ke.scene.BackendType.NATIVE)
    bridge = asset.instantiate(
        scene,
        "/robot",
        "/.Resources/test_empty_root",
        True,
    )

    assert bridge.body_prim(0).get_type() == ke.scene.PrimType.XFORM
    assert [prim.get_path() for prim in bridge.render_prims()] == [
        "/robot/body/visual_0"
    ]


@pytest.mark.parametrize("missing_body", ["base"])
def test_articulation_rejects_failed_mass_inference(tmp_path: Path, missing_body):
    """Missing inertials and shapes must not silently add unit mass/inertia."""
    import xml.etree.ElementTree as ET

    root = ET.fromstring("""<mujoco><worldbody>
      <body name="base">
        <inertial mass="2" pos="0 0 0" diaginertia=".1 .1 .1"/>
        <body name="marker" pos="0 0 .1">
          <inertial mass="1" pos="0 0 0" diaginertia=".1 .1 .1"/>
        </body>
      </body>
    </worldbody></mujoco>""")
    valid_path = tmp_path / "valid.xml"
    ET.ElementTree(root).write(valid_path)
    body = root.find(f".//body[@name='{missing_body}']")
    body.remove(body.find("inertial"))
    invalid_path = tmp_path / "missing_inertial.xml"
    ET.ElementTree(root).write(invalid_path)

    world = ke.physics.PhysicsWorld(ke.physics.PhysicsConfig.z_up())
    robot = None
    try:
        with pytest.raises(RuntimeError, match=f"Cannot infer mass and inertia for body '{missing_body}'"):
            ke.physics.Articulation.build(
                world, ke.asset.MJCFLoader.load(str(invalid_path))
            )
        # A partial construction must release its PhysX resources on failure.
        robot = ke.physics.Articulation.build(
            world, ke.asset.MJCFLoader.load(str(valid_path))
        )
        world.step()
        np.testing.assert_allclose(robot.get_link_masses(), [2., 1.])
    finally:
        if robot is not None:
            robot.release()


@pytest.mark.parametrize("body_attrs,joint_attrs,expected", [
    ('childclass="body_defaults"', '', (10., 1., .01)),
    ('childclass="body_defaults"', 'class="motor"', (20., 2., .03)),
    ('', 'class="motor"', (20., 2., .03)),
    ('childclass="body_defaults"', 'class="nested_motor"', (20., 4., .03)),
    ('childclass="body_defaults"', 'class="armature_only"', (0., 0., .05)),
    ('childclass="body_defaults"',
     'class="nested_motor" stiffness="0" damping="0" armature="0"',
     (0., 0., 0.)),
])
def test_mjcf_joint_class_precedence(tmp_path, body_attrs, joint_attrs, expected):
    path = tmp_path / "joint_classes.xml"
    path.write_text(f"""<mujoco>
      <default>
        <default class="body_defaults">
          <joint stiffness="10" damping="1" armature=".01"/>
        </default>
        <default class="motor">
          <joint stiffness="20" damping="2" armature=".03"/>
          <default class="nested_motor"><joint damping="4"/></default>
        </default>
        <default class="armature_only"><joint armature=".05"/></default>
      </default>
      <worldbody><body name="root" {body_attrs}>
        <body name="link">
          <joint name="hinge" axis="0 1 0" range="-1 1" {joint_attrs}/>
        </body>
      </body></worldbody>
    </mujoco>""", encoding="utf-8")
    data = ke.asset.MJCFLoader.load(str(path))
    joint = next(iter(data.joints.values()))[0]
    np.testing.assert_allclose((joint.kp, joint.kd, joint.armature), expected)


def _inertial_matrix(inertial):
    q = inertial.quat
    v = np.array([q.x, q.y, q.z], dtype=np.float64)
    skew = np.array([[0., -v[2], v[1]], [v[2], 0., -v[0]], [-v[1], v[0], 0.]])
    rotation = np.eye(3) + 2 * q.w * skew + 2 * skew @ skew
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
    assert np.linalg.det(rotation) == pytest.approx(1., abs=1e-6)
    return rotation @ np.diag(_vec3(inertial.diag_inertia)) @ rotation.T


@pytest.mark.parametrize("values", [
    (2., 2.5, 3., .2, -.3, .4),
    (1., 1., 1., 0., 0., 0.),
    (1e-10, 2e-10, 2.5e-10, 0., 0., 0.),
])
def test_mjcf_fullinertia_preserves_tensor(tmp_path, values):
    path = tmp_path / "full.xml"
    path.write_text(f'''<mujoco><worldbody><body name="body">
      <inertial mass="2" pos=".1 .2 .3"
        fullinertia="{' '.join(map(str, values))}"/>
    </body></worldbody></mujoco>''')
    inertial = ke.asset.MJCFLoader.load(str(path)).inertials[0]
    xx, yy, zz, xy, xz, yz = values
    expected = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
    np.testing.assert_allclose(_inertial_matrix(inertial), expected,
                               rtol=2e-6, atol=1e-7 * np.max(np.abs(expected)))
    assert inertial.mass == pytest.approx(2.)
    np.testing.assert_allclose(_vec3(inertial.com), [.1, .2, .3])


@pytest.mark.parametrize("attributes", [
    'fullinertia=""',
    'fullinertia="1 1 1 0 0"',
    'fullinertia="1 1 1 0 0 0 7"',
    'fullinertia="1 1 1 0 0 0 junk"',
    'fullinertia="nan 1 1 0 0 0"',
    'fullinertia="1 1 inf 0 0 0"',
    'fullinertia="1 1 0 0 0 0"',
    'fullinertia="1 1 1 2 0 0"',
    'fullinertia="1 1 4 0 0 0"',
    'fullinertia="1 1 1 0 0 0" diaginertia="1 1 1"',
    'fullinertia="1 1 1 0 0 0" quat="1 0 0 0"',
    'fullinertia="1 1 1 0 0 0" euler="0 0 0"',
])
def test_mjcf_fullinertia_rejects_invalid_input(tmp_path, attributes):
    path = tmp_path / "invalid.xml"
    path.write_text(f'''<mujoco><worldbody><body name="bad_tensor">
      <inertial mass="1" pos="0 0 0" {attributes}/>
    </body></worldbody></mujoco>''')
    with pytest.raises(RuntimeError, match="Invalid MJCF fullinertia on body 'bad_tensor'"):
        ke.asset.MJCFLoader.load(str(path))


def test_mjcf_diaginertia_orientation_is_preserved(tmp_path):
    path = tmp_path / "diagonal.xml"
    path.write_text('''<mujoco><worldbody><body name="body">
      <inertial mass="2" pos="0 0 0" diaginertia="1 2 3"
        quat="0.7071067811865476 0 0 0.7071067811865476"/>
    </body></worldbody></mujoco>''')
    inertial = ke.asset.MJCFLoader.load(str(path)).inertials[0]
    np.testing.assert_allclose(_inertial_matrix(inertial), np.diag([2., 1., 3.]), atol=1e-6)


def test_k1_fullinertia_matches_authored_tensors():
    import xml.etree.ElementTree as ET

    path = (Path(__file__).resolve().parents[3] / "assets/external/characters"
            / "ai_sapiens_description/mujoco/k1/k1.xml")
    if not path.is_file():
        pytest.skip("K1 external asset is not installed")
    data = ke.asset.MJCFLoader.load(str(path))
    names = list(data.skeleton_tree.node_names())
    count = 0
    for body in ET.parse(path).getroot().find("worldbody").iter("body"):
        source = body.find("inertial")
        if source is None or source.get("fullinertia") is None:
            continue
        xx, yy, zz, xy, xz, yz = map(float, source.get("fullinertia").split())
        expected = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
        inertial = data.inertials[names.index(body.get("name"))]
        np.testing.assert_allclose(_inertial_matrix(inertial), expected,
                                   rtol=2e-5, atol=1e-8)
        assert inertial.mass == pytest.approx(float(source.get("mass")))
        count += 1
    assert count == 25


@pytest.mark.parametrize("declaration", ["direct", "class", "childclass", "main"])
def test_mjcf_ball_builds_three_axis_spherical_joint(tmp_path, declaration):
    attrs = 'type="ball" pos="0 0 -.25" damping=".2" armature=".03"'
    defaults, body_attrs, joint_attrs = '', '', attrs
    if declaration in ('class', 'childclass'):
        defaults = f'<default><default class="socket"><joint {attrs}/></default></default>'
        body_attrs = 'childclass="socket"' if declaration == 'childclass' else ''
        joint_attrs = 'class="socket"' if declaration == 'class' else ''
    elif declaration == 'main':
        defaults, joint_attrs = f'<default><joint {attrs}/></default>', ''
    path = tmp_path / 'ball.xml'
    path.write_text(f'''<mujoco>{defaults}<worldbody>
      <body name="root" {body_attrs}>
        <inertial mass="1" pos="0 0 0" diaginertia="1 1 1"/>
        <body name="child" pos="0 0 1">
          <joint name="socket" {joint_attrs}/>
          <inertial mass="1" pos="0 0 0" diaginertia="1 1 1"/>
        </body>
      </body></worldbody></mujoco>''')
    data = ke.asset.MJCFLoader.load(str(path), scale=2.)
    joints = data.joints[1]
    assert len(joints) == 3
    np.testing.assert_allclose([_vec3(j.axis) for j in joints], np.eye(3))
    for j in joints:
        np.testing.assert_allclose(j.joint_offset, [0, 0, -.5])
        assert j.kd == pytest.approx(.2)
        assert j.armature == pytest.approx(.03)
    # Keep the existing per-axis coordinate representation (three, not nine DOFs).
    layout = ke.animation.ArticulationCoordinateLayout.from_data(data)
    assert len(layout.blocks) == 3
    world = ke.physics.PhysicsWorld(ke.physics.PhysicsConfig.z_up())
    robot = ke.physics.Articulation.build(world, data)
    try:
        assert robot.num_dofs() == 3
        assert robot.get_dof_names() == ['socket/0', 'socket/1', 'socket/2']
        assert [i % 6 for i in robot.get_dof_joint_force_indices()] == [3, 4, 5]
        np.testing.assert_allclose(robot.get_local_debug_frames()[1]['joint'][:3], [0, 0, -.5])
        # Each rotational DOF moves the body around the authored attachment.
        for axis in range(3):
            q = np.zeros(3, dtype=np.float32)
            q[axis] = .4
            robot.set_dof_state(q, np.zeros(3, dtype=np.float32))
            np.testing.assert_allclose(robot.get_dof_positions(), q, atol=1e-6)
        world.step()
        assert np.isfinite(robot.get_dof_positions()).all()
    finally:
        robot.release()


@pytest.mark.parametrize('defaults,attrs', [
    ('', 'limited="true" range="0 30"'),
    ('', 'range="0 30"'),
    ('<default><joint limited="true" range="0 30"/></default>', ''),
    ('<default><default class="limited"><joint limited="true" range="0 30"/></default></default>',
     'class="limited"'),
])
def test_mjcf_ball_rejects_unsupported_rotation_angle_limit(tmp_path, defaults, attrs):
    path = tmp_path / 'limited_ball.xml'
    path.write_text(f'''<mujoco>{defaults}<worldbody><body name="root">
      <body name="child"><joint name="socket" type="ball" {attrs}/></body>
    </body></worldbody></mujoco>''')
    with pytest.raises(RuntimeError, match="ball joint 'socket'.*rotation-angle limit"):
        ke.asset.MJCFLoader.load(str(path))


def test_mjcf_ball_explicitly_disables_inherited_limit(tmp_path):
    path = tmp_path / 'unlimited_ball.xml'
    path.write_text('''<mujoco><default><joint limited="true" range="0 30"/></default>
      <worldbody><body name="root"><body name="child">
        <joint name="socket" type="ball" limited="false" axis="0 0 0"/>
      </body></body></worldbody></mujoco>''')
    joints = ke.asset.MJCFLoader.load(str(path)).joints[1]
    assert len(joints) == 3
    np.testing.assert_allclose([_vec3(j.axis) for j in joints], np.eye(3))


@pytest.mark.parametrize('ball_first', [True, False])
def test_mjcf_ball_rejects_combined_joints(tmp_path, ball_first):
    ball = '<joint name="socket" type="ball"/>'
    hinge = '<joint name="hinge" axis="1 0 0"/>'
    path = tmp_path / 'combined.xml'
    path.write_text(f'''<mujoco><worldbody><body name="root"><body name="child">
      {ball + hinge if ball_first else hinge + ball}
    </body></body></worldbody></mujoco>''')
    with pytest.raises(RuntimeError, match='ball joint.*cannot be combined'):
        ke.asset.MJCFLoader.load(str(path))


@pytest.mark.parametrize('defaults,material,body_attrs,geom_attrs,expected', [
    ('', '<material name="paint" rgba=".7 .6 .5 .4"/>', '', 'material="paint"', [.7, .6, .5, .4]),
    ('', '<material name="paint" rgba=".7 .6 .5 .4"/>', '',
     'material="paint" rgba=".1 .2 .3 .8"', [.1, .2, .3, .8]),
    ('', '<material name="paint" rgba=".7 .6 .5 .4"/>', '',
     'material="paint" rgba=".5 .5 .5 1"', [.7, .6, .5, .4]),
    ('<default><geom material="paint"/></default>',
     '<material name="paint" rgba=".7 .6 .5 .4"/>', '', '', [.7, .6, .5, .4]),
    ('<default><default class="outer"><geom material="paint"/><default class="inner"/></default></default>',
     '<material name="paint" rgba=".7 .6 .5 .4"/>', 'childclass="inner"', '', [.7, .6, .5, .4]),
    ('<default><default class="outer"><geom material="paint" rgba=".2 .3 .4 .5"/></default></default>',
     '<material name="paint" rgba=".7 .6 .5 .4"/>', '', 'class="outer"', [.2, .3, .4, .5]),
    ('<default><geom material="paint"/><default class="clear"><geom material=""/></default></default>',
     '<material name="paint" rgba=".7 .6 .5 .4"/>', '', 'class="clear"', [.15, .15, .15, 1]),
    ('<default><geom material="paint"/></default>',
     '<material name="paint" rgba=".7 .6 .5 .4"/>', '', 'material=""', [.15, .15, .15, 1]),
    ('<default><material rgba=".1 .2 .3 .4"/></default>',
     '<material name="paint"/>', '', 'material="paint"', [.1, .2, .3, .4]),
    ('<default><material rgba=".1 .2 .3 .4"/><default class="outer"><material rgba=".6 .5 .4 .3"/><default class="inner"/></default></default>',
     '<material name="paint" class="inner"/>', '', 'material="paint"', [.6, .5, .4, .3]),
    ('', '<material name="paint"/>', '', 'material="paint"', [1, 1, 1, 1]),
    ('', '', '', 'rgba=".2 .4 .6 .8"', [.2, .4, .6, .8]),
])
def test_mjcf_material_color_precedence(tmp_path, defaults, material, body_attrs, geom_attrs, expected):
    (tmp_path / 'triangle.obj').write_text('v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n')
    path = tmp_path / 'material.xml'
    path.write_text(f'''<mujoco>{defaults}
      <asset><mesh name="mesh" file="triangle.obj"/>{material}</asset>
      <worldbody><body name="body" {body_attrs}>
        <geom type="mesh" mesh="mesh" contype="0" conaffinity="0" {geom_attrs}/>
      </body></worldbody></mujoco>''')
    result = ke.asset.MJCFLoader.parse(str(path))
    assert result.diagnostics.warnings == []
    color = result.articulation.visual_geoms[0].rgba
    np.testing.assert_allclose([color.x, color.y, color.z, color.w], expected, atol=1e-6)
    asset = ke.visual.ArticulationVisualAsset.from_data(result.articulation)
    scene = ke.scene.create_backend(ke.scene.BackendType.NATIVE)
    bridge = asset.instantiate(scene, '/robot', '/.Resources/material', True)
    color = bridge.render_prims()[0].get_display_color_alpha()
    np.testing.assert_allclose([color.x, color.y, color.z, color.w], expected, atol=1e-6)


def test_mjcf_unknown_visual_material_warns(tmp_path):
    path = tmp_path / 'unknown_material.xml'
    path.write_text('''<mujoco><asset><mesh name="mesh" file="unused.obj"/></asset>
      <worldbody><body name="body">
        <geom type="mesh" mesh="mesh" material="missing" contype="0" conaffinity="0"/>
      </body></worldbody></mujoco>''')
    result = ke.asset.MJCFLoader.parse(str(path))
    assert any("unknown material 'missing'" in w for w in result.diagnostics.warnings)


@pytest.mark.parametrize('value', [
    None, '', '1 1', '1 1 1 1', '1 1 1 junk', 'nan 1 1', '1 inf 1',
    '0 1 1', '-1 1 1', '4 1 1', '1e-60 1e-60 1e-60',
    '1e40 1e40 1e40', '1 1 2.00001',
])
def test_mjcf_diaginertia_rejects_invalid_authored_moments(tmp_path, value):
    attribute = '' if value is None else f'diaginertia="{value}"'
    path = tmp_path / 'invalid_diagonal.xml'
    path.write_text(f'''<mujoco><worldbody><body name="bad_link">
      <inertial mass="1" pos="0 0 0" {attribute}/>
    </body></worldbody></mujoco>''')
    with pytest.raises(RuntimeError, match="Invalid MJCF diaginertia on body 'bad_link'"):
        ke.asset.MJCFLoader.load(str(path))


@pytest.mark.parametrize('moments', [(3e-10, 1e-10, 2e-10), (2.000001, 1., 1.)])
def test_mjcf_diaginertia_preserves_small_and_near_boundary_values(tmp_path, moments):
    path = tmp_path / 'small_diagonal.xml'
    path.write_text(f'''<mujoco><worldbody><body name="small_link">
      <inertial mass="1" pos="0 0 0" diaginertia="{' '.join(map(str, moments))}"/>
    </body></worldbody></mujoco>''')
    value = ke.asset.MJCFLoader.load(str(path)).inertials[0]
    np.testing.assert_allclose(_vec3(value.diag_inertia), moments, rtol=1e-6, atol=0)
