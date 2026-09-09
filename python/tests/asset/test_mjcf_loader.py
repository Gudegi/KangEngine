from pathlib import Path

import numpy as np

import kangengine as ke


def _vec3(value) -> np.ndarray:
    return np.array([value.x, value.y, value.z], dtype=np.float32)


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
