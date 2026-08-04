"""
Test XML Robot Viewer — Python equivalent of physx_h1_ragdoll.cpp style.
Loads Unitree H1 from MJCF and optionally plays back motion from poselib.
"""

import os
import sys
from pathlib import Path

import kangengine as ke
from kangengine import visual


# ---------------------------------------------------------------------------
# Optional: motion playback via poselib
# ---------------------------------------------------------------------------
_motion_data = None

try:
    project_root = Path(__file__).resolve().parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from references.isaacsim.poselib.poselib.skeleton.my_skeleton3d import (
        SkeletonMotion,
    )

    _motion_file = "/Users/asaid/Dev/sample_isaac/assets/IsaacSim_motions/unitree_h1/samples/SwingDancing1.npy"
    _mot = SkeletonMotion.from_file(_motion_file)
    _root_trans = _mot.global_translation[:, 0].numpy().copy()
    _root_trans[:, 2] += 0.3
    _local_rot = _mot.local_rotation.numpy()
    _motion_data = (_root_trans, _local_rot)
    print(f"Motion loaded: {_motion_file}  ({_root_trans.shape[0]} frames)")
except Exception as e:
    print(f"Motion playback disabled: {e}")


# ---------------------------------------------------------------------------
def asset_path(*parts):
    base = os.path.join(os.path.dirname(ke.__file__), "assets")
    return os.path.join(base, *parts)


# ---------------------------------------------------------------------------
class RobotViewer(ke.App):
    def setup(self):
        self.frame_idx = 0
        materials = self.create_standard_materials()
        self.robot_material = materials.common
        self.ground_material = materials.ground

        # Ground plane (Z-up)
        self.scene.add_ground(scale=100.0, material=self.ground_material)

        # Robot
        mjcf = asset_path("external", "retargetted", "unitree_h1", "unitree_h1.xml")
        self.robot = visual.ArticulationVisual.from_mjcf(
            mjcf, self.scene.native, "/robot", 1.0, "BFS"
        )
        for prim in self.robot.body_prims():
            self.scene.add_renderable(prim, self.robot_material)

        print(f"Robot loaded: {self.robot.num_bodies()} bodies")
        self.check_error()

    def preRender(self):
        if _motion_data is None:
            return

        root_trans, local_rot = _motion_data
        idx = self.frame_idx

        root = root_trans[idx]
        self.robot.set_root_translation(
            # ke.vec3(float(root[0]), float(root[1]), float(root[2]))
            root
        )
        for i in range(self.robot.num_bodies()):
            q = local_rot[idx][i]
            self.robot.set_joint_rotation(
                i, ke.quat(float(q[0]), float(q[1]), float(q[2]), float(q[3]))
            )
        self.robot.apply_pose()

        self.frame_idx = (self.frame_idx + 1) % root_trans.shape[0]
        self.check_error()

    def render(self):
        pass

    def postRender(self):
        pass


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app = RobotViewer()
    app.initialize(1920, 1080, False, ke.UpAxis.Z)
    app.start()
