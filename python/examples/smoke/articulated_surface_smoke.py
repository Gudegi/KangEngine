"""Validate kinematic ArticulatedSurface instancing without a window."""

from pathlib import Path

import numpy as np

import kangengine as ke


class _View:
    def __init__(self, prim):
        self.prim = prim

    def set_base_color(self, color):
        self.prim.set_display_color_alpha(color)

    def get_base_color(self):
        return self.prim.get_display_color_alpha()

    def set_alpha_mode(self, mode, cutoff=0.5):
        self.alpha_mode = (mode, cutoff)

    def set_visible(self, visible):
        self.visible = bool(visible)

    def set_casts_shadow(self, casts_shadow):
        self.casts_shadow = bool(casts_shadow)


class _Scene:
    def __init__(self):
        self.native = ke.scene.create_backend(ke.scene.BackendType.NATIVE)

    def add_renderable(self, prim, material):
        return _View(prim)


class _App:
    def __init__(self):
        self.scene = _Scene()

    def remove_prim(self, path):
        return self.scene.native.remove_prim(path)


def main():
    root = Path(__file__).resolve().parents[3]
    mjcf = root / "python/kangengine/assets/characters/kw/kw5.xml"
    app = _App()
    robot = ke.visual.ArticulatedSurface.create_from_mjcf(
        app, "/robot", mjcf, object()
    )

    joints = robot.skeleton_tree.num_joints()
    rotations = np.zeros((joints, 4), np.float32)
    rotations[:, 0] = 1.0
    state = ke.animation.SkeletonState.from_rotation_and_root_translation(
        robot.skeleton_tree, rotations, np.asarray([0.5, 0.0, 0.0], np.float32)
    )
    robot.apply_state(state)

    ghost = robot.create_instance(
        "/robot_ghost", color=(0.3, 0.7, 1.0, 0.2)
    )
    assert ghost.asset is robot.asset
    assert ghost.skeleton_tree.num_joints() == joints
    assert all(abs(view.get_base_color().w - 0.2) < 1e-6 for view in ghost.views)
    ghost.set_alpha(0.1).set_casts_shadow(False).set_visible(True)
    assert all(abs(view.get_base_color().w - 0.1) < 1e-6 for view in ghost.views)
    assert ghost.remove()
    assert not ghost.remove()

    print("PASS: ArticulatedSurface kinematic instance")


if __name__ == "__main__":
    main()
