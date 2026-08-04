"""Visual playback for the local KW Walking motion."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import kangengine as ke
from kangengine import imgui, keys, visual


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def package_asset_path(*parts: str) -> str:
    return str(Path(ke.__file__).resolve().parent / "assets" / Path(*parts))


def default_char_file() -> Path:
    return repo_root() / "assets" / "characters" / "kw" / "kw.xml"


def default_motion_file() -> Path:
    return (
        repo_root() / "assets" / "characters" / "kw" / "motions" / "Walking.npy"
    )  # from Mixamo


def _array_from_saved(value) -> np.ndarray:
    if isinstance(value, dict) and "arr" in value:
        value = value["arr"]
    return np.asarray(value, dtype=np.float32)


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        axis=-1,
    ).astype(np.float32)


def _quat_conjugate(q: np.ndarray) -> np.ndarray:
    out = q.copy()
    out[..., 1:] *= -1.0
    return out


def _convert_walking_axes(root_pos: np.ndarray, local_rot: np.ndarray):
    """Convert  from y-up/z-forward into z-up/x-forward."""
    root_pos = root_pos[:, [2, 0, 1]].astype(np.float32)
    local_rot = local_rot[..., [0, 3, 1, 2]].astype(np.float32)
    # # Source vector [x, y, z] maps to [z, x, y].
    # basis = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    # local_rot = _quat_mul(_quat_mul(basis, local_rot), _quat_conjugate(basis))
    return root_pos, local_rot


def load_motion(path: Path):
    data = np.load(path, allow_pickle=True).item()
    if "localRotations" in data:
        local_rot = _array_from_saved(data["localRotations"])  # wxyz
        root_pos = _array_from_saved(data["rootTranslations"])
        fps = int(np.asarray(data["frameRate"]).item())
        root_pos, local_rot = _convert_walking_axes(root_pos, local_rot)
    else:
        local_rot_xyzw = _array_from_saved(data["rotation"])
        local_rot = np.concatenate(
            (local_rot_xyzw[..., 3:4], local_rot_xyzw[..., :3]),
            axis=-1,
        )
        root_pos = _array_from_saved(data["root_translation"])
        fps = int(np.asarray(data["fps"]).item())
    return root_pos, local_rot, fps


class MotionViewer(ke.App):
    def __init__(self, char_file: Path, motion_file: Path):
        super().__init__()
        self.char_file = str(char_file)
        self.motion_file = str(motion_file)

    def setup(self):
        self.root_pos, self.local_rot, self.fps = load_motion(Path(self.motion_file))
        self.frame_idx = 0
        self.paused = False
        self.playback_speed = 1.0
        self._frame_accum = 0.0

        materials = self.create_standard_materials()
        self.robot_material = materials.common
        self.ground_material = materials.ground

        self.scene.add_ground(scale=100.0, material=self.ground_material)

        self.robot = visual.ArticulationVisual.from_mjcf(
            self.char_file,
            self.scene.native,
            "/kw",
            1.0,
            "DFS",
        )
        for prim in self.robot.body_prims():
            self.scene.add_renderable(prim, self.robot_material)

        self._apply_frame(0)
        print(
            f"KW Walking motion loaded: {self.local_rot.shape[0]} frames, "
            f"{self.robot.num_bodies()} bodies, {self.fps} fps"
        )
        print(f"character: {self.char_file}")
        print(f"motion: {self.motion_file}")
        self.check_error()

    def _apply_frame(self, idx: int):
        root = self.root_pos[idx]
        self.robot.set_root_translation(
            ke.Vec3(float(root[0]), float(root[1]), float(root[2]))
        )
        num_bodies = min(self.robot.num_bodies(), self.local_rot.shape[1])
        for body_idx in range(num_bodies):
            q = self.local_rot[idx][body_idx]  # wxyz
            self.robot.set_joint_rotation(
                body_idx,
                ke.Quat(float(q[0]), float(q[1]), float(q[2]), float(q[3])),
            )
        self.robot.apply_pose()

    def pre_render(self):
        if self.was_key_pressed(keys.SPACE):
            self.paused = not self.paused
        if self.was_key_pressed(keys.R):
            self.frame_idx = 0
            self._frame_accum = 0.0
            self._apply_frame(self.frame_idx)

        if self.paused:
            return

        self._frame_accum += max(0.0, self.playback_speed)
        advance = int(self._frame_accum)
        if advance <= 0:
            return
        self._frame_accum -= advance
        self.frame_idx = (self.frame_idx + advance) % self.local_rot.shape[0]
        self._apply_frame(self.frame_idx)
        self.check_error()

    def render(self):
        state = "PAUSED" if self.paused else "running"
        total = self.local_rot.shape[0]
        imgui.begin("KW Walking View Motion")
        imgui.text(f"Frame: {self.frame_idx} / {total - 1}  |  {state}")
        imgui.text(f"Bodies: {self.robot.num_bodies()}  FPS: {self.fps}")
        imgui.text("Space: pause/resume    R: reset")
        _, self.playback_speed = imgui.slider_float(
            "playback speed",
            self.playback_speed,
            0.0,
            4.0,
        )
        imgui.end()

    def post_render(self):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--char-file", default=str(default_char_file()))
    parser.add_argument("--motion-file", default=str(default_motion_file()))
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    args = parser.parse_args()

    char_file = Path(args.char_file).resolve()
    motion_file = Path(args.motion_file).resolve()
    if not char_file.exists():
        raise FileNotFoundError(char_file)
    if not motion_file.exists():
        raise FileNotFoundError(motion_file)

    app = MotionViewer(char_file, motion_file)
    app.initialize(args.width, args.height, False, ke.UpAxis.Z)
    app.start()


if __name__ == "__main__":
    main()
