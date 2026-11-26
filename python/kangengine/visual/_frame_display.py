"""Show body and joint coordinate axes through scene geometry on either renderer."""

import numpy as np
import torch

from ..utils.batched_rotations import quat_xyzw_to_matrix


_AXIS_COLORS = np.array(
    [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]], dtype=np.float32
)


class FrameDisplay:
    def __init__(self, app):
        self.app = app
        self.path = "/__debug__/physics_frames"
        self.length = 0.12
        self._signature = None
        self._colors = np.empty((0, 4), dtype=np.float32)

    def sync(self, source, bodies, joints):
        frames = source.frame_poses(bodies=bodies, joints=joints)
        signature = (
            bodies,
            joints,
            self.length,
            *(poses.tobytes() for poses in frames),
        )
        if signature == self._signature:
            return
        self._signature = signature

        for name, enabled, poses in zip(("body", "joint"), (bodies, joints), frames):
            path = f"{self.path}/{name}"
            if not enabled or not len(poses):
                self.app.debug_overlay.remove(path)
                continue

            rotation = quat_xyzw_to_matrix(torch.from_numpy(poses[:, 3:])).numpy()
            starts = np.repeat(poses[:, None, :3], 3, axis=1)
            ends = starts + rotation.transpose(0, 2, 1) * self.length
            color_count = len(poses) * 3
            if len(self._colors) < color_count:
                self._colors = np.tile(_AXIS_COLORS, (len(poses), 1))
            self.app.debug_overlay.lines(
                path,
                starts.reshape(-1, 3),
                ends.reshape(-1, 3),
                self._colors[:color_count],
                width=2.0,
                overlay=True,
            )

    def release(self):
        for name in ("body", "joint"):
            self.app.debug_overlay.remove(f"{self.path}/{name}")
        self._signature = None
        self._colors = np.empty((0, 4), dtype=np.float32)
