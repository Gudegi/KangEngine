"""Incremental RGB video recording."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np


class VideoRecorder:
    """Consume RGB frames and optionally encode, retain, or dump them.

    The caller owns frame capture and decides when to call :meth:`write`.
    Video timing is determined by ``fps`` rather than frame arrival time.
    """

    def __init__(
        self,
        output_path: str | Path | None = None,
        fps: float = 30.0,
        *,
        codec: str | None = "libx264",
        retain_frames: bool = False,
        frame_dir: str | Path | None = None,
    ):
        fps = float(fps)
        if not math.isfinite(fps) or fps <= 0.0:
            raise ValueError("fps must be finite and positive")
        if output_path is None and not retain_frames and frame_dir is None:
            raise ValueError(
                "VideoRecorder requires output_path, retain_frames=True, or frame_dir"
            )

        self.output_path = (
            None if output_path is None else Path(output_path).expanduser()
        )
        self.frame_dir = None if frame_dir is None else Path(frame_dir).expanduser()
        self.codec = None if codec is None else str(codec)
        self.retain_frames = bool(retain_frames)
        self._fps = fps
        self._frames: list[np.ndarray] = []
        self._writer = None
        self._streamed_output: Path | None = None
        self._resolution = (0, 0)
        self.frame_count = 0
        self._active = False

    @property
    def out_dir(self) -> Path | None:
        """Compatibility alias for the optional debug frame directory."""
        return self.frame_dir

    def clear(self):
        """Discard retained state and begin a new recording on the next start."""
        self._close_writer()
        self._frames.clear()
        self._streamed_output = None
        self._resolution = (0, 0)
        self.frame_count = 0
        self._active = False

    def start(self):
        """Activate frame collection, clearing any previous recording."""
        self.clear()
        self._active = True
        return self

    def stop(self):
        """Stop frame collection and finalize an incremental encoder."""
        self._active = False
        self._close_writer()
        return self

    def close(self):
        """Finalize owned encoding resources."""
        return self.stop()

    def is_active(self) -> bool:
        return self._active

    def add_frame(self, frame):
        """Add a frame regardless of the active recording state."""
        return self.write(frame, force=True)

    def write(self, frame, *, force: bool = False):
        """Consume one ``uint8`` RGB frame shaped ``[height, width, 3]``."""
        if not self._active and not force:
            return None

        rgb = np.asarray(frame, dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError(
                f"expected RGB frame with shape [H, W, 3], got {rgb.shape}"
            )
        resolution = (int(rgb.shape[0]), int(rgb.shape[1]))
        if self.frame_count == 0:
            self._resolution = resolution
        elif resolution != self._resolution:
            raise ValueError(
                f"frame resolution changed from {self._resolution} to {resolution}"
            )

        rgb = np.ascontiguousarray(rgb)
        if self.retain_frames:
            self._frames.append(rgb.copy())
        if self.frame_dir is not None:
            self._write_ppm(rgb)
        if self.output_path is not None:
            if self._writer is None:
                self._writer = self._open_writer(self.output_path)
                self._streamed_output = self.output_path
            self._writer.append_data(rgb)

        self.frame_count += 1
        return self.frame_count - 1

    def get_fps(self) -> float:
        return self._fps

    def get_num_frames(self) -> int:
        return self.frame_count

    def get_resolution(self) -> tuple[int, int]:
        return self._resolution

    def get_frames(self) -> list[np.ndarray]:
        if not self.retain_frames:
            raise RuntimeError(
                "frames were not retained; construct VideoRecorder with retain_frames=True"
            )
        return self._frames

    def save(self, file_path: str | Path | None = None):
        """Finalize or encode retained frames to ``file_path``."""
        path = self.output_path if file_path is None else Path(file_path).expanduser()
        if path is None:
            raise ValueError("save() requires a file path")
        if self.frame_count == 0:
            return None

        self.stop()
        if self._same_path(path, self._streamed_output):
            return path
        if not self.retain_frames:
            raise RuntimeError("saving to a new path requires retain_frames=True")

        writer = self._open_writer(path)
        try:
            for frame in self._frames:
                writer.append_data(frame)
        finally:
            writer.close()
        return path

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        try:
            self._close_writer()
        except Exception:
            pass

    def _open_writer(self, path: Path):
        try:
            import imageio.v2 as imageio
        except ImportError as exc:
            raise RuntimeError(
                "KangEngine video dependencies are missing; reinstall kangengine"
            ) from exc

        path.parent.mkdir(parents=True, exist_ok=True)
        kwargs = {"fps": self._fps}
        if path.suffix.lower() != ".gif":
            kwargs["macro_block_size"] = 1
            if self.codec is not None:
                kwargs["codec"] = self.codec
        return imageio.get_writer(str(path), **kwargs)

    def _close_writer(self):
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def _write_ppm(self, rgb: np.ndarray):
        self.frame_dir.mkdir(parents=True, exist_ok=True)
        path = self.frame_dir / f"frame_{self.frame_count:06d}.ppm"
        height, width, _ = rgb.shape
        with path.open("wb") as stream:
            stream.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
            stream.write(rgb.tobytes())

    @staticmethod
    def _same_path(lhs: Path, rhs: Path | None) -> bool:
        if rhs is None:
            return False
        return lhs.resolve() == rhs.resolve()
