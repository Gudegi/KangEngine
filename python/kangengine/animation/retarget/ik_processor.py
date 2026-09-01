"""High-level offline IK retarget processor."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

from ..coordinates import CoordinateSystem
from .ik_profile import IKRetargetConfig
from .io import load_retarget_motion

if TYPE_CHECKING:
    from ...adapters.pyroki import (
        PyrokiOfflineRetargetResult,
        PyrokiProgressCallback,
        PyrokiTrajectoryIKConfig,
        PyrokiTrajectoryState,
    )


@dataclass(frozen=True)
class IKRetargetResult:
    """Saved articulation motion and its solver diagnostics."""

    input_path: Path
    output_path: Path
    frame_count: int
    result: PyrokiOfflineRetargetResult


class IKRetargetProcessor:
    """Reusable model, layout, and coordinate pipeline for one IK pair."""

    def __init__(self, config_path: str | Path) -> None:
        from ...adapters.pyroki import load_urdf_model
        from ... import asset
        from ..articulation_motion import ArticulationCoordinateLayout

        self.config_path = Path(config_path).expanduser().resolve()
        self.config = IKRetargetConfig.load(self.config_path)
        visual = self.config.target_profile.visual_skeleton
        if visual is None:
            raise ValueError("IK target profile requires visual_skeleton")
        self.model = load_urdf_model(self.config)
        data = asset.MJCFLoader.load(str(visual), order="DFS")
        self.layout = ArticulationCoordinateLayout.from_data(data=data, free_root=True)

    def load_source(self, path: str | Path):
        return load_retarget_motion(
            path,
            self.config.source_profile,
            target_coordinate_system=CoordinateSystem(
                self.config.target_profile.coordinate_system
            ),
            pair_scale=self.config.translation_scale,
        )

    def process_motion(
        self,
        motion,
        *,
        initial_state: PyrokiTrajectoryState | None = None,
        contact_mask=None,
        detect_contacts: bool = True,
        solver_config: PyrokiTrajectoryIKConfig | None = None,
        progress_callback: PyrokiProgressCallback | None = None,
    ) -> PyrokiOfflineRetargetResult:
        from ...adapters.pyroki import (
            PyrokiTrajectoryIKConfig,
            retarget_motion_offline,
        )

        return retarget_motion_offline(
            motion,
            self.model,
            self.config,
            self.layout,
            initial_state=initial_state,
            contact_mask=contact_mask,
            detect_contacts=detect_contacts,
            config=(
                PyrokiTrajectoryIKConfig() if solver_config is None else solver_config
            ),
            progress_callback=progress_callback,
        )

    def process_file(
        self,
        input_path: str | Path,
        output_path: str | Path,
        **kwargs,
    ) -> IKRetargetResult:
        from ..articulation_io import save_articulation_motion_npz

        source_path = Path(input_path).expanduser().resolve()
        destination = Path(output_path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        result = self.process_motion(self.load_source(source_path), **kwargs)
        saved = save_articulation_motion_npz(result.motion, destination)
        return IKRetargetResult(
            source_path,
            Path(saved),
            result.motion.num_frames(),
            result,
        )

    def process_files(
        self,
        input_paths: Iterable[str | Path],
        output_directory: str | Path,
        *,
        suffix: str = "_retargeted",
        **kwargs,
    ) -> list[IKRetargetResult]:
        directory = Path(output_directory).expanduser().resolve()
        return [
            self.process_file(
                path,
                directory / f"{Path(path).stem}{suffix}.npz",
                **kwargs,
            )
            for path in input_paths
        ]


__all__ = ["IKRetargetProcessor", "IKRetargetResult"]
