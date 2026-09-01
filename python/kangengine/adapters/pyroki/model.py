"""URDF-backed PyRoki model loading and profile validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ...animation.retarget.ik_profile import IKRetargetConfig, IKTargetProfile
from ._dependency import load_pyroki


@dataclass(frozen=True)
class PyrokiRobotModel:
    """Loaded backend objects and their stable name metadata."""

    urdf: object
    robot: object
    link_names: tuple[str, ...]
    actuated_joint_names: tuple[str, ...]

    def forward_kinematics(
        self, configuration: npt.ArrayLike | None = None
    ) -> npt.NDArray[np.float32]:
        """Return root-local link poses as ``(links, 7)`` WXYZ+XYZ rows."""

        if configuration is None:
            configuration = np.zeros(len(self.actuated_joint_names), dtype=np.float32)
        values = np.asarray(configuration, dtype=np.float32)
        expected_width = len(self.actuated_joint_names)
        if values.ndim < 1 or values.shape[-1] != expected_width:
            raise ValueError(
                f"configuration shape is {values.shape}, expected (..., {expected_width})"
            )
        poses = np.asarray(self.robot.forward_kinematics(values), dtype=np.float32)
        expected_pose_shape = (*values.shape[:-1], len(self.link_names), 7)
        if poses.shape != expected_pose_shape:
            raise RuntimeError(
                f"PyRoki FK shape is {poses.shape}, expected {expected_pose_shape}"
            )
        return poses

    def connection_mask(self, link_names: tuple[str, ...]) -> npt.NDArray[np.float32]:
        """Return direct selected-link connections along the kinematic tree."""

        try:
            selected = [self.link_names.index(name) for name in link_names]
        except ValueError as error:
            raise ValueError("connection mask contains an unknown link") from error
        parent_joints = tuple(
            int(index) for index in self.robot.links.parent_joint_indices
        )
        joint_parents = tuple(int(index) for index in self.robot.joints.parent_indices)
        link_from_parent_joint = {
            joint: link for link, joint in enumerate(parent_joints) if joint >= 0
        }
        root_links = [link for link, joint in enumerate(parent_joints) if joint < 0]
        if len(root_links) != 1:
            raise ValueError("PyRoki model must contain exactly one root link")
        root_link = root_links[0]
        selected_set = set(selected)
        mask = np.eye(len(selected), dtype=np.float32)

        for child_slot, child_link in enumerate(selected):
            current_joint = parent_joints[child_link]
            while current_joint >= 0:
                parent_joint = joint_parents[current_joint]
                parent_link = (
                    root_link
                    if parent_joint < 0
                    else link_from_parent_joint[parent_joint]
                )
                if parent_link in selected_set:
                    parent_slot = selected.index(parent_link)
                    mask[child_slot, parent_slot] = 1.0
                    mask[parent_slot, child_slot] = 1.0
                    break
                current_joint = parent_joint
        return mask


def load_urdf_model(
    profile: IKRetargetConfig | IKTargetProfile,
) -> PyrokiRobotModel:
    """Load a custom URDF and verify all profile names against PyRoki."""

    urdf_path = (
        profile.urdf_path
        if isinstance(profile, IKRetargetConfig)
        else profile.skeleton
    ).expanduser().resolve()
    if not urdf_path.is_file():
        raise FileNotFoundError(f"URDF file does not exist: {urdf_path}")
    pyroki, yourdfpy = load_pyroki()

    def filename_handler(fname: str) -> str:
        return yourdfpy.filename_handler_magic(fname, dir=urdf_path.parent)

    urdf = yourdfpy.URDF.load(urdf_path, filename_handler=filename_handler)
    robot = pyroki.Robot.from_urdf(urdf)
    model = PyrokiRobotModel(
        urdf=urdf,
        robot=robot,
        link_names=tuple(str(name) for name in robot.links.names),
        actuated_joint_names=tuple(str(name) for name in robot.joints.actuated_names),
    )
    if isinstance(profile, IKRetargetConfig):
        validate_profile(profile, model)
    else:
        missing = sorted(set(profile.joint_order) - set(model.actuated_joint_names))
        if missing:
            raise ValueError(
                "target profile contains unknown URDF joints: " + ", ".join(missing)
            )
    return model


def validate_profile(profile: IKRetargetConfig, model: PyrokiRobotModel) -> None:
    """Validate target links and canonical joint ordering."""

    known_links = set(model.link_names)
    requested_links = {mapping.target_link for mapping in profile.link_mappings}
    requested_links.update(point.target_link for point in profile.auxiliary_points)
    requested_links.update(profile.contact_links)
    requested_links.add(profile.target_root_link)
    missing_links = sorted(requested_links - known_links)
    if missing_links:
        raise ValueError(
            "retarget profile references links absent from URDF: "
            + ", ".join(missing_links)
        )

    known_joints = set(model.actuated_joint_names)
    requested_joints = set(profile.joint_order)
    missing_joints = sorted(requested_joints - known_joints)
    if missing_joints:
        raise ValueError(
            "retarget profile references actuated joints absent from URDF: "
            + ", ".join(missing_joints)
        )
    omitted_joints = sorted(known_joints - requested_joints)
    if omitted_joints:
        raise ValueError(
            "joint_order must list every actuated URDF joint; omitted: "
            + ", ".join(omitted_joints)
        )


__all__ = ["PyrokiRobotModel", "load_urdf_model", "validate_profile"]
