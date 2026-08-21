import numpy as np
from enum import Enum

from .._core import _ke
from .color import preset_rgba
from .debug_draw import log_debug_axes
from .joint_mapping import (
    COMMON,
    DEFAULT_CONTACT_SEMANTICS,
    DEFAULT_PROFILE_ORDER,
    DEFAULT_TRACKING_SEMANTICS,
    GENO,
    JOINT_ALIASES,
    JOINT_PROFILES,
    JointMapper,
    JointSemantic,
    KW,
    KW5,
    MIXAMO,
    normalize_joint_name,
)
from .math import (
    normalize_vector,
    quat_wxyz_to_xyzw,
    quat_wxyz_twist_angle,
    quat_xyzw_conjugate,
    quat_xyzw_from_two_vectors,
    quat_xyzw_multiply,
    quat_xyzw_normalize,
    quat_xyzw_rotate,
)


class CoordinateSystem(str, Enum):
    """Right-handed coordinate-system presets used by shared converters."""

    Y_UP_Z_FORWARD = "y_up_z_forward"
    Y_UP_NEG_Z_FORWARD = "y_up_neg_z_forward"
    Z_UP_X_FORWARD = "z_up_x_forward"


def coordinate_conversion_matrix(
    source: CoordinateSystem | str, target: CoordinateSystem | str
) -> np.ndarray:
    """Return the 3x3 rotation matrix converting source vectors to target axes."""
    source_value = getattr(source, "value", source)
    target_value = getattr(target, "value", target)
    return _ke.utils.coordinate_conversion_matrix(source_value, target_value)


_LAZY_IMPORTS = {
    "as_cpu_numpy": (".tensor", "as_cpu_numpy"),
    "as_sim_buffer": (".sim_buffer", "as_sim_buffer"),
    "as_tensor": (".tensor", "as_tensor"),
    "resolve_device": (".tensor", "resolve_device"),
    "SimBuffer": (".sim_buffer", "SimBuffer"),
    "to_gpu_array_view": (".sim_buffer", "to_gpu_array_view"),
    "to_external_transform_desc": (".sim_buffer", "to_external_transform_desc"),
    "quat_wxyz_conjugate": (".batched_rotations", "quat_wxyz_conjugate"),
    "quat_wxyz_from_angle_axis": (
        ".batched_rotations",
        "quat_wxyz_from_angle_axis",
    ),
    "quat_wxyz_heading_quat": (
        ".batched_rotations",
        "quat_wxyz_heading_quat",
    ),
    "quat_wxyz_heading_quat_inverse": (
        ".batched_rotations",
        "quat_wxyz_heading_quat_inverse",
    ),
    "quat_wxyz_heading_xy": (".batched_rotations", "quat_wxyz_heading_xy"),
    "quat_wxyz_heading_xz": (".batched_rotations", "quat_wxyz_heading_xz"),
    "quat_wxyz_heading_yz": (".batched_rotations", "quat_wxyz_heading_yz"),
    "quat_wxyz_multiply": (".batched_rotations", "quat_wxyz_multiply"),
    "quat_wxyz_normalize": (".batched_rotations", "quat_wxyz_normalize"),
    "quat_wxyz_rotate": (".batched_rotations", "quat_wxyz_rotate"),
    "quat_wxyz_rotate_inverse": (
        ".batched_rotations",
        "quat_wxyz_rotate_inverse",
    ),
    "quat_wxyz_to_matrix": (".batched_rotations", "quat_wxyz_to_matrix"),
    "quat_wxyz_to_tangent_normal": (
        ".batched_rotations",
        "quat_wxyz_to_tangent_normal",
    ),
    "matrix_to_rotation_6d": (".batched_rotations", "matrix_to_rotation_6d"),
    "matrix_to_quat_wxyz": (".batched_rotations", "matrix_to_quat_wxyz"),
    "quat_xyzw_from_angle_axis": (
        ".batched_rotations",
        "quat_xyzw_from_angle_axis",
    ),
    "quat_xyzw_heading_quat": (
        ".batched_rotations",
        "quat_xyzw_heading_quat",
    ),
    "quat_xyzw_heading_quat_inverse": (
        ".batched_rotations",
        "quat_xyzw_heading_quat_inverse",
    ),
    "quat_xyzw_heading_xy": (".batched_rotations", "quat_xyzw_heading_xy"),
    "quat_xyzw_heading_xz": (".batched_rotations", "quat_xyzw_heading_xz"),
    "quat_xyzw_heading_yz": (".batched_rotations", "quat_xyzw_heading_yz"),
    "quat_xyzw_rotate_inverse": (
        ".batched_rotations",
        "quat_xyzw_rotate_inverse",
    ),
    "quat_xyzw_to_matrix": (".batched_rotations", "quat_xyzw_to_matrix"),
    "quat_xyzw_to_tangent_normal": (
        ".batched_rotations",
        "quat_xyzw_to_tangent_normal",
    ),
    "rotation_6d_to_matrix": (".batched_rotations", "rotation_6d_to_matrix"),
}


def __getattr__(name):
    try:
        module_name, attr_name = _LAZY_IMPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    from importlib import import_module

    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value


__all__ = [
    "CoordinateSystem",
    "coordinate_conversion_matrix",
    "preset_rgba",
    "log_debug_axes",
    "COMMON",
    "DEFAULT_CONTACT_SEMANTICS",
    "DEFAULT_PROFILE_ORDER",
    "DEFAULT_TRACKING_SEMANTICS",
    "GENO",
    "JOINT_ALIASES",
    "JOINT_PROFILES",
    "JointMapper",
    "JointSemantic",
    "KW",
    "KW5",
    "MIXAMO",
    "normalize_joint_name",
    "normalize_vector",
    "as_cpu_numpy",
    "as_sim_buffer",
    "as_tensor",
    "SimBuffer",
    "to_gpu_array_view",
    "to_external_transform_desc",
    "quat_wxyz_conjugate",
    "quat_wxyz_from_angle_axis",
    "quat_wxyz_heading_quat",
    "quat_wxyz_heading_quat_inverse",
    "quat_wxyz_heading_xy",
    "quat_wxyz_heading_xz",
    "quat_wxyz_heading_yz",
    "quat_wxyz_multiply",
    "quat_wxyz_normalize",
    "quat_wxyz_rotate",
    "quat_wxyz_rotate_inverse",
    "quat_wxyz_to_matrix",
    "quat_wxyz_to_tangent_normal",
    "matrix_to_rotation_6d",
    "matrix_to_quat_wxyz",
    "quat_xyzw_heading_quat",
    "quat_xyzw_heading_quat_inverse",
    "quat_xyzw_heading_xy",
    "quat_xyzw_heading_xz",
    "quat_xyzw_heading_yz",
    "quat_xyzw_rotate_inverse",
    "quat_xyzw_to_matrix",
    "quat_xyzw_to_tangent_normal",
    "rotation_6d_to_matrix",
    "quat_wxyz_to_xyzw",
    "quat_wxyz_twist_angle",
    "quat_xyzw_conjugate",
    "quat_xyzw_from_two_vectors",
    "quat_xyzw_multiply",
    "quat_xyzw_normalize",
    "quat_xyzw_rotate",
    "resolve_device",
]
