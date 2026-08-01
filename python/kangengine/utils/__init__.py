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
    "quat_wxyz_multiply": (".batched_rotations", "quat_wxyz_multiply"),
    "quat_wxyz_normalize": (".batched_rotations", "quat_wxyz_normalize"),
    "quat_wxyz_rotate": (".batched_rotations", "quat_wxyz_rotate"),
    "quat_wxyz_rotate_inverse": (
        ".batched_rotations",
        "quat_wxyz_rotate_inverse",
    ),
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
    "quat_wxyz_multiply",
    "quat_wxyz_normalize",
    "quat_wxyz_rotate",
    "quat_wxyz_rotate_inverse",
    "quat_wxyz_to_xyzw",
    "quat_wxyz_twist_angle",
    "quat_xyzw_conjugate",
    "quat_xyzw_from_two_vectors",
    "quat_xyzw_multiply",
    "quat_xyzw_normalize",
    "quat_xyzw_rotate",
    "resolve_device",
]
