"""Visual assets and instances, loaded when requested."""

from importlib import import_module

_IMPORTS = {
    "ArticulationVisual": (".articulation_visual", "ArticulationVisual"),
    "ArticulationVisualAsset": (".articulation_visual", "ArticulationVisualAsset"),
    "ArticulatedSurface": (".articulated_surface", "ArticulatedSurface"),
    "ArticulatedSurfaceAsset": (".articulated_surface", "ArticulatedSurfaceAsset"),
    "DeformableSurface": (".deformable", "DeformableSurface"),
    "SkinnedSurface": (".deformable", "SkinnedSurface"),
    "SkinnedSurfaceAsset": (".deformable", "SkinnedSurfaceAsset"),
    "RigidVisual": (".rigid_visual", "RigidVisual"),
    "SkinVisual": (".skin", "SkinVisual"),
    "SkeletalVisual": (".skeletal", "SkeletalVisual"),
    "SkeletalVisualConfig": (".skeletal", "SkeletalVisualConfig"),
}

__all__ = [
    "ArticulationVisual",
    "ArticulationVisualAsset",
    "ArticulatedSurface",
    "ArticulatedSurfaceAsset",
    "DeformableSurface",
    "RigidVisual",
    "SkinVisual",
    "SkinnedSurface",
    "SkinnedSurfaceAsset",
    "SkeletalVisual",
    "SkeletalVisualConfig",
    "sim",
]


def __getattr__(name):
    if name == "sim":
        value = import_module(".sim", __name__)
    elif name in _IMPORTS:
        module, attribute = _IMPORTS[name]
        value = getattr(import_module(module, __name__), attribute)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
