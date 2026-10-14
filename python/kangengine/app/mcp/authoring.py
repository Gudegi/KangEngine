"""Owner-thread authoring of static scene objects through common App APIs."""

from pathlib import PurePosixPath
import re

import numpy as np

from ... import geometry, render, scene
from ...utils.math import quat_wxyz_to_matrix_numpy
from ..capture import AutomationError
from .control import camera_get, camera_set
from .inspection import _bounded_integer, _prim_snapshot, _resolve_prim


COMMANDS = {
    "scene_create_primitive",
    "scene_set_transform",
    "scene_set_material",
    "scene_delete",
    "object_inspect",
    "resource_inspect",
    "camera_focus",
}
MAX_OBJECTS = 256
MAX_MATERIALS = 128


def _vector(value, name, size, *, minimum=-1e6, maximum=1e6):
    result = np.asarray(value, dtype=float)
    if (
        result.shape != (size,)
        or not np.isfinite(result).all()
        or np.any(result < minimum)
        or np.any(result > maximum)
    ):
        raise AutomationError(
            "INVALID_ARGUMENT",
            f"{name} needs {size} finite values in {minimum}..{maximum}",
        )
    return result


def _transform(translation, rotation_wxyz, scale):
    translation = _vector(translation, "translation", 3)
    scale = _vector(scale, "scale", 3, minimum=1e-4, maximum=1e4)
    rotation = _vector(rotation_wxyz, "rotation_wxyz", 4)
    length = np.linalg.norm(rotation)
    if length < 1e-8:
        raise AutomationError("INVALID_ARGUMENT", "rotation_wxyz must be nonzero")
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = quat_wxyz_to_matrix_numpy(rotation / length) * scale
    matrix[:3, 3] = translation
    # Mat4's input conversion consumes columns; its output buffer exposes rows.
    return matrix.T.copy()


def _static_ancestry(prim):
    """Reject simulation ownership before changing a visual or its parent frame."""
    current = prim
    while current is not None:
        component = current.get_render_component()
        if (
            current.has_articulation_component()
            or current.has_articulation_binding_component()
            or current.has_collision_shape_component()
            or (
                component is not None
                and component.transform_source == render.TransformSource.EXTERNAL_BUFFER
            )
        ):
            raise AutomationError(
                "UNSUPPORTED", "Physics and ExternalBuffer objects are not editable"
            )
        current = current.get_parent()


def _update_transform(matrix, translation, rotation_wxyz, scale):
    """Preserve omitted affine components; never silently discard shear/reflection."""
    matrix = np.array(matrix, dtype=np.float64, copy=True)
    if not np.isfinite(matrix).all() or not np.allclose(matrix[3], [0, 0, 0, 1]):
        raise AutomationError("UNSUPPORTED", "Transform must be finite and affine")
    if translation is not None:
        matrix[:3, 3] = _vector(translation, "translation", 3)
    if rotation_wxyz is not None and scale is not None:
        matrix[:3, :3] = _transform([0, 0, 0], rotation_wxyz, scale).T[:3, :3]
    elif rotation_wxyz is not None or scale is not None:
        linear = matrix[:3, :3]
        lengths = np.linalg.norm(linear, axis=0)
        if np.any(lengths < 1e-8):
            raise AutomationError(
                "UNSUPPORTED", "Cannot partially edit a singular transform"
            )
        rotation = linear / lengths
        if (
            not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
            or np.linalg.det(rotation) < 0
        ):
            raise AutomationError(
                "UNSUPPORTED",
                "Sheared or reflected transforms require both rotation_wxyz and scale; "
                "translation-only edits preserve them",
            )
        if rotation_wxyz is not None:
            rotation = _transform([0, 0, 0], rotation_wxyz, [1, 1, 1]).T[:3, :3]
        if scale is not None:
            lengths = _vector(scale, "scale", 3, minimum=1e-4, maximum=1e4)
        matrix[:3, :3] = rotation * lengths
    return matrix.astype(np.float32).T.copy()


class SceneAuthoring:
    """Retain bounded reusable resources, never non-owning Prim pointers."""

    def __init__(self):
        self._objects = {}
        self._meshes = {}
        self._mesh_bounds = {}
        self._materials = {}

    def _material(self, app, color, shininess):
        color = _vector(color, "color", 3, minimum=0, maximum=1)
        if not np.isfinite(shininess) or not 1 <= shininess <= 256:
            raise AutomationError("INVALID_ARGUMENT", "shininess must be in 1..256")
        key = (*color, float(shininess))
        if key not in self._materials:
            if len(self._materials) >= MAX_MATERIALS:
                raise AutomationError(
                    "BUSY", "This session already has 128 authoring materials"
                )
            self._materials[key] = app.create_phong_material(
                ambient=(color * 0.2).tolist(),
                diffuse=color.tolist(),
                specular=[0.3, 0.3, 0.3],
                shininess=shininess,
            )
        return self._materials[key]

    def scene_create_primitive(
        self,
        app,
        *,
        path,
        shape="box",
        translation=(0, 0, 0),
        rotation_wxyz=(1, 0, 0, 0),
        scale=(1, 1, 1),
        color=(0.7, 0.7, 0.7),
    ):
        if (
            not isinstance(path, str)
            or len(path) > 512
            or not re.fullmatch(r"(?:/[A-Za-z_][A-Za-z_0-9]*)+", path)
        ):
            raise AutomationError(
                "INVALID_ARGUMENT", "Use an absolute path with identifier segments"
            )
        if shape not in {"box", "sphere", "plane"}:
            raise AutomationError(
                "INVALID_ARGUMENT", "shape must be box, sphere or plane"
            )
        matrix = _transform(translation, rotation_wxyz, scale)
        parent = _resolve_prim(app, str(PurePosixPath(path).parent))
        if parent.get_type() not in {scene.PrimType.ROOT, scene.PrimType.XFORM}:
            raise AutomationError(
                "UNSUPPORTED", "Parent must be an existing root or Xform"
            )
        _static_ancestry(parent)
        if app.scene.get_prim_at_path(path) is not None:
            raise AutomationError("INVALID_ARGUMENT", "The scene path already exists")
        # Detached shared components remain safe to inspect after external deletion.
        self._objects = {p: c for p, c in self._objects.items() if c.attached}
        if len(self._objects) >= MAX_OBJECTS:
            raise AutomationError("BUSY", "At most 256 authoring objects may exist")
        material = self._material(app, color, 32)
        if shape not in self._meshes:
            if shape == "box":
                mesh = geometry.create_box_data(1, 1, 1)
            elif shape == "sphere":
                mesh = geometry.create_sphere_data(0.5, 32, 16)
            else:
                mesh = geometry.create_plane_data(1, app.up_axis)
            self._meshes[shape] = mesh
            # These small built-in meshes are measured once, without GPU readback.
            vertices = np.array([[v.x, v.y, v.z] for v in mesh.vertices])
            self._mesh_bounds[shape] = np.stack(
                (vertices.min(axis=0), vertices.max(axis=0))
            )
        try:
            view = app.scene.add_mesh(
                path,
                self._meshes[shape],
                material=material,
                uri=f"generated://automation/{shape}",
            )
            view.set_local_matrix(matrix)
        except Exception:
            app.remove_prim(path)
            raise
        self._objects[path] = view.prim.get_transform_component()
        return self.object_inspect(app, path=path)

    def scene_set_transform(
        self,
        app,
        *,
        path,
        translation=None,
        rotation_wxyz=None,
        scale=None,
    ):
        if translation is None and rotation_wxyz is None and scale is None:
            raise AutomationError(
                "INVALID_ARGUMENT", "Supply at least one transform field"
            )
        prim = _resolve_prim(app, path)
        _static_ancestry(prim)
        if path == "/" or prim.get_children() or not prim.has_transform_component():
            raise AutomationError(
                "UNSUPPORTED", "Transform editing requires a leaf with a transform"
            )
        matrix = _update_transform(
            prim.compute_local_matrix(), translation, rotation_wxyz, scale
        )
        prim.set_local_matrix(matrix)
        return self.object_inspect(app, path=path)

    def scene_set_material(self, app, *, path, color, shininess=32.0):
        prim = _resolve_prim(app, path)
        _static_ancestry(prim)
        component = prim.get_render_component()
        if component is None:
            raise AutomationError("UNSUPPORTED", "The prim has no render component")
        material = self._material(app, color, shininess)
        if not app.get_scene_render_system().set_material(component, material.native):
            raise AutomationError(
                "INTERNAL_ERROR",
                "Material replacement failed; previous material restored",
            )
        prim.set_display_color_alpha([1, 1, 1, 1])
        return self.object_inspect(app, path=path)

    def _owned(self, prim):
        component = self._objects.get(prim.get_path())
        return (
            component is not None
            and component.attached
            and component == prim.get_transform_component()
        )

    def scene_delete(self, app, *, path):
        prim = _resolve_prim(app, path)
        _static_ancestry(prim)
        if not self._owned(prim) or prim.get_children():
            raise AutomationError(
                "UNSUPPORTED",
                "Only childless automation-created objects can be deleted",
            )
        removed = app.remove_prim(path)
        if removed:
            del self._objects[path]
        return {"path": path, "removed": removed}

    def object_inspect(self, app, *, path):
        prim = _resolve_prim(app, path)
        result = _prim_snapshot(app, prim, 0)
        component = prim.get_render_component()
        result["automation_created"] = self._owned(prim)
        result["render"] = (
            None
            if component is None
            else {
                "double_sided": component.double_sided,
                "casts_shadow": component.casts_shadow,
                "alpha_mode": component.alpha_mode.name,
            }
        )
        color = prim.get_display_color_alpha()
        result["color_multiplier"] = (
            None if color is None else [color.x, color.y, color.z, color.w]
        )
        result["physics_metadata"] = {
            "articulation": prim.has_articulation_component(),
            "articulation_binding": prim.has_articulation_binding_component(),
            "collision_shape": prim.has_collision_shape_component(),
        }
        return result

    def resource_inspect(self, app, *, handle, max_paths=100):
        _bounded_integer(handle, "handle", 1, 2**64 - 1)
        _bounded_integer(max_paths, "max_paths", 1, 1000)
        prim = app.resources.resource_prim(handle)
        if prim is None:
            raise AutomationError("NOT_FOUND", "Resource handle does not exist")
        result = _prim_snapshot(app, prim, 0)["resource"]
        paths = list(app.resources.usage_paths(handle))
        return {
            **result,
            "usage_count": app.resources.usage_count(handle),
            "usage_paths": paths[:max_paths],
            "truncated": len(paths) > max_paths,
        }

    def camera_focus(self, app, *, path, distance=None):
        if distance is not None and (
            not np.isfinite(distance) or not 1e-3 <= distance <= 1e6
        ):
            raise AutomationError(
                "INVALID_ARGUMENT", "distance must be in 0.001..1000000"
            )
        prim = _resolve_prim(app, path)
        _static_ancestry(prim)
        if not prim.has_transform_component():
            raise AutomationError("UNSUPPORTED", "The prim has no transform")
        target = np.asarray(prim.compute_world_matrix(), dtype=float)[:3, 3]
        bounds = None
        if distance is None:
            shape = next(
                (
                    name
                    for name, mesh in self._meshes.items()
                    if mesh == prim.get_mesh_data()
                ),
                None,
            )
            if not self._owned(prim) or shape is None:
                raise AutomationError(
                    "UNSUPPORTED",
                    "Automatic focus requires an automation-created primitive; "
                    "supply distance to focus another static prim's origin",
                )
            lower, upper = self._mesh_bounds[shape]
            corners = np.array(
                [
                    [x, y, z, 1]
                    for x in (lower[0], upper[0])
                    for y in (lower[1], upper[1])
                    for z in (lower[2], upper[2])
                ]
            )
            world = (np.asarray(prim.compute_world_matrix()) @ corners.T).T[:, :3]
            bounds = np.stack((world.min(axis=0), world.max(axis=0)))
            target = bounds.mean(axis=0)
            radius = np.linalg.norm(bounds[1] - bounds[0]) / 2
            width, height = app.get_width(), app.get_height()
            if width <= 0 or height <= 0 or not np.isfinite(bounds).all():
                raise AutomationError(
                    "NOT_READY", "Valid viewport and geometry bounds are required"
                )
            camera = app.get_camera()
            half_vertical = np.deg2rad(camera.get_fov()) / 2
            half_horizontal = np.arctan(np.tan(half_vertical) * width / height)
            half_angle = min(half_vertical, half_horizontal)
            distance = max(
                1e-3,
                1.1 * radius / np.sin(half_angle),
                radius + camera.get_near_plane() * 1.1,
            )
            if not np.isfinite(distance) or distance + radius >= camera.get_far_plane():
                raise AutomationError(
                    "UNSUPPORTED", "Geometry does not fit within camera clipping planes"
                )
        camera = camera_get(app)
        direction = np.array(camera["position"]) - camera["target"]
        length = np.linalg.norm(direction)
        if length < 1e-4:
            raise AutomationError("NOT_READY", "Set a valid navigation camera first")
        result = camera_set(
            app, position=target + direction / length * distance, target=target
        )
        return {
            **result,
            "distance": float(distance),
            "focus_mode": "bounds" if bounds is not None else "origin",
            "world_bounds": None if bounds is None else bounds.tolist(),
        }
