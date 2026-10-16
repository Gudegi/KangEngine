"""Shared color, material, and texture helpers for visual creation."""

from pathlib import Path

import numpy as np

from .. import render as render_api
from .._core import _ke
from .._public import unwrap_native
from ..utils.color import preset_rgba


def normalize_color(color):
    arr = normalize_color_array(color)
    if arr is None:
        return None
    return _ke.Vec4(float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3]))


def normalize_color_array(color):
    if color is None:
        return None
    arr = np.asarray(color, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return None
    if arr.size == 1:
        arr = np.repeat(arr, 3)
    if arr.size == 3:
        arr = np.concatenate([arr, np.ones(1, dtype=np.float32)])
    if arr.size < 4:
        raise ValueError(f"color must have 1, 3, or 4 values; got {arr.size}")
    return np.clip(arr[:4], 0.0, 1.0).astype(np.float32, copy=False)


def _texture_values(app, source, slots, asset_dir):
    paths = source.texture_paths
    unknown = paths.keys() - slots
    if unknown:
        raise ValueError(f"unknown material texture slots: {sorted(unknown)}")

    return {
        slot: (
            app.load_texture(Path(asset_dir) / paths[slot], flip=True)
            if paths.get(slot)
            else getattr(source, slot)
        )
        for slot in slots
    }


def material_instance(app, material, *, asset_dir=""):
    """Copy material factors and resolve texture paths without changing the source."""
    if material is None:
        return app.create_pbr_material(
            base_color=_ke.Vec4(*preset_rgba(_ke.ColorType.LIGHT_GRAY)),
            metallic=0.0,
            roughness=0.8,
        )

    source = unwrap_native(material)
    if isinstance(source, _ke.VertexColorMaterial):
        return app.create_vertex_color_material(style=source.style)
    if isinstance(source, _ke.PhongMaterial):
        textures = _texture_values(
            app,
            source,
            ("diffuse_map", "specular_map", "alpha_map", "normal_map"),
            asset_dir,
        )
        return app.create_phong_material(
            ambient=source.ambient,
            diffuse=source.diffuse,
            specular=source.specular,
            shininess=source.shininess,
            **textures,
        )
    if isinstance(source, _ke.PBRMaterial):
        textures = _texture_values(
            app,
            source,
            (
                "base_color_texture",
                "normal_texture",
                "metallic_roughness_texture",
                "metallic_texture",
                "roughness_texture",
                "ao_texture",
                "orm_texture",
                "emissive_texture",
            ),
            asset_dir,
        )
        return app.create_pbr_material(
            base_color=source.base_color,
            metallic=source.metallic,
            roughness=source.roughness,
            emissive_color=source.emissive_color,
            emissive_strength=source.emissive_strength,
            **textures,
        )
    raise TypeError("visuals require a built-in material")


def visual_color(geom):
    # Material alpha remains in the shader; instance tint must not apply it again.
    return np.array(geom.rgba, dtype=np.float32)


def visual_material(app, geom, *, asset_dir="", override=None):
    source = geom.material if geom is not None else None
    if override is not None:
        source = override
    return material_instance(app, source, asset_dir=asset_dir)


def apply_visual_alpha(view, material, color):
    material = unwrap_native(material)
    if isinstance(material, _ke.PhongMaterial) and material.alpha_map is not None:
        view.set_alpha_mode(render_api.AlphaMode.MASK)
        return

    alpha = color[3]
    if isinstance(material, _ke.PBRMaterial):
        alpha *= material.base_color.w
    if alpha < 1.0:
        view.set_alpha_mode(render_api.AlphaMode.BLEND)
