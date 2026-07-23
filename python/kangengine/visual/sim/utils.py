"""Shared helpers for simulation visual sync."""

from __future__ import annotations

import hashlib
import os
import re

import numpy as np

from ..._core import _ke


def normalize_color(color):
    arr = normalize_color_array(color)
    if arr is None:
        return None
    return _ke.vec4(float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3]))


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


def batch_colors(color, env_ids):
    env_ids = tuple(env_ids)
    if color is None:
        return np.tile(
            np.asarray([0.15, 0.15, 0.15, 1.0], dtype=np.float32),
            (len(env_ids), 1),
        )
    rows = [
        normalize_color_array(select_env_visual_value(color, index, len(env_ids), env_id))
        for index, env_id in enumerate(env_ids)
    ]
    return np.stack(rows, axis=0).astype(np.float32, copy=False)


def apply_prim_color(prims, color):
    rgba = normalize_color(color)
    if rgba is None:
        return
    for prim in prims:
        prim.set_display_color_alpha(rgba)


def select_env_visual_value(value, index: int, env_count: int, env_id: int):
    if callable(value):
        return value(env_id)
    arr = np.asarray(value)
    if arr.ndim > 1 and arr.shape[0] == env_count:
        return arr[index]
    return value


def safe_prim_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", str(name)).strip("_")
    return safe or "shape"


def debug_instancing(kind, env_id, obj_id, num_bodies, handles, mesh_asset_base_path):
    if os.environ.get("KANGENGINE_DEBUG_RENDER_INSTANCING", "").lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return
    unique_handles = len(set(int(h) for h in handles))
    print(
        "[kangengine render instancing] "
        f"kind={kind} env={env_id} obj={obj_id} bodies={num_bodies} "
        f"handles={len(handles)} unique_handles={unique_handles} "
        f"mesh_asset={mesh_asset_base_path}"
    )


def mesh_asset_base_path(mjcf_path: str, scale: float, order: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_]+", "_", str(mjcf_path).split("/")[-1]).strip("_")
    if not stem:
        stem = "character"
    digest_src = f"{mjcf_path}|{float(scale):.9g}|{order}".encode("utf-8")
    digest = hashlib.sha1(digest_src).hexdigest()[:10]
    return f"/.Resources/Meshes/Skeletons/{stem}_{digest}"
