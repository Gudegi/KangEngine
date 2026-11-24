"""Persistent debug batches and SPG inputs for OVRTX output images."""

import math
import os
from pathlib import Path

import numpy as np
import ovstage


class OvrtxDebugDraw:
    def __init__(self, adapter):
        self._adapter = adapter
        self._batches = {}
        self._dirty = False
        self._camera = None
        self._arrays = {}
        self._active = False

    @staticmethod
    def schema():
        source = Path(os.environ["KANGENGINE_ASSETS_ROOT"]) / "shaders/ovrtx/debug_draw.slang"
        return f'''
        def RenderVar "DebugDepth" {{
            string sourceName = "DistanceToImagePlaneSD"
            opaque omni:rtx:aov
        }}
        def RenderVar "DebugColor" {{
            string sourceName = "DebugColor"
            opaque omni:rtx:aov.connect = <../DebugDraw.outputs:Color>
        }}
        def Shader "DebugDraw" {{
            uniform token info:implementationSource = "sourceAsset"
            uniform asset info:spg:sourceAsset = @{source}@
            uniform token info:spg:sourceAsset:subIdentifier = "debugDraw"
            float4[] inputs:primitives = []
            int[] inputs:offsets = []
            int[] inputs:ids = []
            opaque inputs:Image.connect = <../LdrColor.omni:rtx:aov>
            opaque inputs:Depth.connect = <../DebugDepth.omni:rtx:aov>
            opaque outputs:Color
        }}
'''

    @property
    def output_path(self):
        return "/Render/Camera/DebugColor" if self._active else "/Render/Camera/LdrColor"

    def lines(self, path, starts, ends, colors, width, hidden, overlay):
        self._batches[(0, path)] = dict(starts=starts, ends=ends, colors=colors,
                                       size=width, hidden=hidden, overlay=overlay)
        self._dirty = True

    def points(self, path, points, colors, size, hidden, overlay):
        self._batches[(1, path)] = dict(starts=points, ends=points, colors=colors,
                                       size=size, hidden=hidden, overlay=overlay)
        self._dirty = True

    def clear_lines(self, path):
        self._clear((0, path))

    def clear_points(self, path):
        self._clear((1, path))

    def _clear(self, key):
        if key in self._batches:
            batch = self._batches[key]
            batch.update(starts=np.empty((0, 3)), ends=np.empty((0, 3)),
                         colors=np.empty((0, 4)))
            self._dirty = True

    def set_visible(self, path, visible):
        for kind in (0, 1):
            batch = self._batches.get((kind, path))
            if batch is not None and batch["hidden"] == visible:
                batch["hidden"] = not visible
                self._dirty = True

    def remove(self, path):
        for kind in (0, 1):
            if self._batches.pop((kind, path), None) is not None:
                self._dirty = True

    def remove_all(self):
        if self._batches:
            self._batches.clear()
            self._dirty = True

    def close(self):
        self._batches.clear()
        self._arrays.clear()
        self._adapter = None

    def _write_array(self, name, values, lanes=1):
        if name in self._arrays and np.array_equal(values, self._arrays[name]):
            return
        adapter = self._adapter
        tensor = ovstage.make_dltensor(
            values, shape=[len(values)],
            dtype=ovstage.numpy_to_dldatatype(values.dtype, lanes=lanes),
        )
        with adapter._query("/Render/Camera/DebugDraw") as query:
            adapter._stage.write_attribute(query, "inputs:" + name,
                                           ordinal=adapter._ordinal, tensors=tensor,
                                           is_array=True).wait()
        self._arrays[name] = values

    def prepare(self, camera, structural_change):
        active = any(len(batch["starts"]) and not batch["hidden"] for batch in self._batches.values())
        if active != self._active or (active and structural_change):
            names = ["LdrColor", "DebugDepth", "DebugColor"] if active else ["LdrColor"]
            with ovstage.PathDictionary(self._adapter._stage) as paths:
                ids = np.array([paths.intern_path("/Render/Camera/" + name).raw_id for name in names], dtype=np.uint64)
                with self._adapter._query("/Render/Camera") as query:
                    self._adapter._stage.write_attribute(
                        query, "orderedVars", ordinal=self._adapter._ordinal,
                        tensors=ids, is_array=True,
                        semantic=ovstage.AttributeSemantic.RELATIONSHIP_PATH_ID,
                    ).wait()
            self._active = active
        if structural_change:
            self._arrays.clear()
        if not active:
            if self._dirty:
                self._write_array("primitives", np.empty((0, 4), dtype=np.float32), 4)
                self._write_array("offsets", np.empty(0, dtype=np.int32))
                self._write_array("ids", np.empty(0, dtype=np.int32))
                self._dirty = False
                self._camera = None
            return

        width, height = self._adapter._resolution
        signature = (camera["world"].tobytes(), camera["vertical_fov"], camera["near"], camera["far"], width, height)
        if not self._dirty and not structural_change and signature == self._camera:
            return
        view = np.linalg.inv(camera["world"])
        scale = height / (2 * math.tan(math.radians(camera["vertical_fov"]) / 2))
        columns, rows = (width + 15) // 16, (height + 15) // 16
        tiles = {}
        primitives = []

        # Preserve Native's depth-first, lines-before-points ordering.
        batches = sorted(self._batches.items())
        for overlay in (False, True):
            for _, batch in batches:
                if batch["hidden"] or batch["overlay"] != overlay or not len(batch["starts"]):
                    continue
                starts, ends, colors = batch["starts"], batch["ends"], batch["colors"]
                a = starts @ view[:3, :3].T + view[:3, 3]
                b = ends @ view[:3, :3].T + view[:3, 3]
                a[:, 2] *= -1
                b[:, 2] *= -1
                for idx, (start, end) in enumerate(zip(a, b)):
                    # Clip in camera space before dividing by depth.
                    delta = end - start
                    lo, hi = 0.0, 1.0
                    if delta[2] == 0:
                        if not camera["near"] <= start[2] <= camera["far"]:
                            continue
                    else:
                        t0, t1 = sorted(((camera["near"] - start[2]) / delta[2],
                                         (camera["far"] - start[2]) / delta[2]))
                        lo, hi = max(lo, t0), min(hi, t1)
                        if lo > hi:
                            continue
                    x0, y0, z0 = start + lo * delta
                    x1, y1, z1 = start + hi * delta
                    x0, y0 = x0 / z0 * scale + width / 2, -y0 / z0 * scale + height / 2
                    x1, y1 = x1 / z1 * scale + width / 2, -y1 / z1 * scale + height / 2
                    radius = batch["size"] / 2
                    if not all(math.isfinite(value) for value in (x0, y0, x1, y1, radius)):
                        raise ValueError("Debug projection and pixel size must be finite")
                    left = max(math.floor((min(x0, x1) - radius) / 16), 0)
                    top = max(math.floor((min(y0, y1) - radius) / 16), 0)
                    right = min(math.floor((max(x0, x1) + radius) / 16), columns - 1)
                    bottom = min(math.floor((max(y0, y1) + radius) / 16), rows - 1)
                    if left > right or top > bottom:
                        continue
                    color = colors[min(idx, len(colors) - 1)] if len(colors) else np.ones(4)
                    primitive_idx = len(primitives) // 3
                    primitives.extend([
                        [x0, y0, 1 / z0, radius],
                        [x1, y1, 1 / z1, int(overlay)],
                        color,
                    ])
                    for y in range(top, bottom + 1):
                        for x in range(left, right + 1):
                            tiles.setdefault(y * columns + x, []).append(primitive_idx)

        # Allocate lists only for occupied tiles; the shader still uses a dense offset table.
        offsets = np.zeros(columns * rows + 1, dtype=np.int32)
        ids = []
        for tile, entries in sorted(tiles.items()):
            offsets[tile + 1] = len(entries)
            ids.extend(entries)
        np.cumsum(offsets, out=offsets)
        self._write_array("primitives", np.asarray(primitives, dtype=np.float32).reshape(-1, 4), 4)
        self._write_array("offsets", offsets)
        self._write_array("ids", np.asarray(ids, dtype=np.int32))
        self._camera = signature
        self._dirty = False
