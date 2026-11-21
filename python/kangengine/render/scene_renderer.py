"""Static scene rendering through Native or optional ovrtx adapters."""

from pathlib import Path
import threading

from .._core import _ke


def _validate_frame(frame, texture_paths, validated_meshes):
    for item in frame["objects"]:
        material = item["material"]
        if (material["model"] != "pbr" or material["unlit"] or
                material["has_native_textures"] or material["base_color"][3] != 1 or
                item["color"][3] != 1):
            raise ValueError(f'{item["path"]}: use opaque lit PBR and texture_resources')
        if item["mesh_id"] not in validated_meshes:
            if item["mesh"].colors:
                raise ValueError("Scene renderer does not yet support vertex colors")
            validated_meshes.add(item["mesh_id"])
        for slot, texture in item["textures"].items():
            if slot != "base_color_texture":
                raise ValueError(f"Scene renderer does not yet support texture slot {slot}")
            if texture["pixels"] is not None:
                continue
            if texture["id"] not in texture_paths:
                texture_paths[texture["id"]] = str(Path(texture["path"]).resolve(strict=True))
            texture["path"] = texture_paths[texture["id"]]
    return frame


class SceneRenderer:
    """Render a scene to RGB images without changing its authoring code.

    Supports opaque, lit PBR triangle meshes, base-color file textures, scene
    lights and a perspective camera. Sources stay
    immutable after registration; register a new resource to replace geometry
    or an image.
    Native also accepts memory images; ovrtx currently requires file sources.
    The session retains its scene and resource manager. Use on its creating
    thread and close to release backend resources.
    """

    def __init__(self, scene, resources, *, camera_path: str, backend: str = "native",
                 width: int = 512, height: int = 512):
        if backend not in ("native", "ovrtx"):
            raise ValueError("backend must be 'native' or 'ovrtx'")
        if width <= 0 or height <= 0:
            raise ValueError("Image dimensions must be positive")
        self._scene = scene
        self._resources = resources
        self._camera_path = camera_path
        self._collector = _ke._RenderSceneDataCollector()
        self._texture_paths = {}
        self._validated_meshes = set()
        self._owner_thread = threading.get_ident()
        self._adapter = None
        self._collect()
        if backend == "native":
            from ._native_scene import NativeSceneAdapter
            adapter_type = NativeSceneAdapter
        else:
            from ._ovrtx_scene import OvrtxSceneAdapter
            adapter_type = OvrtxSceneAdapter
        self._adapter = adapter_type(width, height)

    def _collect(self):
        frame = self._collector.collect(self._scene, self._resources, self._camera_path)
        return _validate_frame(frame, self._texture_paths, self._validated_meshes)

    def render(self):
        """Return an owned, top-down uint8 RGB [height, width, 3] image.

        Scene changes are evaluated on each call. The image remains valid
        after the next frame or close. Rendering and readback are synchronous.
        """
        self._check_thread()
        if self._adapter is None:
            raise RuntimeError("SceneRenderer is closed")
        image = self._adapter.render(self._collect())
        self._collector.accept()
        return image

    def _check_thread(self):
        if threading.get_ident() != self._owner_thread:
            raise RuntimeError("SceneRenderer must be used on its creating thread")

    def close(self):
        """Release renderer resources; the authored scene remains usable."""
        self._check_thread()
        if self._adapter is not None:
            adapter, self._adapter = self._adapter, None
            adapter.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
