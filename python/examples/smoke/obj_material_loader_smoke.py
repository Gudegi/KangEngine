"""Validate OBJ loader material metadata imported from MTL files."""

from __future__ import annotations

import tempfile
from pathlib import Path

import kangengine as ke


def _close(actual, expected, eps=1.0e-5):
    if abs(float(actual) - float(expected)) > eps:
        raise AssertionError(f"{actual} != {expected}")


def main():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        obj = root / "quad.obj"
        mtl = root / "quad.mtl"

        mtl.write_text(
            "\n".join(
                [
                    "newmtl OrangePaint",
                    "Ka 0.1 0.1 0.1",
                    "Kd 0.8 0.3 0.02",
                    "Ks 0.5 0.5 0.5",
                    "Ns 64.0",
                    "d 0.75",
                    "map_Kd textures/orange.png",
                    "norm textures/orange_n.png",
                    "",
                    "newmtl BluePaint",
                    "Kd 0.0 0.2 1.0",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        obj.write_text(
            "\n".join(
                [
                    "mtllib quad.mtl",
                    "v -0.5 -0.5 0.0",
                    "v 0.5 -0.5 0.0",
                    "v 0.5 0.5 0.0",
                    "v -0.5 0.5 0.0",
                    "vt 0.0 0.0",
                    "vt 1.0 0.0",
                    "vt 1.0 1.0",
                    "vt 0.0 1.0",
                    "vn 0.0 0.0 1.0",
                    "usemtl OrangePaint",
                    "f 1/1/1 2/2/1 3/3/1",
                    "usemtl BluePaint",
                    "f 1/1/1 3/3/1 4/4/1",
                    "",
                ]
            ),
            encoding="utf-8",
        )

        plain_mesh = ke.asset.load_obj(str(obj))
        info = ke.asset.load_obj_with_materials(str(obj))

        if len(plain_mesh.vertices) != len(info.mesh_data.vertices):
            raise AssertionError("load_obj and load_obj_with_materials mesh mismatch")
        if info.material_count != 2:
            raise AssertionError("expected two MTL materials")
        if info.primary_material_index != 0:
            raise AssertionError("expected primary material index 0")
        if info.subset_count != 2:
            raise AssertionError("expected two material subsets")

        mat = info.materials[0]
        if mat.name != "OrangePaint":
            raise AssertionError("material name mismatch")
        _close(mat.diffuse_color[0], 0.8)
        _close(mat.diffuse_color[1], 0.3)
        _close(mat.diffuse_color[2], 0.02)
        _close(mat.diffuse_color[3], 0.75)
        _close(mat.shininess, 64.0)
        if not mat.has_diffuse_texture or not mat.has_normal_texture:
            raise AssertionError("expected diffuse and normal texture references")
        if mat.diffuse_texture_path != str(root / "textures" / "orange.png"):
            raise AssertionError("diffuse texture path was not resolved")
        if mat.normal_texture_path != str(root / "textures" / "orange_n.png"):
            raise AssertionError("normal texture path was not resolved")

        subsets = list(info.subsets)
        if [subset.material_index for subset in subsets] != [0, 1]:
            raise AssertionError("subset material indices mismatch")
        if [subset.name for subset in subsets] != ["OrangePaint", "BluePaint"]:
            raise AssertionError("subset names mismatch")
        for subset in subsets:
            if len(subset.mesh_data.indices) != 3:
                raise AssertionError("each material subset should contain one triangle")

    print("PASS: OBJ MTL material loader smoke completed")


if __name__ == "__main__":
    main()
