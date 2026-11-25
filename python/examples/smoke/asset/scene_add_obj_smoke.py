"""Validate SceneContext.add_obj material-subset scene creation."""

from __future__ import annotations

import tempfile
from pathlib import Path

import kangengine as ke


def main():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        obj = root / "robot.obj"
        mtl = root / "robot.mtl"

        mtl.write_text(
            "\n".join(
                [
                    "newmtl Body",
                    "Ka 0.1 0.1 0.1",
                    "Kd 0.7 0.7 0.7",
                    "Ks 0.1 0.1 0.1",
                    "Ns 8",
                    "",
                    "newmtl Foot",
                    "Ka 0.05 0.05 0.05",
                    "Kd 0.4 0.4 0.4",
                    "Ks 0.05 0.05 0.05",
                    "Ns 4",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        obj.write_text(
            "\n".join(
                [
                    "mtllib robot.mtl",
                    "v -0.5 -0.5 0.0",
                    "v 0.5 -0.5 0.0",
                    "v 0.5 0.5 0.0",
                    "v -0.5 0.5 0.0",
                    "vt 0.0 0.0",
                    "vt 1.0 0.0",
                    "vt 1.0 1.0",
                    "vt 0.0 1.0",
                    "vn 0.0 0.0 1.0",
                    "usemtl Body",
                    "f 1/1/1 2/2/1 3/3/1",
                    "usemtl Foot",
                    "f 1/1/1 3/3/1 4/4/1",
                    "",
                ]
            ),
            encoding="utf-8",
        )

        app = ke.App()
        app.initialize(width=64, height=64, hide_ui=True, headless=True)
        result = app.scene.add_obj("/Robot", obj)

        if result.root.get_path() != "/Robot":
            raise AssertionError("add_obj root path mismatch")
        if len(result.views) != 2:
            raise AssertionError("add_obj should create one view per material subset")
        if result.info.subset_count != 2:
            raise AssertionError("OBJ result should expose two subsets")

        child_paths = [view.prim.get_path() for view in result.views]
        if child_paths != ["/Robot/Body", "/Robot/Foot"]:
            raise AssertionError(f"unexpected subset child paths: {child_paths}")

        for view in result.views:
            mesh_component = view.prim.get_mesh_component()
            if mesh_component is None:
                raise AssertionError("subset prim missing MeshComponent")
            if mesh_component.resource_handle == ke.scene.InvalidResourceHandle:
                raise AssertionError("subset mesh was not registered as resource")
            resource_prim = app.resources.resource_prim(mesh_component.resource_handle)
            if resource_prim is None:
                raise AssertionError("subset mesh resource prim missing")
            if not resource_prim.get_path().startswith("/.Resources/Meshes/"):
                raise AssertionError("subset mesh resource path mismatch")
            if view.prim.get_material() is None:
                raise AssertionError("subset prim missing auto material")

    print("PASS: SceneContext.add_obj smoke completed")


if __name__ == "__main__":
    main()
