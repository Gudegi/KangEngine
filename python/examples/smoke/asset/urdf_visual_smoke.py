"""Load and render the packaged G1 URDF through the native visual bridge."""

from __future__ import annotations

from pathlib import Path

import kangengine as ke


class URDFVisualSmoke(ke.App):
    def __init__(self):
        super().__init__()
        self.frames = 0

    def setup(self):
        urdf_path = (
            Path(ke.__file__).resolve().parent
            / "assets"
            / "characters"
            / "g1"
            / "g1_29dof.urdf"
        )
        materials = self.create_standard_materials()
        self.robot = ke.visual.ArticulationVisual.from_urdf(
            urdf_path=str(urdf_path),
            scene=self.scene.native,
            path="/g1",
            scale=1.0,
            order="DFS",
        )
        for prim in self.robot.render_prims():
            if prim.resolve_mesh_data() is not None:
                self.scene.add_renderable(prim, materials.common)

        if self.robot.num_bodies() != 41:
            raise AssertionError(
                f"expected 41 G1 bodies, got {self.robot.num_bodies()}"
            )
        if not self.robot.render_prims():
            raise AssertionError("URDF visual bridge created no render prims")

    def pre_render(self):
        self.frames += 1
        if self.frames >= 2:
            self.request_close()


def main():
    app = URDFVisualSmoke()
    app.initialize(640, 480, True, ke.UpAxis.Z, headless=True)
    app.start()
    print(
        "PASS: native URDF G1 visualization "
        f"({app.robot.num_bodies()} bodies, {len(app.robot.render_prims())} render prims)"
    )


if __name__ == "__main__":
    main()
