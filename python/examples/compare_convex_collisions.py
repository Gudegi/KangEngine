"""Compare a PhysX single hull with a CoACD convex compound.

The input OBJ is loaded once with KangEngine. Its vertices are sent directly
to PhysX for the single-hull baseline and to CoACD for decomposition. Every
``run_coacd()`` result part becomes one KangEngine ``ConvexMeshPart`` and one
PhysX shape on a single rigid body. No intermediate OBJ files are written.

Example:
    python python/examples/compare_convex_collisions.py model.obj \
        --threshold 0.05 --seed 0
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import coacd
import kangengine as ke
import numpy as np
from kangengine import imgui, keys


def convex_part(vertices: Any, faces: Any) -> Any:
    return ke.physics.ConvexMeshPart(
        np.asarray(vertices, dtype=np.float32),
        np.asarray(faces, dtype=np.int64),
    )


@dataclass(frozen=True)
class TriangleMeshInput:
    mesh_data: Any
    vertices: np.ndarray
    faces: np.ndarray


def load_triangle_mesh(path: Path) -> TriangleMeshInput:
    mesh_data = ke.asset.load_obj(str(path))
    vertices = np.asarray(mesh_data.vertices, dtype=np.float32)
    indices = np.asarray(mesh_data.indices, dtype=np.int64)

    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise ValueError(f"KangEngine loaded no valid vertices from {path}")
    if indices.ndim != 1 or len(indices) == 0 or len(indices) % 3 != 0:
        raise ValueError(f"CoACD requires indexed triangles: {path}")

    faces = indices.reshape(-1, 3)
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError(f"OBJ contains an out-of-range vertex index: {path}")
    return TriangleMeshInput(mesh_data, vertices, faces)


def actor_summary(actor: Any) -> dict[str, Any]:
    return {
        "shape_count": actor.num_shapes(),
        "mass": actor.get_mass(),
        "position": actor.get_root_position().tolist(),
    }


class ConvexCollisionViewer(ke.App):
    def __init__(
        self,
        args: argparse.Namespace,
        input_mesh: TriangleMeshInput,
        physx_part: Any,
        coacd_parts: list[Any],
        decomposition_seconds: float,
    ):
        super().__init__()
        self.args = args
        self.input_mesh = input_mesh
        self.physx_part = physx_part
        self.coacd_parts = coacd_parts
        self.decomposition_seconds = decomposition_seconds
        self.show_source = True
        self.show_collisions = True
        self.reset_rng = np.random.default_rng()

    def setup(self):
        self.set_light_direction(ke.Vec3(-0.35, 0.82, -0.45))
        self.set_light_intensity(1.25)
        self.set_light_ambient(ke.Vec3(0.3, 0.3, 0.3))

        config = ke.physics.PhysicsConfig.y_up()
        self.physics = ke.physics.PhysicsWorld(config)
        self.timing = self.configure_timing(
            ke.SimulationTimingConfig.from_dt(
                physics_dt=config.dt,
                fixed_dt=config.dt,
                render_hz=60.0,
            )
        )
        self.set_simulation_hotkeys_enabled(True)
        self.physics.add_default_ground()

        extent = float(max(np.ptp(self.input_mesh.vertices, axis=0).max(), 1.0))
        ground_material = self.create_standard_materials().ground
        self.scene.add_ground(
            scale=max(extent * 6.0, self.args.spacing * 3.0), material=ground_material
        )

        cooking = ke.physics.ConvexCookingOptions()
        cooking.vertex_limit = self.args.vertex_limit
        self.initial_positions = {
            "physx": [-self.args.spacing * 0.5, self.args.height, 0.0],
            "coacd": [self.args.spacing * 0.5, self.args.height, 0.0],
        }
        self.collision_resources = {
            "physx": self.physics.create_convex_collision(
                [self.physx_part], cooking=cooking
            ),
            "coacd": self.physics.create_convex_collision(
                self.coacd_parts, cooking=cooking
            ),
        }
        self.actors = {
            "physx": self.physics.create_dynamic_from_collision(
                self.collision_resources["physx"],
                self.initial_positions["physx"],
                density=self.args.density,
            ),
            "coacd": self.physics.create_dynamic_from_collision(
                self.collision_resources["coacd"],
                self.initial_positions["coacd"],
                density=self.args.density,
            ),
        }

        self.collision_material = self.create_phong_material(
            ambient=ke.Vec3(0.12, 0.12, 0.12),
            diffuse=ke.Vec3(1.0, 1.0, 1.0),
            specular=ke.Vec3(0.08, 0.08, 0.08),
            shininess=18.0,
        )
        self.source_material = self.create_phong_material(
            ambient=ke.Vec3(0.08, 0.08, 0.08),
            diffuse=ke.Vec3(1.0, 1.0, 1.0),
            specular=ke.Vec3(0.0, 0.0, 0.0),
            shininess=1.0,
        )

        self.collision_views = {
            "physx": self._add_collision_views("physx", ke.Vec4(0.15, 0.5, 1.0, 1.0)),
            "coacd": self._add_coacd_collision_views(),
        }
        self.source_views = {
            name: self.scene.add_mesh(
                f"/{name}/source",
                self.input_mesh.mesh_data,
                self.source_material,
                color=ke.Vec4(0.9, 0.9, 0.9, 0.16),
            )
            for name in self.actors
        }
        for view in self.source_views.values():
            view.set_alpha_mode(ke.render.AlphaMode.BLEND)
            view.set_double_sided(True)
            view.set_casts_shadow(False)

        self._sync_visuals()
        self._setup_camera(extent)
        self._print_report()

    def _add_collision_views(self, name: str, color: Any) -> list[Any]:
        meshes = self.actors[name].get_convex_collision_meshes()
        views = []
        for index, mesh in enumerate(meshes):
            view = self.scene.add_mesh(
                f"/{name}/collision_{index}",
                mesh,
                self.collision_material,
                color=color,
            )
            views.append(view)
        return views

    def _add_coacd_collision_views(self) -> list[Any]:
        palette = (
            (0.95, 0.28, 0.16),
            (0.98, 0.62, 0.12),
            (0.28, 0.78, 0.35),
            (0.18, 0.72, 0.82),
            (0.58, 0.38, 0.92),
            (0.95, 0.32, 0.68),
        )
        meshes = self.actors["coacd"].get_convex_collision_meshes()
        views = []
        for index, mesh in enumerate(meshes):
            rgb = palette[index % len(palette)]
            view = self.scene.add_mesh(
                f"/coacd/collision_{index}",
                mesh,
                self.collision_material,
                color=ke.Vec4(*rgb, 1.0),
            )
            views.append(view)
        return views

    def _sync_visuals(self):
        for name, actor in self.actors.items():
            pos = actor.get_root_position()
            rot = actor.get_root_rotation()
            views = [*self.collision_views[name], self.source_views[name]]
            for view in views:
                view.prim.set_local_translation(
                    ke.Vec3(float(pos[0]), float(pos[1]), float(pos[2]))
                )
                view.prim.set_local_rotation(
                    ke.Quat(float(rot[3]), float(rot[0]), float(rot[1]), float(rot[2]))
                )

    def _reset(self):
        u1, u2, u3 = self.reset_rng.random(3)
        sqrt_u1 = np.sqrt(u1)
        sqrt_one_minus_u1 = np.sqrt(1.0 - u1)
        rotation_xyzw = [
            sqrt_one_minus_u1 * np.sin(2.0 * np.pi * u2),
            sqrt_one_minus_u1 * np.cos(2.0 * np.pi * u2),
            sqrt_u1 * np.sin(2.0 * np.pi * u3),
            sqrt_u1 * np.cos(2.0 * np.pi * u3),
        ]
        for name, actor in self.actors.items():
            actor.set_root_state(
                self.initial_positions[name],
                rotation_xyzw,
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            )
        self._sync_visuals()

    def _setup_camera(self, extent: float):
        center = (
            self.input_mesh.vertices.min(axis=0) + self.input_mesh.vertices.max(axis=0)
        ) * 0.5
        target = ke.Vec3(
            float(center[0]),
            float(center[1] + self.args.height),
            float(center[2]),
        )
        radius = max(extent, self.args.spacing, 1.0)
        camera = self.get_camera()
        camera.set_near_plane(max(radius * 0.002, 0.01))
        camera.set_far_plane(max(radius * 100.0, 100.0))
        camera.set_fov(55.0)
        camera.set_target_pos(target)
        camera.set_camera_pos(
            target + ke.Vec3(radius * 0.65, radius * 0.55, radius * 1.5)
        )
        self.set_camera_move_speed(max(radius * 0.3, 1.0))

    def _print_report(self):
        report = {
            "input": {
                "path": str(self.args.input_obj),
                "vertex_count": int(len(self.input_mesh.vertices)),
                "triangle_count": int(len(self.input_mesh.faces)),
            },
            "coacd": {
                "part_count": len(self.coacd_parts),
                "threshold": self.args.threshold,
                "max_convex_hull": self.args.max_convex_hull,
                "seed": self.args.seed,
                "decomposition_seconds": self.decomposition_seconds,
            },
            "actors": {
                name: actor_summary(actor) for name, actor in self.actors.items()
            },
            "collision_resources": {
                name: {
                    "part_count": resource.part_count,
                    "vertex_limit": resource.vertex_limit,
                    "gpu_compatible": resource.gpu_compatible,
                }
                for name, resource in self.collision_resources.items()
            },
        }
        print(json.dumps(report, indent=2))

    def pre_update(self):
        if self.was_key_pressed(keys.R):
            self._reset()

    def fixed_update(self, fixed_dt):
        self.physics.step()

    def pre_render(self):
        self._sync_visuals()

    def render(self):
        imgui.begin("Convex Collision Comparison")
        imgui.text("Left: PhysX single convex hull")
        imgui.text(f"Right: CoACD compound ({len(self.coacd_parts)} shapes)")
        imgui.separator()
        imgui.text(f"CoACD time: {self.decomposition_seconds:.3f} s")
        imgui.text(f"threshold: {self.args.threshold:g}    seed: {self.args.seed}")
        imgui.text(
            f"mass: PhysX={self.actors['physx'].get_mass():.5g}  "
            f"CoACD={self.actors['coacd'].get_mass():.5g}"
        )
        changed, self.show_source = imgui.checkbox(
            "Show source surface", self.show_source
        )
        if changed:
            for view in self.source_views.values():
                view.set_visible(self.show_source)
        changed, self.show_collisions = imgui.checkbox(
            "Show collision hulls", self.show_collisions
        )
        if changed:
            for views in self.collision_views.values():
                for view in views:
                    view.set_visible(self.show_collisions)
        paused = self.is_simulation_paused()
        changed, paused = imgui.checkbox("Pause simulation", paused)
        if changed:
            self.set_simulation_paused(paused)
        if imgui.button("Reset (R)"):
            self._reset()
        imgui.text("Enter: play/pause    Space: pause/step    R: reset")
        imgui.end()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_obj", type=Path)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--max-convex-hull", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--verbose-coacd", action="store_true", help="Print CoACD progress logs."
    )
    parser.add_argument(
        "--preprocess-mode", choices=("auto", "on", "off"), default="auto"
    )
    parser.add_argument("--real-metric", action="store_true")
    parser.add_argument("--height", type=float, default=3.0)
    parser.add_argument("--spacing", type=float, default=3.0)
    parser.add_argument("--density", type=float, default=1.0)
    parser.add_argument("--vertex-limit", type=int, default=255)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--window-height", type=int, default=900)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.input_obj = args.input_obj.expanduser().resolve()
    if not args.input_obj.is_file():
        raise FileNotFoundError(args.input_obj)

    if not args.verbose_coacd:
        coacd.set_log_level("error")

    input_mesh = load_triangle_mesh(args.input_obj)
    physx_part = convex_part(input_mesh.vertices, input_mesh.faces)

    coacd_input = coacd.Mesh(input_mesh.vertices, input_mesh.faces)
    start = perf_counter()
    coacd_result = coacd.run_coacd(
        coacd_input,
        threshold=args.threshold,
        max_convex_hull=args.max_convex_hull,
        preprocess_mode=args.preprocess_mode,
        seed=args.seed,
        real_metric=args.real_metric,
    )
    decomposition_seconds = perf_counter() - start
    coacd_parts = [convex_part(vertices, faces) for vertices, faces in coacd_result]
    if not coacd_parts:
        raise RuntimeError("CoACD returned no convex parts")

    app = ConvexCollisionViewer(
        args,
        input_mesh,
        physx_part,
        coacd_parts,
        decomposition_seconds,
    )
    app.initialize(width=args.width, height=args.window_height)
    app.start()


if __name__ == "__main__":
    main()
