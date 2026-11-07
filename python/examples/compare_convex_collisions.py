"""Compare a PhysX single hull with a saved or newly decomposed convex compound.

The input OBJ is loaded once with KangEngine. Its vertices are sent directly
to PhysX for the single-hull baseline and to the selected decomposition backend.
Each result part becomes one KangEngine ``ConvexMeshPart`` and one PhysX shape.
Existing NPZ results are reused; use --rebuild after changing input or settings.
In Force mode, Shift + left-drag either object to pull at the picked point.

Example:
    python python/examples/compare_convex_collisions.py model.obj \
        --threshold 0.05 --seed 0
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import kangengine as ke
import numpy as np
from kangengine import imgui, keys
from kangengine.utils.math import quat_xyzw_conjugate, quat_xyzw_rotate

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import convex_decomposition as decomposition


DRAG_STIFFNESS = 40.0  # Per-mass spring gain, s^-2.
DRAG_DAMPING = 12.0  # Per-mass point-velocity damping, s^-1.
DRAG_MAX_ACCELERATION = 100.0  # m/s^2.


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
        raise ValueError(f"Convex decomposition requires indexed triangles: {path}")

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
        parts: list[Any],
        metadata: dict[str, Any],
    ):
        super().__init__()
        self.args = args
        self.input_mesh = input_mesh
        self.physx_part = physx_part
        self.parts = parts
        self.metadata = metadata
        self.show_source = True
        self.show_collisions = True
        self.show_contacts = True
        self.show_com = True
        magenta = ke.ColorLibrary.get(ke.ColorType.MAGENTA)
        self.com_color = np.array(
            [[magenta.r, magenta.g, magenta.b, magenta.a]], dtype=np.float32
        )
        self.contacts = {"physx": [], "decomposition": []}
        self.contact_colors = {}
        for name, preset in (
            ("physx", ke.ColorType.YELLOW),
            ("decomposition", ke.ColorType.CYAN),
        ):
            color = ke.ColorLibrary.get(preset)
            self.contact_colors[name] = np.array(
                [[color.r, color.g, color.b, color.a]], dtype=np.float32
            )
        self.reset_rng = np.random.default_rng()
        self._drag_actor = None
        self._drag_local_anchor = None
        self._drag_target = None
        red = ke.ColorLibrary.get(ke.ColorType.RED)
        self.drag_color = np.array([[red.r, red.g, red.b, red.a]], dtype=np.float32)

    def setup(self):
        self.set_light_direction(ke.Vec3(0.35, -0.82, 0.45))
        self.set_light_intensity(1.25)
        self.set_light_ambient(ke.Vec3(0.3, 0.3, 0.3))

        config = ke.physics.PhysicsConfig.y_up()
        config.enable_contact_reports = True
        self.physics_dt = config.dt
        self.physics = ke.physics.PhysicsWorld(config)
        self.timing = self.configure_timing(
            ke.SimulationTimingConfig.from_dt(
                physics_dt=config.dt,
                fixed_dt=config.dt,
                render_hz=60.0,
            )
        )
        self.set_simulation_hotkeys_enabled(True)
        self.set_interaction_mode(ke.InteractionMode.FORCE)
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
            "decomposition": [self.args.spacing * 0.5, self.args.height, 0.0],
        }
        self.actors = {
            "physx": self.physics.create_dynamic_convex_compound(
                [self.physx_part],
                self.initial_positions["physx"],
                density=self.args.density,
                cooking=cooking,
            ),
            "decomposition": self.physics.create_dynamic_convex_compound(
                self.parts,
                self.initial_positions["decomposition"],
                density=self.args.density,
                cooking=cooking,
            ),
        }
        # Uncomment to give both colliders the single hull's mass properties.
        # Both geometries must use the same body-local coordinate frame.
        # reference = self.actors["physx"]
        # target = self.actors["decomposition"]
        # target.set_mass(mass=reference.get_mass())
        # target.set_mass_space_inertia(inertia=reference.get_mass_space_inertia())
        # target.set_local_com_pose(pose=reference.get_local_com_pose())

        # Use one meters-per-newton scale for both colliders, sized to this asset.
        self.default_force_scale = (
            2.0
            * extent
            / (max(actor.get_mass() for actor in self.actors.values()) * 9.81)
        )
        self.contact_force_scale = self.default_force_scale

        self.collision_material = self.create_pbr_material(
            base_color=ke.Vec4(1.0, 1.0, 1.0, 1.0),
            metallic=0.0,
            roughness=0.55,
        )
        self.source_material = self.create_pbr_material(
            base_color=ke.Vec4(1.0, 1.0, 1.0, 1.0),
            metallic=0.0,
            roughness=0.85,
        )

        self.collision_views = {
            "physx": self._add_collision_views("physx", ke.Vec4(0.15, 0.5, 1.0, 1.0)),
            "decomposition": self._add_decomposition_collision_views(),
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
        self._pick_actors = {
            view.prim.get_path(): self.actors[name]
            for name in self.actors
            for view in [*self.collision_views[name], self.source_views[name]]
        }

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

    def _add_decomposition_collision_views(self) -> list[Any]:
        palette = (
            (0.95, 0.28, 0.16),
            (0.98, 0.62, 0.12),
            (0.28, 0.78, 0.35),
            (0.18, 0.72, 0.82),
            (0.58, 0.38, 0.92),
            (0.95, 0.32, 0.68),
        )
        meshes = self.actors["decomposition"].get_convex_collision_meshes()
        views = []
        for index, mesh in enumerate(meshes):
            rgb = palette[index % len(palette)]
            view = self.scene.add_mesh(
                f"/decomposition/collision_{index}",
                mesh,
                self.collision_material,
                color=ke.Vec4(*rgb, 1.0),
            )
            views.append(view)
        return views

    def _sync_visuals(self):
        for name, actor in self.actors.items():
            pos = actor.get_root_position()
            rot = actor.get_root_quat_xyzw()
            views = [*self.collision_views[name], self.source_views[name]]
            for view in views:
                view.prim.set_local_translation(
                    ke.Vec3(float(pos[0]), float(pos[1]), float(pos[2]))
                )
                view.prim.set_local_quat_wxyz(
                    ke.Quat(float(rot[3]), float(rot[0]), float(rot[1]), float(rot[2]))
                )

    def _update_centers_of_mass(self):
        if not self.show_com:
            self.debug.overlay.clear_points("/debug/centers_of_mass")
            return
        # PhysX stores COM relative to the actor, not in world coordinates.
        points = np.array([
            actor.get_root_position() + quat_xyzw_rotate(
                actor.get_root_quat_xyzw(), actor.get_local_com_position()
            )
            for actor in self.actors.values()
        ], dtype=np.float32)
        self.debug.overlay.points(
            "/debug/centers_of_mass", points, self.com_color, size=14.0, overlay=True
        )

    def _reset(self):
        self.on_force_drag_end()
        u1, u2, u3 = self.reset_rng.random(3)
        sqrt_u1 = np.sqrt(u1)
        sqrt_one_minus_u1 = np.sqrt(1.0 - u1)
        quat_xyzw = [
            sqrt_one_minus_u1 * np.sin(2.0 * np.pi * u2),
            sqrt_one_minus_u1 * np.cos(2.0 * np.pi * u2),
            sqrt_u1 * np.sin(2.0 * np.pi * u3),
            sqrt_u1 * np.cos(2.0 * np.pi * u3),
        ]
        for name, actor in self.actors.items():
            actor.set_root_state(
                self.initial_positions[name],
                quat_xyzw,
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            )
        self.physics.clear_contacts()
        for name in self.actors:
            self.contacts[name] = []
            self.debug.overlay.clear_points(f"/debug/contacts/{name}")
            self.debug.overlay.clear_lines(f"/debug/contact_forces/{name}")
        self._sync_visuals()

    def on_force_drag_begin(self, result, target):
        self.on_force_drag_end()
        if not result.hit or result.prim is None:
            return
        self._drag_actor = self._pick_actors.get(result.prim.get_path())
        if self._drag_actor is None:
            return
        hit = np.array([result.position.x, result.position.y, result.position.z])
        self._drag_local_anchor = quat_xyzw_rotate(
            quat_xyzw_conjugate(self._drag_actor.get_root_quat_xyzw()),
            hit - self._drag_actor.get_root_position(),
        )
        self.on_force_drag_update(result, target)

    def on_force_drag_update(self, result, target):
        if self._drag_actor is not None:
            self._drag_target = np.array(
                [target.x, target.y, target.z], dtype=np.float32
            )

    def on_force_drag_end(self):
        self._drag_actor = None
        self._drag_local_anchor = None
        self._drag_target = None
        self.debug.overlay.clear_lines("/debug/mouse_force")
        self.debug.overlay.clear_points("/debug/mouse_target")

    def _drag_anchor_world(self):
        actor = self._drag_actor
        return actor.get_root_position() + quat_xyzw_rotate(
            actor.get_root_quat_xyzw(), self._drag_local_anchor
        )

    def _apply_mouse_force(self):
        if self._drag_actor is None:
            return
        if self.get_interaction_mode() != ke.InteractionMode.FORCE:
            self.on_force_drag_end()
            return
        actor = self._drag_actor
        anchor = self._drag_anchor_world()
        com = actor.get_root_position() + quat_xyzw_rotate(
            actor.get_root_quat_xyzw(), actor.get_local_com_position()
        )
        # PhysX linear velocity is at the COM, including for asymmetric compounds.
        velocity = actor.get_root_linear_velocity() + np.cross(
            actor.get_root_angular_velocity(), anchor - com
        )
        acceleration = (
            DRAG_STIFFNESS * (self._drag_target - anchor) - DRAG_DAMPING * velocity
        )
        magnitude = np.linalg.norm(acceleration)
        if magnitude > DRAG_MAX_ACCELERATION:
            acceleration *= DRAG_MAX_ACCELERATION / magnitude
        actor.add_force_at_position(
            force=actor.get_mass() * acceleration, position=anchor
        )

    def _draw_mouse_force(self):
        if self._drag_actor is None:
            return
        anchor = self._drag_anchor_world().reshape(1, 3)
        target = self._drag_target.reshape(1, 3)
        self.debug.overlay.lines(
            "/debug/mouse_force",
            anchor,
            target,
            self.drag_color,
            width=2.0,
            overlay=True,
        )
        self.debug.overlay.points(
            "/debug/mouse_target", target, self.drag_color, size=8.0, overlay=True
        )

    def _update_contact_forces(self):
        for name, contacts in self.contacts.items():
            point_path = f"/debug/contacts/{name}"
            force_path = f"/debug/contact_forces/{name}"
            if not self.show_contacts or not contacts:
                self.debug.overlay.clear_points(point_path)
                self.debug.overlay.clear_lines(force_path)
                continue

            points = np.array(
                [[c.position.x, c.position.y, c.position.z] for c in contacts],
                dtype=np.float32,
            )
            forces = (
                np.array(
                    [[c.impulse.x, c.impulse.y, c.impulse.z] for c in contacts],
                    dtype=np.float32,
                )
                / self.physics_dt
            )
            # Draw over scene depth so forces inside a collider remain visible.
            color = self.contact_colors[name]
            self.debug.overlay.points(point_path, points, color, size=6.0, overlay=True)
            self.debug.overlay.arrows(
                force_path, points, points + forces * self.contact_force_scale,
                color, width=3.0, overlay=True,
            )

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
        self.set_camera_move_speed(50.0)

    def _print_report(self):
        report = {
            "input": {
                "path": str(self.args.input_obj),
                "vertex_count": int(len(self.input_mesh.vertices)),
                "triangle_count": int(len(self.input_mesh.faces)),
            },
            "decomposition": self.metadata,
            "result_path": str(self.args.result),
            "actors": {
                name: actor_summary(actor) for name, actor in self.actors.items()
            },
        }
        print(json.dumps(report, indent=2))

    def pre_update(self):
        if self.was_key_pressed(keys.R):
            self._reset()

    def fixed_update(self, fixed_dt):
        self._apply_mouse_force()
        self.physics.step()
        for name, actor in self.actors.items():
            contacts = self.physics.get_contacts(rigid=actor)
            # Sleeping pairs stop reporting; preserve their last measured sample.
            if contacts or not actor.is_sleeping():
                self.contacts[name] = contacts

    def pre_render(self):
        self._sync_visuals()
        self._update_centers_of_mass()
        self._update_contact_forces()
        self._draw_mouse_force()

    def render(self):
        imgui.begin("Convex Collision Comparison")
        imgui.text("Left: PhysX single convex hull")
        backend = self.metadata["backend"]
        imgui.text(f"Right: {backend} compound ({len(self.parts)} shapes)")
        imgui.separator()
        source = "Saved result" if self.metadata["cache_hit"] else "New result"
        imgui.text(
            f"{source}; original decomposition: "
            f"{self.metadata['decomposition_seconds']:.3f} s"
        )
        imgui.text(json.dumps(self.metadata["options"], sort_keys=True))
        imgui.text(
            f"mass: PhysX={self.actors['physx'].get_mass():.5g}  "
            f"{backend}={self.actors['decomposition'].get_mass():.5g}"
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
        changed, self.show_com = imgui.checkbox("Show CoM (magenta)", self.show_com)
        if changed:
            self._update_centers_of_mass()
        changed, self.show_contacts = imgui.checkbox(
            "Show contact forces", self.show_contacts
        )
        if changed:
            self._update_contact_forces()
        changed, self.contact_force_scale = imgui.slider_float(
            "Force scale (m/N)",
            self.contact_force_scale,
            0.0,
            self.default_force_scale * 10.0,
        )
        if changed:
            self._update_contact_forces()
        for name, label in (
            ("physx", "PhysX / yellow"),
            ("decomposition", f"{backend} / cyan"),
        ):
            status = (
                " (sleeping: last sample)" if self.actors[name].is_sleeping() else ""
            )
            imgui.text(f"{label}: {len(self.contacts[name])} contacts{status}")
        paused = self.is_simulation_paused()
        changed, paused = imgui.checkbox("Pause simulation", paused)
        if changed:
            self.set_simulation_paused(paused)
        if imgui.button("Reset (R)"):
            self._reset()
        imgui.text("Enter: play/pause    Space: pause/step    R: reset")
        imgui.text("Force mode: Shift + left-drag an object (red tether)")
        imgui.end()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_obj", type=Path)
    parser.add_argument("--backend", default="coacd")
    parser.add_argument(
        "--backend-options",
        type=json.loads,
        default={},
        metavar="JSON",
        help="Algorithm keyword arguments; override CoACD flags when provided.",
    )
    parser.add_argument(
        "--result",
        type=Path,
        help="Existing result to load, or NPZ destination for a new result.",
    )
    parser.add_argument("--rebuild", action="store_true")
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

    input_mesh = load_triangle_mesh(args.input_obj)
    physx_part = convex_part(input_mesh.vertices, input_mesh.faces)

    args.backend = args.backend.strip().lower()
    if args.result is None:
        args.result = Path("build/convex") / args.input_obj.stem / f"{args.backend}.npz"
    args.result = args.result.expanduser().resolve()
    options = {}
    if args.backend == "coacd":
        options = {
            "threshold": args.threshold,
            "max_convex_hull": args.max_convex_hull,
            "preprocess_mode": args.preprocess_mode,
            "seed": args.seed,
            "real_metric": args.real_metric,
        }
        if args.rebuild or not args.result.is_file():
            import coacd

            coacd.set_log_level("info" if args.verbose_coacd else "error")
    options.update(args.backend_options)
    parts, metadata = decomposition.decompose_mesh(
        input_mesh.vertices,
        input_mesh.faces,
        backend=args.backend,
        options=options,
        result_path=args.result,
        rebuild=args.rebuild,
    )
    native_parts = [convex_part(part.vertices, part.faces) for part in parts]

    app = ConvexCollisionViewer(
        args,
        input_mesh,
        physx_part,
        native_parts,
        metadata,
    )
    app.initialize(width=args.width, height=args.window_height)
    app.start()


if __name__ == "__main__":
    main()
