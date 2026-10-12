"""Shared CLI, benchmark, and viewer for Motion Matching examples."""

from __future__ import annotations

import argparse
from collections.abc import Callable
import math
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np

import kangengine as ke
from kangengine import animation, imgui, keys, visual
from kangengine.animation.matching import (
    make_walking_query,
    predict_walking_trajectory,
)


def parse_args(
    argv: list[str] | None = None,
    *,
    default_dataset: Path,
    description: str | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--dataset", type=Path, default=default_dataset)
    parser.add_argument(
        "--benchmark-search",
        action="store_true",
        help="Compare Python/native search correctness and timing, then exit without opening a viewer",
    )
    args = parser.parse_args(argv)
    args.dataset = args.dataset.expanduser().resolve()
    return args


def benchmark_search(database: animation.MotionMatchingDatabase) -> None:
    """Compare warmed Python/native search correctness and timing."""
    rng = np.random.default_rng(83)
    rows = rng.integers(database.num_frames, size=128)
    queries = database.features[rows] + rng.normal(
        scale=0.02, size=(len(rows), database.features.shape[1])
    )
    results = {
        backend: [database.search(query, backend=backend) for query in queries]
        for backend in ("python", "native")
    }
    for reference, native in zip(results["python"], results["native"]):
        if (reference.motion_index, reference.frame_index) != (
            native.motion_index,
            native.frame_index,
        ):
            raise RuntimeError("Python/native search selected different frames")
        if not math.isclose(reference.cost, native.cost, rel_tol=1e-10, abs_tol=1e-10):
            raise RuntimeError("Python/native search costs differ")
    timings = {"python": [], "native": []}
    for repeat in range(7):
        # Alternate order to reduce timing bias.
        for backend in (
            ("python", "native") if repeat % 2 == 0 else ("native", "python")
        ):
            start = perf_counter()
            for query in queries:
                database.search(query, backend=backend)
            timings[backend].append((perf_counter() - start) * 1e6 / len(queries))
    print(
        f"Search benchmark: {database.num_frames} candidates x "
        f"{database.features.shape[1]} features; 128 queries x 7 rounds, warmed CPU"
    )
    print("Parity: all 128 selected frames and costs agree")
    for backend, values in timings.items():
        print(
            f"  {backend}: median {np.median(values):.1f} us/query "
            f"(round range {min(values):.1f}-{max(values):.1f})"
        )
    print(
        f"  Native speedup: {np.median(timings['python']) / np.median(timings['native']):.2f}x"
    )


def target_direction(
    config: animation.MotionMatchingConfig, angle: float
) -> np.ndarray:
    up = np.eye(3, dtype=np.float32)["XYZ".index(config.up_axis)]
    forward = np.eye(3, dtype=np.float32)["XYZ".index(config.forward_axis)]
    return math.cos(angle) * forward + math.sin(angle) * np.cross(up, forward)


def follow_character(app):
    """Follow translation while preserving camera orbit and zoom."""
    root = app.controller.root_position
    if app.follow_camera:
        delta = root - app._camera_follow_root
        camera = app.get_camera()
        app.set_camera_view(
            np.asarray(camera.get_camera_pos()) + delta,
            np.asarray(camera.get_target_pos()) + delta,
        )
    app._camera_follow_root = root


def input_source_ui(source):
    sources = ("auto", "keyboard", "gamepad")
    if imgui.button(f"Input: {source} (switch)"):
        return sources[(sources.index(source) + 1) % len(sources)]
    return source


def draw_trajectory(app, points, directions, look_direction):
    """Orange trajectory and cyan input direction in world space."""
    up = "XYZ".index(app.controller.database.config.up_axis)
    points = points.copy()
    points[:, up] = 0.06
    orange = ke.preset_rgba(ke.ColorType.ORANGE)
    app.debug_overlay.lines(
        path="/motion_matching/desired_trajectory",
        starts=points[:-1],
        ends=points[1:],
        colors=np.tile(orange, (len(points) - 1, 1)),
        width=3.0,
        hidden=not app.show_trajectory,
    )
    starts, ends, colors = [], [], []
    if app.show_look:
        origin = app.controller.root_position.copy()
        origin[up] = 0.1
        starts.append(origin)
        ends.append(origin + np.asarray(look_direction) * 0.8)
        colors.append(ke.preset_rgba(ke.ColorType.CYAN))
    if app.show_trajectory:
        starts.extend(points[1:])
        ends.extend(points[1:] + directions[1:] * 0.25)
        colors.extend([orange] * (len(points) - 1))
    starts = np.asarray(starts, dtype=np.float32).reshape(-1, 3)
    ends = np.asarray(ends, dtype=np.float32).reshape(-1, 3)
    colors = np.asarray(colors, dtype=np.float32).reshape(-1, 4)
    arrows = getattr(app, "_look_arrows", None)
    if arrows is None:
        if len(starts):
            app._look_arrows = app.scene.debug_geometry.add_arrows(
                path="/motion_matching/look_arrows",
                starts=starts,
                ends=ends,
                colors=colors,
                radius=0.012,
            )
    else:
        arrows.update_arrows(starts=starts, ends=ends, colors=colors)


def control_ui(app, *, autoplay=True):
    imgui.text("WASD / left stick: move | Shift / RB: run | right stick: face")
    imgui.text("Space: pause | R: reset | Mouse: camera")
    app.input_source = input_source_ui(app.input_source)
    if autoplay:
        changed, app.autoplay = imgui.checkbox("Autoplay (24 s loop)", app.autoplay)
        if changed and app.autoplay:
            app.manual_goals = False
    changed, app.manual_goals = imgui.checkbox(
        "Use speed/heading sliders", app.manual_goals
    )
    if changed and app.manual_goals:
        app.autoplay = False
    _, app.paused = imgui.checkbox("Paused", app.paused)
    _, app.face_movement = imgui.checkbox(
        "Face movement (off: strafe/backwards)", app.face_movement
    )
    if app.manual_goals:
        _, app.speed = imgui.slider_float("Desired speed (m/s)", app.speed, 0.0, 5.0)
        _, app.heading_degrees = imgui.slider_float(
            "Desired heading (deg)", app.heading_degrees, -180.0, 180.0
        )
    else:
        _, app.walk_speed = imgui.slider_float(
            "Walk speed (m/s)", app.walk_speed, 0.0, 3.0
        )
        _, app.run_speed = imgui.slider_float(
            "Run speed (m/s)", app.run_speed, 0.0, 5.0
        )
    changed, half_life = imgui.slider_float(
        "Transition half-life (s)", app.controller.transition_half_life, 0.0, 0.5
    )
    if changed:
        app.controller.transition_half_life = half_life
    imgui.text("Set half-life to 0 to compare unsmoothed transitions")
    _, app.trajectory_half_life = imgui.slider_float(
        "Trajectory half-life (s)", app.trajectory_half_life, 0.0, 0.5
    )
    _, app.follow_camera = imgui.checkbox("Follow camera", app.follow_camera)
    _, app.show_trajectory = imgui.checkbox(
        "Show desired trajectory", app.show_trajectory
    )
    _, app.show_look = imgui.checkbox("Show look arrow", app.show_look)
    imgui.text("Cyan: input look | Orange: desired path and future facing")
    _, app.show_keyboard = imgui.checkbox("Show keyboard input", app.show_keyboard)
    if app.show_keyboard:
        if not hasattr(app, "keyboard_visualizer"):
            app.keyboard_visualizer = ke.input.KeyboardVisualizer(app)
        app.keyboard_visualizer.draw()
    if imgui.button("Reset"):
        app.reset()


class MotionMatchingViewer(ke.App):
    def __init__(
        self,
        database: animation.MotionMatchingDatabase,
        *,
        transition_half_life: float,
        walk_speed: float = 1.4,
        run_speed: float = 3.5,
        demo_input: Callable[[float], tuple[np.ndarray, float, str]] | None = None,
        trajectory_half_life: float = 0.2,
        search_backend: str = "native",
    ):
        super().__init__()
        self.database = database
        self.controller: animation.MotionMatchingController = (
            animation.MotionMatchingController(
                database,
                transition_half_life=transition_half_life,
                search_backend=search_backend,
            )
        )
        self.speed = 0.0
        self.trajectory_half_life = trajectory_half_life
        self.heading_degrees = 0.0
        self.paused = False
        self.show_trajectory = True
        self.show_look = True
        self.show_keyboard = True
        self.follow_camera = True
        self.camera_keyboard_enabled = False
        self.controls = ke.input.LocomotionInput(
            self,
            up_axis=database.config.up_axis,
            forward_axis=database.config.forward_axis,
        )
        self.input_source = "auto"
        self.manual_goals = False
        self.face_movement = True
        self.autoplay = False
        self._demo_input = demo_input
        self.input_label = "Idle"
        self.walk_speed = walk_speed
        self.run_speed = run_speed
        self.velocity = np.zeros(3)

    def setup(self):
        materials: SimpleNamespace = self.create_standard_materials()
        self.add_ground(material=materials.ground)
        config: visual.SkeletalVisualConfig = visual.SkeletalVisualConfig(
            bone_radius=0.015, joint_radius=0.025
        )
        self.character_visual: visual.SkeletalVisual = visual.SkeletalVisual.define(
            app=self,
            material=materials.common,
            path="/motion_matching",
            state=self.controller.state,
            config=config,
        )
        self.update_camera()

    def apply_pose(self):
        self.character_visual.apply_state(state=self.controller.state)

    def update_camera(self):
        up = np.eye(3)["XYZ".index(self.database.config.up_axis)]
        forward = np.eye(3)["XYZ".index(self.database.config.forward_axis)]
        root = self.controller.root_position
        # Place the camera behind the configured forward axis.
        self.set_camera_view(root + 2 * up - 6 * forward, root + 0.35 * up)
        self._camera_follow_root = root

    def reset(self):
        self.controller.reset()
        self.speed = 0.0
        self.heading_degrees = 0.0
        self.velocity[:] = 0
        self.apply_pose()
        self.update_camera()

    def update_input(self):
        if not self.keyboard_input_captured:
            if self.input.keyboard.pressed(keys.SPACE):
                self.paused = not self.paused
            if self.input.keyboard.pressed(keys.R):
                self.reset()
        if self.autoplay and self._demo_input is not None:
            self.velocity, angle, self.input_label = self._demo_input(
                self.controller.elapsed_time
            )
            self.speed = float(np.linalg.norm(self.velocity))
            self.heading_degrees = math.degrees(angle)
            return
        config = self.database.config
        if self.manual_goals:
            self.input_label = "Sliders"
            self.velocity = self.speed * target_direction(
                config, math.radians(self.heading_degrees)
            )
            return
        self.velocity, facing = self.controls.sample(
            facing=target_direction(config, math.radians(self.heading_degrees)),
            walk_speed=self.walk_speed,
            run_speed=self.run_speed,
            source=self.input_source,
            face_movement=self.face_movement,
        )
        self.speed = float(np.linalg.norm(self.velocity))
        self.input_label = "Move" if self.speed > 0 else "Idle"
        forward = np.eye(3)["XYZ".index(config.forward_axis)]
        self.heading_degrees = math.degrees(
            math.atan2(
                np.cross(forward, facing)["XYZ".index(config.up_axis)],
                np.dot(forward, facing),
            )
        )

    def pre_render(self):
        self.update_input()
        angle = math.radians(self.heading_degrees)
        if not self.paused:
            self.controller.step_with_query(
                dt=min(self.get_delta_time(), 0.1),
                query_provider=lambda controller: make_walking_query(
                    controller,
                    desired_velocity=self.velocity,
                    facing_angle=angle,
                    half_life=self.trajectory_half_life,
                ),
            )
            self.apply_pose()
        follow_character(self)
        points, directions = predict_walking_trajectory(
            self.controller,
            desired_velocity=self.velocity,
            facing_angle=angle,
            half_life=self.trajectory_half_life,
        )
        draw_trajectory(
            self, points, directions, target_direction(self.database.config, angle)
        )

    def render(self):
        imgui.begin("Motion Matching")
        control_ui(self, autoplay=self._demo_input is not None)
        motion = self.database.motions[self.controller.motion_index]
        imgui.text(f"Clip: {motion.motion_name()}")
        imgui.text(f"Time: {self.controller.time:.3f} s")
        imgui.text(f"Transitions: {self.controller.transition_count}")
        if self.controller.last_match is not None:
            imgui.text(f"Search cost: {self.controller.last_match.cost:.4f}")
        imgui.end()
