"""Static FBX mesh viewer with skeleton overlay."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import torch

import kangengine as ke
from kangengine import animation, asset, imgui, keys, scene


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def package_asset_path(*parts: str) -> str:
    return str(Path(ke.__file__).resolve().parent / "assets" / Path(*parts))


def default_fbx_file() -> Path:
    return repo_root() / "assets" / "external" / "Capoeira.fbx"
    #return repo_root() / "assets" / "external" / "Dying.fbx"


def prim_safe_name(name: str, fallback: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_")
    return clean or fallback


class FbxCharacterViewer(ke.App):
    def __init__(
        self,
        fbx_file: Path,
        clip_index: int,
        fps: float,
        scale: float,
        line_radius: float,
    ):
        super().__init__()
        self.fbx_file = str(fbx_file)
        self.clip_index = clip_index
        self.fps = fps
        self.scale = scale
        self.line_radius = line_radius

    def setup(self):
        self.show_mesh = True
        self.show_skeleton = True
        self.cast_shadows = True
        self.animate = True
        self.playback_speed = 1.0
        self.time = 0.0
        self.line_handle = None
        self.mesh_handles = []
        self.mesh_prims = []
        self.mesh_colors = []
        self.textures = []
        self.skeleton_starts = None
        self.skeleton_ends = None
        self.skeleton_colors = None
        self.skinned_meshes = []

        device = self.get_renderer().device()
        vs = package_asset_path("shaders", "common.vs")
        skinned_vs = package_asset_path("shaders", "skinned_mesh.vs")
        fs = package_asset_path("shaders", "common.fs")
        tex_fs = package_asset_path("shaders", "commonTex.fs")
        checker_fs = package_asset_path("shaders", "checkerboard.fs")

        self.mesh_shader = device.create_shader_from_file(skinned_vs, fs)
        self.textured_mesh_shader = device.create_shader_from_file(skinned_vs, tex_fs)
        self.skeleton_shader = device.create_shader_from_file(vs, fs)
        self.ground_shader = device.create_shader_from_file(vs, checker_fs)

        for shader in (
            self.mesh_shader,
            self.textured_mesh_shader,
            self.skeleton_shader,
            self.ground_shader,
        ):
            shader.use()
            shader.set_uniform_block_binding("cameraUBO", 0)
            shader.set_uniform_block_binding("lightUBO", 1)
            shader.set_uniform_block_binding("shadowUBO", 2)
        self.textured_mesh_shader.use()
        self.textured_mesh_shader.set_int("uTexture", 0)

        self.ground_shader.use()
        self.ground_shader.set_vec4("checkerColor1", ke.vec4(1.0, 1.0, 1.0, 1.0))
        self.ground_shader.set_vec4("checkerColor2", ke.vec4(0.77, 0.93, 0.78, 1.0))

        ground = self.get_scene().define_prim("/ground", scene.PrimType.Mesh)
        ground.set_mesh_data(scene.Prim.create_plane_data(20.0, self.up_axis))
        self.add_renderable(self.ground_shader, ground)

        camera = self.get_camera()
        camera.set_camera_pos(ke.vec3(0.0, 1.45, 3.2))
        camera.set_target_pos(ke.vec3(0.0, 0.85, 0.0))

        self.clips = asset.FBXLoader.load_animation_clip_infos(self.fbx_file)
        if self.clip_index == -1 and self.fps <= 0.0 and self.scale == 0.01:
            self.load_call = "FBXLoader.load_character(path)"
            character = asset.FBXLoader.load_character(self.fbx_file)
        else:
            self.load_call = "FBXLoader.load_character(path, clip_index, fps, scale)"
            character = asset.FBXLoader.load_character(
                self.fbx_file,
                clip_index=self.clip_index,
                fps=self.fps,
                scale=self.scale,
            )
        self.meshes = character.skinned_meshes
        self.motion = character.motion
        self.parents = self.motion.parent_indices()
        self.names = self.motion.node_names()

        self._create_mesh_prims()
        self._apply_time(0.0)
        self._apply_visibility()
        self._apply_shadow_casting()

        self._print_fbx_import_info()
        self.check_error()

    def _print_fbx_import_info(self):
        print(f"FBX character loaded: {Path(self.fbx_file).name}")
        print(f"load: {self.load_call}")
        fps_text = "source" if self.fps <= 0.0 else f"{self.fps:g}"
        print(
            f"request: clip_index={self.clip_index} "
            f"fps={fps_text} "
            f"scale={self.scale:g}"
        )
        print("clips:")
        for idx, clip in enumerate(self.clips):
            duration = clip.end_time - clip.start_time
            selected = " <- selected" if clip.name == self.motion.motion_name() else ""
            print(
                f"  [{idx}] {clip.name} "
                f"start={clip.start_time:.3f}s end={clip.end_time:.3f}s "
                f"duration={duration:.3f}s frame_rate={clip.frame_rate:g}{selected}"
            )
        print(
            f"motion: {self.motion.motion_name()} "
            f"joints={self.motion.num_joints()} frames={self.motion.num_frames()} "
            f"fps={self.motion.fps():g} duration={self.motion.duration():.3f}s"
        )
        print(f"meshes: {len(self.meshes)}")
        for idx, mesh in enumerate(self.meshes):
            materials = list(mesh.materials)
            primary = self._primary_material(mesh)
            texture = ""
            if primary is not None:
                texture_bits = []
                if primary.has_diffuse_texture:
                    texture_bits.append("diffuse")
                if primary.has_normal_texture:
                    texture_bits.append("normal")
                texture = f" primary_material={primary.name}"
                if texture_bits:
                    texture += f" textures={','.join(texture_bits)}"
            print(
                f"  [{idx}] {mesh.name} "
                f"vertices={mesh.vertex_count} indices={mesh.index_count} "
                f"materials={len(materials)} skin={mesh.has_skin} "
                f"clusters={len(mesh.skin_cluster_names)} "
                f"unweighted={mesh.unweighted_vertex_count} "
                f"overweight={mesh.overweight_vertex_count}{texture}"
            )

    def _create_mesh_prims(self):
        palette = [
            ke.vec4(0.82, 0.78, 0.68, 1.0),
            ke.vec4(0.42, 0.58, 0.92, 1.0),
            ke.vec4(0.88, 0.45, 0.34, 1.0),
            ke.vec4(0.55, 0.78, 0.48, 1.0),
        ]
        for idx, mesh in enumerate(self.meshes):
            name = prim_safe_name(mesh.name, f"mesh_{idx}")
            prim = self.get_scene().define_prim(
                f"/fbx_character/{idx}_{name}",
                scene.PrimType.Mesh,
            )
            prim.set_mesh_data(mesh.mesh_data)
            color = self._mesh_material_color(mesh, palette[idx % len(palette)])
            prim.set_display_color_alpha(color)
            texture = self._load_mesh_texture(mesh)
            normal_texture = self._load_mesh_normal_texture(mesh)
            shader = self.textured_mesh_shader if texture is not None else self.mesh_shader
            handle = self.add_skinned_renderable(
                shader,
                prim,
                mesh.skinned_mesh_data,
            )
            if texture is not None:
                self.set_renderable_texture(handle, texture, 0)
                if normal_texture is not None:
                    self.set_renderable_texture(handle, normal_texture, 5)
            # self.set_renderable_double_sided(handle, True)
            self.mesh_prims.append(prim)
            self.mesh_colors.append(color)
            self.mesh_handles.append(handle)
            inverse_bind_matrices = np.asarray(
                mesh.inverse_bind_matrices,
                dtype=np.float32,
            )
            bone_node_indices = np.asarray(
                mesh.bone_node_indices,
                dtype=np.int32,
            )
            self.skinned_meshes.append(
                {
                    "bone_node_indices": bone_node_indices,
                    "inverse_bind_matrices": inverse_bind_matrices,
                    "bone_matrices": np.tile(
                        np.eye(4, dtype=np.float32),
                        (len(inverse_bind_matrices), 1, 1),
                    ),
                    "handle": handle,
                }
            )

    def _primary_material(self, mesh):
        materials = list(mesh.materials)
        primary = int(mesh.primary_material_index)
        if 0 <= primary < len(materials):
            return materials[primary]
        return None

    def _mesh_material_color(self, mesh, fallback):
        material = self._primary_material(mesh)
        if material is None:
            return fallback
        color = material.diffuse_color
        if color is None:
            return fallback
        return ke.vec4(float(color[0]), float(color[1]), float(color[2]), float(color[3]))

    def _load_mesh_texture(self, mesh):
        material = self._primary_material(mesh)
        if material is None or not material.has_diffuse_texture:
            return None
        texture_path = Path(material.diffuse_texture_path)
        if not texture_path.exists():
            return None
        texture = self.get_renderer().device().create_texture(str(texture_path), True)
        self.textures.append(texture)
        return texture

    def _load_mesh_normal_texture(self, mesh):
        material = self._primary_material(mesh)
        if material is None or not material.has_normal_texture:
            return None
        texture_path = Path(material.normal_texture_path)
        if not texture_path.exists():
            return None
        texture = self.get_renderer().device().create_texture(str(texture_path), True)
        self.textures.append(texture)
        return texture

    def _apply_time(self, time: float):
        state = self.motion.sample(time, loop=True)
        global_mats = np.asarray(state.compute_global_matrices(), dtype=np.float32)
        self._update_skeleton_lines_from_state(state)
        self._update_skinned_meshes(global_mats)

    def _update_skinned_meshes(self, global_mats: np.ndarray):
        for mesh in self.skinned_meshes:
            if not len(mesh["bone_node_indices"]) or not len(
                mesh["inverse_bind_matrices"]
            ):
                continue

            """
            Reference Python implementation kept for studying the skinning
            equation. The active path below only uploads GPU bone matrices.
            To re-enable this CPU path, cache these arrays in _create_mesh_prims:
            vertices, normals, bone_indices, bone_weights.

            vertices = mesh["vertices"]
            normals = mesh["normals"]
            bone_indices = mesh["bone_indices"]
            bone_weights = mesh["bone_weights"]
            bone_node_indices = mesh["bone_node_indices"]
            inv_bind = mesh["inverse_bind_matrices"]

            ones = np.ones((vertices.shape[0], 1), dtype=np.float32)
            v_h = np.concatenate((vertices, ones), axis=1)
            skinned = np.zeros_like(v_h)
            skinned_normals = np.zeros_like(normals)

            for slot in range(4):
                bone_ids = bone_indices[:, slot]
                weights = bone_weights[:, slot]
                active = (bone_ids >= 0) & (weights > 1e-6)
                if not np.any(active):
                    continue

                node_ids = bone_node_indices[bone_ids[active]]
                valid = node_ids >= 0
                if not np.any(valid):
                    continue

                active_indices = np.nonzero(active)[0][valid]
                active_bones = bone_ids[active][valid]
                active_nodes = node_ids[valid]
                active_weights = weights[active][valid]
                skin_mats = global_mats[active_nodes] @ inv_bind[active_bones]

                transformed = np.einsum(
                    "nij,nj->ni",
                    skin_mats,
                    v_h[active_indices],
                )
                skinned[active_indices] += transformed * active_weights[:, None]

                normal_mats = skin_mats[:, :3, :3]
                transformed_normals = np.einsum(
                    "nij,nj->ni",
                    normal_mats,
                    normals[active_indices],
                )
                skinned_normals[active_indices] += (
                    transformed_normals * active_weights[:, None]
                )

            normal_norms = np.linalg.norm(skinned_normals, axis=1, keepdims=True)
            skinned_normals = np.divide(
                skinned_normals,
                np.maximum(normal_norms, 1e-8),
                out=np.zeros_like(skinned_normals),
                where=normal_norms > 1e-8,
            )
            """

            """
            Reference C++ CPU helper path kept for comparing the GPU path.
            To re-enable it, cache vertices/normals/bone_indices/bone_weights
            in _create_mesh_prims and call update_renderable_geometry below.

            skinned = animation.cpu_skin(
                mesh["vertices"],
                mesh["normals"],
                mesh["bone_indices"],
                mesh["bone_weights"],
                mesh["bone_node_indices"],
                mesh["inverse_bind_matrices"],
                global_mats,
            )

            self.update_renderable_geometry(
                mesh["handle"],
                skinned["positions"],
                skinned["normals"],
            )
            """

            bone_node_indices = mesh["bone_node_indices"]
            inverse_bind = mesh["inverse_bind_matrices"]
            bone_matrices = mesh["bone_matrices"]
            animation.compute_skinning_matrices_into(
                bone_node_indices,
                inverse_bind,
                global_mats,
                bone_matrices,
            )
            self.update_renderable_skinning_matrices(
                mesh["handle"],
                bone_matrices,
            )

    def _update_skeleton_lines_from_state(self, state):
        if not self.show_skeleton and self.line_handle is not None:
            return

        positions = state.compute_global_positions()
        starts = []
        ends = []
        colors = []
        for body_idx, parent_idx in enumerate(self.parents):
            if parent_idx < 0:
                continue
            parent_pos = positions[parent_idx]
            body_pos = positions[body_idx]
            starts.append(
                [float(parent_pos.x), float(parent_pos.y), float(parent_pos.z)]
            )
            ends.append([float(body_pos.x), float(body_pos.y), float(body_pos.z)])

            name = self.names[body_idx].lower()
            if "left" in name:
                colors.append([0.2, 0.45, 1.0, 1.0])
            elif "right" in name:
                colors.append([1.0, 0.25, 0.18, 1.0])
            else:
                colors.append([0.98, 0.98, 0.98, 1.0])

        starts_t = torch.tensor(starts, dtype=torch.float32)
        ends_t = torch.tensor(ends, dtype=torch.float32)
        colors_t = torch.tensor(colors, dtype=torch.float32)
        if not self.show_skeleton:
            colors_t[:, 3] = 0.0
        self.skeleton_starts = starts_t
        self.skeleton_ends = ends_t
        self.skeleton_colors = torch.tensor(colors, dtype=torch.float32)

        if self.line_handle is None:
            self.line_handle = scene.DebugDraw.log_lines(
                self,
                self.skeleton_shader,
                "/debug/fbx_character_skeleton",
                starts_t,
                ends_t,
                colors_t,
                self.line_radius,
                8,
            )
        else:
            scene.DebugDraw.update_lines(
                self,
                self.line_handle,
                starts_t,
                ends_t,
                colors_t,
            )

    def _apply_visibility(self):
        mesh_alpha = 1.0 if self.show_mesh else 0.0

        for prim, color in zip(self.mesh_prims, self.mesh_colors):
            prim.set_display_color_alpha(
                ke.vec4(float(color.x), float(color.y), float(color.z), mesh_alpha)
            )
        self._apply_skeleton_visibility()

    def _apply_shadow_casting(self):
        for handle in self.mesh_handles:
            self.set_renderable_casts_shadow(handle, self.cast_shadows)

    def _apply_skeleton_visibility(self):
        if (
            self.line_handle is None
            or self.skeleton_starts is None
            or self.skeleton_ends is None
            or self.skeleton_colors is None
        ):
            return

        colors = self.skeleton_colors.clone()
        colors[:, 3] = 1.0 if self.show_skeleton else 0.0
        scene.DebugDraw.update_lines(
            self,
            self.line_handle,
            self.skeleton_starts,
            self.skeleton_ends,
            colors,
        )

    def preRender(self):
        changed = False
        if self.was_key_pressed(keys.M):
            self.show_mesh = not self.show_mesh
            changed = True
        if self.was_key_pressed(keys.L):
            self.show_skeleton = not self.show_skeleton
            changed = True
        if self.was_key_pressed(keys.H):
            self.cast_shadows = not self.cast_shadows
            self._apply_shadow_casting()
        if self.was_key_pressed(keys.SPACE):
            self.animate = not self.animate
        if changed:
            self._apply_visibility()
        if self.animate:
            self.time += self.get_delta_time() * max(0.0, self.playback_speed)
            self._apply_time(self.time)
        self.check_error()

    def render(self):
        imgui.begin("FBX Character")
        imgui.text(f"{Path(self.fbx_file).name}")
        imgui.text(f"Meshes: {len(self.meshes)}  Joints: {self.motion.num_joints()}")
        state = "running" if self.animate else "paused"
        duration = max(self.motion.duration(), 1e-6)
        imgui.text(f"Time: {self.time % duration:.3f}s / {duration:.3f}s  |  {state}")
        imgui.text("Space: pause/resume    M: mesh    L: skeleton    H: shadow")
        mesh_changed, self.show_mesh = imgui.checkbox("show mesh", self.show_mesh)
        skeleton_changed, self.show_skeleton = imgui.checkbox(
            "show skeleton",
            self.show_skeleton,
        )
        shadow_changed, self.cast_shadows = imgui.checkbox(
            "cast shadows",
            self.cast_shadows,
        )
        _, self.animate = imgui.checkbox("animate", self.animate)
        _, self.playback_speed = imgui.slider_float(
            "playback speed",
            self.playback_speed,
            0.0,
            4.0,
        )
        if mesh_changed or skeleton_changed:
            self._apply_visibility()
        if shadow_changed:
            self._apply_shadow_casting()
        imgui.end()

    def postRender(self):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fbx-file", default=str(default_fbx_file()))
    parser.add_argument("--clip-index", type=int, default=-1)
    parser.add_argument("--fps", type=float, default=-1.0)
    parser.add_argument("--scale", type=float, default=0.01)
    parser.add_argument("--line-radius", type=float, default=0.008)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    args = parser.parse_args()

    fbx_file = Path(args.fbx_file).expanduser().resolve()
    if not fbx_file.exists():
        raise FileNotFoundError(fbx_file)

    app = FbxCharacterViewer(
        fbx_file,
        args.clip_index,
        args.fps,
        args.scale,
        args.line_radius,
    )
    app.initialize(args.width, args.height, False, ke.UpAxis.Y)
    app.start()


if __name__ == "__main__":
    main()
