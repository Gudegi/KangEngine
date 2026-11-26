"""Explicit smoke cases; optional devices and render adapters have separate entries."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Case:
    source: str
    suites: tuple[str, ...]
    requirements: tuple[str, ...] = ()
    args: tuple[str, ...] = ()
    variant: str = ""
    timeout: float = 180.0

    @property
    def id(self) -> str:
        language = "py" if self.source.endswith(".py") else "cpp"
        suffix = f":{self.variant}" if self.variant else ""
        return f"{language}:{Path(self.source).stem}{suffix}"


PY = "python/examples/smoke/"
CPP = "examples/smokes/"
CASES = (
    Case(PY + "api/public_api_surface_smoke.py", ("core",)),
    Case(PY + "api/public_stub_surface_smoke.py", ("core",)),
    Case(PY + "api/quaternion_binding_smoke.py", ("core",)),
    Case(PY + "app/frame_execution_smoke.py", ("app",), ("opengl",)),
    Case(PY + "asset/mjcf_visual_duplicate_mesh_smoke.py", ("asset",)),
    Case(PY + "asset/mjcf_visual_rgba_smoke.py", ("asset",)),
    Case(PY + "asset/obj_material_loader_smoke.py", ("asset",)),
    Case(PY + "asset/pbr_opacity_smoke.py", ("asset",), ("opengl",)),
    Case(PY + "asset/ply_loader_smoke.py", ("asset",), ("opengl",)),
    Case(PY + "asset/rigid_asset_visual_smoke.py", ("asset",), ("opengl", "physx"), ("--device", "cpu")),
    Case(PY + "asset/scene_add_obj_smoke.py", ("asset",), ("opengl",)),
    Case(PY + "asset/texture_color_space_smoke.py", ("asset",), ("opengl",)),
    Case(PY + "asset/urdf_visual_smoke.py", ("asset",), ("opengl",)),
    Case(PY + "asset/usd_articulation_texture_smoke.py", ("asset",), ("opengl", "usd", "physx")),
    Case(PY + "asset/usd_lighting_smoke.py", ("asset",), ("opengl",)),
    Case(PY + "mcp/automation_idle_smoke.py", ("mcp",), ("opengl", "mcp")),
    Case(PY + "mcp/capture_lifecycle_smoke.py", ("mcp",), ("opengl", "mcp")),
    Case(PY + "mcp/control_smoke.py", ("mcp",), ("opengl", "mcp")),
    Case(PY + "mcp/observation_smoke.py", ("mcp",), ("opengl", "mcp")),
    Case(PY + "mcp/scenario_smoke.py", ("mcp",), ("opengl", "mcp")),
    Case(PY + "physics/articulation_builder_smoke.py", ("physics",), ("physx",)),
    Case(PY + "physics/d6_joint_batch_smoke.py", ("physics",), ("physx",)),
    Case(PY + "physics/physics_cpu_system_smoke.py", ("physics",), ("physx",)),
    Case(PY + "physics/physics_gpu_system_smoke.py", ("gpu",), ("physx", "cuda"), timeout=600),
    Case(PY + "rendering/app_backend_smoke.py", ("render",), ("opengl", "physx")),
    Case(PY + "rendering/app_texture_source_smoke.py", ("render",), ("opengl",)),
    Case(PY + "rendering/debug_draw_transform_smoke.py", ("render", "core")),
    Case(PY + "rendering/debug_geometry_ownership_smoke.py", ("render",), ("opengl",)),
    Case(PY + "rendering/mesh_debug_backend_smoke.py", ("render",), ("opengl",)),
    Case(PY + "rendering/mesh_debug_backend_smoke.py", ("ovrtx",), ("ovrtx",),
         ("--renderer", "ovrtx"), variant="ovrtx", timeout=600),
    Case(PY + "rendering/debug_overlay_depth_smoke.py", ("render",), ("opengl",)),
    Case(PY + "rendering/debug_backend_smoke.py", ("render",), ("opengl",)),
    Case(PY + "rendering/raycast_backend_smoke.py", ("render",), ("opengl",)),
    Case(PY + "rendering/signed_distance_smoke.py", ("render",), ("opengl",)),
    Case(PY + "rendering/raycast_backend_smoke.py", ("ovrtx",), ("ovrtx",),
         ("--renderer", "ovrtx"), variant="ovrtx", timeout=600),
    Case(PY + "rendering/text_backend_smoke.py", ("render",), ("opengl",),
         ("--window",)),
    Case(PY + "rendering/text_backend_smoke.py", ("ovrtx",), ("ovrtx",),
         ("--renderer", "ovrtx"), variant="ovrtx", timeout=600),
    Case(PY + "rendering/text_backend_smoke.py", ("ovrtx",), ("ovrtx", "opengl", "cuda"),
         ("--renderer", "ovrtx", "--window"), variant="ovrtx-window", timeout=600),
    Case(PY + "rendering/debug_backend_smoke.py", ("ovrtx",), ("ovrtx",),
         ("--renderer", "ovrtx"), variant="ovrtx", timeout=600),
    Case(PY + "rendering/debug_backend_smoke.py", ("ovrtx",), ("ovrtx", "opengl", "cuda"),
         ("--renderer", "ovrtx", "--window"), variant="ovrtx-window", timeout=600),
    Case(PY + "rendering/dome_light_smoke.py", ("render",), ("opengl",)),
    Case(PY + "rendering/display_settings_smoke.py", ("render",), ("opengl",)),
    Case(PY + "rendering/display_settings_smoke.py", ("ovrtx",), ("ovrtx",),
         ("--renderer", "ovrtx"), variant="ovrtx", timeout=600),
    Case(PY + "rendering/dynamic_geometry_smoke.py", ("core",)),
    Case(PY + "rendering/dynamic_geometry_smoke.py", ("render",), ("opengl",),
         ("--renderer", "native"), variant="native"),
    Case(PY + "rendering/dynamic_geometry_smoke.py", ("ovrtx",), ("ovrtx",),
         ("--renderer", "ovrtx"), variant="ovrtx", timeout=600),
    Case(PY + "rendering/external_instances_smoke.py", ("core",)),
    Case(PY + "rendering/external_instances_smoke.py", ("render",), ("opengl",),
         ("--renderer", "native"), variant="native"),
    Case(PY + "rendering/external_instances_smoke.py", ("ovrtx",), ("ovrtx",),
         ("--renderer", "ovrtx"), variant="ovrtx", timeout=600),
    Case(PY + "rendering/gpu_physics_inspector_smoke.py", ("gpu",), ("opengl", "physx", "cuda")),
    Case(PY + "rendering/contact_inspector_smoke.py", ("render",), ("opengl", "physx")),
    Case(PY + "rendering/com_inspector_smoke.py", ("render",), ("opengl", "physx")),
    Case(PY + "rendering/com_inspector_smoke.py", ("gpu",), ("opengl", "physx", "cuda"),
         ("--gpu",), variant="gpu"),
    Case(PY + "rendering/com_inspector_smoke.py", ("ovrtx",), ("ovrtx", "physx"),
         ("--renderer", "ovrtx"), variant="ovrtx", timeout=600),
    Case(PY + "rendering/com_inspector_smoke.py", ("ovrtx",), ("ovrtx", "physx", "cuda"),
         ("--renderer", "ovrtx", "--gpu"), variant="ovrtx-gpu", timeout=600),
    Case(PY + "rendering/contact_inspector_smoke.py", ("gpu",), ("opengl", "physx", "cuda"),
         ("--gpu",), variant="gpu"),
    Case(PY + "rendering/contact_inspector_smoke.py", ("ovrtx",), ("ovrtx", "physx"),
         ("--renderer", "ovrtx"), variant="ovrtx", timeout=600),
    Case(PY + "rendering/contact_inspector_smoke.py", ("ovrtx",), ("ovrtx", "physx", "cuda"),
         ("--renderer", "ovrtx", "--gpu"), variant="ovrtx-gpu", timeout=600),
    Case(PY + "rendering/gpu_physics_inspector_smoke.py", ("ovrtx",), ("ovrtx", "physx", "cuda"),
         ("--renderer", "ovrtx"), variant="ovrtx", timeout=600),
    Case(PY + "rendering/light_coordinates_smoke.py", ("render",), ("opengl",)),
    Case(PY + "rendering/light_overlay_smoke.py", ("render",), ("opengl",)),
    Case(PY + "rendering/material_texture_scope_smoke.py", ("render",), ("opengl",)),
    Case(PY + "rendering/memory_texture_source_smoke.py", ("render",), ("opengl",)),
    Case(PY + "rendering/render_scene_data_collector_smoke.py", ("render", "core")),
    Case(PY + "rendering/renderer_profiler_smoke.py", ("render",), ("opengl",)),
    Case(PY + "rendering/scene_binding_updates_smoke.py", ("render",), ("opengl",)),
    Case(PY + "rendering/scene_hook_custom_pipeline_smoke.py", ("render",), ("opengl",)),
    Case(PY + "rendering/scene_light_sync_smoke.py", ("render",), ("opengl",)),
    Case(PY + "rendering/scene_policy_smoke.py", ("render",), ("opengl",)),
    Case(PY + "rendering/scene_render_instancing_smoke.py", ("render",), ("opengl",)),
    Case(PY + "rendering/scene_render_system_smoke.py", ("render",), ("opengl",)),
    Case(PY + "rendering/scene_renderer_boundary_smoke.py", ("render",), ("opengl",)),
    Case(PY + "rendering/scene_renderer_smoke.py", ("render",), ("opengl",), ("--output", "{output}")),
    Case(PY + "rendering/scene_updates_smoke.py", ("render", "core")),
    Case(PY + "rendering/skinning_backend_smoke.py", ("core",)),
    Case(PY + "rendering/ovrtx/ovrtx_picking_smoke.py", ("ovrtx",), ("ovrtx", "opengl")),
    Case(PY + "rendering/ovrtx/ovrtx_material_smoke.py", ("ovrtx",), ("ovrtx",)),
    Case(PY + "rendering/ovrtx/ovrtx_cuda_instances_smoke.py", ("ovrtx",), ("ovrtx", "cuda")),
    Case(PY + "rendering/skinning_backend_smoke.py", ("render",), ("opengl",),
         ("--renderer", "native"), variant="native"),
    Case(PY + "rendering/skinning_backend_smoke.py", ("ovrtx",), ("ovrtx",),
         ("--renderer", "ovrtx"), variant="ovrtx", timeout=600),
    Case(PY + "rendering/shared_material_updates_smoke.py", ("render",), ("opengl",)),
    Case(PY + "rendering/unlit_material_smoke.py", ("render",), ("opengl",)),
    Case(PY + "rendering/vertex_color_smoke.py", ("render",), ("opengl",)),
    Case(PY + "rendering/view_scene_renderer_smoke.py", ("render",), ("opengl",)),
    Case(PY + "rendering/window_render_backend_smoke.py", ("render",), ("opengl",)),
    Case(PY + "scene/camera_component_smoke.py", ("core",)),
    Case(PY + "scene/light_component_smoke.py", ("core",)),
    Case(PY + "scene/mesh_component_smoke.py", ("core",)),
    Case(PY + "scene/python_resource_manager_smoke.py", ("core",)),
    Case(PY + "scene/render_component_lifecycle_smoke.py", ("core",)),
    Case(PY + "scene/resource_component_smoke.py", ("core",)),
    Case(PY + "scene/scene_data_smoke.py", ("core",)),
    Case(PY + "scene/scene_resource_manager_smoke.py", ("core",)),
    Case(PY + "scene/shader_pipeline_resource_smoke.py", ("core",)),
    Case(PY + "scene/transform_component_smoke.py", ("core",)),
    Case(PY + "sim/animation_sim_roundtrip_smoke.py", ("physics",), ("physx",)),
    Case(PY + "sim/cuda_gl_visual_smoke.py", ("gpu",), ("physx", "cuda", "opengl"), timeout=600),
    Case(PY + "sim/gpu_articulation_visual_backend_smoke.py", ("gpu",),
         ("physx", "cuda", "opengl"), timeout=600),
    Case(PY + "sim/gpu_articulation_visual_backend_smoke.py", ("ovrtx",),
         ("physx", "cuda", "ovrtx"), ("--renderer", "ovrtx"), variant="ovrtx", timeout=600),
    Case(PY + "sim/gpu_rigid_visual_backend_smoke.py", ("gpu",),
         ("physx", "cuda", "opengl"), timeout=600),
    Case(PY + "sim/gpu_rigid_visual_backend_smoke.py", ("ovrtx",),
         ("physx", "cuda", "ovrtx"), ("--renderer", "ovrtx"), variant="ovrtx", timeout=600),
    Case(PY + "sim/fixed_update_gpu_stress.py", ("stress",), ("physx", "cuda", "opengl"), timeout=600),
    Case(PY + "sim/kangsimworld_gpu_articulation_control_smoke.py", ("gpu",), ("physx", "cuda"), timeout=600),
    Case(PY + "sim/kangsimworld_gpu_articulation_dynamics_smoke.py", ("gpu",), ("physx", "cuda"), timeout=600),
    Case(PY + "sim/kangsimworld_gpu_articulation_state_smoke.py", ("gpu",), ("physx", "cuda"), timeout=600),
    Case(PY + "sim/kangsimworld_gpu_contact_sensor_smoke.py", ("gpu",), ("physx", "cuda"), timeout=600),
    Case(PY + "sim/kangsimworld_gpu_root_state_smoke.py", ("gpu",), ("physx", "cuda"), timeout=600),
    Case(PY + "sim/mimickit_engine_gpu_reset_smoke.py", ("gpu",), ("physx", "cuda"), timeout=600),
    Case(PY + "sim/sim_visual_batch_smoke.py", ("physics",), ("physx",)),
    Case(PY + "sim/sim_visual_backend_smoke.py", ("render",), ("opengl", "physx")),
    Case(PY + "sim/sim_visual_backend_smoke.py", ("ovrtx",), ("ovrtx", "physx"),
         ("--renderer", "ovrtx"), variant="ovrtx", timeout=600),
    Case(PY + "visual/articulated_surface_smoke.py", ("render", "core")),
    Case(PY + "visual/articulation_binding_component_smoke.py", ("render", "core")),
    Case(PY + "visual/articulation_vertex_color_smoke.py", ("render", "core")),
    Case(PY + "visual/skinned_instance_smoke.py", ("render",), ("opengl",)),
    Case(PY + "visual/skinned_surface_multi_part_smoke.py", ("render", "core")),
    Case(PY + "visual/terrain_visual_smoke.py", ("render",), ("opengl", "physx")),
    Case(PY + "visual/visual_resource_registration_smoke.py", ("render",), ("opengl",)),
    Case(CPP + "animation/motion_matching_search_smoke.cpp", ("core",)),
    Case(CPP + "animation/motion_playback_shortcuts_smoke.cpp", ("core",)),
    Case(CPP + "animation/skeleton_motion_player_smoke.cpp", ("core",)),
    Case(CPP + "animation/skeleton_pose_smoother_smoke.cpp", ("core",)),
    Case(CPP + "animation/trajectory_spring_smoke.cpp", ("core",)),
    Case(CPP + "physics/d6_joint_registry_smoke.cpp", ("physics",), ("physx",)),
    Case(CPP + "physics/environment_filter_smoke.cpp", ("physics",), ("physx",), ("cpu", "{root}/python/kangengine/assets/characters/kw/kw5.xml")),
    Case(CPP + "physics/physx_gpu_step_smoke.cpp", ("gpu",), ("physx", "cuda"), timeout=600),
    Case(CPP + "physics/terrain_collision_smoke.cpp", ("physics",), ("physx",), ("cpu",)),
    Case(CPP + "rendering/cuda_gl_buffer_interop_smoke.cpp", ("gpu",), ("opengl", "cuda"), timeout=600),
    Case(CPP + "rendering/gpu_picking_smoke.cpp", ("render",), ("opengl",)),
    Case(CPP + "rendering/mouse_selection_smoke.cpp", ("render",), ("opengl",)),
    Case(CPP + "rendering/native_render_system_smoke.cpp", ("render",), ("opengl",)),
    Case(CPP + "rendering/render_backend_smoke.cpp", ("core",)),
    Case(CPP + "rendering/renderer_profiler_smoke.cpp", ("core",)),
    Case(CPP + "rendering/resource_monitor_smoke.cpp", ("core",)),
    Case(CPP + "rendering/rhi_command_smoke.cpp", ("render",), ("opengl",)),
    Case(CPP + "rendering/rhi_mrt_smoke.cpp", ("render",), ("opengl",)),
    Case(CPP + "rendering/rhi_phase1_scene_smoke.cpp", ("render",), ("opengl",)),
    Case(CPP + "rendering/rhi_post_processor_smoke.cpp", ("render",), ("opengl",)),
    Case(CPP + "rendering/rhi_profiler_async_smoke.cpp", ("render",), ("opengl",)),
    Case(CPP + "rendering/rhi_profiler_contract_smoke.cpp", ("core",)),
    Case(CPP + "rendering/rhi_resource_smoke.cpp", ("render",), ("opengl",)),
    Case(CPP + "rendering/rhi_selection_outline_smoke.cpp", ("render",), ("opengl",)),
    Case(CPP + "rendering/rhi_types_smoke.cpp", ("core",)),
    Case(CPP + "rendering/scene_raycast_smoke.cpp", ("render",), ("opengl",)),
    Case(CPP + "rendering/scene_render_bridge_smoke.cpp", ("core",)),
    Case(CPP + "rendering/window_initial_size_smoke.cpp", ("render",), ("opengl",)),
    Case(CPP + "scene/transform_component_smoke.cpp", ("core",)),
    Case(PY + "rendering/shared_material_updates_smoke.py", ("render",), ("opengl",),
         ("--scene-renderer",), variant="scene-renderer"),
    Case(CPP + "rendering/render_backend_smoke.cpp", ("render",), ("opengl",),
         ("--app",), variant="app"),
    Case(PY + "asset/rigid_asset_visual_smoke.py", ("gpu",), ("physx", "cuda", "opengl"),
         ("--device", "cuda"), variant="cuda", timeout=600),
    Case(PY + "physics/d6_joint_batch_smoke.py", ("gpu",), ("physx", "cuda"),
         ("--gpu",), variant="cuda", timeout=600),
    Case(CPP + "physics/environment_filter_smoke.cpp", ("gpu",), ("physx", "cuda"),
         ("gpu", "{root}/python/kangengine/assets/characters/kw/kw5.xml"),
         variant="cuda", timeout=600),
    Case(CPP + "physics/terrain_collision_smoke.cpp", ("gpu",), ("physx", "cuda"),
         ("gpu",), variant="cuda", timeout=600),
)

# Optional adapters use the same assertions in their own processes.
for name in (
    "app_backend", "app_texture_source", "dome_light", "scene_light_sync",
    "scene_renderer", "shared_material_updates", "view_scene_renderer",
    "window_render_backend",
):
    args = ("--renderer", "ovrtx")
    requirements = ("ovrtx",)
    if name in ("app_texture_source", "view_scene_renderer", "window_render_backend"):
        requirements += ("opengl",)
    if name == "window_render_backend":
        requirements += ("cuda",)
    if name == "app_backend":
        requirements += ("physx",)
    if name == "scene_renderer":
        args += ("--output", "{output}")
    CASES += (Case(PY + f"rendering/{name}_smoke.py", ("ovrtx",),
                   requirements, args, variant="ovrtx", timeout=600),)

SUITES = {
    suite: tuple(case.id for case in CASES if suite in case.suites)
    for suite in sorted({suite for case in CASES for suite in case.suites})
}
SUITES["all"] = tuple(case.id for case in CASES)

# Preserve the existing Makefile validation scope, including argument variants.
SUITES.update({
    "physx-gpu": (
        "py:physics_gpu_system_smoke",
        "py:kangsimworld_gpu_root_state_smoke",
        "py:kangsimworld_gpu_articulation_state_smoke",
        "py:kangsimworld_gpu_articulation_control_smoke",
        "py:kangsimworld_gpu_contact_sensor_smoke",
        "py:mimickit_engine_gpu_reset_smoke",
    ),
    "physx-gpu-cpp": (
        "cpp:physx_gpu_step_smoke",
    ),
    "fixed-update-gpu": (
        "py:fixed_update_gpu_stress",
    ),
    "sim-visual-batch": (
        "py:sim_visual_batch_smoke",
    ),
    "python-api": (
        "py:public_api_surface_smoke",
        "py:public_stub_surface_smoke",
    ),
    "render-component": (
        "py:obj_material_loader_smoke",
        "py:ply_loader_smoke",
        "py:scene_add_obj_smoke",
        "py:mjcf_visual_rgba_smoke",
        "py:mjcf_visual_duplicate_mesh_smoke",
        "py:articulation_binding_component_smoke",
        "py:articulation_vertex_color_smoke",
        "py:debug_draw_transform_smoke",
        "py:debug_geometry_ownership_smoke",
        "py:quaternion_binding_smoke",
        "py:transform_component_smoke",
        "py:mesh_component_smoke",
        "py:resource_component_smoke",
        "py:scene_resource_manager_smoke",
        "py:python_resource_manager_smoke",
        "py:render_component_lifecycle_smoke",
        "py:scene_render_system_smoke",
        "py:scene_render_instancing_smoke",
        "py:scene_binding_updates_smoke",
        "py:skinned_instance_smoke",
        "py:scene_policy_smoke",
        "py:visual_resource_registration_smoke",
        "py:material_texture_scope_smoke",
        "py:scene_updates_smoke",
        "py:shared_material_updates_smoke",
        "py:shared_material_updates_smoke:scene-renderer",
        "py:vertex_color_smoke",
        "py:unlit_material_smoke",
        "py:app_texture_source_smoke",
        "py:app_backend_smoke",
        "py:memory_texture_source_smoke",
    ),
})

MANUAL_CASES = {
    PY + "asset/usd_scene_smoke.py": "requires a user-supplied USD file",
    CPP + "rendering/render_scene_input_export.cpp": "exports scene input; not a smoke assertion",
}
