"""Validate graphics/compute shader authored resources without a GPU window."""

from __future__ import annotations

import kangengine as ke
from kangengine import render


def main():
    backend = ke.scene.create_backend(ke.scene.BackendType.NATIVE)
    resources = ke.scene.SceneResourceManager(backend)

    shader = ke.scene.ShaderSourceResource()
    shader.stage = render.ShaderType.COMPUTE
    shader.language = ke.scene.ShaderLanguage.WGSL
    shader.entry_point = "main"
    shader.source = "@compute @workgroup_size(64) fn main() {}"
    shader_handle = resources.register_shader_source(
        "particle_update", shader, "memory://particle_update.wgsl"
    )

    pipeline = ke.scene.PipelineResource()
    pipeline.type = ke.scene.PipelineType.COMPUTE
    pipeline.shader_sources = [shader_handle]
    pipeline.state_summary = "workgroup_size=64"
    pipeline_handle = resources.register_pipeline("particle_compute", pipeline)

    shader_prim = resources.resource_prim(shader_handle)
    pipeline_prim = resources.resource_prim(pipeline_handle)
    if not shader_prim.get_path().startswith("/.Resources/ShaderSources/"):
        raise AssertionError("shader source mirror path mismatch")
    if not pipeline_prim.get_path().startswith("/.Resources/Pipelines/"):
        raise AssertionError("pipeline mirror path mismatch")
    if resources.shader_source(shader_handle).stage != render.ShaderType.COMPUTE:
        raise AssertionError("compute shader stage was not preserved")
    if resources.pipeline(pipeline_handle).shader_sources != [shader_handle]:
        raise AssertionError("pipeline shader dependency was not preserved")
    if resources.usage_paths(shader_handle) != [pipeline_prim.get_path()]:
        raise AssertionError("pipeline-to-shader usage was not reported")
    resources.add_external_usage(pipeline_handle, "render-hook://77")
    if resources.usage_paths(pipeline_handle) != ["render-hook://77"]:
        raise AssertionError("external pipeline usage was not reported")
    resources.remove_external_usage(pipeline_handle, "render-hook://77")
    if resources.usage_count(pipeline_handle) != 0:
        raise AssertionError("removed external pipeline usage was retained")
    print("PASS: authored shader/pipeline resources")


if __name__ == "__main__":
    main()
