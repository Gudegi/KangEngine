"""Observation contracts using native prims without a graphics context."""

import json
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

import kangengine as ke
from kangengine.app.capture import AutomationError
from kangengine.app.inspection import engine_status, renderer_profile, scene_inspect


@pytest.fixture
def app():
    root = ke.scene.Prim("", ke.scene.PrimType.ROOT)
    group = root.add_child("Group", ke.scene.PrimType.XFORM)
    group.set_local_translation([10, 20, 30])
    group.set_active(False)
    mesh = group.add_child("Mesh", ke.scene.PrimType.MESH)
    mesh.set_local_translation([1, 2, 3])
    mesh.set_visible(False)
    mesh.add_mesh_component().resource_handle = 7
    root.add_child("Sibling", ke.scene.PrimType.XFORM)
    return SimpleNamespace(
        scene=SimpleNamespace(
            get_root_prim=lambda: root,
            get_prim_at_path=root.get_prim_at_path,
        ),
        _existing_resource_handle=lambda material: None,
        graphics_backend_type=ke.render.BackendType.OPENGL,
        scene_backend_type=ke.scene.BackendType.NATIVE,
        up_axis=ke.UpAxis.Z,
        headless=True,
        run_config=None,
    )


def test_scene_copies_native_hierarchy_and_row_major_transforms(app):
    result = scene_inspect(app)
    json.dumps(result, allow_nan=False)
    records = {prim["path"]: prim for prim in result["prims"]}
    mesh = records["/Group/Mesh"]
    assert mesh["parent"] == "/Group"
    assert mesh["type"] == "MESH"
    assert mesh["active"] is True
    assert mesh["effective_active"] is False
    assert mesh["visible"] is False
    assert mesh["effective_render_state"] is False
    assert mesh["resources"] == {"mesh": 7, "material": None}
    np.testing.assert_array_equal(np.array(mesh["local_matrix"])[:3, 3], [1, 2, 3])
    np.testing.assert_array_equal(np.array(mesh["world_matrix"])[:3, 3], [11, 22, 33])
    app.scene.get_prim_at_path("/Group").set_local_translation([0, 0, 0])
    np.testing.assert_array_equal(np.array(mesh["world_matrix"])[:3, 3], [11, 22, 33])
    assert result["truncated"] is False


def test_external_buffer_matrix_describes_batch_owner(app):
    prim = app.scene.get_prim_at_path("/Group/Mesh")
    prim.add_render_component().transform_source = (
        ke.render.TransformSource.EXTERNAL_BUFFER
    )
    record = scene_inspect(app, path="/Group/Mesh")["prims"][0]
    assert record["transform_source"] == "ExternalBuffer"
    assert record["matrix_scope"] == "batch_owner_prim"
    assert record["depth"] == 0


def test_material_identity_uses_existing_registry_without_registration(app):
    material = ke.material.VertexColorMaterial()
    app._resource_handles_by_object_id = {}
    app._existing_resource_handle = MethodType(ke.App._existing_resource_handle, app)
    ke.App._remember_resource_handle(app, material, 31)
    app.scene.get_prim_at_path("/Group/Mesh").set_material(material.native)
    record = scene_inspect(app, path="/Group/Mesh")["prims"][0]
    assert record["resources"]["material"] == 31
    assert len(app._resource_handles_by_object_id) == 2


def test_resource_metadata_is_copied_without_mesh_buffers(app):
    prim = app.scene.get_root_prim().add_child("Resource", ke.scene.PrimType.RESOURCE)
    resource = prim.add_resource_component()
    resource.handle = 9
    resource.type = ke.scene.ResourceType.MESH
    resource.uri = "generated://mesh"
    resource.display_name = "Test mesh"
    record = scene_inspect(app, path="/Resource")["prims"][0]
    assert record["resource"] == {
        "handle": 9,
        "type": "MESH",
        "uri": "generated://mesh",
        "display_name": "Test mesh",
    }
    assert record["local_matrix"] is None
    assert record["world_matrix"] is None
    assert record["transform_source"] is None
    assert "vertices" not in record
    json.dumps(record, allow_nan=False)


@pytest.mark.parametrize(
    "options,count,truncated",
    [
        ({"max_depth": 0}, 1, True),
        ({"max_depth": 1}, 3, True),
        ({"max_prims": 2}, 2, True),
        ({"max_prims": 4}, 4, False),
        ({"path": "/Group/Mesh", "max_depth": 0, "max_prims": 1}, 1, False),
    ],
)
def test_subtree_limits_report_actual_truncation(app, options, count, truncated):
    result = scene_inspect(app, **options)
    assert len(result["prims"]) == count
    assert result["truncated"] is truncated


@pytest.mark.parametrize(
    "options",
    [
        {"path": "relative"},
        {"path": "/Group/../Sibling"},
        {"path": "/Group/"},
        {"path": "/Group//Mesh"},
        {"path": None},
        {"max_depth": -1},
        {"max_depth": 17},
        {"max_prims": 0},
        {"max_prims": 1001},
        {"max_prims": True},
    ],
)
def test_invalid_scene_requests_are_stable_errors(app, options):
    with pytest.raises(AutomationError) as error:
        scene_inspect(app, **options)
    assert error.value.code == "INVALID_ARGUMENT"


def test_missing_prim_and_unsupported_backend_are_distinct(app):
    with pytest.raises(AutomationError) as error:
        scene_inspect(app, path="/Missing")
    assert error.value.code == "NOT_FOUND"
    app.scene.get_root_prim = lambda: None
    with pytest.raises(AutomationError) as error:
        scene_inspect(app)
    assert error.value.code == "UNSUPPORTED"


def test_status_does_not_assume_imported_asset_units(app):
    status = engine_status(app)
    assert status["graphics_backend"] == "OPENGL"
    assert status["up_axis"] == "Z"
    assert status["units"]["meters_per_unit"] is None
    assert status["capabilities"]["scene_inspect"] is True
    assert status["render_enabled"] is True
    app.run_config = SimpleNamespace(render_enabled=False)
    assert engine_status(app)["render_enabled"] is False


def _profile_renderer(*, enabled, frames):
    capability_names = (
        "cpu_scopes",
        "gpu_timestamps",
        "pass_timestamps",
        "command_timestamps",
        "external_timestamps",
        "in_pass_timestamps",
        "debug_groups",
    )
    capabilities = SimpleNamespace(
        **dict.fromkeys(capability_names, False), timestamp_capacity=0
    )
    summary = SimpleNamespace(
        capture_id=3 if frames else 0,
        frame_count=len(frames),
        first_frame_index=frames[0].frame_index if frames else None,
        last_frame_index=frames[-1].frame_index if frames else None,
        scopes=(),
    )
    return SimpleNamespace(
        profiler_enabled=enabled,
        profiler_capabilities=capabilities,
        frame_profile_history=lambda: tuple(frames),
        profile_summary=lambda *, max_frames: summary,
    )


def _frame(frame_index, status):
    counter_names = (
        "draw_calls",
        "indexed_draw_calls",
        "instances",
        "triangles",
        "buffer_upload_bytes",
        "texture_upload_bytes",
        "external_buffer_bytes",
        "buffer_allocations",
        "buffer_allocated_bytes",
    )
    return SimpleNamespace(
        capture_id=3,
        frame_index=frame_index,
        revision=2,
        finalized=status != ke.render.ProfileSampleStatus.PENDING,
        backend=ke.render.BackendType.OPENGL,
        gpu_latency_frames=None,
        dropped_samples=0,
        counters=SimpleNamespace(**dict.fromkeys(counter_names, 0)),
        metadata={"coverage": "RHI"},
        samples=(
            SimpleNamespace(
                sample_id=1,
                parent_sample_id=None,
                path="scene",
                duration_ms=None,
                available=False,
                domain=ke.render.ProfileTimingDomain.GPU,
                status=status,
            ),
        ),
    )


def test_profile_retains_pending_unsupported_and_revision_without_waiting(app):
    frames = [
        _frame(10, ke.render.ProfileSampleStatus.UNSUPPORTED),
        _frame(11, ke.render.ProfileSampleStatus.PENDING),
    ]
    renderer = _profile_renderer(enabled=False, frames=frames)
    app.get_renderer = lambda: renderer
    result = renderer_profile(app)
    assert result["enabled"] is False
    assert result["data_status"] == "retained"
    assert [frame["samples"][0]["status"] for frame in result["frames"]] == [
        "UNSUPPORTED",
        "PENDING",
    ]
    assert result["frames"][1]["finalized"] is False
    assert result["frames"][1]["revision"] == 2
    assert result["frames"][1]["samples"][0]["duration_ms"] is None
    assert len(renderer_profile(app, max_frames=1)["frames"]) == 1
    frames[0].metadata["coverage"] = "changed"
    assert result["frames"][0]["metadata"]["coverage"] == "RHI"
    json.dumps(result, allow_nan=False)


def test_profile_summary_does_not_turn_missing_times_into_zero(app):
    renderer = _profile_renderer(enabled=True, frames=[])
    summary = renderer.profile_summary(max_frames=1)
    summary.scopes = (
        SimpleNamespace(
            path="scene",
            sample_count=1,
            ready_frames=0,
            pending_frames=1,
            unavailable_frames=0,
            mean_ms=None,
            median_ms=None,
            p95_ms=None,
            max_ms=None,
            domain=ke.render.ProfileTimingDomain.GPU,
        ),
    )
    app.get_renderer = lambda: renderer
    result = renderer_profile(app)
    assert result["summary"]["scopes"][0]["mean_ms"] is None
    assert result["summary"]["scopes"][0]["pending_frames"] == 1


@pytest.mark.parametrize("enabled", [False, True])
def test_profile_empty_and_disabled_are_independent(app, enabled):
    app.get_renderer = lambda: _profile_renderer(enabled=enabled, frames=[])
    result = renderer_profile(app)
    assert result["enabled"] is enabled
    assert result["data_status"] == "empty"
    assert result["summary"]["first_frame_index"] is None


@pytest.mark.parametrize("max_frames", [0, 241, True, 1.5])
def test_profile_window_is_bounded(app, max_frames):
    with pytest.raises(AutomationError) as error:
        renderer_profile(app, max_frames=max_frames)
    assert error.value.code == "INVALID_ARGUMENT"
