"""Shared renderer frame data for App-side MCP tests."""

from types import SimpleNamespace

import kangengine as ke


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
