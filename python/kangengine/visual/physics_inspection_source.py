"""Data contract between simulation providers and the viewport inspector."""

from typing import Any, Protocol, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from torch import Tensor


class PhysicsInspectionSource(Protocol):
    """Supply collider snapshots and checked body references after simulation fetch.

    Body references expose a cheap ``valid`` property, false after removal.
    Poses use xyz/xyzw in scene units. Shape dictionaries contain ``type``,
    ``shape_idx``, ``local_pose``, ``actor_pose`` and geometry dimensions/data.
    Dynamic shapes use ``body_idx`` to index the returned body references.
    Capsules and plane normals use local X. Meshes supply vertices, triangles,
    mesh_scale and mesh_scale_rotation_xyzw; heightfield vertices are already scaled.
    Queries must not step physics or silently download GPU poses. GPU snapshots
    use actor-local geometry and actor_pose=None.
    Providers without GPU collider raycasts expose supports_dynamic_picking=False.
    Optional pose_revision skips unchanged snapshots; None keeps CPU pose polling.
    A GPU body list stays fixed until Refresh or GPU system reinitialization.
    Optional contact inspection exposes supports_contacts, contacts_unavailable_reason,
    contact_revision, contact_display_settings(), contact_data(velocities=...), and
    clear_contact_cache(). Settings return glyph length and force gain. Data returns
    an (N,13) NumPy/CUDA array plus the last physics dt: world position, normal,
    normal impulse on endpoint 0, separation, and relative point velocity (0 minus 1).
    contact_revision changes on step/reset/clear; it must not fetch device data.
    Optional COM inspection exposes supports_com,
    com_positions(bodies=True, articulations=True), clear_com_cache().
    com_positions returns two world-space NumPy (N,3) arrays: body/link centers and
    mass-weighted centers per articulation. Static actors have no COM marker.
    Optional frame inspection exposes supports_frames,
    frame_poses(bodies=True, joints=True), clear_frame_cache().
    frame_poses returns world body/link and child-side joint frames as NumPy (N,7)
    XYZ/XYZW arrays, excluding static actors. Both queries take keyword-only display
    flags; disabled outputs are empty arrays and skip calculation and readback.
    """

    @property
    def supports_dynamic(self) -> bool: ...

    @property
    def dynamic_unavailable_reason(self) -> str: ...

    def body_reference(self, body: Any) -> Any:
        """Return a checked reference suitable for selection and pose queries."""
        ...

    def static_shapes(self) -> list[dict]: ...

    def dynamic_shapes(self) -> tuple[list[dict], list[Any]]: ...

    def body_poses(self, bodies: list[Any]) -> tuple["np.ndarray | Tensor", np.ndarray]:
        """Return world poses (N, 7), NumPy or CUDA, and a CPU validity mask (N,).

        CUDA results must be ready on the current torch stream and retained
        until the next call. Optional clear_pose_cache() releases cached poses.
        """
        ...

    def body_info(self, body: Any) -> dict | None:
        """Return kind, available, shapes; None means the body was removed.

        Optional display_body_idx selects a body from the current dynamic snapshot.
        """
        ...

    def pick(self, origin, direction, distance, statics, dynamics) -> tuple[Any, int] | None:
        """Return the nearest body/shape index, or None for a miss."""
        ...

    def resolve_selection(self, selection) -> tuple[Any, dict]:
        """Return a body and optional path/env_id/obj_id/body_id, or (None, {})."""
        ...
