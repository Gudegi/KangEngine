"""Data contract between simulation providers and the viewport inspector."""

from typing import Any, Protocol, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from torch import Tensor


class PhysicsInspectionSource(Protocol):
    """Supply physics debug data to the viewport inspector after simulation fetch.

    Queries never step physics. Poses use xyz/xyzw in scene units.
    The methods below cover colliders and selection; other features are optional.

    Contacts (optional):
        - supports_contacts and contacts_unavailable_reason control availability.
        - contact_revision changes on step/reset/clear without device queries.
        - contact_display_settings() returns glyph length and force gain.
        - clear_contact_cache() releases cached contact data.

        contact_data(velocities=...) returns an (N, 13) NumPy/CUDA array and
        the last physics dt in seconds. Columns are:
        - 0:3: world position
        - 3:6: world normal
        - 6:9: normal impulse (N s), or force (N) as specified below
        - 9: separation
        - 10:13: relative point velocity, receiving endpoint minus the other

        The provider chooses the receiving endpoint. Normal, force, and relative
        velocity must all use that endpoint.

        supports_contact_forces=False disables force and estimated contact modes.
        contact_forces_are_impulses=False supplies force directly; dt is unused.
        Both flags default to True.

    Centers of mass (optional):
        - supports_com enables COM display.
        - com_positions(*, bodies=True, articulations=True) returns two world-space
          NumPy (N, 3) arrays: body/link centers and mass-weighted articulation centers.
        - clear_com_cache() releases cached COM data.

    Body and joint frames (optional):
        - supports_frames enables frame display.
        - frame_poses(*, bodies=True, joints=True) returns two world-space NumPy
          (N, 7) arrays: body/link poses and child-side joint poses.
        - clear_frame_cache() releases cached frame data.

    COM and frame queries exclude static actors. Disabled outputs are empty
    arrays and skip calculation and readback.
    """

    @property
    def supports_dynamic(self) -> bool: ...

    @property
    def dynamic_unavailable_reason(self) -> str: ...

    def body_reference(self, body: Any) -> Any:
        """Return a checked reference for selection and pose queries.

        Its cheap ``valid`` property becomes False after removal.
        """
        ...

    def static_shapes(self) -> list[dict]:
        """Return static collider snapshots in the shared shape format.

        Shape fields: type, shape_idx, local_pose, actor_pose, and geometry data.
        Capsules and plane normals use local X. Meshes supply vertices, triangles,
        mesh_scale, and mesh_scale_rotation_xyzw. Heightfield vertices are scaled.
        """
        ...

    def dynamic_shapes(self) -> tuple[list[dict], list[Any]]:
        """Return shape snapshots and body references.

        Uses the static_shapes() format; body_idx indexes the returned references.
        GPU snapshots keep actor-local geometry and actor_pose=None, without pose
        readback. The GPU body list stays fixed until Refresh or reinitialization.
        """
        ...

    def body_poses(self, bodies: list[Any]) -> tuple["np.ndarray | Tensor", np.ndarray]:
        """Return world poses (N, 7), NumPy or CUDA, and a CPU validity mask (N,).

        CUDA results must be ready on the current torch stream and retained
        until the next call; never silently download GPU poses.
        Optional pose_revision skips unchanged poses; None keeps CPU pose polling.
        Optional clear_pose_cache() releases cached poses.
        """
        ...

    def body_info(self, body: Any) -> dict | None:
        """Return kind, available, shapes; None means the body was removed.

        Optional display_body_idx selects a body from the current dynamic snapshot.
        """
        ...

    def pick(self, origin, direction, distance, statics, dynamics) -> tuple[Any, int] | None:
        """Return the nearest body/shape index, or None for a miss.

        Set supports_dynamic_picking=False when GPU collider raycasts are unavailable.
        """
        ...

    def resolve_selection(self, selection) -> tuple[Any, dict]:
        """Return a body and optional path/env_id/obj_id/body_id, or (None, {})."""
        ...
