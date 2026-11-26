"""Provide actual PhysX collision data to the viewport inspector."""

import weakref

import numpy as np

from .._core import _ke
from .wrappers import unwrap_native


class PhysXInspectionSource:
    """Read a PhysicsWorld without owning or stepping its simulation objects.

    Args:
        world: PhysicsWorld retained while this source is connected.
        visualizer: Optional SimWorldVisualizer for scene selection mapping.
        gpu_system: Initialized PhysicsGpuSystem for GPU worlds without a visualizer.
    """

    def __init__(self, world, *, visualizer=None, gpu_system=None):
        self.world = unwrap_native(world)
        if visualizer is not None and visualizer.world.physics is not self.world:
            raise ValueError("visualizer must use the same PhysicsWorld")

        self._visualizer = weakref.ref(visualizer) if visualizer is not None else None
        self._gpu_system = gpu_system
        self._pose_cache = None
        self._pose_stream = None
        self._contact_metadata = None
        self._contact_cache = None
        self._contact_stream = None
        self.contacts_truncated = False
        self._com_metadata = None
        self._com_pose_cache = None
        self._com_cache = None
        self._frame_metadata = None
        self._frame_pose_cache = None
        self._frame_cache = None

    def _gpu(self):
        visualizer = self._visualizer() if self._visualizer is not None else None
        if visualizer is not None and visualizer.is_valid:
            return visualizer.world._gpu_system
        return self._gpu_system

    @property
    def supports_dynamic(self) -> bool:
        gpu = self._gpu()
        return self.world._cpu_collision_poses_available or (
            gpu is not None and gpu.is_initialized()
        )

    @property
    def supports_dynamic_picking(self) -> bool:
        return self.world._cpu_collision_poses_available

    @property
    def dynamic_unavailable_reason(self) -> str:
        return "GPU colliders require an initialized PhysicsGpuSystem."

    @property
    def pose_revision(self):
        return (
            None
            if self.world._cpu_collision_poses_available
            else tuple(self._gpu()._collision_pose_revision)
        )

    def body_reference(self, body):
        body = unwrap_native(body)
        return (
            body
            if isinstance(body, _ke.physics.JointBody)
            else self.world.joint_body(body)
        )

    def static_shapes(self):
        return self.world.get_static_collision_geometry()

    def dynamic_shapes(self):
        self._pose_cache = None
        return self.world._get_dynamic_collision_geometry()

    def clear_pose_cache(self):
        self._pose_cache = None
        self._pose_stream = None

    def body_poses(self, bodies):
        if self.world._cpu_collision_poses_available:
            return self.world._get_collision_body_poses(bodies)

        poses, valid, self._pose_cache = self._gpu_body_poses(bodies, self._pose_cache)
        return poses, valid

    def _gpu_body_poses(self, bodies, cache):
        import torch

        gpu = self._gpu()
        revision = tuple(gpu._collision_pose_revision)
        if any(not body.valid for body in bodies):
            raise RuntimeError(
                "GPU bodies changed; reinitialize PhysicsGpuSystem and refresh the inspector"
            )
        if cache is not None and cache[5] != revision[0]:
            raise RuntimeError("GPU system was reinitialized; refresh the inspector")

        if cache is None or cache[0] is not bodies:
            rows = np.asarray(gpu._collision_body_rows(bodies), dtype=np.int64).reshape(
                -1, 3
            )
            device = gpu.rigid_data().device_id
            groups = []
            for kind in (0, 1):
                ids = np.flatnonzero(rows[:, 0] == kind)
                groups.append(
                    tuple(
                        torch.as_tensor(values, device=f"cuda:{device}")
                        for values in (ids, rows[ids, 1], rows[ids, 2])
                    )
                )
            poses = torch.zeros(
                (len(bodies), 7), dtype=torch.float32, device=f"cuda:{device}"
            )
            poses[:, 6] = 1
            valid = np.ones(len(bodies), dtype=bool)
            cache = (bodies, valid, groups, poses, rows, revision[0])

        _, valid, groups, poses, _, _ = cache
        stream = torch.cuda.current_stream(poses.device)
        if self._pose_stream is not None and self._pose_stream != stream:
            stream.wait_stream(self._pose_stream)
        self._pose_stream = stream

        # Visual batches may already have fetched this completed physics state.
        gpu._fetch_collision_poses(bool(len(groups[0][0])), bool(len(groups[1][0])))

        # Gather on CUDA; the producer must not overwrite poses until the gather finishes.
        for kind, (ids, rows, links) in enumerate(groups):
            if not len(ids):
                continue
            if kind == 0:
                view = gpu.rigid_data()
            else:
                view = gpu.articulation_link_data()
            # Fetch records ready_event_handle on this stream before returning.
            producer = torch.cuda.ExternalStream(
                view.stream_handle, device=poses.device
            )
            if stream != producer:
                stream.wait_stream(producer)
            state = view.torch()
            poses[ids] = state[rows, :7] if kind == 0 else state[rows, links, :7]
            if stream != producer:
                producer.wait_stream(stream)
        return poses, valid, cache

    @property
    def supports_com(self) -> bool:
        return self.supports_dynamic

    def clear_com_cache(self):
        self._com_metadata = None
        self._com_pose_cache = None
        self._com_cache = None

    def com_positions(self, *, bodies=True, articulations=True):
        """Return body/link and articulation COM positions in world space.

        Disabled outputs are empty (0, 3) arrays and are not downloaded.
        Mass/local COM edits require Refresh.
        """
        import torch
        from ..utils.batched_rotations import quat_xyzw_to_matrix

        if not (bodies or articulations):
            return np.empty((0, 3), np.float32), np.empty((0, 3), np.float32)
        revision = self.pose_revision
        key = (revision, bodies, articulations)
        if (
            revision is not None
            and self._com_cache is not None
            and self._com_cache[0] == key
        ):
            return self._com_cache[1:]

        # Cache body masses and local centers until Refresh.
        if self._com_metadata is None:
            body_refs, centers, masses, groups = self.world._com_display_info()
            device = (
                "cpu"
                if revision is None
                else f"cuda:{self._gpu().rigid_data().device_id}"
            )
            self._com_metadata = (
                body_refs,
                torch.as_tensor(
                    np.asarray(centers, dtype=np.float32).reshape(-1, 3), device=device
                ),
                torch.tensor(masses, dtype=torch.float32, device=device),
                torch.tensor(groups, dtype=torch.long, device=device),
                max(groups, default=-1) + 1,
            )

        body_refs, centers, masses, groups, robot_count = self._com_metadata
        if revision is None:
            poses, valid = self.world._get_collision_body_poses(body_refs)
            if (
                self._com_cache is not None
                and self._com_cache[0] == key
                and self._com_pose_cache is not None
                and np.array_equal(poses, self._com_pose_cache[0])
                and np.array_equal(valid, self._com_pose_cache[1])
            ):
                return self._com_cache[1:]
            self._com_pose_cache = (poses, valid)
            poses = torch.from_numpy(poses)
        else:
            poses, valid, self._com_pose_cache = self._gpu_body_poses(
                body_refs, self._com_pose_cache
            )
            for tensor in (centers, masses, groups):
                tensor.record_stream(torch.cuda.current_stream(poses.device))

        positions = poses[:, :3] + (
            quat_xyzw_to_matrix(poses[:, 3:]) @ centers[..., None]
        ).squeeze(-1)

        alive = torch.as_tensor(valid, device=positions.device)
        robot_rows = positions.new_empty((0, 4))
        if articulations and robot_count:
            # Sum each articulation independently, including fixed-base link masses.
            weights = masses * alive
            ids = groups.clamp_min(0)
            robot_weights = torch.where(groups >= 0, weights, 0)
            totals = positions.new_zeros(robot_count)
            sums = positions.new_zeros((robot_count, 3))
            totals.index_add_(0, ids, robot_weights)
            sums.index_add_(0, ids, positions * robot_weights[:, None])
            robots = sums / totals[:, None].clamp_min(1e-12)
            robot_rows = torch.cat((robots, totals[:, None]), dim=1)

        # One readback includes validity so empty groups need no extra synchronization.
        body_rows = (
            torch.cat((positions, alive[:, None]), dim=1)
            if bodies
            else positions.new_empty((0, 4))
        )
        body_count = len(body_rows)
        packed = torch.cat((body_rows, robot_rows), dim=0).cpu().numpy()
        body_rows, robot_rows = packed[:body_count], packed[body_count:]
        result = (
            body_rows[body_rows[:, 3] > 0, :3],
            robot_rows[robot_rows[:, 3] > 0, :3],
        )
        self._com_cache = (key, *result)
        return result

    @property
    def supports_frames(self) -> bool:
        return self.supports_dynamic

    def clear_frame_cache(self):
        self._frame_metadata = None
        self._frame_pose_cache = None
        self._frame_cache = None

    def frame_poses(self, *, bodies=True, joints=True):
        """Return world body and child-side joint frames as NumPy XYZ/XYZW rows.

        Disabled outputs are empty (0, 7) arrays and are not downloaded.
        """
        import torch
        from ..utils.batched_rotations import quat_xyzw_rotate, quat_xyzw_multiply

        if not (bodies or joints):
            return np.empty((0, 7), np.float32), np.empty((0, 7), np.float32)
        revision = self.pose_revision
        key = (revision, bodies, joints)
        if (
            revision is not None
            and self._frame_cache is not None
            and self._frame_cache[0] == key
        ):
            return self._frame_cache[1:]

        # Cache each joint's child-local frame until Refresh.
        if self._frame_metadata is None:
            body_refs, ids, local = self.world._frame_display_info()
            device = (
                "cpu"
                if revision is None
                else f"cuda:{self._gpu().rigid_data().device_id}"
            )
            self._frame_metadata = (
                body_refs,
                np.asarray(ids, dtype=np.int64),
                torch.tensor(ids, device=device, dtype=torch.long),
                torch.as_tensor(
                    np.asarray(local, dtype=np.float32).reshape(-1, 7), device=device
                ),
            )

        body_refs, host_ids, ids, local = self._frame_metadata
        if revision is None:
            poses, valid = self.world._get_collision_body_poses(body_refs)
            if (
                self._frame_cache is not None
                and self._frame_cache[0] == key
                and self._frame_pose_cache is not None
                and np.array_equal(poses, self._frame_pose_cache[0])
                and np.array_equal(valid, self._frame_pose_cache[1])
            ):
                return self._frame_cache[1:]
            self._frame_pose_cache = (poses, valid)
            poses = torch.from_numpy(poses)
        else:
            poses, valid, self._frame_pose_cache = self._gpu_body_poses(
                body_refs, self._frame_pose_cache
            )
            for tensor in (ids, local):
                tensor.record_stream(torch.cuda.current_stream(poses.device))

        # Compose joint frames from the current child-body poses.
        joint_rows = poses[:0]
        if joints:
            owners = poses[ids]
            joint_rows = torch.cat(
                (
                    owners[:, :3] + quat_xyzw_rotate(owners[:, 3:], local[:, :3]),
                    quat_xyzw_multiply(owners[:, 3:], local[:, 3:]),
                ),
                dim=1,
            )
        body_rows = poses if bodies else poses[:0]
        body_count = len(body_rows)
        packed = torch.cat((body_rows, joint_rows)).cpu().numpy()
        result = (
            packed[:body_count][valid] if bodies else packed[:0],
            packed[body_count:][valid[host_ids]] if joints else packed[:0],
        )
        self._frame_cache = (key, *result)
        return result

    def body_info(self, body):
        info = self.world._get_collision_body_geometry(body)
        if (
            info is not None
            and info["kind"] != "static"
            and self._pose_cache is not None
        ):
            row = self._gpu()._collision_body_rows([body])[0]
            ids = np.flatnonzero(np.all(self._pose_cache[4] == row, axis=1))
            info["display_body_idx"] = int(ids[0]) if len(ids) else None
        return info

    @property
    def supports_contacts(self) -> bool:
        if self.world._cpu_collision_poses_available:
            return self.world._contact_reports_enabled
        gpu = self._gpu()
        return (
            gpu is not None
            and gpu.is_initialized()
            and gpu.contact_points().shape[0] > 0
            and gpu.contact_pair_body_refs().shape[0] > 0
        )

    @property
    def contacts_unavailable_reason(self) -> str:
        return (
            "Create the CPU world with enable_contact_reports=True."
            if self.world._cpu_collision_poses_available
            else "GPU contacts require an initialized PhysicsGpuSystem with contact buffers."
        )

    @property
    def contact_revision(self):
        revision = tuple(self.world._contact_revision)
        if not self.world._cpu_collision_poses_available:
            gpu = self._gpu()
            revision += tuple(gpu._collision_pose_revision) + (
                gpu.contact_point_count().version,
            )
        return revision

    def contact_display_settings(self):
        """Calibrate glyph length and force gain once on enable or Refresh."""
        length, force_scale, bodies, centers = self.world._contact_display_info()
        self.clear_contact_cache()
        if not self.world._cpu_collision_poses_available:
            import torch

            gpu = self._gpu()
            rows = np.asarray(gpu._collision_body_rows(bodies), dtype=np.int64).reshape(
                -1, 3
            )
            device = f"cuda:{gpu.contact_points().device_id}"
            rigid = torch.zeros(
                (gpu.rigid_data().shape[0], 3), device=device, dtype=torch.float32
            )
            links = torch.zeros(
                (*gpu.articulation_link_data().shape[:2], 3),
                device=device,
                dtype=torch.float32,
            )
            centers = np.asarray(centers, dtype=np.float32).reshape(-1, 3)
            for kind, target in ((0, rigid), (1, links)):
                ids = np.flatnonzero(rows[:, 0] == kind)
                if not len(ids):
                    continue
                row = torch.as_tensor(rows[ids, 1], device=device)
                values = torch.as_tensor(centers[ids], device=device)
                if kind == 0:
                    target[row] = values
                else:
                    link = torch.as_tensor(rows[ids, 2], device=device)
                    target[row, link] = values
            self._contact_metadata = (
                tuple(gpu._collision_pose_revision)[0],
                rigid,
                links,
            )
            self._contact_stream = torch.cuda.current_stream(device)
        return length, force_scale

    def clear_contact_cache(self):
        self._contact_metadata = None
        self._contact_cache = None
        self._contact_stream = None
        self.contacts_truncated = False

    def contact_data(self, *, velocities: bool):
        """Return [position, normal, impulse, separation, relative velocity] (N,13).

        Values are world-space; impulse is the normal impulse on endpoint 0.
        GPU arrays stay on CUDA; only the contact count is read on the CPU.
        Relative velocity is post-step velocity at the contact point (0 minus 1).
        """
        revision = self.contact_revision
        if (
            self._contact_cache is not None
            and self._contact_cache[0] == revision
            and (self._contact_cache[1] or not velocities)
        ):
            if self._contact_stream is not None:
                import torch

                current = torch.cuda.current_stream(self._contact_stream.device)
                if current != self._contact_stream:
                    current.wait_stream(self._contact_stream)
                    self._contact_cache[2].record_stream(current)
            return self._contact_cache[2], self.world._contact_dt

        if self.world._cpu_collision_poses_available:
            data = self.world._get_contact_display_data(velocities)
        else:
            data = self._gpu_contact_data(velocities)
        self._contact_cache = (self.contact_revision, velocities, data)
        return data, self.world._contact_dt

    def _gpu_contact_data(self, velocities):
        import torch
        from ..utils.batched_rotations import quat_xyzw_to_matrix

        gpu = self._gpu()
        if self._contact_metadata[0] != tuple(gpu._collision_pose_revision)[0]:
            raise RuntimeError("GPU system was reinitialized; refresh the inspector")

        gpu._fetch_inspection_contacts(False)
        view = gpu.contact_points()
        device = f"cuda:{view.device_id}"
        stream = torch.cuda.current_stream(device)
        producer = torch.cuda.ExternalStream(view.stream_handle, device=device)
        if self._contact_stream is not None and stream != self._contact_stream:
            stream.wait_stream(self._contact_stream)
            for centers in self._contact_metadata[1:]:
                centers.record_stream(stream)
        if stream != producer:
            stream.wait_stream(producer)

        # One counter readback avoids drawing empty capacity and detects both overflow limits.
        if (
            self._contact_cache is not None
            and self._contact_cache[0] == self.contact_revision
        ):
            count = len(self._contact_cache[2])
        else:
            count, pairs = (
                torch.stack(
                    (
                        gpu.contact_point_count().torch()[0],
                        gpu.contact_pair_count().torch()[0],
                    )
                )
                .cpu()
                .tolist()
            )
            self.contacts_truncated = (
                count > view.shape[0] or pairs > gpu.contact_pair_body_refs().shape[0]
            )
        count = min(count, view.shape[0])
        data = torch.zeros((count, 13), device=device, dtype=torch.float32)
        data[:, :10] = view.torch()[:count]

        # Relative point velocity is needed only for contact-mode display.
        if velocities and count:
            gpu._fetch_inspection_contacts(True)
            if stream != producer:
                stream.wait_stream(producer)
            pairs = gpu.contact_point_pair_indices().torch()[:count].long()
            refs = gpu.contact_pair_body_refs().torch().reshape(-1, 2, 3)[pairs]
            for kind, state_view, centers in (
                (0, gpu.rigid_data(), self._contact_metadata[1]),
                (1, gpu.articulation_link_data(), self._contact_metadata[2]),
            ):
                if not centers.numel():
                    continue
                state = state_view.torch()
                for endpoint, sign in ((0, 1), (1, -1)):
                    ref = refs[:, endpoint]
                    row = ref[:, 1].long().clamp(0, state.shape[0] - 1)
                    if kind == 0:
                        values, local_com = state[row], centers[row]
                    else:
                        link = ref[:, 2].long().clamp(0, state.shape[1] - 1)
                        values, local_com = state[row, link], centers[row, link]
                    rotation = quat_xyzw_to_matrix(values[:, 3:7])
                    com = values[:, :3] + (rotation @ local_com[..., None]).squeeze(-1)
                    velocity = values[:, 7:10] + torch.cross(
                        values[:, 10:13],
                        data[:, :3] - com,
                        dim=-1,
                    )
                    data[:, 10:13] += sign * torch.where(
                        (ref[:, 0] == kind)[:, None],
                        velocity,
                        0.0,
                    )

        if stream != producer:
            producer.wait_stream(stream)
        self._contact_stream = stream
        return data

    def pick(self, origin, direction, distance, statics, dynamics):
        return self.world._pick_collision_body(
            origin,
            direction,
            distance,
            statics,
            dynamics and self.supports_dynamic_picking,
        )

    def resolve_selection(self, selection):
        visualizer = self._visualizer() if self._visualizer is not None else None
        if visualizer is None or not visualizer.is_valid:
            return None, {}
        picked = visualizer.pick_body(selection)
        if picked is None or picked.env_id is None:
            return None, {}
        world = visualizer.world
        key = (picked.env_id, picked.obj_id)
        record = world.articulations.get(key)
        if record is not None:
            body = record.articulation.link(picked.body_id)
        else:
            record = world.rigids.get(key) or world.static_rigids.get(key)
            if record is None:
                return None, {}
            body = self.world.joint_body(record.rigid)
        return body, {
            "env_id": picked.env_id,
            "obj_id": picked.obj_id,
            "body_id": picked.body_id,
        }
