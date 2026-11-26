"""Supply Newton colliders and COM positions to the shared viewport inspector."""

from __future__ import annotations

from dataclasses import dataclass
import weakref

import newton
import numpy as np
import torch
import warp as wp

from ._dependency import load_raycast_kernel


@wp.kernel
def _body_poses(
    body_ids: wp.array[int],
    body_q: wp.array[wp.transform],
    body_world: wp.array[int],
    offsets: wp.array[wp.vec3],
    layer: wp.transform,
    output: wp.array[wp.transform],
):
    idx = wp.tid()
    body_idx = body_ids[idx]
    pose = body_q[body_idx]
    world_idx = body_world[body_idx]
    if world_idx >= 0 and world_idx < offsets.shape[0]:
        pose = wp.transform(pose.p + offsets[world_idx], pose.q)
    output[idx] = wp.transform_multiply(layer, pose)


@wp.kernel
def _com_positions(
    body_ids: wp.array[int],
    poses: wp.array[wp.transform],
    local_com: wp.array[wp.vec3],
    output: wp.array[wp.vec3],
):
    idx = wp.tid()
    output[idx] = wp.transform_point(poses[idx], local_com[body_ids[idx]])


@wp.func
def _point_velocity(
    body_idx: int,
    point: wp.vec3,
    poses: wp.array[wp.transform],
    velocities: wp.array[wp.spatial_vector],
    centers: wp.array[wp.vec3],
):
    result = wp.vec3()
    if body_idx >= 0:
        velocity = velocities[body_idx]
        center = wp.transform_point(poses[body_idx], centers[body_idx])
        result = wp.spatial_top(velocity) + wp.cross(
            wp.spatial_bottom(velocity), point - center
        )
    return result


@wp.kernel
def _pack_contacts(
    shape0: wp.array[int],
    shape1: wp.array[int],
    points: wp.array[wp.vec3],
    distances: wp.array[float],
    normals: wp.array[wp.vec3],
    forces: wp.array[wp.spatial_vector],
    shape_body: wp.array[int],
    shape_world: wp.array[int],
    poses: wp.array[wp.transform],
    velocities: wp.array[wp.spatial_vector],
    centers: wp.array[wp.vec3],
    offsets: wp.array[wp.vec3],
    visible: wp.array[int],
    layer: wp.transform,
    include_velocity: bool,
    output: wp.array2d[float],
):
    idx = wp.tid()
    a, b = shape0[idx], shape1[idx]
    for col in range(13):
        output[idx, col] = 0.0
    output[idx, 9] = 1.0
    if a < 0 or b < 0 or a == b:
        return
    world = shape_world[a]
    if world < 0:
        world = shape_world[b]
    if visible and world >= 0 and visible[world] == 0:
        return

    point = points[idx]
    # Use Newton's A -> B display normal, force on B, and velocity B - A.
    normal = normals[idx]
    force = wp.vec3()
    if forces:
        force = normal * wp.max(0.0, wp.dot(-wp.spatial_top(forces[idx]), normal))
    relative = wp.vec3()
    if include_velocity:
        relative = _point_velocity(
            shape_body[b], point, poses, velocities, centers
        ) - _point_velocity(shape_body[a], point, poses, velocities, centers)
    if world >= 0:
        point += offsets[world]
    point = wp.transform_point(layer, point)
    normal = wp.transform_vector(layer, normal)
    force = wp.transform_vector(layer, force)
    relative = wp.transform_vector(layer, relative)
    for col in range(3):
        output[idx, col] = point[col]
        output[idx, col + 3] = normal[col]
        output[idx, col + 6] = force[col]
        output[idx, col + 10] = relative[col]
    output[idx, 9] = distances[idx]


@dataclass(frozen=True)
class _BodyReference:
    source: object
    body_idx: int
    shape_idx: int = -1

    @property
    def valid(self):
        return self.source.valid


class NewtonInspectionSource:
    """Read debug data lazily; CUDA collider poses stay on the device.

    Geometry is a snapshot until Refresh. COM markers download positions only
    while enabled. World offsets and the active layer transform match the visuals.
    """

    supports_dynamic = True
    dynamic_unavailable_reason = ""
    supports_com = True
    supports_frames = True
    contact_forces_are_impulses = False

    def __init__(self, visualizer):
        self._visualizer = weakref.ref(visualizer)
        self.model = visualizer.model
        self._shapes = None
        self._shape_bodies = None
        self._references = {}
        self._pose_buffers = {}
        self._display_body_ids = {}
        self._com = None
        self._pick_buffers = {}
        self._frames = None
        self._contact_input = None
        self._contact_state = None
        self._contact_frame = -1
        self._contact_solver = None
        self._contact_buffers = None
        self._contact_cache = None
        self._contact_mapping = None
        self._solver_contacts = None
        self._cpu_contact_buffers = None
        self.contacts_truncated = False

    @property
    def valid(self):
        visualizer = self._visualizer()
        return visualizer is not None and visualizer._inspection_source is self

    @property
    def pose_revision(self):
        visualizer = self._visualizer()
        return (visualizer._frame_version, tuple(visualizer.layer.xform))

    def body_reference(self, body):
        if not isinstance(body, _BodyReference) or body.source is not self:
            raise ValueError("Body belongs to a different Newton inspection source")
        return body

    def _reference(self, body_idx, shape_idx=-1):
        key = (int(body_idx), int(shape_idx) if body_idx < 0 else -1)
        if key not in self._references:
            self._references[key] = _BodyReference(self, *key)
        return self._references[key]

    def _snapshot_shapes(self):
        visualizer = self._visualizer()
        model = self.model
        self._shape_bodies = model.shape_body.numpy()
        types = model.shape_type.numpy()
        scales = model.shape_scale.numpy()
        local_poses = model.shape_transform.numpy()
        flags = model.shape_flags.numpy()
        worlds = model.shape_world.numpy()
        offsets = visualizer.world_offsets.numpy()
        meshes = {
            shape_idx: visualizer._meshes[batch.mesh]
            for batch in visualizer._shape_instances.values()
            for shape_idx in batch.model_shapes
        }
        identity = np.array([0, 0, 0, 0, 0, 0, 1], np.float32)
        # Newton capsules and planes use Z; the inspector primitives use X.
        x_to_z = wp.transform(wp.vec3(), wp.quat(0, -np.sqrt(0.5), 0, np.sqrt(0.5)))
        mesh_payloads = {}
        shapes = {}
        collision_flags = int(
            newton.ShapeFlags.COLLIDE_SHAPES | newton.ShapeFlags.COLLIDE_PARTICLES
        )
        for idx, kind in enumerate(types):
            if not flags[idx] & collision_flags:
                continue
            if not visualizer._should_render_world(int(worlds[idx])):
                continue
            local_pose = local_poses[idx].copy()
            actor = wp.transform_identity()
            if worlds[idx] >= 0:
                actor = wp.transform(wp.vec3(*offsets[worlds[idx]]), wp.quat_identity())
            actor = wp.transform_multiply(visualizer.layer.xform, actor)
            shape = dict(
                type=newton.GeoType(int(kind)).name.lower(),
                shape_idx=idx,
                local_pose=local_pose,
                actor_pose=np.array(actor, np.float32),
            )
            scale = scales[idx]
            if kind == newton.GeoType.BOX:
                shape["half_extents"] = scale
            elif kind == newton.GeoType.SPHERE:
                shape["radius"] = float(scale[0])
            elif kind == newton.GeoType.CAPSULE or (
                kind == newton.GeoType.PLANE and not np.all(scale[:2] > 0)
            ):
                shape["local_pose"] = np.array(
                    wp.transform_multiply(wp.transform_expand(local_pose), x_to_z),
                    np.float32,
                )
                if kind == newton.GeoType.CAPSULE:
                    shape.update(radius=float(scale[0]), half_height=float(scale[1]))
            elif kind in (newton.GeoType.MESH, newton.GeoType.CONVEX_MESH):
                source = model.shape_source[idx]
                shape.update(
                    type="convex_mesh"
                    if kind == newton.GeoType.CONVEX_MESH
                    else "triangle_mesh",
                    vertices=source.vertices,
                    triangles=np.asarray(source.indices).reshape(-1, 3),
                    mesh_scale=scale,
                    mesh_scale_rotation_xyzw=identity[3:],
                )
            elif idx in meshes:
                # Reuse Newton's generated cylinder/cone/ellipsoid/heightfield mesh.
                mesh = meshes[idx]
                if id(mesh) not in mesh_payloads:
                    mesh_payloads[id(mesh)] = (
                        np.asarray(mesh.vertices, np.float32),
                        np.asarray(mesh.indices, np.int32).reshape(-1, 3),
                    )
                vertices, triangles = mesh_payloads[id(mesh)]
                shape.update(
                    type="triangle_mesh",
                    vertices=vertices,
                    triangles=triangles,
                    mesh_scale=np.ones(3),
                    mesh_scale_rotation_xyzw=identity[3:],
                )
            shapes[idx] = shape
        self._shapes = shapes
        self._pick_buffers.clear()

    def static_shapes(self):
        self._snapshot_shapes()
        return [s for idx, s in self._shapes.items() if self._shape_bodies[idx] < 0]

    def dynamic_shapes(self):
        self._snapshot_shapes()
        body_ids = sorted(
            {
                int(self._shape_bodies[idx])
                for idx in self._shapes
                if self._shape_bodies[idx] >= 0
            }
        )
        self._display_body_ids = {
            body_idx: idx for idx, body_idx in enumerate(body_ids)
        }
        shapes = [
            dict(
                s,
                body_idx=self._display_body_ids[int(self._shape_bodies[idx])],
                actor_pose=None,
            )
            for idx, s in self._shapes.items()
            if self._shape_bodies[idx] >= 0
        ]
        return shapes, [self._reference(idx) for idx in body_ids]

    def _poses(self, body_ids):
        visualizer = self._visualizer()
        key = tuple(body_ids)
        state = visualizer._last_state
        stream = visualizer._inspection_stream
        if stream is not None:
            # Do not overwrite a pose buffer still consumed by Torch debug drawing.
            current = torch.cuda.current_stream(wp.device_to_torch(self.model.device))
            wp.stream_to_torch(stream).wait_stream(current)
        with wp.ScopedDevice(self.model.device), wp.ScopedStream(stream):
            if key not in self._pose_buffers:
                self._pose_buffers[key] = (
                    wp.array(body_ids, dtype=int),
                    wp.empty(len(body_ids), dtype=wp.transform),
                )
            ids, output = self._pose_buffers[key]
            if len(body_ids):
                wp.launch(
                    _body_poses,
                    dim=len(body_ids),
                    inputs=[
                        ids,
                        self.model.body_q if state is None else state.body_q,
                        self.model.body_world,
                        visualizer.world_offsets,
                        visualizer.layer.xform,
                    ],
                    outputs=[output],
                )
        return ids, output

    def body_poses(self, bodies):
        _, output = self._poses([body.body_idx for body in bodies])
        valid = np.full(len(bodies), self.valid, dtype=bool)
        if not self.model.device.is_cuda:
            return output.numpy().copy(), valid
        # The inspector consumes poses on Torch's current stream after pre_render.
        stream = self._visualizer()._inspection_stream
        current = torch.cuda.current_stream(wp.device_to_torch(self.model.device))
        current.wait_stream(wp.stream_to_torch(stream))
        return wp.to_torch(output), valid

    def clear_pose_cache(self):
        self._pose_buffers.clear()

    def body_info(self, body):
        if not body.valid:
            return None
        if self._shapes is None:
            self._snapshot_shapes()
        static = body.body_idx < 0
        shapes = (
            [s for idx, s in self._shapes.items() if idx == body.shape_idx]
            if static
            else [
                s
                for idx, s in self._shapes.items()
                if self._shape_bodies[idx] == body.body_idx
            ]
        )
        return dict(
            kind="static" if static else "dynamic",
            available=True,
            shapes=shapes,
            display_body_idx=self._display_body_ids.get(body.body_idx),
        )

    def resolve_selection(self, selection):
        visualizer = self._visualizer()
        if not selection.hit:
            return None, {}
        for batch in visualizer._shape_instances.values():
            instances = visualizer._instances.get(batch.name)
            if instances is None:
                continue
            view = instances.view
            if selection.prim is not None:
                if selection.prim != view.prim:
                    continue
            elif selection.handle != view._render_system._handle(
                view.component, required=False
            ):
                continue
            idx = int(selection.instance_index)
            if 0 <= idx < len(batch.model_shapes):
                shape_idx = batch.model_shapes[idx]
                if self._shape_bodies is None:
                    self._shape_bodies = self.model.shape_body.numpy()
                return self._reference(self._shape_bodies[shape_idx], shape_idx), {}
        return None, {}

    def pick(self, origin, direction, distance, statics, dynamics):
        if self._shapes is None:
            self._snapshot_shapes()
        key = (statics, dynamics)
        if key not in self._pick_buffers:
            shape_ids = [
                idx
                for idx in self._shapes
                if (statics if self._shape_bodies[idx] < 0 else dynamics)
            ]
            with wp.ScopedStream(self._visualizer()._inspection_stream):
                arrays = [
                    wp.array(
                        getattr(self.model, name).numpy()[shape_ids],
                        dtype=dtype,
                        device=self.model.device,
                    )
                    for name, dtype in (
                        ("shape_body", int),
                        ("shape_transform", wp.transform),
                        ("shape_type", int),
                        ("shape_scale", wp.vec3),
                        ("shape_source_ptr", wp.uint64),
                        ("shape_world", int),
                    )
                ]
            self._pick_buffers[key] = shape_ids, arrays
        shape_ids, arrays = self._pick_buffers[key]
        if not shape_ids:
            return None
        visualizer = self._visualizer()
        inverse = wp.transform_inverse(visualizer.layer.xform)
        origin = wp.transform_point(inverse, wp.vec3(*origin))
        direction = wp.transform_vector(inverse, wp.vec3(*direction))
        with (
            wp.ScopedDevice(self.model.device),
            wp.ScopedStream(visualizer._inspection_stream),
        ):
            minimum = wp.array([distance], dtype=float)
            hit = wp.array([-1], dtype=int)
            body = wp.array([-1], dtype=int)
            lock = wp.zeros(1, dtype=int)
            state = visualizer._last_state
            wp.launch(
                load_raycast_kernel(),
                dim=len(shape_ids),
                inputs=[
                    self.model.body_q if state is None else state.body_q,
                    *arrays[:5],
                    origin,
                    direction,
                    lock,
                    minimum,
                    hit,
                    body,
                    arrays[5],
                    visualizer.world_offsets,
                    visualizer._visible_worlds_mask,
                ],
            )
            idx = int(hit.numpy()[0])
        if idx < 0:
            return None
        shape_idx = shape_ids[idx]
        return self._reference(self._shape_bodies[shape_idx], shape_idx), shape_idx

    def com_positions(self, *, bodies=True, articulations=True):
        empty = np.empty((0, 3), np.float32)
        if not bodies and not articulations:
            return empty, empty
        visualizer = self._visualizer()
        if self._com is None:
            mass = self.model.body_mass.numpy()
            worlds = self.model.body_world.numpy()
            body_ids = np.array(
                [
                    idx
                    for idx, value in enumerate(mass)
                    if value > 0 and visualizer._should_render_world(int(worlds[idx]))
                ],
                dtype=np.int32,
            )
            groups = []
            child = self.model.joint_child.numpy()
            rows = {int(body_idx): idx for idx, body_idx in enumerate(body_ids)}
            for start, end in zip(
                self.model.articulation_start.numpy()[:-1],
                self.model.articulation_end.numpy(),
            ):
                group = np.array(
                    [rows[int(idx)] for idx in child[start:end] if idx in rows],
                    dtype=np.int32,
                )
                if len(group):
                    groups.append(group)
            self._com = dict(
                ids=body_ids,
                mass=mass[body_ids],
                groups=groups,
                output=wp.empty(len(body_ids), dtype=wp.vec3, device=self.model.device),
                revision=None,
                result=None,
            )
        cache = self._com
        revision = (self.pose_revision, bodies, articulations)
        if cache["revision"] == revision:
            return cache["result"]
        ids, poses = self._poses(cache["ids"])
        with (
            wp.ScopedDevice(self.model.device),
            wp.ScopedStream(visualizer._inspection_stream),
        ):
            if len(cache["ids"]):
                wp.launch(
                    _com_positions,
                    dim=len(cache["ids"]),
                    inputs=[ids, poses, self.model.body_com],
                    outputs=[cache["output"]],
                )
            # Enabling COM explicitly opts into this small marker-position readback.
            positions = cache["output"].numpy().copy()
        robots = empty
        if articulations:
            robots = np.array(
                [
                    np.average(positions[group], axis=0, weights=cache["mass"][group])
                    for group in cache["groups"]
                ],
                np.float32,
            ).reshape(-1, 3)
        cache["result"] = (positions if bodies else empty, robots)
        cache["revision"] = revision
        return cache["result"]

    def clear_com_cache(self):
        self._com = None

    def clear_frame_cache(self):
        self._frames = None

    @property
    def supports_contacts(self):
        return self._contact_input is not None or self._contact_solver is not None

    contacts_unavailable_reason = "Contact data has not been supplied yet."

    @property
    def supports_contact_forces(self):
        return self._contact_solver is not None or (
            self._contact_input is not None and self._contact_input.force is not None
        )

    @property
    def contact_revision(self):
        return self.pose_revision, self._contact_frame

    def set_contacts(self, contacts, state, *, solver=None):
        if contacts is not self._contact_input or solver is not self._contact_solver:
            self.clear_contact_cache()
        self._contact_input = contacts
        self._contact_solver = solver
        self._contact_state = state
        self._contact_frame = self._visualizer()._frame_version
        self._contact_cache = None

    def contact_display_settings(self):
        visualizer = self._visualizer()
        self.clear_contact_cache()
        # Use half the typical shape radius, matching the PhysX display convention.
        return float(visualizer.scene_scale * 0.5), float(
            visualizer.contact_force_scale
        )

    def clear_contact_cache(self):
        self._contact_buffers = None
        self._contact_cache = None
        self._contact_mapping = None
        self._solver_contacts = None
        self._cpu_contact_buffers = None
        self.contacts_truncated = False

    def _cpu_mujoco_contacts(self):
        import mujoco

        solver = self._contact_solver
        data = solver.mj_data
        count = int(data.ncon)
        if self._contact_mapping is None:
            self._contact_mapping = solver.mjc_geom_to_newton_shape.numpy().reshape(-1)
        shape_ids = self._contact_mapping[np.asarray(data.contact.geom[:count])]
        normals = np.asarray(data.contact.frame[:count, :3], np.float32)
        if (
            self._cpu_contact_buffers is None
            or len(self._cpu_contact_buffers[0]) < count
        ):
            capacity = max(1, count)
            self._cpu_contact_buffers = tuple(
                wp.empty(capacity, dtype=dtype, device="cpu")
                for dtype in (int, int, wp.vec3, float, wp.vec3, wp.spatial_vector)
            )
        buffers = self._cpu_contact_buffers
        for buffer, values in zip(
            buffers[:5],
            (
                shape_ids[:, 0],
                shape_ids[:, 1],
                data.contact.pos[:count],
                data.contact.dist[:count],
                normals,
            ),
        ):
            buffer.numpy()[:count] = values
        force = buffers[5].numpy()[:count]
        force.fill(0)
        local = np.empty(6)
        for idx in range(count):
            mujoco.mj_contactForce(solver.mj_model, data, idx, local)
            force[idx, :3] = -normals[idx] * local[0]
        return count, buffers

    def contact_data(self, *, velocities: bool):
        visualizer = self._visualizer()
        key = (self.contact_revision, velocities)
        if self._contact_frame != visualizer._frame_version:
            return np.empty((0, 13), np.float32), 0.0
        if self._contact_cache is not None and self._contact_cache[0] == key:
            if self.model.device.is_cuda:
                current = torch.cuda.current_stream(
                    wp.device_to_torch(self.model.device)
                )
                current.wait_stream(wp.stream_to_torch(visualizer._inspection_stream))
            return self._contact_cache[1], 0.0
        solver = self._contact_solver
        cpu_mujoco = solver is not None and solver.use_mujoco_cpu
        stream = visualizer._inspection_stream
        if stream is not None:
            current = torch.cuda.current_stream(wp.device_to_torch(self.model.device))
            wp.stream_to_torch(stream).wait_stream(current)
        with wp.ScopedDevice(self.model.device), wp.ScopedStream(stream):
            if cpu_mujoco:
                count, inputs = self._cpu_mujoco_contacts()
                capacity = max(1, count)
            else:
                contacts = self._contact_input
                if solver is not None:
                    if self._solver_contacts is None:
                        self._solver_contacts = newton.Contacts(
                            solver.get_max_contact_count(),
                            0,
                            device=self.model.device,
                            requested_attributes={"force"},
                        )
                    contacts = self._solver_contacts
                    solver.update_contacts(contacts, self._contact_state)
                capacity = contacts.rigid_contact_max
                reported = int(contacts.rigid_contact_count.numpy()[0])
                self.contacts_truncated = reported > capacity
                count = min(reported, capacity)
            if (
                self._contact_buffers is None
                or len(self._contact_buffers[0]) < capacity
            ):
                self._contact_buffers = (
                    wp.empty((capacity, 13), dtype=float),
                    wp.empty(capacity, dtype=wp.vec3),
                    wp.empty(capacity, dtype=float),
                )
            output, points, distances = self._contact_buffers
            if count:
                if not cpu_mujoco:
                    newton.eval_rigid_contact_kinematics(
                        self.model,
                        self._contact_state,
                        contacts,
                        out_point0_world=points,
                        out_distance=distances,
                    )
                    inputs = (
                        contacts.rigid_contact_shape0,
                        contacts.rigid_contact_shape1,
                        points,
                        distances,
                        contacts.rigid_contact_normal,
                        contacts.force,
                    )
                wp.launch(
                    _pack_contacts,
                    dim=count,
                    inputs=[
                        *inputs,
                        self.model.shape_body,
                        self.model.shape_world,
                        self._contact_state.body_q,
                        self._contact_state.body_qd,
                        self.model.body_com,
                        visualizer.world_offsets,
                        visualizer._visible_worlds_mask,
                        visualizer.layer.xform,
                        velocities,
                        output,
                    ],
                )
            if stream is not None:
                current.wait_stream(wp.stream_to_torch(stream))
                result = wp.to_torch(output)[:count]
            else:
                result = output.numpy()[:count].copy()
        self._contact_cache = (key, result)
        return result, 0.0

    def frame_poses(self, *, bodies=True, joints=True):
        """Return body and child-side joint frames, XYZ/XYZW in display coordinates."""
        from ...utils.batched_rotations import quat_xyzw_multiply, quat_xyzw_rotate

        empty = np.empty((0, 7), np.float32)
        if not bodies and not joints:
            return empty, empty
        visualizer = self._visualizer()
        key = (self.pose_revision, bodies, joints)
        if self._frames is None:
            worlds = self.model.body_world.numpy()
            body_ids = [
                idx
                for idx, world in enumerate(worlds)
                if visualizer._should_render_world(int(world))
            ]
            rows = {body_idx: idx for idx, body_idx in enumerate(body_ids)}
            children = self.model.joint_child.numpy()
            enabled = self.model.joint_enabled.numpy()
            joint_ids = [
                idx
                for idx, child in enumerate(children)
                if enabled[idx] and child in rows
            ]
            device = wp.device_to_torch(self.model.device)
            self._frames = dict(
                bodies=[self._reference(idx) for idx in body_ids],
                rows=torch.tensor(
                    [rows[int(children[idx])] for idx in joint_ids],
                    device=device,
                    dtype=torch.long,
                ),
                local=torch.tensor(
                    self.model.joint_X_c.numpy()[joint_ids], device=device
                ),
                key=None,
                result=None,
            )
        cache = self._frames
        if cache["key"] == key:
            return cache["result"]
        poses, _ = self.body_poses(cache["bodies"])
        poses = torch.as_tensor(poses)
        joint_rows = poses[:0]
        if joints:
            owners = poses[cache["rows"]]
            local = cache["local"]
            joint_rows = torch.cat(
                (
                    owners[:, :3] + quat_xyzw_rotate(owners[:, 3:], local[:, :3]),
                    quat_xyzw_multiply(owners[:, 3:], local[:, 3:]),
                ),
                dim=1,
            )
        body_rows = poses if bodies else poses[:0]
        packed = torch.cat((body_rows, joint_rows)).cpu().numpy()
        cache["result"] = (packed[: len(body_rows)], packed[len(body_rows) :])
        cache["key"] = key
        return cache["result"]
