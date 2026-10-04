import numpy as np
import torch

from ..utils import to_gpu_array_view
from .wrappers import D6Axis, D6JointConfig, JointBody


class D6Batch:
    """Python batch interface over individual D6 handles, with batched CUDA readback.

    Removing an endpoint invalidates only the joints connected to it.
    The caller owns environment/limb-to-index mapping and simulation timing.
    """

    def __init__(self, joints):
        self.joints = tuple(joints)
        self._gpu_system = None
        self._wrenches = None

    @classmethod
    def create_world_frames(cls, world, articulations, frame_names, *, world_frames, config=None):
        """Attach named model frames to world poses using their physical owners.

        Example: attach a robot's named frame at a world-space xyz/xyzw pose::

            joints = D6Batch.create_world_frames(
                physics_world, [robot], ["attachment_frame"],
                world_frames=[[0, 0, 0, 0, 0, 0, 1]],
            )

        Resolves the owning body and local frame offset automatically.
        """
        if isinstance(frame_names, str):
            raise TypeError("frame_names must be a sequence of names")
        articulations = tuple(articulations)
        names = tuple(frame_names)
        if len(articulations) != len(names):
            raise ValueError("articulations and frame_names must have equal lengths")
        indices, poses = [], []
        for articulation, name in zip(articulations, names):
            if articulation.num_links() == 0:
                raise RuntimeError("Articulation has been released")
            frame = next((f for f in articulation.template.fixed_frames if f.name == name), None)
            if frame is None:
                raise KeyError(name)
            indices.append(frame.body_index)
            poses.append([frame.pos.x, frame.pos.y, frame.pos.z,
                          frame.quat.x, frame.quat.y, frame.quat.z, frame.quat.w])
        return cls.create_world_links(world, articulations, indices,
                                      world_frames=world_frames, link_frames=poses, config=config)

    @staticmethod
    def _frames(frames, count):
        if isinstance(frames, torch.Tensor):
            values = frames.detach().to(device="cpu", dtype=torch.float32)
            if values.shape != (count, 7) or not torch.isfinite(values).all().item():
                raise ValueError("frames must be finite host [N,7] xyz/xyzw poses")
            if ((values[:, 3:].square().sum(dim=1) - 1).abs() > 1e-3).any().item():
                raise ValueError("frame quaternions must be normalized")
            # Convert the whole batch once, not each row inside the setter loop.
            return values.tolist()
        values = np.asarray(frames, dtype=np.float32)
        if values.shape != (count, 7) or not np.isfinite(values).all():
            raise ValueError("frames must be finite host [N,7] xyz/xyzw poses")
        if np.any(np.abs(np.sum(values[:, 3:] ** 2, axis=1) - 1) > 1e-3):
            raise ValueError("frame quaternions must be normalized")
        return values

    @classmethod
    def create_world_links(
        cls,
        world,
        articulations,
        link_indices,
        *,
        world_frames,
        link_frames,
        config=None,
    ):
        """Connect each link to the fixed world.

        world_frames: world-space attachment poses; link_frames: link-local poses.
        Both use [x, y, z, qx, qy, qz, qw]. CPU/CUDA tensors are accepted;
        CUDA frames are copied to CPU before setting PhysX constraints.
        """
        count = len(articulations)
        if len(link_indices) != count:
            raise ValueError("articulations and link_indices must have equal lengths")
        bodies = [a.link(i) for a, i in zip(articulations, link_indices)]
        return cls.create(
            world,
            [None] * count,
            bodies,
            frames0=world_frames,
            frames1=link_frames,
            config=config,
        )

    @classmethod
    def create(cls, world, bodies0, bodies1, *, frames0, frames1, config=None):
        """Connect paired body/link handles; None denotes the fixed world.

        Frames are [N,7] xyz/xyzw poses in each endpoint's local coordinates
        (world coordinates for None). Explicit frames override config frames.
        Raw rigid bodies are also accepted and converted to endpoint handles.
        Torch frames are detached and converted to CPU float32, including CUDA
        inputs, then converted to Python lists once per frame batch. This device
        transfer synchronizes before the CPU PhysX calls.
        """
        if len(bodies0) != len(bodies1):
            raise ValueError("bodies0 and bodies1 must have equal lengths")
        count = len(bodies0)
        frames0 = cls._frames(frames0, count)
        frames1 = cls._frames(frames1, count)

        def endpoint(body):
            return (
                body
                if body is None or isinstance(body, JointBody)
                else world.joint_body(body)
            )

        endpoints0 = [endpoint(body) for body in bodies0]
        endpoints1 = [endpoint(body) for body in bodies1]
        source = D6JointConfig() if config is None else config
        joints = []
        try:
            for body0, body1, frame0, frame1 in zip(
                endpoints0, endpoints1, frames0, frames1
            ):
                item = D6JointConfig(source)
                item.frame0, item.frame1 = frame0, frame1
                joints.append(world.create_d6_joint(body0, body1, config=item))
        except Exception:
            for joint in joints:
                joint.release()
            raise
        return cls(joints)

    @property
    def size(self):
        return len(self.joints)

    @property
    def enabled(self):
        return [joint.enabled for joint in self.joints]

    def _select(self, indices):
        indices = list(indices)
        if any(not isinstance(index, (int, np.integer)) for index in indices):
            raise TypeError("joint indices must be integers")
        indices = [int(index) for index in indices]
        if len(set(indices)) != len(indices) or any(
            i < 0 or i >= self.size for i in indices
        ):
            raise ValueError("joint indices must be unique and in range")
        selected = [self.joints[i] for i in indices]
        if any(not joint.valid for joint in selected):
            raise RuntimeError("selected joint or endpoint was released")
        return selected

    def set_enabled(self, indices, enabled):
        selected = self._select(indices)
        if enabled and any(joint.broken for joint in selected):
            raise RuntimeError("a broken D6 must be released and recreated")
        for joint in selected:
            joint.set_enabled(enabled)

    def set_world_frames(self, indices, frames):
        """World-space frame0 updates for batches created with create_world_links."""
        self.set_frames0(indices, frames)

    def set_link_frames(self, indices, frames):
        """Link-local frame1 updates for batches created with create_world_links."""
        self.set_frames1(indices, frames)

    def set_frames0(self, indices, frames):
        """Update endpoint 0 poses; Torch inputs are detached/copied to CPU."""
        selected = self._select(indices)
        frames = self._frames(frames, len(selected))
        for joint, frame in zip(selected, frames):
            joint.set_frame0(frame)

    def set_frames1(self, indices, frames):
        """Update endpoint 1 poses; Torch inputs are detached/copied to CPU."""
        selected = self._select(indices)
        frames = self._frames(frames, len(selected))
        for joint, frame in zip(selected, frames):
            joint.set_frame1(frame)

    def _configure(self, indices, change, method, *args):
        selected = self._select(indices)
        # Validate every proposed configuration before mutating any selected joint.
        for joint in selected:
            config = joint.config
            change(config)
            config.validate()
        for joint in selected:
            getattr(joint, method)(*args)

    def set_motion(self, indices, axis, motion):
        if not isinstance(axis, D6Axis) or not 0 <= int(axis) < 6:
            raise ValueError("invalid D6 axis")
        def change(config):
            motions = list(config.motions)
            motions[int(axis)] = motion
            config.motions = motions
        self._configure(indices, change, "set_motion", axis, motion)

    def set_linear_limit(self, indices, axis, lower, upper):
        if axis not in (D6Axis.X, D6Axis.Y, D6Axis.Z):
            raise ValueError("linear limits require X, Y, or Z")
        def change(config):
            limits = list(config.linear_limits)
            limits[int(axis)] = [lower, upper]
            config.linear_limits = limits
        self._configure(indices, change, "set_linear_limit", axis, lower, upper)

    def set_twist_limit(self, indices, lower, upper):
        self._configure(indices, lambda c: setattr(c, "twist_limits", [lower, upper]),
                        "set_twist_limit", lower, upper)

    def set_swing_limit(self, indices, y_angle, z_angle):
        self._configure(indices, lambda c: setattr(c, "swing_limits", [y_angle, z_angle]),
                        "set_swing_limit", y_angle, z_angle)

    def set_drive(self, indices, axis, config):
        def change(item):
            drives = dict(item.drives)
            drives[axis] = config
            item.drives = drives
        self._configure(indices, change, "set_drive", axis, config)

    def set_drive_targets(self, indices, poses):
        selected = self._select(indices)
        poses = self._frames(poses, len(selected))
        for joint, pose in zip(selected, poses):
            joint.set_drive_target(pose)

    def set_drive_velocity(self, indices, linear, angular):
        def change(config):
            config.drive_linear_velocity = linear
            config.drive_angular_velocity = angular
        self._configure(indices, change, "set_drive_velocity", linear, angular)

    def set_break_force(self, indices, force, torque):
        def change(config):
            config.break_force, config.break_torque = force, torque
        self._configure(indices, change, "set_break_force", force, torque)

    def get_wrenches(self, indices=None):
        """Return host float32 [N,6] wrenches; Direct GPU reads synchronize per joint."""
        selected = self._select(range(self.size) if indices is None else indices)
        return np.asarray([joint.get_wrench() for joint in selected], dtype=np.float32).reshape(-1, 6)

    def fetch_wrenches(self, *, gpu_system=None):
        """Refresh a reusable Torch CUDA [N,6] tensor without host force copies.

        Pass an initialized PhysicsGpuSystem on the first call. Later calls
        reuse that system and the output tensor. Operations run on the current
        Torch stream. Use the same stream for consumers, or synchronize them
        before overwriting this shared tensor in another stream.
        """
        system = self._gpu_system if gpu_system is None else gpu_system
        if system is None:
            raise RuntimeError("pass an initialized gpu_system on the first fetch_wrenches call")
        system.check_initialized()
        device = torch.device("cuda", system.rigid_data().device_id)
        output = self._wrenches
        if output is None or output.device != device or output.shape != (self.size, 6):
            output = torch.empty((self.size, 6), dtype=torch.float32, device=device)
        stream = torch.cuda.current_stream(device)
        view = to_gpu_array_view(output, name="d6_wrenches")
        view.stream_handle = stream.cuda_stream
        # Native work writes into Torch's allocation on this stream, so retained
        # tensors survive batch/system teardown and the allocator tracks reuse.
        output.record_stream(stream)
        system.fetch_d6_wrenches(self.joints, view)
        self._gpu_system = system
        self._wrenches = output

    @property
    def wrenches(self):
        """Torch CUDA [N,6] from the last fetch; clone() to preserve a snapshot."""
        if self._wrenches is None:
            raise RuntimeError("call fetch_wrenches before accessing wrenches")
        return self._wrenches

    @property
    def broken(self):
        return [joint.broken for joint in self.joints]

    def release(self):
        for joint in self.joints:
            joint.release()
        self._wrenches = None
        self._gpu_system = None
