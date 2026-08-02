"""Sensors bound to the high-level simulation world."""

from __future__ import annotations

from dataclasses import dataclass

import torch


def _contact_sensor_aggregator():
    """Return the optional native CUDA contact aggregator."""
    from .._core import _ke

    aggregator = getattr(
        getattr(_ke, "physics", None),
        "aggregate_contact_sensors_cuda",
        None,
    )
    if aggregator is None:
        raise RuntimeError("KangEngine was built without CUDA contact aggregation")
    return aggregator


@dataclass(frozen=True, slots=True)
class _ContactSensorDescriptor:
    body_kind: int
    row_map_offset: int
    row_map_count: int
    body_map_offset: int
    body_map_count: int
    output_offset: int
    environment_count: int
    body_count: int

    def packed(self):
        return (
            self.body_kind,
            self.row_map_offset,
            self.row_map_count,
            self.body_map_offset,
            self.body_map_count,
            self.output_offset,
            self.environment_count,
            self.body_count,
        )

    @property
    def output_end(self):
        return self.output_offset + self.environment_count * self.body_count


@dataclass(slots=True)
class ContactSensorData:
    """Per-environment, per-body Torch contact state.

    Shapes:
        contact_count: ``(N, B)``.
        in_contact: ``(N, B)``.
        net_impulse: ``(N, B, 3)``.
        net_force: ``(N, B, 3)``.
    """

    contact_count: torch.Tensor
    in_contact: torch.Tensor
    net_impulse: torch.Tensor
    sim_dt: float

    @property
    def net_force(self) -> torch.Tensor:
        """Shape: ``(N, B, 3)``."""
        return self.net_impulse / self.sim_dt


class ContactSensor:
    """GPU contact sensor attached to a rigid or articulation object view."""

    requires_contact_data = True

    def __init__(self, world, target, body_ids=None, *, name: str = ""):
        if not getattr(world, "_uses_gpu_sim", False):
            raise RuntimeError("ContactSensor currently requires sim_device='cuda'")
        self.world = world
        self.target = target
        self.name = str(name)
        self.env_ids = tuple(int(value) for value in target.env_ids)
        self.obj_id = int(target.obj_id)
        self.body_ids = self._normalize_body_ids(body_ids)
        self._kind = self._resolve_kind()
        self._contact_count = None
        self._in_contact = None
        self._net_impulse = None

    @property
    def data(self) -> ContactSensorData:
        if self._contact_count is None:
            raise RuntimeError(
                f"contact sensor {self.name!r} has not been refreshed yet"
            )
        return ContactSensorData(
            self._contact_count,
            self._in_contact,
            self._net_impulse,
            float(self.world.sim_dt),
        )

    @property
    def contact_count(self) -> torch.Tensor:
        """Shape: ``(N, B)``."""
        return self.data.contact_count

    @property
    def in_contact(self) -> torch.Tensor:
        """Shape: ``(N, B)``."""
        return self.data.in_contact

    @property
    def net_impulse(self) -> torch.Tensor:
        """Shape: ``(N, B, 3)``."""
        return self.data.net_impulse

    @property
    def net_force(self) -> torch.Tensor:
        """Shape: ``(N, B, 3)``."""
        return self.data.net_force

    def refresh(self) -> ContactSensorData:
        """Refresh the world sensor batch and return this sensor's CUDA data."""
        self.world.refresh_sensors()
        return self.data

    def release(self):
        self.world = None
        self.target = None
        self._contact_count = None
        self._in_contact = None
        self._net_impulse = None

    def _bind_outputs(self, contact_count, in_contact, net_impulse):
        shape = (len(self.env_ids), len(self.body_ids))
        self._contact_count = contact_count.reshape(shape)
        self._in_contact = in_contact.reshape(shape)
        self._net_impulse = net_impulse.reshape((*shape, 3))

    def _gpu_rows(self):
        if self._kind == 0:
            return tuple(
                self.world.rigid_gpu_row(env_id, self.obj_id) for env_id in self.env_ids
            )
        return tuple(
            self.world.articulation_gpu_row(env_id, self.obj_id)
            for env_id in self.env_ids
        )

    def _normalize_body_ids(self, body_ids):
        if body_ids is None:
            values = tuple(range(int(self.target.num_bodies)))
        else:
            values = tuple(int(value) for value in body_ids)
        if not values:
            raise ValueError("ContactSensor requires at least one body id")
        if len(set(values)) != len(values):
            raise ValueError("ContactSensor body ids must be unique")
        num_bodies = int(self.target.num_bodies)
        if any(value < 0 or value >= num_bodies for value in values):
            raise IndexError(
                f"ContactSensor body ids must be in [0, {num_bodies}), got {values}"
            )
        return values

    def _resolve_kind(self) -> int:
        keys = {(env_id, self.obj_id) for env_id in self.env_ids}
        if keys.issubset(self.world.rigids):
            return 0
        if keys.issubset(self.world.articulations):
            return 1
        raise KeyError("ContactSensor target is not registered in its KangSimWorld")


class ForceSensor(ContactSensor):
    """Normal contact-force sensor backed by PhysX solver impulses.

    This does not include tangential friction impulse or a complete wrench.
    """

    @property
    def force(self) -> torch.Tensor:
        """Shape: ``(N, B, 3)``."""
        return self.net_force

    @property
    def impulse(self) -> torch.Tensor:
        """Shape: ``(N, B, 3)``."""
        return self.net_impulse


class _ContactSensorBatch:
    """World-owned packed storage and one-pass CUDA contact aggregation."""

    def __init__(self, world):
        self.world = world
        self._dirty = True
        self._torch = None
        self._native_views = None
        self._storage = None

    def mark_dirty(self):
        self._dirty = True

    def refresh(self, sensors) -> None:
        if self._dirty:
            self._prepare(tuple(sensors))

        stream = int(
            self._torch.cuda.current_stream(self._storage[3].device).cuda_stream
        )
        for view in self._native_views:
            view.stream_handle = stream
        _contact_sensor_aggregator()(
            self.world.gpu_system.contact_pair_body_refs(),
            self.world.gpu_system.contact_pair_count(),
            self.world.gpu_system.contact_points(),
            self.world.gpu_system.contact_point_count(),
            self.world.gpu_system.contact_point_pair_indices(),
            *self._native_views,
        )

    def clear_outputs(self, sensors) -> None:
        if self._dirty:
            self._prepare(tuple(sensors))
        self._storage[3].zero_()
        self._storage[4].zero_()
        self._storage[5].zero_()

    def release(self):
        self.world = None
        self._torch = None
        self._native_views = None
        self._storage = None
        self._dirty = True

    def _prepare(self, sensors):
        import torch

        from ..utils import to_gpu_array_view

        if not sensors:
            raise RuntimeError("contact sensor batch requires at least one sensor")
        _contact_sensor_aggregator()

        device_id = int(self.world.gpu_system.contact_pair_body_refs().device_id)
        device = torch.device(f"cuda:{device_id}")
        descriptors = []
        row_maps = []
        body_maps = []
        output_offset = 0
        for sensor in sensors:
            rows = sensor._gpu_rows()
            row_map = [-1] * (max(rows) + 1)
            for environment, row in enumerate(rows):
                row_map[row] = environment
            body_map = [-1] * (max(sensor.body_ids) + 1)
            for slot, body in enumerate(sensor.body_ids):
                body_map[body] = slot

            descriptors.append(
                _ContactSensorDescriptor(
                    body_kind=sensor._kind,
                    row_map_offset=len(row_maps),
                    row_map_count=len(row_map),
                    body_map_offset=len(body_maps),
                    body_map_count=len(body_map),
                    output_offset=output_offset,
                    environment_count=len(rows),
                    body_count=len(sensor.body_ids),
                )
            )
            row_maps.extend(row_map)
            body_maps.extend(body_map)
            output_offset += len(rows) * len(sensor.body_ids)

        descriptor_tensor = torch.tensor(
            [descriptor.packed() for descriptor in descriptors],
            dtype=torch.int32,
            device=device,
        )
        row_map_tensor = torch.tensor(row_maps, dtype=torch.int32, device=device)
        body_map_tensor = torch.tensor(body_maps, dtype=torch.int32, device=device)
        contact_count = torch.zeros(output_offset, dtype=torch.int32, device=device)
        in_contact = torch.zeros(output_offset, dtype=torch.bool, device=device)
        net_impulse = torch.zeros(
            (output_offset, 3), dtype=torch.float32, device=device
        )

        for sensor, descriptor in zip(sensors, descriptors):
            begin = descriptor.output_offset
            end = descriptor.output_end
            sensor._bind_outputs(
                contact_count[begin:end],
                in_contact[begin:end],
                net_impulse[begin:end],
            )

        self._torch = torch
        self._storage = (
            descriptor_tensor,
            row_map_tensor,
            body_map_tensor,
            contact_count,
            in_contact,
            net_impulse,
        )
        self._native_views = tuple(
            to_gpu_array_view(tensor, dtype=tensor.dtype, name=name)
            for tensor, name in zip(
                self._storage,
                (
                    "contact_sensor_descriptors",
                    "contact_sensor_row_maps",
                    "contact_sensor_body_maps",
                    "contact_sensor_counts",
                    "contact_sensor_masks",
                    "contact_sensor_net_impulses",
                ),
            )
        )
        self._dirty = False
