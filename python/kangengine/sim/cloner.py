"""Utilities for constructing spatially separated replicated environments."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from ..utils.env_utils import env_id_list


class GridCloner:
    """Create batched objects on a grid before GPU broadphase initialization.

    Articulations share one immutable native template. PhysX actors remain
    environment-local and receive their final grid pose immediately during
    construction, so a later GPU initialization never observes them stacked
    at the origin.
    """

    def __init__(
        self,
        world,
        *,
        num_envs: int | None = None,
        spacing: float | Sequence[float] = 3.0,
        columns: int | None = None,
    ):
        self.world = world
        self.num_envs = world.num_envs if num_envs is None else int(num_envs)
        if self.num_envs != world.num_envs:
            raise ValueError(
                "GridCloner num_envs must match world.num_envs: "
                f"{self.num_envs} != {world.num_envs}"
            )
        self.spacing = self._normalize_spacing(spacing)
        self.columns = (
            max(1, math.ceil(math.sqrt(self.num_envs)))
            if columns is None
            else int(columns)
        )
        if self.columns < 1:
            raise ValueError("GridCloner columns must be at least 1")

        self.env_origins = self._make_env_origins()
        self.articulation_templates = {}

    def create_articulation_template(self, name: str, data, config=None):
        name = str(name)
        if name in self.articulation_templates:
            raise ValueError(f"articulation template already exists: {name!r}")
        # The template keeps one shared copy of the SkeletonTree, rest-pose
        # global transforms, body/DOF and joint metadata, inertials, and
        # collision descriptors. PhysX articulations, links, shapes, collision
        # groups, commands, and dynamic state are still created per environment.
        template = self.world.create_articulation_template(data, config)
        self.articulation_templates[name] = template
        return template

    def add_articulation(
        self,
        data,
        *,
        obj_id: int = 0,
        name: str = "",
        config=None,
        env_ids=None,
        initial_root_pos=None,
        initial_root_rot=None,
        initial_linear_velocity=None,
        initial_angular_velocity=None,
    ):
        selected = tuple(env_id_list(env_ids, self.num_envs))
        template_name = str(name) or f"articulation_{int(obj_id)}"
        template = self.articulation_templates.get(template_name)
        if template is None:
            template = self.create_articulation_template(
                template_name,
                data,
                config,
            )
        articulation = self.world.add_articulation_batch(
            template,
            obj_id=obj_id,
            name=name,
            config=config,
            env_ids=selected,
        )

        if initial_root_pos is None:
            initial_root_pos = articulation.first.articulation.get_root_position()
        if initial_root_rot is None:
            initial_root_rot = articulation.first.articulation.get_root_rotation()
        root_pos = self._world_positions(selected, initial_root_pos)
        root_rot = self._expand_value(
            initial_root_rot,
            selected,
            width=4,
            name="initial_root_rot",
        )
        linear_velocity = self._optional_value(
            initial_linear_velocity,
            selected,
            width=3,
            name="initial_linear_velocity",
        )
        angular_velocity = self._optional_value(
            initial_angular_velocity,
            selected,
            width=3,
            name="initial_angular_velocity",
        )
        articulation.set_root_state(
            selected,
            root_pos,
            root_rot,
            linear_velocity=linear_velocity,
            angular_velocity=angular_velocity,
            immediate=True,
        )
        return articulation

    def add_rigid(
        self,
        data,
        *,
        obj_id: int = 0,
        name: str = "",
        env_ids=None,
        initial_root_pos=(0.0, 0.0, 0.0),
        initial_root_rot=(0.0, 0.0, 0.0, 1.0),
        initial_linear_velocity=None,
        initial_angular_velocity=None,
        density: float = 1.0,
        contact_offset: float = 0.02,
        rest_offset: float = 0.0,
        kinematic: bool = False,
        static: bool = False,
    ):
        selected = tuple(env_id_list(env_ids, self.num_envs))
        root_pos = self._world_positions(selected, initial_root_pos)
        root_rot = self._expand_value(
            initial_root_rot,
            selected,
            width=4,
            name="initial_root_rot",
        )
        for row, env_id in enumerate(selected):
            kwargs = dict(
                env_id=env_id,
                obj_id=obj_id,
                name=name,
                pos=root_pos[row],
                rot_xyzw=root_rot[row],
                contact_offset=contact_offset,
                rest_offset=rest_offset,
            )
            if static:
                self.world.add_static_rigid(data, **kwargs)
            else:
                self.world.add_rigid(
                    data,
                    density=density,
                    kinematic=kinematic,
                    **kwargs,
                )
        rigid = self.world.get_rigid_batch(
            env_ids=selected,
            obj_id=obj_id,
            name=name,
        )
        if not static:
            rigid.set_root_state(
                selected,
                root_pos,
                root_rot,
                linear_velocity=self._optional_value(
                    initial_linear_velocity,
                    selected,
                    width=3,
                    name="initial_linear_velocity",
                ),
                angular_velocity=self._optional_value(
                    initial_angular_velocity,
                    selected,
                    width=3,
                    name="initial_angular_velocity",
                ),
                immediate=True,
            )
        return rigid

    @staticmethod
    def _normalize_spacing(spacing):
        if isinstance(spacing, Sequence) and not isinstance(spacing, (str, bytes)):
            values = tuple(float(value) for value in spacing)
            if len(values) != 2:
                raise ValueError("GridCloner spacing sequence must contain x and y")
            return values
        value = float(spacing)
        return value, value

    def _make_env_origins(self):
        env_ids = torch.arange(self.num_envs, device=self.world.device)
        origins = torch.zeros(
            (self.num_envs, 3),
            dtype=torch.float32,
            device=self.world.device,
        )
        origins[:, 0] = (
            env_ids.remainder(self.columns).float() * self.spacing[0]
        )
        origins[:, 1] = (
            torch.div(env_ids, self.columns, rounding_mode="floor").float()
            * self.spacing[1]
        )
        return origins

    def _world_positions(self, selected, local_positions):
        local = self._expand_value(
            local_positions,
            selected,
            width=3,
            name="initial_root_pos",
        )
        indices = torch.as_tensor(
            selected,
            dtype=torch.long,
            device=self.world.device,
        )
        return self.env_origins.index_select(0, indices).detach().cpu() + local

    def _optional_value(self, value, selected, *, width: int, name: str):
        if value is None:
            return None
        return self._expand_value(value, selected, width=width, name=name)

    @staticmethod
    def _expand_value(value, selected, *, width: int, name: str):
        tensor = torch.as_tensor(value, dtype=torch.float32).detach().cpu()
        if tensor.ndim == 1 and tensor.shape[0] == width:
            return tensor.view(1, width).repeat(len(selected), 1)
        if tensor.ndim == 2 and tuple(tensor.shape) == (len(selected), width):
            return tensor.contiguous()
        raise ValueError(
            f"{name} must have shape ({width},) or "
            f"({len(selected)}, {width}), got {tuple(tensor.shape)}"
        )
