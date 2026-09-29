"""Procedural articulation authoring using the same description as asset import."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from .._core import _ke
from .wrappers import ArticulationConfig, ArticulationTemplate

if TYPE_CHECKING:
    from ..asset import ArticulationDesc

_Vec3 = tuple[float, float, float]
_Quat = tuple[float, float, float, float]
_Limits = tuple[float, float]


class ArticulationBuilder:
    """Author a tree before simulation; build() returns an independent description.

    Link poses are parent-local at zero joint position. Joint axes and joint offsets
    are child-link local. Inertia is diagonal about com, aligned with the link.
    Fixed children use joint=None or ArticulationBuilder.FixedJoint(). Closed loops
    use external joints after instantiation, not an extra parent in this tree.
    """

    @dataclass(frozen=True)
    class RevoluteJoint:
        axis: _Vec3 = (0., 0., 1.)
        limits: _Limits = (-math.pi, math.pi)
        joint_offset: _Vec3 = (0., 0., 0.)
        kp: float = 0.
        kd: float = 0.

    @dataclass(frozen=True)
    class PrismaticJoint:
        axis: _Vec3 = (0., 0., 1.)
        limits: _Limits = (-1., 1.)
        joint_offset: _Vec3 = (0., 0., 0.)
        kp: float = 0.
        kd: float = 0.

    @dataclass(frozen=True)
    class SphericalJoint:
        limits: tuple[_Limits, _Limits, _Limits] = ((-math.pi, math.pi),) * 3
        joint_offset: _Vec3 = (0., 0., 0.)
        kp: float = 0.
        kd: float = 0.

    @dataclass(frozen=True)
    class FixedJoint:
        pass

    @dataclass(frozen=True)
    class BoxShape:
        half_extents: _Vec3
        position: _Vec3 = (0., 0., 0.)
        rotation_xyzw: _Quat = (0., 0., 0., 1.)

    @dataclass(frozen=True)
    class SphereShape:
        radius: float
        position: _Vec3 = (0., 0., 0.)

    def __init__(self) -> None:
        self._names, self._parents, self._positions, self._rotations = [], [], [], []
        self._joints, self._shapes, self._inertials = {}, {}, {}

    def add_link(
        self, name: str, *, parent: str | None = None,
        position: ArrayLike = (0., 0., 0.),
        rotation_xyzw: ArrayLike = (0., 0., 0., 1.),
        joint: (
            ArticulationBuilder.RevoluteJoint | ArticulationBuilder.PrismaticJoint
            | ArticulationBuilder.SphericalJoint | ArticulationBuilder.FixedJoint | None
        ) = None,
        mass: float = 1., inertia_diagonal: ArrayLike,
        com: ArrayLike = (0., 0., 0.),
        shapes: Sequence[
            ArticulationBuilder.BoxShape | ArticulationBuilder.SphereShape
        ] = (),
    ) -> str:
        if not isinstance(name, str) or not name or name in self._names:
            raise ValueError("link name must be nonempty and unique")
        if parent is None:
            if self._names or joint is not None:
                raise ValueError("only the first link may be root; root has no inbound joint")
            parent_index = -1
        else:
            if parent not in self._names:
                raise ValueError("parent must name an already-added link")
            parent_index = self._names.index(parent)
        position = np.asarray(position, dtype=np.float32)
        rotation = np.asarray(rotation_xyzw, dtype=np.float32)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("position must be a finite 3-vector")
        if (rotation.shape != (4,) or not np.isfinite(rotation).all()
                or abs(float(rotation @ rotation) - 1) > 1e-3):
            raise ValueError("rotation_xyzw must be a normalized quaternion")
        asset = _ke.asset
        inertial = asset.InertialDesc(mass, diag_inertia=inertia_diagonal, com=com)
        descriptors = []
        if isinstance(joint, ArticulationBuilder.SphericalJoint):
            if len(joint.limits) != 3:
                raise ValueError("spherical joint requires three axis limit pairs")
            axes = np.eye(3)
            limits = joint.limits
            kind = asset.JointDescType.REVOLUTE
        elif isinstance(joint, (
            ArticulationBuilder.RevoluteJoint, ArticulationBuilder.PrismaticJoint,
        )):
            axes, limits = [joint.axis], [joint.limits]
            kind = (
                asset.JointDescType.PRISMATIC
                if isinstance(joint, ArticulationBuilder.PrismaticJoint)
                else asset.JointDescType.REVOLUTE
            )
        elif joint is None or isinstance(joint, ArticulationBuilder.FixedJoint):
            axes, limits = [], []
        else:
            raise TypeError("unsupported inbound joint configuration")
        for index, (axis, limit) in enumerate(zip(axes, limits)):
            lo, hi = limit
            descriptors.append(asset.JointDesc(
                f"{name}/{index}", type=kind, axis=axis, joint_offset=joint.joint_offset,
                lo_limit=lo, hi_limit=hi, kp=joint.kp, kd=joint.kd))
        collisions = []
        for shape in shapes:
            if isinstance(shape, ArticulationBuilder.BoxShape):
                collisions.append(asset.CollisionGeomDesc(
                    type=asset.CollisionGeomDescType.BOX, size=shape.half_extents,
                    position=shape.position, rotation_xyzw=shape.rotation_xyzw))
            elif isinstance(shape, ArticulationBuilder.SphereShape):
                collisions.append(asset.CollisionGeomDesc(
                    type=asset.CollisionGeomDescType.SPHERE, size=(shape.radius, 0., 0.),
                    position=shape.position, rotation_xyzw=(0., 0., 0., 1.)))
            else:
                raise TypeError("unsupported authored collision shape")
        index = len(self._names)
        self._names.append(name)
        self._parents.append(parent_index)
        self._positions.append(position.copy())
        self._rotations.append(rotation[[3, 0, 1, 2]].copy())
        if descriptors:
            self._joints[index] = descriptors
        if collisions:
            self._shapes[index] = collisions
        self._inertials[index] = inertial
        return name

    def build(self) -> ArticulationDesc:
        if not self._names:
            raise ValueError("cannot build an empty articulation")
        tree = _ke.animation.SkeletonTree(
            self._names, self._parents, np.asarray(self._positions), np.asarray(self._rotations))
        return _ke.asset.ArticulationDesc(tree, joints=self._joints,
                                        collision_geoms=self._shapes, inertials=self._inertials)

    def build_template(self, *, config: ArticulationConfig | None = None) -> ArticulationTemplate:
        """Precompute reusable native metadata for multiple runtime instances."""
        return ArticulationTemplate.create(
            self.build(), ArticulationConfig() if config is None else config)
