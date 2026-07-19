"""Low-level physics world and articulation APIs.

This module is a thin public Python layer over the pybind11 ``_ke.physics``
module.  Public wrapper classes are always defined so IDEs and import sites see
a stable API surface even when KangEngine is built without PhysX.  Backend
availability is checked when native objects are actually constructed or used.

Performance rule: wrappers may validate, unwrap, and forward; they must not add
per-body/per-step Python loops around native bulk calls.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from ._core import _ke
from ._public import set_public_module

_native = getattr(_ke, "physics", None)


def _require_native():
    if _native is None:
        raise RuntimeError("KangEngine was built without PhysX bindings.")
    return _native


def _native_attr(name: str) -> Any:
    native_module = _require_native()
    try:
        return getattr(native_module, name)
    except AttributeError as exc:
        raise RuntimeError(
            f"KangEngine PhysX binding is missing {name!r}."
        ) from exc


def unwrap_native(obj: Any) -> Any:
    """Return ``obj._native`` when ``obj`` is a KangEngine Python wrapper."""
    return getattr(obj, "_native", obj)


def _missing_native_type(native_name: str, public_name: str | None = None):
    public_name = public_name or native_name

    class _MissingNativeType:
        def __init__(self, *args, **kwargs):
            _native_attr(native_name)

        def __repr__(self) -> str:
            return f"<unavailable KangEngine physics type {public_name}>"

    _MissingNativeType.__name__ = public_name
    _MissingNativeType.__qualname__ = public_name
    _MissingNativeType.__module__ = __name__
    return _MissingNativeType


def _export_native_type(native_name: str, public_name: str | None = None):
    public_name = public_name or native_name
    if _native is None or not hasattr(_native, native_name):
        value = _missing_native_type(native_name, public_name)
    else:
        value = set_public_module(getattr(_native, native_name), __name__)
    globals()[public_name] = value
    return value


def _export_native_function(native_name: str, public_name: str | None = None):
    public_name = public_name or native_name
    if _native is None or not hasattr(_native, native_name):

        def _missing(*args, **kwargs):
            _native_attr(native_name)

        _missing.__name__ = public_name
        _missing.__qualname__ = public_name
        _missing.__module__ = __name__
        globals()[public_name] = _missing
        return _missing

    value = set_public_module(getattr(_native, native_name), __name__)
    globals()[public_name] = value
    return value


class _NativeWrapper:
    """Small forwarding base for wrapper objects around pybind11 instances."""

    _native: Any

    @property
    def native(self) -> Any:
        """Native pybind11 object for advanced interop and hot paths."""
        return self._native

    def __getattr__(self, name: str) -> Any:
        return getattr(self._native, name)

    def __repr__(self) -> str:
        return repr(self._native)


# Value/config/data types stay native for now.  They are cheap payloads that
# pybind11 already handles well, and keeping them native avoids accidental
# conversion churn when passing them into C++.  If PhysX is unavailable, the
# placeholder class still makes imports/IDE discovery stable and raises at
# construction time.
PhysicsGpuDynamicsConfig = _export_native_type("PhysicsGpuDynamicsConfig")
PhysicsConfig = _export_native_type("PhysicsConfig")
PhysicsMaterialDesc = _export_native_type("PhysicsMaterialDesc")
CollisionMaterialOverride = _export_native_type("CollisionMaterialOverride")
ContactPoint = _export_native_type("ContactPoint")
RigidDynamic = _export_native_type("RigidDynamic")
ArticulationConfig = _export_native_type("ArticulationConfig")
GpuPhysicsConfig = _export_native_type("GpuPhysicsConfig")
PhysicsGpuStateViews = _export_native_type("PhysicsGpuStateViews")

mjcf_friction_to_physx = _export_native_function("mjcf_friction_to_physx")

NativePhysicsWorld = _export_native_type("PhysicsWorld", "NativePhysicsWorld")
NativeArticulation = _export_native_type("Articulation", "NativeArticulation")
NativePhysicsBridge = _export_native_type("PhysicsBridge", "NativePhysicsBridge")
NativePhysicsGpuSystem = _export_native_type(
    "PhysicsGpuSystem", "NativePhysicsGpuSystem"
)


class PhysicsWorld(_NativeWrapper):
    """PhysX simulation world.

    The class is always importable.  Constructing it requires a KangEngine build
    with PhysX bindings.  The wrapper owns a native
    ``_ke.physics.PhysicsWorld`` and forwards calls directly.
    """

    def __init__(self, config: PhysicsConfig | None = None):
        native_module = _require_native()
        if config is None:
            config = native_module.PhysicsConfig()
        self._native = native_module.PhysicsWorld(config)

    def step(self) -> None:
        """Advance simulation by one configured timestep."""
        self._native.step()

    def add_default_ground(self) -> None:
        self._native.add_default_ground()

    def clear_ground_actors(self) -> None:
        self._native.clear_ground_actors()

    def num_ground_actors(self) -> int:
        return self._native.num_ground_actors()

    def num_cached_materials(self) -> int:
        return self._native.num_cached_materials()

    def num_body_actors(self) -> int:
        return self._native.num_body_actors()

    def num_contacts(self) -> int:
        return self._native.num_contacts()

    def get_contacts(self) -> Sequence[ContactPoint]:
        """Return the latest contact records from the native physics backend.

        Treat the returned data conservatively: depending on the active backend,
        contact records may reference native buffers or describe only the most
        recent simulation synchronization point. Do not assume they remain
        unchanged after later step(), sync(), or clear_contacts() calls.
        """
        return self._native.get_contacts()

    def clear_contacts(self) -> None:
        self._native.clear_contacts()

    def add_static_box(
        self,
        half_extents: Sequence[float],
        pos: Sequence[float],
        rot_xyzw: Sequence[float] = (0.0, 0.0, 0.0, 1.0),
        register_as_ground: bool = True,
    ) -> None:
        return self._native.add_static_box(
            half_extents, pos, rot_xyzw, register_as_ground
        )

    def add_heightfield(
        self,
        heights: Any,
        rows: int,
        cols: int,
        horizontal_scale: float = 1.0,
        up_axis: Any = None,
        center: bool = True,
        register_as_ground: bool = True,
        material: PhysicsMaterialDesc | None = None,
    ) -> bool:
        if up_axis is None:
            up_axis = _ke.UpAxis.Y
        if material is None:
            material = _native_attr("PhysicsMaterialDesc")()
        return self._native.add_heightfield(
            heights,
            rows,
            cols,
            horizontal_scale,
            up_axis,
            center,
            register_as_ground,
            material,
        )

    def add_heightmap_collision(
        self,
        path: str,
        up_axis: Any = None,
        horizontal_scale: float = 1.0,
        height_scale: float = 64.0,
        height_offset: float = -16.0,
        sample_stride: int = 1,
        center: bool = True,
        register_as_ground: bool = True,
        material: PhysicsMaterialDesc | None = None,
    ) -> bool:
        if up_axis is None:
            up_axis = _ke.UpAxis.Y
        if material is None:
            material = _native_attr("PhysicsMaterialDesc")()
        return self._native.add_heightmap_collision(
            path,
            up_axis,
            horizontal_scale,
            height_scale,
            height_offset,
            sample_stride,
            center,
            register_as_ground,
            material,
        )

    def create_dynamic_box(
        self,
        half_extents: Sequence[float],
        pos: Sequence[float],
        rot_xyzw: Sequence[float] = (0.0, 0.0, 0.0, 1.0),
        density: float = 1.0,
    ) -> RigidDynamic:
        return self._native.create_dynamic_box(
            half_extents, pos, rot_xyzw, density
        )

    def create_dynamic_sphere(
        self,
        radius: float,
        pos: Sequence[float],
        rot_xyzw: Sequence[float] = (0.0, 0.0, 0.0, 1.0),
        density: float = 1.0,
    ) -> RigidDynamic:
        return self._native.create_dynamic_sphere(radius, pos, rot_xyzw, density)

    def get_contact_forces(
        self, articulation: Articulation | NativeArticulation, ground_only: bool = False
    ) -> Any:
        """Return contact-force data for an articulation.

        The concrete return type is backend-dependent. Treat array-like results
        as borrowed or refreshable data unless you explicitly copy them; later
        simulation steps or synchronization calls may update the underlying
        storage.
        """
        return self._native.get_contact_forces(
            unwrap_native(articulation), ground_only
        )

    def get_ground_contact_forces(self, articulation: Articulation | NativeArticulation) -> Any:
        return self._native.get_ground_contact_forces(unwrap_native(articulation))

    def get_rigid_contact_force(self, rigid: RigidDynamic, ground_only: bool = False) -> Any:
        return self._native.get_rigid_contact_force(unwrap_native(rigid), ground_only)

    def get_rigid_ground_contact_force(self, rigid: RigidDynamic) -> Any:
        return self._native.get_rigid_ground_contact_force(unwrap_native(rigid))

    def set_rigid_collision_material(self, rigid: RigidDynamic, material: PhysicsMaterialDesc) -> int:
        return self._native.set_rigid_collision_material(unwrap_native(rigid), material)

    def set_rigid_collision_material_overrides(
        self,
        rigid: RigidDynamic,
        data: Any,
        material_overrides: Sequence[CollisionMaterialOverride],
    ) -> int:
        return self._native.set_rigid_collision_material_overrides(
            unwrap_native(rigid), data, material_overrides
        )

    def create_dynamic_rigid(
        self,
        data: Any,
        pos: Any,
        rot_xyzw: Any = (0.0, 0.0, 0.0, 1.0),
        density: float = 1.0,
        collision_group: int = 0,
        contact_offset: float = 0.02,
        rest_offset: float = 0.0,
        material_overrides: Sequence[CollisionMaterialOverride] = (),
    ) -> RigidDynamic:
        return self._native.create_dynamic_rigid(
            data,
            pos,
            rot_xyzw,
            density,
            collision_group,
            contact_offset,
            rest_offset,
            material_overrides,
        )

    def set_dt(self, dt: float) -> None:
        self._native.set_dt(float(dt))


class Articulation(_NativeWrapper):
    """PhysX articulation wrapper."""

    def __init__(self, native_articulation: Any | None = None):
        native_module = _require_native()
        self._native = (
            native_module.Articulation()
            if native_articulation is None
            else unwrap_native(native_articulation)
        )

    @staticmethod
    def build(
        physics: PhysicsWorld | NativePhysicsWorld,
        data: Any,
        cfg: ArticulationConfig | None = None,
    ) -> "Articulation":
        native_module = _require_native()
        if cfg is None:
            cfg = native_module.ArticulationConfig()
        return Articulation(
            native_module.Articulation.build(unwrap_native(physics), data, cfg)
        )

    def num_links(self) -> int:
        return self._native.num_links()

    def num_dofs(self) -> int:
        return self._native.num_dofs()

    def release(self) -> None:
        self._native.release()

    def set_collision_material(
        self, physics: PhysicsWorld | NativePhysicsWorld, material: PhysicsMaterialDesc
    ) -> int:
        return self._native.set_collision_material(unwrap_native(physics), material)

    def set_collision_material_overrides(
        self,
        physics: PhysicsWorld | NativePhysicsWorld,
        material_overrides: Sequence[CollisionMaterialOverride],
    ) -> int:
        return self._native.set_collision_material_overrides(
            unwrap_native(physics), material_overrides
        )


class PhysicsBridge(_NativeWrapper):
    """Sync PhysX articulation state into scene/render visuals."""

    def __init__(self):
        self._native = _native_attr("PhysicsBridge")()

    def add(self, artic: Articulation | NativeArticulation, skel_bridge: Any) -> None:
        self._native.add(unwrap_native(artic), unwrap_native(skel_bridge))

    def sync(self) -> None:
        self._native.sync()

    def set_collision_visible(self, visible: bool) -> None:
        self._native.set_collision_visible(bool(visible))

    def add_collision_visuals(
        self,
        artic: Articulation | NativeArticulation,
        scene: Any,
        base_path: str = "/collision",
        visible_by_default: bool = False,
    ):
        return self._native.add_collision_visuals(
            unwrap_native(artic), unwrap_native(scene), base_path, visible_by_default
        )


class PhysicsGpuSystem(_NativeWrapper):
    """Explicit GPU physics synchronization wrapper."""

    def __init__(
        self, world: PhysicsWorld | NativePhysicsWorld, config: GpuPhysicsConfig | None = None
    ):
        native_module = _require_native()
        if config is None:
            config = native_module.GpuPhysicsConfig()
        self._native = native_module.PhysicsGpuSystem(unwrap_native(world), config)

    def articulation_row(self, articulation: Articulation | NativeArticulation) -> int:
        return self._native.articulation_row(unwrap_native(articulation))

    def rigid_row(self, rigid: RigidDynamic) -> int:
        return self._native.rigid_row(unwrap_native(rigid))

    def views(self) -> PhysicsGpuStateViews:
        """Return borrowed GPU state views owned by this PhysicsGpuSystem.

        The returned object may reference buffers managed by the native GPU
        physics system. Keep this PhysicsGpuSystem alive while using the views,
        and do not assume values remain unchanged after later simulation,
        synchronization, or refresh operations.
        """
        return self._native.views()


aggregate_contact_sensors_cuda: Callable[..., Any] | None
aggregate_contact_sensors_cuda = (
    _export_native_function("aggregate_contact_sensors_cuda")
    if _native is not None and hasattr(_native, "aggregate_contact_sensors_cuda")
    else None
)


# Sim buffer / visual batch helpers are still registered at the extension
# top-level.  Keep exporting them here as compatibility names until the sim
# namespace is cleaned up separately.
for _name in (
    "SimMemoryType",
    "SimDType",
    "SimLifetimePolicy",
    "GpuArrayView",
    "SimModel",
    "SimState",
    "SimVisualBatch",
):
    if hasattr(_ke, _name):
        globals()[_name] = set_public_module(getattr(_ke, _name), __name__)


__all__ = [
    name
    for name in (
        "unwrap_native",
        "PhysicsGpuDynamicsConfig",
        "PhysicsConfig",
        "PhysicsMaterialDesc",
        "CollisionMaterialOverride",
        "mjcf_friction_to_physx",
        "ContactPoint",
        "RigidDynamic",
        "PhysicsWorld",
        "ArticulationConfig",
        "Articulation",
        "PhysicsBridge",
        "GpuPhysicsConfig",
        "PhysicsGpuStateViews",
        "PhysicsGpuSystem",
        "NativePhysicsWorld",
        "NativeArticulation",
        "NativePhysicsBridge",
        "NativePhysicsGpuSystem",
        "aggregate_contact_sensors_cuda",
        "SimMemoryType",
        "SimDType",
        "SimLifetimePolicy",
        "GpuArrayView",
        "SimModel",
        "SimState",
        "SimVisualBatch",
    )
    if name in globals() and globals()[name] is not None
]
