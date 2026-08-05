from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

class PhysicsGpuDynamicsConfig:
    temp_buffer_capacity: int
    max_rigid_contact_count: int
    max_rigid_patch_count: int
    heap_capacity: int
    found_lost_pairs_capacity: int
    found_lost_aggregate_pairs_capacity: int
    total_aggregate_pairs_capacity: int
    collision_stack_size: int
    max_num_partitions: int
    def __init__(
        self,
        *,
        temp_buffer_capacity: int = 67108864,
        max_rigid_contact_count: int = 4194304,
        max_rigid_patch_count: int = 524288,
        heap_capacity: int = 268435456,
        found_lost_pairs_capacity: int = 4194304,
        found_lost_aggregate_pairs_capacity: int = 33554432,
        total_aggregate_pairs_capacity: int = 2097152,
        collision_stack_size: int = 268435456,
        max_num_partitions: int = 8,
    ) -> None: ...
    def __repr__(self) -> str: ...

class PhysicsConfig:
    dt: float
    solver_type: int
    static_friction: float
    dynamic_friction: float
    restitution: float
    enable_gpu: bool
    gpu_dynamics: PhysicsGpuDynamicsConfig
    enable_contact_reports: bool
    enable_body_accelerations: bool
    cpu_dispatcher_threads: int
    bounce_threshold_velocity: float
    friction_offset_threshold: float
    friction_correlation_distance: float
    enable_stabilization: bool
    def __init__(
        self,
        *,
        dt: float = 0.0166666667,
        solver_type: int = 1,
        static_friction: float = 1.0,
        dynamic_friction: float = 1.0,
        restitution: float = 0.0,
        enable_gpu: bool = False,
        gpu_dynamics: PhysicsGpuDynamicsConfig = ...,
        enable_contact_reports: bool = True,
        enable_body_accelerations: bool = False,
        cpu_dispatcher_threads: int = 4,
        bounce_threshold_velocity: float = 2.0,
        friction_offset_threshold: float = 0.04,
        friction_correlation_distance: float = 0.025,
        enable_stabilization: bool = False,
    ) -> None: ...
    def __repr__(self) -> str: ...
    @staticmethod
    def y_up() -> PhysicsConfig: ...
    @staticmethod
    def z_up() -> PhysicsConfig: ...

class PhysicsMaterialDesc:
    static_friction: float
    dynamic_friction: float
    restitution: float
    def __init__(
        self,
        static_friction: float | Sequence[float] = 1.0,
        dynamic_friction: float = 1.0,
        restitution: float = 0.0,
    ) -> None: ...
    def as_tuple(self) -> tuple[float, float, float]: ...

class CollisionMaterialOverride:
    body_index: int
    body_name: str
    geom_index: int
    geom_name: str
    material: PhysicsMaterialDesc
    def __init__(
        self, material: PhysicsMaterialDesc | Sequence[float] | None = None
    ) -> None: ...
    @staticmethod
    def all_geoms(
        material: PhysicsMaterialDesc | Sequence[float],
    ) -> CollisionMaterialOverride: ...
    @staticmethod
    def for_body(
        body_name: str, material: PhysicsMaterialDesc | Sequence[float]
    ) -> CollisionMaterialOverride: ...
    @staticmethod
    def for_geom(
        body_name: str,
        geom_name: str,
        material: PhysicsMaterialDesc | Sequence[float],
    ) -> CollisionMaterialOverride: ...
    @staticmethod
    def for_indices(
        body_index: int,
        geom_index: int,
        material: PhysicsMaterialDesc | Sequence[float],
    ) -> CollisionMaterialOverride: ...

def mjcf_friction_to_physx(friction: Sequence[float]) -> PhysicsMaterialDesc: ...

class ContactPoint: ...

class RigidDynamic:
    def get_root_position(self) -> Any: ...
    def get_root_rotation(self) -> Any: ...
    def get_root_linear_velocity(self) -> Any: ...
    def get_root_angular_velocity(self) -> Any: ...
    def get_mass(self) -> float: ...
    def get_inverse_mass(self) -> float: ...
    def get_local_com_position(self) -> Any: ...
    def get_local_com_rotation(self) -> Any: ...
    def get_mass_space_inertia(self) -> Any: ...
    def get_mass_space_inverse_inertia(self) -> Any: ...
    def release(self) -> None: ...
    def set_root_state(
        self,
        pos: Any,
        rot_xyzw: Any,
        linear_velocity: Any = ...,
        angular_velocity: Any = ...,
    ) -> None: ...
    def add_force(self, force: Any) -> None: ...
    def add_force_at_position(self, force: Any, position: Any) -> None: ...

class NativePhysicsWorld: ...
class NativeArticulation: ...
class NativePhysicsBridge: ...
class NativePhysicsGpuSystem: ...

def unwrap_native(obj: Any) -> Any: ...

class PhysicsWorld:
    native: NativePhysicsWorld
    def __init__(self, config: PhysicsConfig | None = None) -> None: ...
    def step(self) -> None: ...
    def add_default_ground(self) -> None: ...
    def clear_ground_actors(self) -> None: ...
    def num_ground_actors(self) -> int: ...
    def num_cached_materials(self) -> int: ...
    def num_body_actors(self) -> int: ...
    def num_contacts(self) -> int: ...
    def get_contacts(self) -> Sequence[ContactPoint]:
        """Return the latest contact records from the native physics backend.

        Treat the returned data conservatively: depending on the active backend,
        contact records may reference native buffers or describe only the most
        recent simulation synchronization point. Do not assume they remain
        unchanged after later step(), sync(), or clear_contacts() calls.
        """
        ...
    def clear_contacts(self) -> None: ...
    def add_static_box(
        self,
        half_extents: Sequence[float],
        pos: Sequence[float],
        rot_xyzw: Sequence[float] = ...,
        register_as_ground: bool = True,
    ) -> None: ...
    def add_heightfield(
        self,
        heights: Any,
        rows: int,
        cols: int,
        horizontal_scale: float = 1.0,
        up_axis: Any = ...,
        center: bool = True,
        register_as_ground: bool = True,
        material: PhysicsMaterialDesc | None = None,
    ) -> bool: ...
    def add_heightmap_collision(
        self,
        path: str,
        up_axis: Any = ...,
        horizontal_scale: float = 1.0,
        height_scale: float = 64.0,
        height_offset: float = -16.0,
        sample_stride: int = 1,
        center: bool = True,
        register_as_ground: bool = True,
        material: PhysicsMaterialDesc | None = None,
    ) -> bool: ...
    def create_dynamic_box(
        self,
        half_extents: Sequence[float],
        pos: Sequence[float],
        rot_xyzw: Sequence[float] = ...,
        density: float = 1.0,
    ) -> RigidDynamic: ...
    def create_dynamic_sphere(
        self,
        radius: float,
        pos: Sequence[float],
        rot_xyzw: Sequence[float] = ...,
        density: float = 1.0,
    ) -> RigidDynamic: ...
    def get_contact_forces(
        self, articulation: Articulation | NativeArticulation, ground_only: bool = False
    ) -> Any:
        """Return contact-force data for an articulation.

        The concrete return type is backend-dependent. Treat array-like results
        as borrowed or refreshable data unless you explicitly copy them; later
        simulation steps or synchronization calls may update the underlying
        storage.
        """
        ...
    def get_ground_contact_forces(
        self, articulation: Articulation | NativeArticulation
    ) -> Any: ...
    def get_rigid_contact_force(
        self, rigid: RigidDynamic, ground_only: bool = False
    ) -> Any: ...
    def get_rigid_ground_contact_force(self, rigid: RigidDynamic) -> Any: ...
    def set_rigid_collision_material(
        self, rigid: RigidDynamic, material: PhysicsMaterialDesc
    ) -> int: ...
    def set_rigid_collision_material_overrides(
        self,
        rigid: RigidDynamic,
        data: Any,
        material_overrides: Sequence[CollisionMaterialOverride],
    ) -> int: ...
    def create_dynamic_rigid(
        self,
        data: Any,
        pos: Any,
        rot_xyzw: Any = ...,
        density: float = 1.0,
        collision_group: int = 0,
        contact_offset: float = 0.02,
        rest_offset: float = 0.0,
        material_overrides: Sequence[CollisionMaterialOverride] = ...,
    ) -> RigidDynamic: ...
    def set_dt(self, dt: float) -> None: ...

class ArticulationConfig:
    fix_base: bool
    disable_self_collision: bool
    use_aggregate: bool
    solver_iterations: int
    solver_position_iteration_count: int
    solver_velocity_iteration_count: int
    collision_group: int
    root_linear_damping: float
    root_angular_damping: float
    link_linear_damping: float
    link_angular_damping: float
    max_angular_velocity: float
    max_depenetration_velocity: float
    sleep_threshold: float
    stabilization_threshold: float
    enable_gyroscopic_forces: bool
    contact_offset: float
    rest_offset: float
    enable_ccd: bool
    material_overrides: list[CollisionMaterialOverride]
    def __init__(
        self,
        *,
        fix_base: bool = True,
        disable_self_collision: bool = True,
        use_aggregate: bool = False,
        solver_position_iteration_count: int = 16,
        solver_velocity_iteration_count: int = 1,
        collision_group: int = 0,
        root_linear_damping: float = 0.0,
        root_angular_damping: float = 0.05,
        link_linear_damping: float = 0.0,
        link_angular_damping: float = 0.05,
        max_angular_velocity: float = 100.0,
        max_depenetration_velocity: float = 1e32,
        sleep_threshold: float = 0.005,
        stabilization_threshold: float = 0.0005,
        enable_gyroscopic_forces: bool = False,
        contact_offset: float = 0.02,
        rest_offset: float = 0.0,
        material_overrides: Sequence[CollisionMaterialOverride] = ...,
        enable_ccd: bool = False,
    ) -> None: ...
    def __repr__(self) -> str: ...
    @staticmethod
    def fixed_base() -> ArticulationConfig: ...
    @staticmethod
    def free_base() -> ArticulationConfig: ...

class Articulation:
    native: NativeArticulation
    def __init__(self, native_articulation: Any | None = None) -> None: ...
    @staticmethod
    def build(
        physics: PhysicsWorld | NativePhysicsWorld,
        data: Any,
        cfg: ArticulationConfig | None = None,
    ) -> Articulation: ...
    def num_links(self) -> int: ...
    def num_dofs(self) -> int: ...
    def release(self) -> None: ...
    def set_collision_material(
        self, physics: PhysicsWorld | NativePhysicsWorld, material: PhysicsMaterialDesc
    ) -> int: ...
    def set_collision_material_overrides(
        self,
        physics: PhysicsWorld | NativePhysicsWorld,
        material_overrides: Sequence[CollisionMaterialOverride],
    ) -> int: ...

class PhysicsBridge:
    native: NativePhysicsBridge
    def __init__(self) -> None: ...
    def add(
        self, artic: Articulation | NativeArticulation, skel_bridge: Any
    ) -> None: ...
    def sync(self) -> None: ...
    def set_collision_visible(self, visible: bool) -> None: ...
    def add_collision_visuals(
        self,
        artic: Articulation | NativeArticulation,
        scene: Any,
        base_path: str = "/collision",
        visible_by_default: bool = False,
    ) -> Any: ...

class GpuPhysicsConfig:
    cuda_device_id: int
    max_contact_pairs: int
    max_contact_points: int
    def __init__(
        self,
        *,
        cuda_device_id: int = 0,
        max_contact_pairs: int = 65536,
        max_contact_points: int = 262144,
    ) -> None: ...
    def __repr__(self) -> str: ...

class PhysicsGpuStateViews:
    rigid_accelerations: GpuArrayView
    articulation_dense_jacobians: GpuArrayView
    articulation_mass_matrices: GpuArrayView
    articulation_gravity_forces: GpuArrayView
    articulation_coriolis_forces: GpuArrayView
    articulation_link_accelerations: GpuArrayView
    articulation_link_forces: GpuArrayView
    articulation_link_torques: GpuArrayView
    articulation_com_world: GpuArrayView
    articulation_com_root: GpuArrayView
    articulation_centroidal_momentum_matrices: GpuArrayView
    articulation_centroidal_bias_forces: GpuArrayView

class PhysicsGpuSystem:
    native: NativePhysicsGpuSystem
    def __init__(
        self,
        world: PhysicsWorld | NativePhysicsWorld,
        config: GpuPhysicsConfig | None = None,
    ) -> None: ...
    def articulation_row(
        self, articulation: Articulation | NativeArticulation
    ) -> int: ...
    def rigid_row(self, rigid: RigidDynamic) -> int: ...
    def rigid_accelerations(self) -> GpuArrayView: ...
    def fetch_rigid_accelerations(self) -> None: ...
    def articulation_generalized_dof_count(self, articulation_row: int) -> int: ...
    def articulation_jacobian_row_count(self, articulation_row: int) -> int: ...
    def articulation_max_generalized_dofs(self) -> int: ...
    def articulation_max_jacobian_rows(self) -> int: ...
    def articulation_dense_jacobians(self) -> GpuArrayView: ...
    def articulation_mass_matrices(self) -> GpuArrayView: ...
    def articulation_gravity_forces(self) -> GpuArrayView: ...
    def articulation_coriolis_forces(self) -> GpuArrayView: ...
    def compute_articulation_dense_jacobians(self) -> None: ...
    def compute_articulation_mass_matrices(self) -> None: ...
    def compute_articulation_gravity_forces(self) -> None: ...
    def compute_articulation_coriolis_forces(self) -> None: ...
    def articulation_link_accelerations(self) -> GpuArrayView: ...
    def articulation_link_forces(self) -> GpuArrayView: ...
    def articulation_link_torques(self) -> GpuArrayView: ...
    def fetch_articulation_link_accelerations(self) -> None: ...
    def prepare_articulation_link_commands(self) -> None: ...
    def apply_articulation_link_forces(self) -> None: ...
    def apply_articulation_link_torques(self) -> None: ...
    def articulation_com_world(self) -> GpuArrayView: ...
    def articulation_com_root(self) -> GpuArrayView: ...
    def articulation_centroidal_momentum_matrices(self) -> GpuArrayView: ...
    def articulation_centroidal_bias_forces(self) -> GpuArrayView: ...
    def compute_articulation_com_world(self) -> None: ...
    def compute_articulation_com_root(self) -> None: ...
    def compute_articulation_centroidal_momentum(self) -> None: ...
    def views(self) -> PhysicsGpuStateViews:
        """Return borrowed GPU state views owned by this PhysicsGpuSystem.

        The returned object may reference buffers managed by the native GPU
        physics system. Keep this PhysicsGpuSystem alive while using the views,
        and do not assume values remain unchanged after later simulation,
        synchronization, or refresh operations.
        """
        ...

aggregate_contact_sensors_cuda: Callable[..., Any] | None

class SimMemoryType: ...
class SimDType: ...
class SimLifetimePolicy: ...
class GpuArrayView: ...
class SimModel: ...
class SimState: ...
class SimVisualBatch: ...
