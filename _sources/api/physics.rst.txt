ke.physics
==========

Low-level PhysX world, rigid body, articulation, and visual sync APIs.

.. currentmodule:: kangengine.physics

API overview
------------

.. autosummary::
   :nosignatures:

   PhysicsConfig
   PhysicsWorld
   ArticulationConfig
   Articulation
   PhysicsBridge
   GpuPhysicsConfig
   PhysicsGpuSystem

Wrapper and lifetime policy
---------------------------

The Python low-level physics API is exposed through thin wrapper classes around
native pybind11 objects. The wrapper classes remain importable for stable IDE
completion and API documentation even when KangEngine is built without PhysX;
constructing or using a PhysX-backed object in such a build raises a runtime
error.

Use the ``.native`` property only when deliberately dropping to native parity.
Helper code that accepts either wrapper or native objects should normalize with
``unwrap_native(obj)``.

Contact queries and GPU state views expose backend-owned state. Treat
``PhysicsWorld.get_contacts()``, ``PhysicsWorld.get_contact_forces(...)``, and
``PhysicsGpuSystem.views()`` conservatively: later simulation, synchronization,
refresh, or clear operations may update or invalidate the underlying storage.
Copy returned array-like data explicitly when a stable snapshot is required.


Return and error contracts
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - API
     - Contract
   * - ``PhysicsGpuSystem.views()`` and GPU buffer accessors
     - Borrowed device storage owned by the GPU system. Fetch and apply methods
       update it; releasing or invalidating the system invalidates the view.
   * - ``PhysicsWorld.get_contacts()`` and ``get_contact_forces(...)``
     - Current-query data only. Copy values that must remain stable across later
       simulation or clear operations.
   * - Resource and actor wrappers
     - Native handles tied to their owning world. Using them after world release
       raises ``RuntimeError`` or results in an invalid native handle.

Invalid indices and names generally raise ``KeyError`` or ``IndexError``;
invalid shapes and configuration values raise ``ValueError``; unavailable
PhysX/CUDA features and invalid lifecycle state raise ``RuntimeError``.

.. autoclass:: PhysicsConfig
   :special-members: __init__

.. autoclass:: PhysicsGpuDynamicsConfig
   :special-members: __init__

.. autoclass:: PhysicsMaterialDesc

.. autoclass:: CollisionMaterialOverride

.. autoclass:: ContactPoint

.. autoclass:: RigidDynamic

.. autofunction:: unwrap_native

.. autofunction:: mjcf_friction_to_physx

.. autoclass:: PhysicsWorld

.. autoclass:: ArticulationConfig
   :special-members: __init__

.. autoclass:: Articulation

.. autoclass:: PhysicsBridge

.. autoclass:: GpuPhysicsConfig
   :special-members: __init__

.. autoclass:: PhysicsGpuStateViews

.. autoclass:: PhysicsGpuSystem

.. py:function:: aggregate_contact_sensors_cuda(...)

   Aggregates contact sensor outputs on the GPU. Available only in CUDA-enabled builds.

Collision visual notes
----------------------

``PhysicsBridge.add_collision_visuals(...)`` creates optional SceneGraph debug
prims for articulation collision shapes. These prims mirror the existing PhysX
collision descriptors and carry scene-side ``CollisionShapeComponent`` metadata;
they do not create additional simulation shapes.

MJCF primitive collision geoms are imported directly. Collidable MJCF mesh geoms
are not cooked yet; bodies with unsupported collidable mesh geoms may receive
KangEngine fallback boxes. Visual-only mesh geoms remain render-only.
