# ke.scene

Scene graph, prim hierarchy, components, resource mirrors, and debug drawing.

## API overview

```{eval-rst}
.. currentmodule:: kangengine.scene

.. autosummary::
   :nosignatures:

   Prim
   Camera
   DirectionalLight
   PointLight
   SpotLight
   SceneBackend
   SceneResourceManager
   MeshData
   SkinnedMeshData
   DebugDraw
```

```{eval-rst}
.. currentmodule:: kangengine.scene

.. autoclass:: Token

.. autoclass:: Prim

.. autoclass:: Camera

.. autoclass:: DirectionalLight

.. autoclass:: PointLight

.. autoclass:: SpotLight

.. autoclass:: TransformComponent

.. autoclass:: MeshComponent

.. autoclass:: RenderComponent

.. autoclass:: MaterialBindingComponent

.. autoclass:: ArticulationComponent

.. autoclass:: ArticulationBindingComponent

.. autoclass:: CollisionShapeComponent

.. autoclass:: ResourceComponent

.. autoclass:: SceneResourceManager

.. autoclass:: MeshData

.. autoclass:: SkinnedMeshData

.. autoclass:: SceneBackend
```
## Return and ownership rules

| API | Return and lifetime contract |
| --- | --- |
| `register_mesh/material/texture()` | A runtime `ResourceHandle`; it is not a persistent asset identifier. |
| `register_shader_source/pipeline()` | A handle for copied authored metadata; compiled backend objects are not stored here. |
| `mesh()` | A shared mesh payload, or `None` for an unknown handle. |
| `material()`, `texture()` | Borrowed engine objects, or `None`; the manager does not own these payloads. |
| `shader_source()`, `pipeline()` | Authored metadata retained by the manager, or `None`. |
| `resource_prim()` | A scene-owned mirror prim, valid until removal, `clear()`, or scene teardown. |
| `usage_paths()` | A Python list converted from the current usage cache. |

`clear()` removes manager records and their `/.Resources` mirror prims. Keep a
separate owned reference when a non-owned material or texture must
outlive its resource registration.


## Debug geometry

Use `app.scene.debug_geometry` for mesh-based debug lines, arrows, and axes.
These objects are ordinary SceneGraph renderables and return
`DebugPrimitiveView` instances.

`ke.scene.DebugDraw` is the low-level native implementation used by that
facade:

```{eval-rst}
.. currentmodule:: kangengine.scene

.. autoclass:: DebugDraw
```

## Resource Prim lifecycle

`SceneResourceManager` is the source of truth for scene-local shared resources.
Registering a mesh/material/texture or authored shader-source/pipeline assigns
a runtime `ResourceHandle` and creates a metadata-only mirror prim under
`/.Resources/...`.

Resource prims are not renderable world objects:

- they are created, updated, and removed by `SceneResourceManager`;
- they carry `ResourceComponent` metadata: type, handle, display name, and URI;
- they do not own a `TransformComponent`;
- they do not own mesh/GPU payloads;
- they are not manipulation targets;
- renderable prims draw through their own `MeshComponent.mesh_data` cache and
  keep `MeshComponent.resource_handle` only as resource identity.

`ResourceHandle` is a runtime identifier. Use URI/path metadata for persistent
asset identity when saving or rebuilding a scene.

Resource usage counts are editor diagnostics. `SceneResourceManager` exposes
`usage_count(handle)` / `is_used(handle)` to report current scene bindings, and
keeps the result in a lazy cache. Normal engine paths invalidate the cache when
resources or renderable bindings change; direct material/component mutation can
call `invalidate_usage_cache()` explicitly.

## Articulation collision metadata

Articulation scene prims use components for metadata rather than a dedicated
`PrimType::Articulation`.

- `ArticulationComponent` belongs on the articulation root prim and stores
  source/root/mesh-asset metadata.
- `ArticulationBindingComponent` records whether a generated prim is a body
  frame, visual geom, or collision geom and which body it belongs to.
- `CollisionShapeComponent` is attached only to optional collision debug prims
  created by `PhysicsBridge.add_collision_visuals(...)`.

`CollisionShapeComponent` mirrors reference collision data for inspection. It
does not own PhysX shapes/materials. When collision debug prims are not created,
physics collision still runs normally but there is no scene component to inspect.
