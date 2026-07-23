# Runtime Architecture

KangEngine supports two complementary runtime tracks.

## Scene track (SceneGraph)

Use scene prims for authored objects, paths, hierarchy, transforms, materials,
selection, and editor interaction.

```text
geometry or imported asset
→ SceneContext
→ Prim + components
→ SceneRenderSystem
→ renderer
```

## Simulation track (ExternalBuffer)

Use `KangSimWorld` and batched views when policies, resets, state tensors, or
many environments matter.

```text
asset description
→ KangSimWorld
→ SimRigid / SimArticulation
→ batched state
→ SimWorldVisualizer
```

Do not treat thousands of scene prims as the canonical simulation state.
ExternalBuffer visual batches exist to keep high-throughput state movement out
of per-prim mutation. Conversely, use explicit SceneGraph visuals when picking
and editor inspection are the goal.
