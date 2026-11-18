#pragma once

// Connects scene Prim state and identity to static Native render batches.

#include "engine/scene/render_scene_data.hpp"
#include "engine/scene/render_material_data.hpp"

#include <vector>

namespace KE {

class Material;

namespace Scene {

class Prim;

// The caller removes registrations before destroying their non-owning Prims.
class SceneRenderBridge {
  public:
    static RenderMeshData meshData(const Prim& prim);
    static RenderMaterialData materialData(const Material& material);
    void add(Prim& prim);
    void remove(const Prim& prim);
    bool empty() const { return _entries.empty(); }
    void collect();
    // Queries may follow selection/visibility edits without a frame update.
    void refreshSelection();
    Prim* resolve(RenderInstanceId id) const;
    int instanceIndex(const Prim& prim) const;
    int instanceIndex(RenderInstanceId id) const;
    const RenderInstanceData& instances() const { return _instances; }

  private:
    struct Entry {
        Prim* prim;
        RenderInstanceId id;
    };
    std::vector<Entry> _entries;
    std::vector<Prim*> _instancePrims;
    RenderInstanceData _instances;
};

} // namespace Scene
} // namespace KE
