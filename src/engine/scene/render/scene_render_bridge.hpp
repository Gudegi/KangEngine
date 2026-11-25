#pragma once

// Connects scene Prim state and identity to renderer-owned batches.

#include "engine/scene/render/render_scene_data.hpp"
#include "engine/scene/render/render_material_data.hpp"

#include <vector>
#include <unordered_set>
#include <utility>

namespace KE {

class Material;
namespace Backend {
class Texture;
} // namespace Backend

namespace Scene {

class Prim;
class SceneResourceManager;

// The caller removes registrations before destroying their non-owning Prims.
class SceneRenderBridge {
  public:
    static RenderMeshData meshData(const Prim& prim);
    static RenderMaterialData materialData(const Material& material);
    static RenderMaterialSource
    materialSource(const Material& material,
                   const SceneResourceManager* resources);
    static std::vector<std::pair<const char*, Backend::Texture*>>
    materialTextures(const Material& material);
    void add(Prim& prim);
    void remove(const Prim& prim);
    bool empty() const { return _entries.empty(); }
    size_t size() const { return _entries.size(); }
    bool contains(const Prim& prim) const;
    void collect();
    // Queries may follow selection/visibility edits without a frame update.
    void refreshSelection();
    Prim* resolve(RenderInstanceId id) const;
    int instanceIndex(const Prim& prim) const;
    int instanceIndex(RenderInstanceId id, int hint = -1) const;
    const RenderInstanceData& instances() const { return _instances; }

  private:
    struct Entry {
        Prim* prim;
        RenderInstanceId id;
    };
    std::vector<Entry> _entries;
    std::unordered_set<const Prim*> _prims;
    std::vector<Prim*> _instancePrims;
    RenderInstanceData _instances;
};

} // namespace Scene
} // namespace KE
