#ifndef _SCENE_RESOURCE_MANAGER_HPP_
#define _SCENE_RESOURCE_MANAGER_HPP_

#include "engine/scene/component/resource_component.hpp"
#include "engine/scene/scene_backend.hpp"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace KE {

class Material;

namespace Backend {
class Shader;
class Texture;
} // namespace Backend

namespace Scene {

// Scene-owned manager for shared resource identities and editor mirrors.
//
// SceneResourceManager is deliberately not a frame-updated system. It records
// shared resource objects, assigns stable handles, and mirrors entries into the
// scene as `/.Resources/...` prims with ResourceComponent metadata.
//
// Resource ownership model:
// - The manager is the conceptual owner/catalog for mesh/material/texture/shader
//   resources in scene scope: handles, names, URIs, retention pointers, and
//   editor-visible Resource Prim mirrors.
// - Renderable prims do not render Resource Prim payloads. They bind resources
//   through components such as MeshComponent and MaterialBindingComponent.
// - MeshComponent may cache the resolved MeshData shared_ptr for the fast render
//   path, but SceneResourceManager remains the source of resource identity.
class SceneResourceManager {
  public:
    struct Entry {
        ResourceHandle handle = InvalidResourceHandle;
        ResourceType type = ResourceType::Unknown;
        std::string name;
        std::string uri;
        Prim* prim = nullptr;

        std::shared_ptr<MeshData> mesh;
        Material* material = nullptr;
        Backend::Texture* texture = nullptr;
        Backend::Shader* shader = nullptr;
    };

    explicit SceneResourceManager(SceneBackend* scene = nullptr);

    void bindScene(SceneBackend* scene);
    SceneBackend* scene() const { return _scene; }

    ResourceHandle registerMesh(const std::string& name,
                                std::shared_ptr<MeshData> mesh,
                                const std::string& uri = {});
    ResourceHandle registerMaterial(const std::string& name, Material* material,
                                    const std::string& uri = {});
    ResourceHandle registerTexture(const std::string& name,
                                   Backend::Texture* texture,
                                   const std::string& uri = {});
    ResourceHandle registerShader(const std::string& name,
                                  Backend::Shader* shader,
                                  const std::string& uri = {});

    const Entry* entry(ResourceHandle handle) const;
    Entry* entry(ResourceHandle handle);

    std::shared_ptr<MeshData> mesh(ResourceHandle handle) const;
    Material* material(ResourceHandle handle) const;
    Backend::Texture* texture(ResourceHandle handle) const;
    Backend::Shader* shader(ResourceHandle handle) const;
    Prim* resourcePrim(ResourceHandle handle) const;
    std::size_t usageCount(ResourceHandle handle) const;
    const std::vector<std::string>& usagePaths(ResourceHandle handle) const;
    bool isUsed(ResourceHandle handle) const { return usageCount(handle) > 0; }
    void invalidateUsageCache() const;

    std::size_t size() const { return _entries.size(); }
    void clear();

  private:
    ResourceHandle registerEntry(Entry entry);
    Prim* ensureResourcePrim(const Entry& entry);
    void rebuildUsageCache() const;
    static const char* folderForType(ResourceType type);
    static std::string safeSegment(const std::string& value);

    SceneBackend* _scene = nullptr;
    ResourceHandle _nextHandle = 1;
    std::unordered_map<ResourceHandle, Entry> _entries;
    mutable bool _usageCacheDirty = true;
    mutable std::unordered_map<ResourceHandle, std::size_t> _usageCache;
    mutable std::unordered_map<ResourceHandle, std::vector<std::string>>
        _usagePathCache;
    mutable std::vector<std::string> _emptyUsagePaths;
};

} // namespace Scene
} // namespace KE

#endif
