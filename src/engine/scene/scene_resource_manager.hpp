#ifndef _SCENE_RESOURCE_MANAGER_HPP_
#define _SCENE_RESOURCE_MANAGER_HPP_

#include "engine/scene/component/resource_component.hpp"
#include "engine/scene/scene_backend.hpp"
#include "engine/graphics/backend/base/graphics_device.hpp"

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace KE {

class Material;

namespace Backend {
class Texture;
} // namespace Backend

namespace Scene {

enum class ShaderLanguage { GLSL, WGSL };
enum class AuthoredPipelineType { Graphics, Compute };

struct ShaderSourceResource {
    Backend::ShaderType stage = Backend::ShaderType::Vertex;
    ShaderLanguage language = ShaderLanguage::GLSL;
    std::string source;
    std::string entryPoint = "main";
};

struct PipelineResource {
    AuthoredPipelineType type = AuthoredPipelineType::Graphics;
    std::vector<ResourceHandle> shaderSources;
    std::vector<std::string> variants;
    std::string stateSummary;
};

// Immutable file or packed RGB8/RGBA8 source; renderers own GPU realizations.
struct TextureSourceResource {
    std::string path;
    bool srgb = false;
    bool flip = true;
    Backend::SamplerDesc sampler;
    int width = 0;
    int height = 0;
    int channels = 0;
    // Original row order; flip applies during upload, not to the stored source.
    std::vector<uint8_t> pixels;
    void validate() const;
};

// Scene-owned manager for shared resource identities and editor mirrors.
//
// SceneResourceManager is deliberately not a frame-updated system. It records
// shared resource objects, assigns stable handles, and mirrors entries into the
// scene as `/.Resources/...` prims with ResourceComponent metadata.
//
// Resource ownership model:
// - The manager is the conceptual owner/catalog for mesh/material/texture
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
        std::shared_ptr<const TextureSourceResource> textureSource;
        std::shared_ptr<ShaderSourceResource> shaderSource;
        std::shared_ptr<PipelineResource> pipeline;
    };

    explicit SceneResourceManager(SceneBackend* scene = nullptr);

    void bindScene(SceneBackend* scene);
    SceneBackend* scene() const { return _scene; }

    // Reuses the identity and metadata of an already registered mesh pointer.
    ResourceHandle registerMesh(const std::string& name,
                                std::shared_ptr<MeshData> mesh,
                                const std::string& uri = {});
    ResourceHandle registerMaterial(const std::string& name, Material* material,
                                    const std::string& uri = {});
    ResourceHandle
    registerTexture(const std::string& name, Backend::Texture* texture,
                    const std::string& uri = {},
                    std::optional<TextureSourceResource> source = std::nullopt);
    ResourceHandle registerTextureSource(const std::string& name,
                                         TextureSourceResource source);
    std::shared_ptr<const TextureSourceResource>
    textureSource(ResourceHandle handle) const;
    ResourceHandle registerShaderSource(
        const std::string& name, ShaderSourceResource shaderSource,
        const std::string& uri = {});
    ResourceHandle registerPipeline(const std::string& name,
                                    PipelineResource pipeline,
                                    const std::string& uri = {});

    const Entry* entry(ResourceHandle handle) const;
    Entry* entry(ResourceHandle handle);

    std::shared_ptr<MeshData> mesh(ResourceHandle handle) const;
    Material* material(ResourceHandle handle) const;
    ResourceHandle materialHandle(const Material* material) const;
    Backend::Texture* texture(ResourceHandle handle) const;
    ResourceHandle textureHandle(const Backend::Texture* texture) const;
    const ShaderSourceResource* shaderSource(ResourceHandle handle) const;
    const PipelineResource* pipeline(ResourceHandle handle) const;
    Prim* resourcePrim(ResourceHandle handle) const;
    std::size_t usageCount(ResourceHandle handle) const;
    const std::vector<std::string>& usagePaths(ResourceHandle handle) const;
    bool isUsed(ResourceHandle handle) const { return usageCount(handle) > 0; }
    void addExternalUsage(ResourceHandle handle, const std::string& path);
    void removeExternalUsage(ResourceHandle handle, const std::string& path);
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
    std::unordered_map<const MeshData*, ResourceHandle> _meshHandles;
    std::unordered_map<const Material*, ResourceHandle> _materialHandles;
    std::unordered_map<const Backend::Texture*, ResourceHandle> _textureHandles;
    mutable bool _usageCacheDirty = true;
    mutable std::unordered_map<ResourceHandle, std::size_t> _usageCache;
    mutable std::unordered_map<ResourceHandle, std::vector<std::string>>
        _usagePathCache;
    std::unordered_map<ResourceHandle, std::unordered_set<std::string>>
        _externalUsagePaths;
    mutable std::vector<std::string> _emptyUsagePaths;
};

} // namespace Scene
} // namespace KE

#endif
