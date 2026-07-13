#include "scene_resource_manager.hpp"

#include "engine/scene/component/resource_component.hpp"
#include "engine/scene/native/prim.hpp"

#include <cctype>
#include <utility>
#include <vector>

namespace KE {
namespace Scene {
namespace {
constexpr const char* kResourceRootPath = "/.Resources";
}

SceneResourceManager::SceneResourceManager(SceneBackend* scene)
    : _scene(scene) {}

void SceneResourceManager::bindScene(SceneBackend* scene) { _scene = scene; }

ResourceHandle SceneResourceManager::registerMesh(
    const std::string& name, std::shared_ptr<MeshData> mesh,
    const std::string& uri) {
    Entry entry;
    entry.type = ResourceType::Mesh;
    entry.name = name;
    entry.uri = uri;
    entry.mesh = std::move(mesh);
    return registerEntry(std::move(entry));
}

ResourceHandle SceneResourceManager::registerMaterial(
    const std::string& name, Material* material, const std::string& uri) {
    Entry entry;
    entry.type = ResourceType::Material;
    entry.name = name;
    entry.uri = uri;
    entry.material = material;
    return registerEntry(std::move(entry));
}

ResourceHandle SceneResourceManager::registerTexture(
    const std::string& name, Backend::Texture* texture,
    const std::string& uri) {
    Entry entry;
    entry.type = ResourceType::Texture;
    entry.name = name;
    entry.uri = uri;
    entry.texture = texture;
    return registerEntry(std::move(entry));
}

ResourceHandle SceneResourceManager::registerShader(
    const std::string& name, Backend::Shader* shader, const std::string& uri) {
    Entry entry;
    entry.type = ResourceType::Shader;
    entry.name = name;
    entry.uri = uri;
    entry.shader = shader;
    return registerEntry(std::move(entry));
}

ResourceHandle SceneResourceManager::registerEntry(Entry entry) {
    const ResourceHandle handle = _nextHandle++;
    entry.handle = handle;
    entry.prim = ensureResourcePrim(entry);
    _entries.emplace(handle, std::move(entry));
    return handle;
}

const SceneResourceManager::Entry*
SceneResourceManager::entry(ResourceHandle handle) const {
    auto it = _entries.find(handle);
    return it == _entries.end() ? nullptr : &it->second;
}

SceneResourceManager::Entry*
SceneResourceManager::entry(ResourceHandle handle) {
    auto it = _entries.find(handle);
    return it == _entries.end() ? nullptr : &it->second;
}

std::shared_ptr<MeshData>
SceneResourceManager::mesh(ResourceHandle handle) const {
    const Entry* e = entry(handle);
    return e ? e->mesh : nullptr;
}

Material* SceneResourceManager::material(ResourceHandle handle) const {
    const Entry* e = entry(handle);
    return e ? e->material : nullptr;
}

Backend::Texture*
SceneResourceManager::texture(ResourceHandle handle) const {
    const Entry* e = entry(handle);
    return e ? e->texture : nullptr;
}

Backend::Shader* SceneResourceManager::shader(ResourceHandle handle) const {
    const Entry* e = entry(handle);
    return e ? e->shader : nullptr;
}

Prim* SceneResourceManager::resourcePrim(ResourceHandle handle) const {
    const Entry* e = entry(handle);
    return e ? e->prim : nullptr;
}

void SceneResourceManager::clear() {
    if (_scene) {
        std::vector<std::string> resourcePrimPaths;
        resourcePrimPaths.reserve(_entries.size());
        for (const auto& [handle, entry] : _entries) {
            (void)handle;
            if (entry.prim)
                resourcePrimPaths.push_back(entry.prim->getPath());
        }
        for (const std::string& path : resourcePrimPaths)
            _scene->removePrim(path);
    }
    _entries.clear();
    _nextHandle = 1;
}

Prim* SceneResourceManager::ensureResourcePrim(const Entry& entry) {
    if (!_scene)
        return nullptr;

    const std::string name =
        entry.name.empty() ? resourceTypeLabel(entry.type) : entry.name;
    const std::string path = std::string(kResourceRootPath) + "/" +
                             folderForType(entry.type) + "/" +
                             safeSegment(name) + "_" +
                             std::to_string(entry.handle);
    Prim* prim = _scene->definePrim(path, PrimType::Resource);
    if (!prim)
        return nullptr;
    if (Prim* root = _scene->getPrimAtPath(kResourceRootPath))
        root->setManipulationPolicy(ManipulationPolicy::Disabled);
    if (Prim* folder = _scene->getPrimAtPath(std::string(kResourceRootPath) +
                                             "/" + folderForType(entry.type)))
        folder->setManipulationPolicy(ManipulationPolicy::Disabled);
    prim->setManipulationPolicy(ManipulationPolicy::Disabled);

    auto component = prim->getResourceComponent();
    if (!component)
        component = prim->addResourceComponent();
    component->setHandle(entry.handle);
    component->setType(entry.type);
    component->setDisplayName(name);
    component->setUri(entry.uri);

    // Resource prims are editor-visible metadata mirrors only. The manager
    // entry owns the resource payload; renderable prims bind mesh geometry
    // directly through MeshComponent and keep this handle as resource identity.
    return prim;
}

const char* SceneResourceManager::folderForType(ResourceType type) {
    switch (type) {
    case ResourceType::Mesh:
        return "Meshes";
    case ResourceType::Material:
        return "Materials";
    case ResourceType::Texture:
        return "Textures";
    case ResourceType::Shader:
        return "Shaders";
    case ResourceType::Unknown:
        return "Unknown";
    }
    return "Unknown";
}

std::string SceneResourceManager::safeSegment(const std::string& value) {
    std::string result;
    result.reserve(value.size());
    for (unsigned char ch : value) {
        if (std::isalnum(ch) || ch == '_' || ch == '-' || ch == '.')
            result.push_back(static_cast<char>(ch));
        else
            result.push_back('_');
    }
    if (result.empty())
        result = "Resource";
    return result;
}

} // namespace Scene
} // namespace KE
