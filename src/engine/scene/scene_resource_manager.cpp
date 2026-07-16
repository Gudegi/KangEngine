#include "scene_resource_manager.hpp"

#include "engine/graphics/material/material.hpp"
#include "engine/scene/component/material_binding_component.hpp"
#include "engine/scene/component/mesh_component.hpp"
#include "engine/scene/component/resource_component.hpp"
#include "engine/scene/native/prim.hpp"

#include <cctype>
#include <unordered_set>
#include <utility>
#include <vector>

namespace KE {
namespace Scene {
namespace {
constexpr const char* kResourceRootPath = "/.Resources";

void collectMaterialTextures(const Material* material,
                             std::vector<Backend::Texture*>& out) {
    if (!material)
        return;
    if (const auto* phong = dynamic_cast<const PhongMaterial*>(material)) {
        if (phong->diffuseMap)
            out.push_back(phong->diffuseMap);
        if (phong->specularMap)
            out.push_back(phong->specularMap);
        if (phong->alphaMap)
            out.push_back(phong->alphaMap);
        if (phong->normalMap)
            out.push_back(phong->normalMap);
        return;
    }
    if (const auto* pbr = dynamic_cast<const PBRMaterial*>(material)) {
        if (pbr->baseColorTexture)
            out.push_back(pbr->baseColorTexture);
        if (pbr->normalTexture)
            out.push_back(pbr->normalTexture);
        if (pbr->metallicRoughnessTexture)
            out.push_back(pbr->metallicRoughnessTexture);
        if (pbr->metallicTexture)
            out.push_back(pbr->metallicTexture);
        if (pbr->roughnessTexture)
            out.push_back(pbr->roughnessTexture);
        if (pbr->aoTexture)
            out.push_back(pbr->aoTexture);
        if (pbr->ormTexture)
            out.push_back(pbr->ormTexture);
        if (pbr->emissiveTexture)
            out.push_back(pbr->emissiveTexture);
    }
}
} // namespace

SceneResourceManager::SceneResourceManager(SceneBackend* scene)
    : _scene(scene) {}

void SceneResourceManager::bindScene(SceneBackend* scene) {
    _scene = scene;
    invalidateUsageCache();
}

ResourceHandle
SceneResourceManager::registerMesh(const std::string& name,
                                   std::shared_ptr<MeshData> mesh,
                                   const std::string& uri) {
    Entry entry;
    entry.type = ResourceType::Mesh;
    entry.name = name;
    entry.uri = uri;
    entry.mesh = std::move(mesh);
    return registerEntry(std::move(entry));
}

ResourceHandle SceneResourceManager::registerMaterial(const std::string& name,
                                                      Material* material,
                                                      const std::string& uri) {
    Entry entry;
    entry.type = ResourceType::Material;
    entry.name = name;
    entry.uri = uri;
    entry.material = material;
    return registerEntry(std::move(entry));
}

ResourceHandle SceneResourceManager::registerTexture(const std::string& name,
                                                     Backend::Texture* texture,
                                                     const std::string& uri) {
    Entry entry;
    entry.type = ResourceType::Texture;
    entry.name = name;
    entry.uri = uri;
    entry.texture = texture;
    return registerEntry(std::move(entry));
}

ResourceHandle SceneResourceManager::registerShader(const std::string& name,
                                                    Backend::Shader* shader,
                                                    const std::string& uri) {
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
    invalidateUsageCache();
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

Backend::Texture* SceneResourceManager::texture(ResourceHandle handle) const {
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

std::size_t SceneResourceManager::usageCount(ResourceHandle handle) const {
    if (_usageCacheDirty)
        rebuildUsageCache();
    const auto it = _usageCache.find(handle);
    if (it == _usageCache.end())
        return 0;
    return it->second;
}

const std::vector<std::string>&
SceneResourceManager::usagePaths(ResourceHandle handle) const {
    if (_usageCacheDirty)
        rebuildUsageCache();
    const auto it = _usagePathCache.find(handle);
    if (it == _usagePathCache.end())
        return _emptyUsagePaths;
    return it->second;
}

void SceneResourceManager::invalidateUsageCache() const {
    _usageCacheDirty = true;
}

void SceneResourceManager::rebuildUsageCache() const {
    _usageCache.clear();
    _usageCache.reserve(_entries.size());
    _usagePathCache.clear();
    _usagePathCache.reserve(_entries.size());
    _emptyUsagePaths.clear();

    std::unordered_map<MeshData*, std::vector<ResourceHandle>> meshHandles;
    std::unordered_map<Material*, std::vector<ResourceHandle>> materialHandles;
    std::unordered_map<Backend::Texture*, std::vector<ResourceHandle>>
        textureHandles;
    std::unordered_map<Backend::Shader*, std::vector<ResourceHandle>>
        shaderHandles;

    for (const auto& [handle, e] : _entries) {
        _usageCache[handle] = 0;
        _usagePathCache[handle] = {};
        switch (e.type) {
        case ResourceType::Mesh:
            if (e.mesh)
                meshHandles[e.mesh.get()].push_back(handle);
            break;
        case ResourceType::Material:
            if (e.material)
                materialHandles[e.material].push_back(handle);
            break;
        case ResourceType::Texture:
            if (e.texture)
                textureHandles[e.texture].push_back(handle);
            break;
        case ResourceType::Shader:
            if (e.shader)
                shaderHandles[e.shader].push_back(handle);
            break;
        case ResourceType::Unknown:
            break;
        }
    }

    if (!_scene || !_scene->getRootPrim()) {
        _usageCacheDirty = false;
        return;
    }

    const auto addHandles = [](std::unordered_set<ResourceHandle>& used,
                               const auto& map, auto* key) {
        if (!key)
            return;
        const auto it = map.find(key);
        if (it == map.end())
            return;
        for (ResourceHandle handle : it->second)
            used.insert(handle);
    };

    _scene->getRootPrim()->traverse([&](Prim* prim) {
        if (!prim || prim->getType() == PrimType::Resource)
            return;

        std::unordered_set<ResourceHandle> usedByPrim;
        if (auto mesh = prim->getMeshComponent()) {
            const ResourceHandle directHandle = mesh->resourceHandle();
            if (directHandle != InvalidResourceHandle &&
                _usageCache.find(directHandle) != _usageCache.end())
                usedByPrim.insert(directHandle);
            if (auto meshData = mesh->meshData())
                addHandles(usedByPrim, meshHandles, meshData.get());
        }

        Material* material = prim->getMaterial();
        addHandles(usedByPrim, materialHandles, material);
        if (material) {
            addHandles(usedByPrim, shaderHandles, material->getShader());
            std::vector<Backend::Texture*> textures;
            collectMaterialTextures(material, textures);
            for (Backend::Texture* texture : textures)
                addHandles(usedByPrim, textureHandles, texture);
        }

        for (ResourceHandle used : usedByPrim) {
            ++_usageCache[used];
            _usagePathCache[used].push_back(prim->getPath());
        }
    });

    _usageCacheDirty = false;
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
    _usageCache.clear();
    _usagePathCache.clear();
    invalidateUsageCache();
    _nextHandle = 1;
}

Prim* SceneResourceManager::ensureResourcePrim(const Entry& entry) {
    if (!_scene)
        return nullptr;

    const std::string name =
        entry.name.empty() ? resourceTypeLabel(entry.type) : entry.name;
    const std::string path =
        std::string(kResourceRootPath) + "/" + folderForType(entry.type) + "/" +
        safeSegment(name) + "_" + std::to_string(entry.handle);
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
