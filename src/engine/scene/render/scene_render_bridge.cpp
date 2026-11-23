// Collects instance state and converts material values for renderers.
#include "scene_render_bridge.hpp"

#include "engine/graphics/material/material.hpp"
#include "engine/scene/component/mesh_component.hpp"
#include "engine/scene/component/selection_component.hpp"
#include "engine/scene/native/prim.hpp"
#include "engine/scene/scene_resource_manager.hpp"

#include <algorithm>
#include <atomic>
#include <stdexcept>

namespace KE {
namespace Scene {
namespace {
std::atomic<RenderInstanceId> nextInstanceId{1};

bool visible(const Prim& prim) {
    return prim.isActiveInHierarchy() && prim.isVisibleInHierarchy();
}

bool pickable(const Prim& prim) {
    const auto selection = prim.getSelectionComponent();
    return !selection || selection->isPickable();
}
} // namespace

RenderMeshData SceneRenderBridge::meshData(const Prim& prim) {
    RenderMeshData input;
    input.mesh = prim.resolveMeshData();
    if (const auto component = prim.getMeshComponent())
        input.resource = component->resourceHandle();
    return input;
}

void SceneRenderBridge::add(Prim& prim) {
    if (!_prims.insert(&prim).second)
        return;
    _entries.push_back(
        {&prim, nextInstanceId.fetch_add(1, std::memory_order_relaxed)});
}

bool SceneRenderBridge::contains(const Prim& prim) const {
    return _prims.count(&prim) != 0;
}

RenderMaterialData SceneRenderBridge::materialData(const Material& material) {
    RenderMaterialData input;
    input.shadingModel = material.shadingModel();
    input.colorStyle = material.colorStyle();
    input.unlit = material.isUnlit();
    input.hasNativeTextures = !material.texturePaths.empty();
    if (input.shadingModel == MaterialShadingModel::Color) {
        const auto& color = static_cast<const ColorMaterial&>(material);
        input.hasNativeTextures |= color.baseColorTexture || color.normalTexture;
    } else if (input.shadingModel == MaterialShadingModel::Phong) {
        const auto& phong = static_cast<const PhongMaterial&>(material);
        input.baseColor = glm::vec4(phong.diffuse, 1.0f);
        input.ambient = phong.ambient;
        input.specular = phong.specular;
        input.shininess = phong.shininess;
        input.hasNativeTextures |= phong.diffuseMap || phong.specularMap ||
                                   phong.alphaMap || phong.normalMap;
    } else if (input.shadingModel == MaterialShadingModel::PBR) {
        const auto& pbr = static_cast<const PBRMaterial&>(material);
        input.baseColor = pbr.baseColor;
        input.metallic = pbr.metallic;
        input.roughness = pbr.roughness;
        input.emissiveColor = pbr.emissiveColor;
        input.emissiveStrength = pbr.emissiveStrength;
        input.hasNativeTextures |=
            pbr.baseColorTexture || pbr.normalTexture ||
            pbr.metallicRoughnessTexture || pbr.metallicTexture ||
            pbr.roughnessTexture || pbr.aoTexture || pbr.ormTexture ||
            pbr.emissiveTexture || pbr.opacityTexture;
    }
    return input;
}

std::vector<std::pair<const char*, Backend::Texture*>>
SceneRenderBridge::materialTextures(const Material& material) {
    std::vector<std::pair<const char*, Backend::Texture*>> textures;
    auto add = [&](const char* slot, Backend::Texture* texture) {
        if (texture)
            textures.emplace_back(slot, texture);
    };
    if (const auto* color = dynamic_cast<const ColorMaterial*>(&material)) {
        add("base_color_texture", color->baseColorTexture);
        add("normal_texture", color->normalTexture);
    } else if (const auto* phong = dynamic_cast<const PhongMaterial*>(&material)) {
        add("diffuse_map", phong->diffuseMap);
        add("specular_map", phong->specularMap);
        add("alpha_map", phong->alphaMap);
        add("normal_map", phong->normalMap);
    } else if (const auto* pbr = dynamic_cast<const PBRMaterial*>(&material)) {
        add("base_color_texture", pbr->baseColorTexture);
        add("normal_texture", pbr->normalTexture);
        add("metallic_roughness_texture", pbr->metallicRoughnessTexture);
        add("metallic_texture", pbr->metallicTexture);
        add("roughness_texture", pbr->roughnessTexture);
        add("ao_texture", pbr->aoTexture);
        add("orm_texture", pbr->ormTexture);
        add("emissive_texture", pbr->emissiveTexture);
        add("opacity_texture", pbr->opacityTexture);
    }
    return textures;
}

RenderMaterialSource
SceneRenderBridge::materialSource(const Material& material,
                                  const SceneResourceManager* resources) {
    RenderMaterialSource result;
    result.values = materialData(material);
    if (!resources) {
        result.values.hasNativeTextures |= !material.textureResources.empty();
        return result;
    }
    result.resource = resources->materialHandle(&material);
    for (const auto& [slot, handle] : material.textureResources) {
        auto source = resources->textureSource(handle);
        if (!source)
            throw std::invalid_argument(
                "Material texture handle has no source: " + slot);
        result.textures.emplace(slot,
                                std::make_pair(handle, std::move(source)));
    }

    // Resolve GPU bindings to catalog sources without readback.
    result.values.hasNativeTextures = !material.texturePaths.empty();
    for (const auto& [slot, texture] : materialTextures(material)) {
        const auto handle = resources->textureHandle(texture);
        auto source = resources->textureSource(handle);
        if (!source) {
            result.values.hasNativeTextures = true;
            continue;
        }
        auto [it, inserted] = result.textures.emplace(
            slot, std::make_pair(handle, std::move(source)));
        if (!inserted && it->second.first != handle)
            throw std::invalid_argument(
                std::string("Native and source texture bindings disagree: ") +
                slot);
    }
    return result;
}

void SceneRenderBridge::remove(const Prim& prim) {
    if (!_prims.erase(&prim))
        return;
    _entries.erase(
        std::remove_if(_entries.begin(), _entries.end(),
                       [&](const Entry& entry) { return entry.prim == &prim; }),
        _entries.end());
    // Preserve frame indices until collect(), but never resolve a deleted Prim.
    for (size_t i = 0; i < _instancePrims.size(); ++i) {
        if (_instancePrims[i] == &prim) {
            _instancePrims[i] = nullptr;
            _instances.ids[i] = InvalidRenderInstance;
            _instances.pickable[i] = false;
        }
    }
}

void SceneRenderBridge::collect() {
    _instances.transforms.clear();
    _instances.colors.clear();
    _instances.transforms.reserve(_entries.size());
    _instances.colors.reserve(_entries.size());
    _instances.ids.resize(_entries.size());
    _instances.pickable.resize(_entries.size());
    _instancePrims.resize(_entries.size());
    size_t instanceCount = 0;
    for (const auto& entry : _entries) {
        if (!visible(*entry.prim))
            continue;
        const auto color = entry.prim->getDisplayColorAlpha();
        _instances.transforms.push_back(entry.prim->computeModelMatrix());
        _instances.colors.push_back(color ? *color : glm::vec4(1.0f));
        _instances.ids[instanceCount] = entry.id;
        _instances.pickable[instanceCount] = true;
        _instancePrims[instanceCount] = entry.prim;
        ++instanceCount;
    }
    _instances.ids.resize(instanceCount);
    _instances.pickable.resize(instanceCount);
    _instancePrims.resize(instanceCount);
}

void SceneRenderBridge::refreshSelection() {
    for (size_t i = 0; i < _instancePrims.size(); ++i) {
        const Prim* prim = _instancePrims[i];
        _instances.pickable[i] = prim && visible(*prim) && pickable(*prim);
    }
}

Prim* SceneRenderBridge::resolve(RenderInstanceId id) const {
    if (id == InvalidRenderInstance)
        return nullptr;
    for (const auto& entry : _entries) {
        if (entry.id == id)
            return entry.prim;
    }
    return nullptr;
}

int SceneRenderBridge::instanceIndex(const Prim& prim) const {
    const auto found =
        std::find(_instancePrims.begin(), _instancePrims.end(), &prim);
    return found == _instancePrims.end()
               ? -1
               : static_cast<int>(found - _instancePrims.begin());
}

int SceneRenderBridge::instanceIndex(RenderInstanceId id) const {
    if (id == InvalidRenderInstance)
        return -1;
    const auto found =
        std::find(_instances.ids.begin(), _instances.ids.end(), id);
    return found == _instances.ids.end()
               ? -1
               : static_cast<int>(found - _instances.ids.begin());
}

} // namespace Scene
} // namespace KE
