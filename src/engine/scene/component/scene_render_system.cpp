#include "scene_render_system.hpp"

#include "engine/graphics/material/material.hpp"
#include "engine/graphics/renderer/renderer.hpp"
#include "engine/scene/component/render_component.hpp"
#include "engine/scene/component/mesh_component.hpp"
#include "engine/scene/native/prim.hpp"
#include "engine/scene/scene_resource_manager.hpp"

#include <stdexcept>
#include <vector>

namespace KE {
namespace Scene {

SceneRenderSystem::~SceneRenderSystem() { clear(); }

void SceneRenderSystem::bind(Renderer* renderer,
                             SceneResourceManager* resources) {
    if (!_registrations.empty() &&
        (renderer != _renderer || resources != _resources))
        throw std::runtime_error(
            "cannot rebind SceneRenderSystem with active registrations");
    _renderer = renderer;
    _resources = resources;
}

void SceneRenderSystem::registerResources(RenderComponent& component,
                                          Material* material) {
    if (!_resources)
        return;

    auto& prim = *component.owner();
    const auto mesh = component.resolveMeshData();
    if (mesh) {
        const auto meshComponent = prim.getMeshComponent();
        const auto handle = _resources->registerMesh(
            prim.getName(), mesh, "scene://mesh" + prim.getPath());
        meshComponent->setResourceHandle(handle);
    }
    if (material &&
        _resources->materialHandle(material) == InvalidResourceHandle)
        _resources->registerMaterial(prim.getName(), material,
                                     "scene://material" + prim.getPath());
    _resources->invalidateUsageCache();
}

void SceneRenderSystem::syncScene(SceneBackend& scene,
                                  const SceneResourceManager& resources) {
    // Batch registration supplies instance arrays without a full snapshot.
    if (!_renderer || _renderer->capabilities().sceneBatches ||
        _renderer->rasterizer() || !_renderer->capabilities().staticScene)
        return;
    _renderer->syncScene(_collector.collectRegistered(scene, resources, *this));
}

void SceneRenderSystem::validateRegistration(
    const std::shared_ptr<RenderComponent>& component) const {
    if (!_renderer)
        throw std::runtime_error(
            "SceneRenderSystem is not bound to a Renderer");
    if (!component || !component->isAttached() || !component->owner())
        throw std::runtime_error(
            "SceneRenderSystem requires an attached RenderComponent");
    if (isRegistered(*component))
        throw std::runtime_error("RenderComponent is already registered");
}

RayPickResult SceneRenderSystem::buildPrimSelection(Prim& prim) const {
    RayPickResult selection;
    if (_renderer && _renderer->buildPrimSelection(prim, selection))
        return selection;
    selection.hit = true;
    selection.prim = &prim;
    selection.sceneInstanceId = _collector.instanceId(prim);
    return selection;
}

RayPickResult SceneRenderSystem::resolvePick(const RenderPickHit& hit) const {
    if (!_renderer || !hit.hit)
        return {};
    if (_renderer->capabilities().sceneBatches || _renderer->rasterizer())
        return _renderer->resolvePick(hit);
    const auto component = _collector.resolve(hit.sceneInstanceId);
    if (!component || !component->isAttached() || !isRegistered(*component))
        return {};
    auto* prim = component->owner();
    if (!prim || !prim->isActiveInHierarchy() || !prim->isVisibleInHierarchy())
        return {};
    RayPickResult result;
    static_cast<RenderPickHit&>(result) = hit;
    result.prim = prim;
    return result;
}

std::shared_ptr<RenderComponent>
SceneRenderSystem::addSkinnedRenderable(Prim& prim, Material* material,
                                        const SkinnedMeshData& skinnedMesh,
                                        TransformSource source) {
    auto component = prim.getRenderComponent();
    const bool created = !component;
    if (!component)
        component = prim.addRenderComponent();
    try {
        component->setTransformSource(source);
        if (!registerSkinnedRenderable(component, material, skinnedMesh)) {
            if (created)
                prim.removeRenderComponent();
            return nullptr;
        }
        prim.setMaterial(material);
    } catch (...) {
        if (created)
            prim.removeRenderComponent();
        throw;
    }
    return component;
}

std::shared_ptr<RenderComponent>
SceneRenderSystem::addRenderable(Prim& prim, Material* material,
                                 TransformSource source) {
    auto component = prim.getRenderComponent();
    const bool created = !component;
    if (!component)
        component = prim.addRenderComponent();
    try {
        component->setTransformSource(source);
        if (!registerRenderable(component, material)) {
            if (created)
                prim.removeRenderComponent();
            return nullptr;
        }
        prim.setMaterial(material);
    } catch (...) {
        if (created)
            prim.removeRenderComponent();
        throw;
    }
    return component;
}

void SceneRenderSystem::finishRegistration(
    const std::shared_ptr<RenderComponent>& component,
    RenderableHandle renderable, Material* material,
    const SkinnedMeshData* skinnedMesh) {
    Registration registration;
    registration.component = component;
    registration.prim = component->owner();
    registration.handle = renderable;
    registration.mesh = component->resolveMeshData().get();
    registration.material = material;
    registration.state = {component->isDoubleSided(), component->castsShadow(),
                          component->alphaMode(), component->alphaCutoff()};
    if (skinnedMesh)
        registration.skinnedMesh = *skinnedMesh;
    _registrations.emplace(component.get(), registration);
    component->setRegistrationCallbacks(
        [this](RenderComponent& detached) { unregister(detached); },
        [this](RenderComponent& changed) { syncState(changed); });
    if (renderable != InvalidHandle)
        _renderer->setSceneBatchState(
            renderable,
            {component->isDoubleSided(), component->castsShadow(),
             component->alphaMode(), component->alphaCutoff()});
}

bool SceneRenderSystem::registerRenderable(
    const std::shared_ptr<RenderComponent>& component, Material* material) {
    validateRegistration(component);
    registerResources(*component, material);
    RenderableHandle handle = InvalidHandle;
    if (_renderer->capabilities().sceneBatches || _renderer->rasterizer()) {
        handle = _renderer->addRenderable(material, component->owner(),
                                          component->transformSource());
        if (handle == InvalidHandle)
            return false;
    } else {
        if (!_renderer->capabilities().staticScene ||
            component->transformSource() != TransformSource::SceneGraph)
            throw std::runtime_error(
                "Render backend requires static SceneGraph registration");
        if (!material)
            throw std::invalid_argument(
                "Scene registration requires a material");
    }
    finishRegistration(component, handle, material);
    return true;
}

bool SceneRenderSystem::registerSkinnedRenderable(
    const std::shared_ptr<RenderComponent>& component, Material* material,
    const SkinnedMeshData& skinnedMesh) {
    validateRegistration(component);
    registerResources(*component, material);
    const auto handle = _renderer->addSkinnedRenderable(
        material, component->owner(), skinnedMesh,
        component->transformSource());
    if (handle == InvalidHandle)
        return false;
    finishRegistration(component, handle, material, &skinnedMesh);
    return true;
}

const SceneRenderSystem::Registration&
SceneRenderSystem::requireRegistration(const RenderComponent& component) const {
    auto it = _registrations.find(&component);
    if (it == _registrations.end())
        throw std::runtime_error("RenderComponent is not registered");
    return it->second;
}

const SceneRenderSystem::Registration&
SceneRenderSystem::requireBatch(const RenderComponent& component) const {
    const auto& registration = requireRegistration(component);
    if (registration.handle == InvalidHandle)
        throw std::runtime_error("This operation requires a renderer batch");
    return registration;
}

void SceneRenderSystem::syncState(RenderComponent& component) {
    auto& registration = _registrations.at(&component);
    if (registration.handle == InvalidHandle)
        return;

    const auto mesh = component.resolveMeshData();
    auto* material = registration.prim->getMaterial();
    const SceneBatchState state{component.isDoubleSided(),
                                component.castsShadow(), component.alphaMode(),
                                component.alphaCutoff()};
    const bool policyChanged =
        component.transformSource() == TransformSource::SceneGraph &&
        !registration.skinnedMesh && state != registration.state;
    if (mesh.get() != registration.mesh || material != registration.material ||
        policyChanged) {
        if (component.transformSource() == TransformSource::ExternalBuffer)
            throw std::runtime_error(
                "cannot replace mesh or material of a registered "
                "ExternalBuffer renderable");
        if (registration.skinnedMesh && mesh.get() != registration.mesh)
            throw std::runtime_error(
                "cannot replace mesh of a registered skinned renderable; "
                "unregister it first");
        if (!material)
            throw std::runtime_error("registered renderable requires a material");

        // Keep the old batch attached until the replacement is ready.
        const auto replacement =
            registration.skinnedMesh
                ? _renderer->addSkinnedRenderable(material, registration.prim,
                                                  *registration.skinnedMesh,
                                                  component.transformSource())
                : _renderer->addRenderable(material, registration.prim,
                                           component.transformSource());
        if (replacement == InvalidHandle)
            throw std::runtime_error(
                "failed to register replacement scene batch");
        try {
            _renderer->setSceneBatchState(replacement, state);
            for (const auto& [texture, slot] : registration.textureBindings)
                _renderer->setRenderableTexture(replacement, texture, slot);
        } catch (...) {
            if (replacement != registration.handle)
                _renderer->removePrim(replacement, registration.prim);
            throw;
        }
        if (replacement != registration.handle)
            _renderer->removePrim(registration.handle, registration.prim);
        registration.handle = replacement;
        registration.mesh = mesh.get();
        registration.material = material;
        registration.state = state;
        return;
    }
    _renderer->setSceneBatchState(registration.handle, state);
    registration.state = state;
}

bool SceneRenderSystem::unregister(RenderComponent& component) {
    auto it = _registrations.find(&component);
    if (it == _registrations.end())
        return false;

    Registration registration = it->second;
    _collector.forget(component);
    _registrations.erase(it);
    component.clearRegistrationCallbacks();
    if (_renderer && registration.prim && registration.handle != InvalidHandle)
        _renderer->removePrim(registration.handle, registration.prim);
    return true;
}

void SceneRenderSystem::detachSubtree(Prim& root) {
    std::vector<Prim*> subtree;
    root.traverse([&subtree](Prim* prim) {
        if (prim)
            subtree.push_back(prim);
    });
    for (Prim* prim : subtree) {
        if (prim->hasRenderComponent())
            prim->removeRenderComponent();
    }
}

void SceneRenderSystem::setDoubleSided(RenderComponent& component,
                                       bool doubleSided) {
    requireRegistration(component);
    component.setDoubleSided(doubleSided);
}

void SceneRenderSystem::setCastsShadow(RenderComponent& component,
                                       bool castsShadow) {
    requireRegistration(component);
    component.setCastsShadow(castsShadow);
}

void SceneRenderSystem::setAlphaMode(RenderComponent& component, AlphaMode mode,
                                     float cutoff) {
    requireRegistration(component);
    component.setAlphaMode(mode, cutoff);
}

bool SceneRenderSystem::setMaterial(RenderComponent& component,
                                    Material* material) {
    const auto& registration = requireRegistration(component);
    if (!material)
        throw std::runtime_error("replacement material must not be null");
    if (component.transformSource() == TransformSource::ExternalBuffer) {
        throw std::runtime_error(
            "dynamic material replacement for ExternalBuffer renderables is "
            "not implemented yet");
    }

    registration.prim->setMaterial(material);
    return true;
}

void SceneRenderSystem::setBatchTexture(RenderComponent& component,
                                        Backend::Texture* texture,
                                        TextureRole role) {
    setBatchTexture(component, texture, textureRoleSlot(role));
}

void SceneRenderSystem::setBatchTexture(RenderComponent& component,
                                        Backend::Texture* texture, int slot) {
    requireBatch(component);
    auto it = _registrations.find(&component);
    _renderer->setRenderableTexture(it->second.handle, texture, slot);

    auto& bindings = it->second.textureBindings;
    bool found = false;
    for (auto& [storedTexture, storedSlot] : bindings) {
        if (storedSlot == slot) {
            storedTexture = texture;
            found = true;
            break;
        }
    }
    if (!found)
        bindings.emplace_back(texture, slot);
}

void SceneRenderSystem::setExternalBuffer(RenderComponent& component,
                                          const ExternalBufferDesc& desc) {
    const auto& registration = requireBatch(component);
    if (component.transformSource() != TransformSource::ExternalBuffer)
        throw std::runtime_error(
            "RenderComponent must be registered with ExternalBuffer "
            "transform source");
    _renderer->setRenderableExternalBuffer(registration.handle, desc);
}

void SceneRenderSystem::setInstanceColors(
    RenderComponent& component, const std::vector<glm::vec4>& colors) {
    const auto& registration = requireBatch(component);
    if (component.transformSource() != TransformSource::ExternalBuffer)
        throw std::runtime_error(
            "Instance colors require an ExternalBuffer renderable; "
            "use Prim display color for SceneGraph objects");
    _renderer->setRenderableColors(registration.handle, colors);
}

void SceneRenderSystem::updateInstances(
    RenderComponent& component, const std::vector<glm::mat4>& transforms,
    const std::vector<glm::vec4>* colors) {
    const auto& registration = requireBatch(component);
    _renderer->updateRenderableTransforms(registration.handle, transforms,
                                          colors);
}

void SceneRenderSystem::updateGeometry(RenderComponent& component,
                                       const std::vector<glm::vec3>& positions,
                                       const std::vector<glm::vec3>& normals) {
    const auto& registration = requireBatch(component);
    _renderer->updateRenderableGeometry(registration.handle, positions,
                                        normals);
}

void SceneRenderSystem::updateSkinning(
    RenderComponent& component, const std::vector<glm::mat4>& boneMatrices) {
    const auto& registration = requireBatch(component);
    _renderer->updateRenderableSkinningMatrices(registration.handle,
                                                boneMatrices);
}

bool SceneRenderSystem::isRegistered(const RenderComponent& component) const {
    return _registrations.count(&component) != 0;
}

bool SceneRenderSystem::sharesBatch(const RenderComponent& first,
                                    const RenderComponent& second) const {
    const auto firstIt = _registrations.find(&first);
    const auto secondIt = _registrations.find(&second);
    if (firstIt == _registrations.end() || secondIt == _registrations.end())
        return false;
    if (firstIt->second.handle == InvalidHandle ||
        secondIt->second.handle == InvalidHandle)
        throw std::runtime_error(
            "Batch sharing requires Native renderer registrations");
    return firstIt->second.handle == secondIt->second.handle;
}

RenderableHandle
SceneRenderSystem::handle(const RenderComponent& component) const {
    auto it = _registrations.find(&component);
    if (it == _registrations.end())
        return InvalidHandle;
    if (it->second.handle == InvalidHandle)
        throw std::runtime_error(
            "Source-only registration has no Native batch handle");
    return it->second.handle;
}

void SceneRenderSystem::clear() {
    while (!_registrations.empty()) {
        auto it = _registrations.begin();
        Registration registration = it->second;
        _registrations.erase(it);
        if (auto component = registration.component.lock())
            component->clearRegistrationCallbacks();
        if (_renderer && registration.prim &&
            registration.handle != InvalidHandle)
            _renderer->removePrim(registration.handle, registration.prim);
    }
}

} // namespace Scene
} // namespace KE
