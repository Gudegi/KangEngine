#include "scene_render_system.hpp"

#include "engine/graphics/renderer/renderer.hpp"
#include "engine/scene/component/render_component.hpp"
#include "engine/scene/native/prim.hpp"

#include <stdexcept>
#include <vector>

namespace KE {
namespace Scene {

SceneRenderSystem::~SceneRenderSystem() { clear(); }

void SceneRenderSystem::bind(Renderer* renderer) {
    if (!_registrations.empty() && renderer != _renderer)
        throw std::runtime_error(
            "cannot rebind SceneRenderSystem with active registrations");
    _renderer = renderer;
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

std::shared_ptr<RenderComponent>
SceneRenderSystem::addRenderable(Prim& prim, Backend::Shader* shader,
                                 TransformSource source) {
    auto component = prim.getRenderComponent();
    const bool created = !component;
    if (!component)
        component = prim.addRenderComponent();
    try {
        component->setTransformSource(source);
        if (registerRenderable(component, shader) == InvalidHandle) {
            if (created)
                prim.removeRenderComponent();
            return nullptr;
        }
    } catch (...) {
        if (created)
            prim.removeRenderComponent();
        throw;
    }
    return component;
}

std::shared_ptr<RenderComponent>
SceneRenderSystem::addSkinnedRenderable(Prim& prim, Backend::Shader* shader,
                                        const SkinnedMeshData& skinnedMesh,
                                        TransformSource source) {
    auto component = prim.getRenderComponent();
    const bool created = !component;
    if (!component)
        component = prim.addRenderComponent();
    try {
        component->setTransformSource(source);
        if (registerSkinnedRenderable(component, shader, skinnedMesh) ==
            InvalidHandle) {
            if (created)
                prim.removeRenderComponent();
            return nullptr;
        }
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
        if (registerRenderable(component, material) == InvalidHandle) {
            if (created)
                prim.removeRenderComponent();
            return nullptr;
        }
    } catch (...) {
        if (created)
            prim.removeRenderComponent();
        throw;
    }
    return component;
}

RenderableHandle SceneRenderSystem::finishRegistration(
    const std::shared_ptr<RenderComponent>& component,
    RenderableHandle renderable) {
    if (renderable == InvalidHandle)
        return InvalidHandle;

    Registration registration;
    registration.component = component;
    registration.prim = component->owner();
    registration.handle = renderable;
    _registrations.emplace(component.get(), registration);
    component->setRegistrationCallbacks(
        [this](RenderComponent& detached) { unregister(detached); },
        [this](RenderComponent& changed) { syncState(changed); });
    syncState(*component);
    return renderable;
}

RenderableHandle SceneRenderSystem::registerRenderable(
    const std::shared_ptr<RenderComponent>& component,
    Backend::Shader* shader) {
    validateRegistration(component);
    return finishRegistration(
        component, _renderer->addRenderable(shader, component->owner(),
                                            component->transformSource()));
}

RenderableHandle SceneRenderSystem::registerSkinnedRenderable(
    const std::shared_ptr<RenderComponent>& component, Backend::Shader* shader,
    const SkinnedMeshData& skinnedMesh) {
    validateRegistration(component);
    return finishRegistration(
        component,
        _renderer->addSkinnedRenderable(shader, component->owner(), skinnedMesh,
                                        component->transformSource()));
}

RenderableHandle SceneRenderSystem::registerRenderable(
    const std::shared_ptr<RenderComponent>& component, Material* material) {
    validateRegistration(component);
    return finishRegistration(
        component, _renderer->addRenderable(material, component->owner(),
                                            component->transformSource()));
}

const SceneRenderSystem::Registration&
SceneRenderSystem::requireRegistration(const RenderComponent& component) const {
    auto it = _registrations.find(&component);
    if (it == _registrations.end())
        throw std::runtime_error("RenderComponent is not registered");
    return it->second;
}

void SceneRenderSystem::syncState(RenderComponent& component) {
    const auto& registration = requireRegistration(component);
    _renderer->setRenderableDoubleSided(registration.handle,
                                        component.isDoubleSided());
    _renderer->setRenderableCastsShadow(registration.handle,
                                        component.castsShadow());
    _renderer->setRenderableAlphaMode(
        registration.handle, component.alphaMode(), component.alphaCutoff());
}

bool SceneRenderSystem::unregister(RenderComponent& component) {
    auto it = _registrations.find(&component);
    if (it == _registrations.end())
        return false;

    Registration registration = it->second;
    _registrations.erase(it);
    component.clearRegistrationCallbacks();
    if (_renderer && registration.prim)
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

void SceneRenderSystem::setTexture(RenderComponent& component,
                                   Backend::Texture* texture,
                                   TextureRole role) {
    const auto& registration = requireRegistration(component);
    _renderer->setRenderableTexture(registration.handle, texture, role);
}

void SceneRenderSystem::setTexture(RenderComponent& component,
                                   Backend::Texture* texture, int slot) {
    const auto& registration = requireRegistration(component);
    _renderer->setRenderableTexture(registration.handle, texture, slot);
}

void SceneRenderSystem::setExternalBuffer(RenderComponent& component,
                                          const ExternalBufferDesc& desc) {
    const auto& registration = requireRegistration(component);
    if (component.transformSource() != TransformSource::ExternalBuffer)
        throw std::runtime_error(
            "RenderComponent must be registered with ExternalBuffer "
            "transform source");
    _renderer->setRenderableExternalBuffer(registration.handle, desc);
}

void SceneRenderSystem::updateInstances(
    RenderComponent& component, const std::vector<glm::mat4>& transforms,
    const std::vector<glm::vec4>* colors) {
    const auto& registration = requireRegistration(component);
    _renderer->updateRenderableTransforms(registration.handle, transforms,
                                          colors);
}

void SceneRenderSystem::updateGeometry(RenderComponent& component,
                                       const std::vector<glm::vec3>& positions,
                                       const std::vector<glm::vec3>& normals) {
    const auto& registration = requireRegistration(component);
    _renderer->updateRenderableGeometry(registration.handle, positions,
                                        normals);
}

void SceneRenderSystem::updateSkinning(
    RenderComponent& component, const std::vector<glm::mat4>& boneMatrices) {
    const auto& registration = requireRegistration(component);
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
    return firstIt != _registrations.end() &&
           secondIt != _registrations.end() &&
           firstIt->second.handle == secondIt->second.handle;
}

RenderableHandle
SceneRenderSystem::handle(const RenderComponent& component) const {
    auto it = _registrations.find(&component);
    return it == _registrations.end() ? InvalidHandle : it->second.handle;
}

void SceneRenderSystem::clear() {
    while (!_registrations.empty()) {
        auto it = _registrations.begin();
        Registration registration = it->second;
        _registrations.erase(it);
        if (auto component = registration.component.lock())
            component->clearRegistrationCallbacks();
        if (_renderer && registration.prim)
            _renderer->removePrim(registration.handle, registration.prim);
    }
}

} // namespace Scene
} // namespace KE
