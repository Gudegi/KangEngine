#include "render_component.hpp"

#include "engine/scene/native/prim.hpp"

#include <stdexcept>
#include <utility>

namespace KE {
namespace Scene {

RenderComponent::RenderComponent(Prim* owner)
    : ComponentBase(owner, "RenderComponent") {}

void RenderComponent::markChanged() {
    ComponentBase::markChanged();
    if (_changeCallback)
        _changeCallback(*this);
}

void RenderComponent::detach() {
    if (!_owner)
        return;
    auto callback = std::move(_detachCallback);
    if (callback)
        callback(*this);
    detachBase();
}

void RenderComponent::setRegistrationCallbacks(
    std::function<void(RenderComponent&)> detachCallback,
    std::function<void(RenderComponent&)> changeCallback) {
    _detachCallback = std::move(detachCallback);
    _changeCallback = std::move(changeCallback);
    _registered = true;
}

void RenderComponent::clearRegistrationCallbacks() {
    _detachCallback = {};
    _changeCallback = {};
    _registered = false;
}

bool RenderComponent::isVisible() const {
    requireAttached();
    return _owner->isVisible();
}

void RenderComponent::setVisible(bool visible) {
    requireAttached();
    if (_owner->isVisible() == visible)
        return;
    _owner->setVisible(visible);
}

std::optional<glm::vec4> RenderComponent::displayColor() const {
    requireAttached();
    return _displayColor;
}

void RenderComponent::setDisplayColor(glm::vec4 color) {
    requireAttached();
    if (_displayColor && *_displayColor == color)
        return;
    _displayColor = color;
    // Colors are collected per instance, without changing batch render state.
    ComponentBase::markChanged();
}

bool RenderComponent::isDoubleSided() const {
    requireAttached();
    return _doubleSided;
}

void RenderComponent::setDoubleSided(bool doubleSided) {
    requireAttached();
    if (_doubleSided == doubleSided)
        return;
    _doubleSided = doubleSided;
    try {
        markChanged();
    } catch (...) {
        _doubleSided = !doubleSided;
        throw;
    }
}

bool RenderComponent::castsShadow() const {
    requireAttached();
    return _castsShadow;
}

void RenderComponent::setCastsShadow(bool castsShadow) {
    requireAttached();
    if (_castsShadow == castsShadow)
        return;
    _castsShadow = castsShadow;
    try {
        markChanged();
    } catch (...) {
        _castsShadow = !castsShadow;
        throw;
    }
}

AlphaMode RenderComponent::alphaMode() const {
    requireAttached();
    return _alphaMode;
}

float RenderComponent::alphaCutoff() const {
    requireAttached();
    return _alphaCutoff;
}

void RenderComponent::setAlphaMode(AlphaMode mode, float cutoff) {
    requireAttached();
    if (_alphaMode == mode && _alphaCutoff == cutoff)
        return;
    const auto previousMode = _alphaMode;
    const auto previousCutoff = _alphaCutoff;
    _alphaMode = mode;
    _alphaCutoff = cutoff;
    try {
        markChanged();
    } catch (...) {
        _alphaMode = previousMode;
        _alphaCutoff = previousCutoff;
        throw;
    }
}

TransformSource RenderComponent::transformSource() const {
    requireAttached();
    return _transformSource;
}

void RenderComponent::setTransformSource(TransformSource source) {
    requireAttached();
    if (_transformSource == source)
        return;
    if (_registered)
        throw std::runtime_error(
            "cannot change transform source while RenderComponent is "
            "registered");
    _transformSource = source;
    markChanged();
}

std::shared_ptr<MeshData> RenderComponent::resolveMeshData() const {
    requireAttached();
    return _owner->resolveMeshData();
}

} // namespace Scene
} // namespace KE
