#include "material_binding_component.hpp"

#include "engine/scene/component/render_component.hpp"
#include "engine/scene/native/prim.hpp"

#include <stdexcept>

namespace KE {
namespace Scene {

MaterialBindingComponent::MaterialBindingComponent(Prim* owner)
    : ComponentBase(owner, "MaterialBindingComponent") {}

void MaterialBindingComponent::detach() {
    if (!_owner)
        return;
    _material = nullptr;
    detachBase();
}

Material* MaterialBindingComponent::material() const {
    requireAttached();
    return _material;
}

void MaterialBindingComponent::setMaterial(Material* material) {
    requireAttached();
    if (_material == material)
        return;
    auto* previous = _material;
    _material = material;
    try {
        if (auto render = _owner->getRenderComponent())
            render->markChanged();
    } catch (...) {
        _material = previous;
        throw;
    }
    markChanged();
}

void MaterialBindingComponent::clearMaterial() {
    setMaterial(nullptr);
}

} // namespace Scene
} // namespace KE
