#include "material_binding_component.hpp"

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
    _material = material;
    markChanged();
}

void MaterialBindingComponent::clearMaterial() {
    setMaterial(nullptr);
}

} // namespace Scene
} // namespace KE
