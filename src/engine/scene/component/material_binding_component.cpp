#include "material_binding_component.hpp"

#include <stdexcept>

namespace KE {
namespace Scene {

MaterialBindingComponent::MaterialBindingComponent(Prim* owner)
    : _owner(owner) {
    requireAttached();
}

void MaterialBindingComponent::requireAttached() const {
    if (!_owner)
        throw std::runtime_error(
            "MaterialBindingComponent is detached from its Prim");
}

void MaterialBindingComponent::markChanged() { ++_version; }

void MaterialBindingComponent::detach() {
    if (!_owner)
        return;
    _owner = nullptr;
    _material = nullptr;
    markChanged();
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
