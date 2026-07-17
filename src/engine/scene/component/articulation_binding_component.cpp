#include "engine/scene/component/articulation_binding_component.hpp"

#include <utility>

namespace KE {
namespace Scene {

ArticulationBindingComponent::ArticulationBindingComponent(Prim* owner)
    : ComponentBase(owner, "ArticulationBindingComponent") {}

void ArticulationBindingComponent::detach() { detachBase(); }

void ArticulationBindingComponent::setRole(ArticulationPrimRole role) {
    requireAttached();
    if (_role == role)
        return;
    _role = role;
    markChanged();
}

void ArticulationBindingComponent::setBodyIndex(int bodyIndex) {
    requireAttached();
    if (_bodyIndex == bodyIndex)
        return;
    _bodyIndex = bodyIndex;
    markChanged();
}

void ArticulationBindingComponent::setBodyName(std::string bodyName) {
    requireAttached();
    if (_bodyName == bodyName)
        return;
    _bodyName = std::move(bodyName);
    markChanged();
}

void ArticulationBindingComponent::setArticulationRootPath(
    std::string rootPath) {
    requireAttached();
    if (_articulationRootPath == rootPath)
        return;
    _articulationRootPath = std::move(rootPath);
    markChanged();
}

void ArticulationBindingComponent::setBinding(ArticulationPrimRole role,
                                              int bodyIndex,
                                              std::string bodyName,
                                              std::string rootPath) {
    requireAttached();
    const bool changed =
        _role != role || _bodyIndex != bodyIndex || _bodyName != bodyName ||
        _articulationRootPath != rootPath;
    if (!changed)
        return;
    _role = role;
    _bodyIndex = bodyIndex;
    _bodyName = std::move(bodyName);
    _articulationRootPath = std::move(rootPath);
    markChanged();
}

const char* articulationPrimRoleLabel(ArticulationPrimRole role) {
    switch (role) {
    case ArticulationPrimRole::Root:
        return "Root";
    case ArticulationPrimRole::BodyFrame:
        return "BodyFrame";
    case ArticulationPrimRole::VisualGeom:
        return "VisualGeom";
    case ArticulationPrimRole::CollisionGeom:
        return "CollisionGeom";
    }
    return "Unknown";
}

} // namespace Scene
} // namespace KE
