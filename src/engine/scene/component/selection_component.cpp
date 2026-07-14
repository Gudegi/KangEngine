#include "engine/scene/component/selection_component.hpp"

#include <utility>

namespace KE {
namespace Scene {

SelectionComponent::SelectionComponent(Prim* owner)
    : ComponentBase(owner, "SelectionComponent") {}

void SelectionComponent::detach() { detachBase(); }

void SelectionComponent::setPickable(bool pickable) {
    requireAttached();
    if (_pickable == pickable)
        return;
    _pickable = pickable;
    markChanged();
}

void SelectionComponent::setSelectable(bool selectable) {
    requireAttached();
    if (_selectable == selectable)
        return;
    _selectable = selectable;
    markChanged();
}

void SelectionComponent::setManipulatable(bool manipulatable) {
    requireAttached();
    if (_manipulatable == manipulatable)
        return;
    _manipulatable = manipulatable;
    markChanged();
}

void SelectionComponent::setForceDraggable(bool forceDraggable) {
    requireAttached();
    if (_forceDraggable == forceDraggable)
        return;
    _forceDraggable = forceDraggable;
    markChanged();
}

void SelectionComponent::setInteractionKind(InteractionKind kind) {
    requireAttached();
    if (_interactionKind == kind)
        return;
    _interactionKind = kind;
    markChanged();
}

void SelectionComponent::setEnvId(int envId) {
    requireAttached();
    if (_envId == envId)
        return;
    _envId = envId;
    markChanged();
}

void SelectionComponent::setObjId(int objId) {
    requireAttached();
    if (_objId == objId)
        return;
    _objId = objId;
    markChanged();
}

void SelectionComponent::setBodyId(int bodyId) {
    requireAttached();
    if (_bodyId == bodyId)
        return;
    _bodyId = bodyId;
    markChanged();
}

void SelectionComponent::setUserTag(std::string tag) {
    requireAttached();
    if (_userTag == tag)
        return;
    _userTag = std::move(tag);
    markChanged();
}

const char* interactionKindLabel(InteractionKind kind) {
    switch (kind) {
    case InteractionKind::ScenePrim:
        return "ScenePrim";
    case InteractionKind::SimBody:
        return "SimBody";
    case InteractionKind::DebugVisual:
        return "DebugVisual";
    case InteractionKind::Helper:
        return "Helper";
    case InteractionKind::Resource:
        return "Resource";
    }
    return "Unknown";
}

} // namespace Scene
} // namespace KE
