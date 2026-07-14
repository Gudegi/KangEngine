#ifndef _SCENE_SELECTION_COMPONENT_HPP_
#define _SCENE_SELECTION_COMPONENT_HPP_

#include "engine/scene/component/component.hpp"

#include <string>

namespace KE {
namespace Scene {

// Broad editor interaction category for a Prim.
//
// This is intentionally editor/interaction metadata, not render identity.
// Runtime selection state still belongs to App/InteractionController because it
// can be viewport-specific. SelectionComponent only answers whether this Prim
// is allowed to participate in picking, selection, manipulation, and force
// drag.
enum class InteractionKind {
    ScenePrim,
    SimBody,
    DebugVisual,
    Helper,
    Resource,
};

class SelectionComponent : public ComponentBase {
  public:
    bool isPickable() const { return _pickable; }
    void setPickable(bool pickable);

    bool isSelectable() const { return _selectable; }
    void setSelectable(bool selectable);

    bool isManipulatable() const { return _manipulatable; }
    void setManipulatable(bool manipulatable);

    bool isForceDraggable() const { return _forceDraggable; }
    void setForceDraggable(bool forceDraggable);

    InteractionKind interactionKind() const { return _interactionKind; }
    void setInteractionKind(InteractionKind kind);

    int envId() const { return _envId; }
    void setEnvId(int envId);

    int objId() const { return _objId; }
    void setObjId(int objId);

    int bodyId() const { return _bodyId; }
    void setBodyId(int bodyId);

    const std::string& userTag() const { return _userTag; }
    void setUserTag(std::string tag);

  private:
    friend class Prim;

    explicit SelectionComponent(Prim* owner);
    void detach();

    bool _pickable = true;
    bool _selectable = true;
    bool _manipulatable = true;
    bool _forceDraggable = true;
    InteractionKind _interactionKind = InteractionKind::ScenePrim;
    int _envId = -1;
    int _objId = -1;
    int _bodyId = -1;
    std::string _userTag;
};

const char* interactionKindLabel(InteractionKind kind);

} // namespace Scene
} // namespace KE

#endif
