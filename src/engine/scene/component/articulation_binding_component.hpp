#ifndef _SCENE_ARTICULATION_BINDING_COMPONENT_HPP_
#define _SCENE_ARTICULATION_BINDING_COMPONENT_HPP_

#include "engine/scene/component/component.hpp"

#include <string>

namespace KE {
namespace Scene {

// Semantic metadata for prims generated from an articulated character/robot.
//
// This component intentionally does not own FK state or hierarchy arrays.
// It only records how this Prim is bound to a future/root articulation
// controller: which body it belongs to and what role this Prim plays.
enum class ArticulationPrimRole {
    Root,
    BodyFrame,
    VisualGeom,
    CollisionGeom,
};

class ArticulationBindingComponent : public ComponentBase {
  public:
    ArticulationPrimRole role() const { return _role; }
    void setRole(ArticulationPrimRole role);

    int bodyIndex() const { return _bodyIndex; }
    void setBodyIndex(int bodyIndex);

    const std::string& bodyName() const { return _bodyName; }
    void setBodyName(std::string bodyName);

    const std::string& articulationRootPath() const {
        return _articulationRootPath;
    }
    void setArticulationRootPath(std::string rootPath);

    void setBinding(ArticulationPrimRole role, int bodyIndex,
                    std::string bodyName, std::string rootPath);

  private:
    friend class Prim;

    explicit ArticulationBindingComponent(Prim* owner);
    void detach();

    ArticulationPrimRole _role = ArticulationPrimRole::BodyFrame;
    int _bodyIndex = -1;
    std::string _bodyName;
    std::string _articulationRootPath;
};

const char* articulationPrimRoleLabel(ArticulationPrimRole role);

} // namespace Scene
} // namespace KE

#endif
