#ifndef _SCENE_ARTICULATION_BINDING_COMPONENT_HPP_
#define _SCENE_ARTICULATION_BINDING_COMPONENT_HPP_

#include "engine/scene/component/component.hpp"

#include <string>

namespace KE {
namespace Scene {

enum class ArticulationPrimRole {
    Root,
    BodyFrame,
    VisualGeom,
    CollisionGeom,
};

// Maps a Prim to an articulation root path, body index, and role.
// Stores names, paths, and indices; does not own FK or physics state.
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
