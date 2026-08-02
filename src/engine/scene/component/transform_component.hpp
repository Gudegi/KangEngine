#ifndef _SCENE_TRANSFORM_COMPONENT_HPP_
#define _SCENE_TRANSFORM_COMPONENT_HPP_

#include "engine/scene/component/component.hpp"

#include <glm/gtc/quaternion.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

namespace KE {
namespace Scene {

// Local/world transform state attached to every Prim.
//
// Prim keeps identity, path, and hierarchy. TransformComponent owns the cached
// local/world matrices and the dirty propagation logic for that hierarchy.
// Existing Prim transform APIs forward here for compatibility.
class TransformComponent : public ComponentBase {
  public:
    void setLocalTranslation(glm::vec3 translation);
    void setLocalScale(glm::vec3 scale);
    void setLocalRotation(glm::quat rotation);
    void setLocalRotationAxisAngle(glm::vec3 axis, float angleRadians);
    void setLocalMatrix(const glm::mat4& matrix);

    void setWorldTranslation(glm::vec3 translation);
    void setWorldRotation(glm::quat rotation);
    void setWorldRotationAxisAngle(glm::vec3 axis, float angleRadians);
    void setWorldMatrix(const glm::mat4& matrix);

    glm::vec3 getLocalTranslation();
    glm::quat getLocalRotation();
    glm::vec3 getWorldTranslation();
    glm::quat getWorldRotation();

    glm::mat4 computeLocalMatrix();
    glm::mat4 computeWorldMatrix();
    glm::mat4 computeModelMatrix();

  private:
    friend class Prim;

    explicit TransformComponent(Prim* owner);
    void detach();
    void markLocalTransformDirty();
    void markWorldTransformDirtyRecursive(bool countSelfVersion = true);

    bool _suppressLocalDirtyVersion = false;
    bool _localDirty = true;
    bool _worldDirty = true;
    glm::mat4 _cachedLocalMat = glm::mat4(1.0f);
    glm::mat4 _cachedWorldMat = glm::mat4(1.0f);
};

} // namespace Scene
} // namespace KE

#endif
