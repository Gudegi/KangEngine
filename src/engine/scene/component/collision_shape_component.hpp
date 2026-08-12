#ifndef _SCENE_COLLISION_SHAPE_COMPONENT_HPP_
#define _SCENE_COLLISION_SHAPE_COMPONENT_HPP_

#include "engine/scene/component/component.hpp"

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

namespace KE {
namespace Scene {

// Scene-side metadata for a collision shape.
//
// This component deliberately does not own PhysX objects or runtime contact
// state. It mirrors the imported/reference collision descriptor onto debug
// prims so Inspector, Python tools, and future editors can reason about shape
// type, local pose, and material settings without touching the simulation hot
// path.(GPU batch)
enum class CollisionShapeType {
    Sphere,
    Capsule,
    Cylinder,
    Box,
    ConvexMesh,
};

class CollisionShapeComponent : public ComponentBase {
  public:
    CollisionShapeType shapeType() const { return _shapeType; }
    void setShapeType(CollisionShapeType shapeType);

    const glm::vec3& size() const { return _size; }
    void setSize(const glm::vec3& size);

    const glm::vec3& localPosition() const { return _localPosition; }
    void setLocalPosition(const glm::vec3& position);

    const glm::quat& localRotation() const { return _localRotation; }
    void setLocalRotation(const glm::quat& rotation);

    bool hasFromTo() const { return _hasFromTo; }
    const glm::vec3& fromPosition() const { return _fromPosition; }
    const glm::vec3& toPosition() const { return _toPosition; }
    void setFromTo(const glm::vec3& from, const glm::vec3& to);
    void clearFromTo();

    float staticFriction() const { return _staticFriction; }
    void setStaticFriction(float value);

    float dynamicFriction() const { return _dynamicFriction; }
    void setDynamicFriction(float value);

    float restitution() const { return _restitution; }
    void setRestitution(float value);

    int condim() const { return _condim; }
    void setCondim(int value);

    float margin() const { return _margin; }
    void setMargin(float value);

    int sourceGeomIndex() const { return _sourceGeomIndex; }
    void setSourceGeomIndex(int index);

    void setShapeMetadata(CollisionShapeType shapeType, const glm::vec3& size,
                          const glm::vec3& localPosition,
                          const glm::quat& localRotation, float staticFriction,
                          float dynamicFriction, float restitution, int condim,
                          float margin, int sourceGeomIndex);

  private:
    friend class Prim;

    explicit CollisionShapeComponent(Prim* owner);
    void detach();

    CollisionShapeType _shapeType = CollisionShapeType::Sphere;
    glm::vec3 _size{0.0f};
    glm::vec3 _localPosition{0.0f};
    glm::quat _localRotation{1.0f, 0.0f, 0.0f, 0.0f};
    bool _hasFromTo = false;
    glm::vec3 _fromPosition{0.0f};
    glm::vec3 _toPosition{0.0f};
    float _staticFriction = 1.0f;
    float _dynamicFriction = 1.0f;
    float _restitution = 0.0f;
    int _condim = -1;
    float _margin = -1.0f;
    int _sourceGeomIndex = -1;
};

const char* collisionShapeTypeLabel(CollisionShapeType type);

} // namespace Scene
} // namespace KE

#endif
