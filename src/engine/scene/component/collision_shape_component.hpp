#ifndef _SCENE_COLLISION_SHAPE_COMPONENT_HPP_
#define _SCENE_COLLISION_SHAPE_COMPONENT_HPP_

#include "engine/scene/component/component.hpp"

#include <glm/glm.hpp>
#include <functional>
#include <memory>

namespace KE {

namespace Physics {
class ConvexCollisionResource;
}

namespace Scene {

// Scene-side metadata for a collision shape.
//
// This component deliberately does not own PhysX objects or runtime contact
// state. It mirrors the imported/reference collision descriptor onto debug
// prims so Inspector, Python tools, and future editors can reason about shape
// type, normalized size, source metadata, and material settings without
// touching the simulation hot
// path (GPU batch). Convex shapes may retain a reusable collision-resource
// handle, but the originating PhysicsWorld still owns the native PxConvexMesh
// payload and controls its validity.
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

    // Preserves the original MJCF endpoint authoring form for diagnostics and
    // round-tripping. Runtime pose/size are already normalized onto the Prim.
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

    std::shared_ptr<Physics::ConvexCollisionResource> convexResource() const {
        return _convexResource;
    }
    void setConvexResource(
        std::shared_ptr<Physics::ConvexCollisionResource> resource);
    void clearConvexResource();

    void setShapeMetadata(CollisionShapeType shapeType, const glm::vec3& size,
                          float staticFriction,
                          float dynamicFriction, float restitution, int condim,
                          float margin, int sourceGeomIndex);

  private:
    friend class Prim;
    friend class ScenePhysicsSystem;

    explicit CollisionShapeComponent(Prim* owner);
    void detach();
    void setRegistrationCallbacks(
        std::function<void(CollisionShapeComponent&)> detachCallback,
        std::function<void(CollisionShapeComponent&)> changeCallback);
    void clearRegistrationCallbacks();
    void markChanged();

    CollisionShapeType _shapeType = CollisionShapeType::Sphere;
    glm::vec3 _size{0.0f};
    bool _hasFromTo = false;
    glm::vec3 _fromPosition{0.0f};
    glm::vec3 _toPosition{0.0f};
    float _staticFriction = 1.0f;
    float _dynamicFriction = 1.0f;
    float _restitution = 0.0f;
    int _condim = -1;
    float _margin = -1.0f;
    int _sourceGeomIndex = -1;
    std::shared_ptr<Physics::ConvexCollisionResource> _convexResource;
    std::function<void(CollisionShapeComponent&)> _registrationDetachCallback;
    std::function<void(CollisionShapeComponent&)> _registrationChangeCallback;
};

const char* collisionShapeTypeLabel(CollisionShapeType type);

} // namespace Scene
} // namespace KE

#endif
