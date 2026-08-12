#pragma once

#include "engine/scene/component/component.hpp"

#include <cstdint>
#include <functional>

namespace KE {
namespace Scene {

enum class RigidBodyType {
    Static,
    Dynamic,
    Kinematic,
};

enum class PhysicsTransformMode {
    PhysicsToScene,
    SceneToPhysics,
};

// Scene authoring state for one rigid body root.
//
// This component deliberately does not own a PhysX actor. ScenePhysicsSystem
// owns registration and creates/destroys the optional runtime actor.
class RigidBodyComponent : public ComponentBase {
  public:
    RigidBodyType bodyType() const;
    void setBodyType(RigidBodyType type);

    PhysicsTransformMode transformMode() const;

    float density() const;
    void setDensity(float density);

    uint32_t collisionGroup() const;
    void setCollisionGroup(uint32_t group);

    float contactOffset() const;
    void setContactOffset(float offset);

    float restOffset() const;
    void setRestOffset(float offset);

    void setContactOffsets(float contactOffset, float restOffset);

    bool isEnabled() const;
    void setEnabled(bool enabled);

  private:
    friend class Prim;
    friend class ScenePhysicsSystem;

    explicit RigidBodyComponent(Prim* owner);
    void detach();
    void setRegistrationCallbacks(
        std::function<void(RigidBodyComponent&)> detachCallback,
        std::function<void(RigidBodyComponent&)> changeCallback);
    void clearRegistrationCallbacks();
    void markChanged();

    RigidBodyType _bodyType = RigidBodyType::Static;
    float _density = 1000.0f;
    uint32_t _collisionGroup = 0;
    float _contactOffset = 0.02f;
    float _restOffset = 0.0f;
    bool _enabled = true;
    std::function<void(RigidBodyComponent&)> _detachCallback;
    std::function<void(RigidBodyComponent&)> _changeCallback;
};

const char* rigidBodyTypeLabel(RigidBodyType type);
const char* physicsTransformModeLabel(PhysicsTransformMode mode);

} // namespace Scene
} // namespace KE
