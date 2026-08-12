#pragma once

#include "PxPhysicsAPI.h"

#include <cstddef>
#include <cstdint>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <memory>
#include <vector>

namespace KE {

class PhysicsWorld;

namespace Physics {

// Empty identity token used only to detect whether PhysicsWorld's cooked
// resource storage is still alive. Runtime pointers do not belong here.
struct PhysicsResourceLifetimeToken final {};

// Backend-neutral convex part payload. Decomposition tools such as CoACD can
// populate this without depending on PhysX. PhysX cooks one hull per part.
struct ConvexMeshPart {
    std::vector<glm::vec3> vertices;
    std::vector<uint32_t> indices;
    glm::vec3 localPosition = glm::vec3(0.0f);
    glm::quat localRotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
};

struct ConvexCookingOptions {
    uint32_t vertexLimit = 255;
    bool gpuCompatible = false;
};

// Reusable collision payload for one rigid actor with one or more exclusive
// PxShape instances. PhysicsWorld owns and releases the native PxConvexMesh
// objects; this handle preserves their part-local poses and validates that its
// originating world is still alive.
class ConvexCollisionResource {
  public:
    size_t partCount() const { return _parts.size(); }
    const ConvexCookingOptions& cookingOptions() const { return _options; }
    bool isValid() const { return !_resourceLifetime.expired(); }

  private:
    friend class ::KE::PhysicsWorld;

    struct Part {
        physx::PxConvexMesh* mesh = nullptr;
        glm::vec3 localPosition{0.0f};
        glm::quat localRotation{1.0f, 0.0f, 0.0f, 0.0f};
    };

    ConvexCollisionResource(
        PhysicsWorld* owner,
        std::weak_ptr<PhysicsResourceLifetimeToken> resourceLifetime,
        std::vector<Part> parts, const ConvexCookingOptions& options)
        : _owner(owner), _resourceLifetime(std::move(resourceLifetime)),
          _parts(std::move(parts)), _options(options) {}

    PhysicsWorld* _owner = nullptr;
    std::weak_ptr<PhysicsResourceLifetimeToken> _resourceLifetime;
    std::vector<Part> _parts;
    ConvexCookingOptions _options;
};

} // namespace Physics
} // namespace KE
