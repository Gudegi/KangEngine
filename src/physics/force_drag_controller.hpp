#pragma once

#include "PxPhysicsAPI.h"
#include "physics.hpp"
#include "engine/graphics/renderer/renderer_types.hpp"
#include <glm/vec3.hpp>
#include <unordered_map>
#include <vector>

namespace KE {

class Articulation;

struct ForceDragConfig {
    float stiffness = 250.0f;
    float damping = 8.0f;
    float maxForce = 800.0f;
};

class ForceDragController {
  public:
    ForceDragController();
    explicit ForceDragController(ForceDragConfig config);

    void setConfig(ForceDragConfig config) { _config = config; }
    const ForceDragConfig& config() const { return _config; }

    // Handles are expected to be ordered by body/link index.
    void registerArticulation(Articulation& articulation,
                              const std::vector<RenderableHandle>& bodyHandles);
    void registerRigid(physx::PxRigidDynamic& rigid, RenderableHandle handle);
    void clearBindings();

    bool begin(const RayPickResult& pick, const glm::vec3& target);
    bool beginDirect(physx::PxRigidDynamic& rigid,
                     const glm::vec3& hitPosition);
    void update(const glm::vec3& target);
    void computeForce(const glm::vec3& target);
    void applyCachedForce();
    void end();
    bool active() const { return _active; }
    const glm::vec3& lastBodyPosition() const { return _lastBodyPosition; }
    const glm::vec3& lastAnchorPosition() const { return _lastAnchorPosition; }
    const glm::vec3& lastTarget() const { return _lastTarget; }
    const glm::vec3& lastForce() const { return _lastForce; }

  private:
    struct Binding {
        Articulation* articulation = nullptr;
        physx::PxRigidDynamic* rigid = nullptr;
        int linkIndex = -1;
    };

    ForceDragConfig _config;
    std::unordered_map<RenderableHandle, Binding> _bindings;
    Binding _activeBinding;
    physx::PxVec3 _localAnchor = physx::PxVec3(0.0f);
    glm::vec3 _lastBodyPosition = glm::vec3(0.0f);
    glm::vec3 _lastAnchorPosition = glm::vec3(0.0f);
    glm::vec3 _lastTarget = glm::vec3(0.0f);
    glm::vec3 _lastForce = glm::vec3(0.0f);
    bool _active = false;

    void applyForce(const glm::vec3& target);
    static glm::vec3 clampForce(glm::vec3 force, float maxForce);
};

} // namespace KE
