#ifndef _CLOTH_SOLVER_HPP_
#define _CLOTH_SOLVER_HPP_

#include "cloth_model.hpp"
#include "cloth_state.hpp"
#include <glm/glm.hpp>
#include <glm/vec3.hpp>
#include <vector>

namespace KE {
namespace Sim {

struct ClothSolverConfig {
    glm::vec3 gravity = {0.f, 0.f, -9.81f};
    float damping = 0.99f; // velocity damping per step
    // int iterations = 8;    // constraint projection iterations
    int numSubsteps = 4;
    float stretchStiffness = 1.0f;
    float shearStiffness = 0.8f;
    float bendStiffness = 0.2f;
};

class ClothSolver {
  public:
    void step(const ClothModel& model, ClothState& state, float dt,
              const ClothSolverConfig& cfg = {});

    // Call after step() and before uploading to GPU.
    static std::vector<glm::vec3> computeNormals(const ClothModel& model,
                                                 const ClothState& state);

  private:
    void applyExternalForces(ClothState& state, const glm::vec3& gravity,
                             float dt);
    void projectConstraints(const ClothModel& model, ClothState& state,
                            const ClothSolverConfig& cfg, float dt);

    virtual void projectDistanceConstraint(ClothState& state, int a, int b,
                                           float restLength, float stiffness,
                                           float dt);
    void updateVelocities(ClothState& state, float dt, float damping);
};

class ClothXPBDSolver : public ClothSolver {
  private:
    void projectDistanceConstraint(ClothState& state, int a, int b,
                                   float restLength, float stiffness,
                                   float dt) override;
};

} // namespace Sim
} // namespace KE

#endif