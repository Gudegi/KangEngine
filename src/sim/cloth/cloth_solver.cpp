#include "cloth_solver.hpp"
#include "cloth_model.hpp"
#include "cloth_state.hpp"
#include <glm/geometric.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <vector>

namespace KE {
namespace Sim {

// TODO : add collision

void ClothSolver::step(const ClothModel& model, ClothState& state, float dt,
                       const ClothSolverConfig& cfg) {
    /*
    applyExternalForces(state, cfg.gravity, dt);
    for (int iter = 0; iter < cfg.iterations; iter++) {
        projectConstraints(model, state, cfg, dt);
    }
    updateVelocities(state, dt, cfg.damping);
    */
    float sdt = dt / cfg.numSubsteps;
    for (int iter = 0; iter < cfg.numSubsteps; iter++) {
        applyExternalForces(state, cfg.gravity, sdt);
        projectConstraints(model, state, cfg, sdt);
        updateVelocities(state, sdt, cfg.damping);
    }
}

void ClothSolver::applyExternalForces(ClothState& state,
                                      const glm::vec3& gravity, float dt) {
    int n = state.numParticles();
    for (int i = 0; i < n; i++) {
        if (state.invMass[i] == 0.f)
            continue;
        // current sovler only assumes gravitational force
        state.vel[i] += gravity * dt;
        state.prevPos[i] = state.pos[i];
        state.pos[i] += state.vel[i] * dt;
    }
}

void ClothSolver::projectConstraints(const ClothModel& model, ClothState& state,
                                     const ClothSolverConfig& cfg, float dt) {
    for (const auto& c : model.stretches)
        projectDistanceConstraint(state, c.a, c.b, c.restLength,
                                  cfg.stretchStiffness, dt);
    for (const auto& c : model.shears)
        projectDistanceConstraint(state, c.a, c.b, c.restLength,
                                  cfg.shearStiffness, dt);
    for (const auto& c : model.bends)
        projectDistanceConstraint(state, c.a, c.b, c.restLength,
                                  cfg.bendStiffness, dt);
}

void ClothSolver::projectDistanceConstraint(ClothState& state, int a, int b,
                                            float restLength, float stiffness,
                                            float dt) {
    float wA = state.invMass[a];
    float wB = state.invMass[b];
    float wSum = wA + wB;
    if (wSum == 0.f)
        return;

    glm::vec3 diff = state.pos[a] - state.pos[b];
    float len = glm::length(diff);
    if (len < 1e-6f)
        return;

    float corrMag = len - restLength; // distance constraint func
    glm::vec3 dir = diff / len;

    glm::vec3 correction = stiffness * (corrMag * dir);
    state.pos[a] -= (wA / wSum) * correction;
    state.pos[b] += (wB / wSum) * correction;
}

void ClothXPBDSolver::projectDistanceConstraint(ClothState& state, int a, int b,
                                                float restLength,
                                                float stiffness, float dt) {
    float wA = state.invMass[a];
    float wB = state.invMass[b];
    float wSum = wA + wB;
    if (wSum == 0.f)
        return;

    float compliance = 1 / stiffness;
    float alphaTilde = compliance / (dt * dt);

    glm::vec3 diff = state.pos[a] - state.pos[b];
    float len = glm::length(diff);
    if (len < 1e-6f)
        return;

    float corrMag = len - restLength;
    glm::vec3 dir = diff / len;

    float multiplier = corrMag / (wSum + alphaTilde);
    glm::vec3 correction = multiplier * dir;

    state.pos[a] -= wA * correction;
    state.pos[b] += wB * correction;
}

void ClothSolver::updateVelocities(ClothState& state, float dt, float damping) {
    int n = state.numParticles();
    float invDt = (dt > 0.f) ? 1.f / dt : 0.f;
    for (int i = 0; i < n; i++) {
        if (state.invMass[i] == 0.f)
            continue;
        state.vel[i] = (state.pos[i] - state.prevPos[i]) * invDt * damping;
    }
}

std::vector<glm::vec3> ClothSolver::computeNormals(const ClothModel& model,
                                                   const ClothState& state) {
    int n = state.numParticles();
    std::vector<glm::vec3> normals(n, glm::vec3(0.f));

    for (int i = 0; i + 2 < (int)model.indices.size(); i += 3) {
        int a = model.indices[i];
        int b = model.indices[i + 1];
        int c = model.indices[i + 2];
        glm::vec3 faceNormal = glm::cross(state.pos[b] - state.pos[a],
                                          state.pos[c] - state.pos[a]);
        normals[a] += faceNormal;
        normals[b] += faceNormal;
        normals[c] += faceNormal;
    }

    for (auto& n : normals) {
        float len = glm::length(n);
        if (len > 1e-6f)
            n /= len;
    }
    return normals;
}

} // namespace Sim
} // namespace KE
