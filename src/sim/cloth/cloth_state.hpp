#ifndef _CLOTH_STATE_HPP_
#define _CLOTH_STATE_HPP_

#include "cloth_model.hpp"
#include <glm/glm.hpp>
#include <glm/vec3.hpp>
#include <vector>

namespace KE {
namespace Sim {

// Dynamic state — updated every frame.
struct ClothState {
    std::vector<glm::vec3> pos; // current positions  [num_particles]
    std::vector<glm::vec3>
        prevPos;                // previous positions [num_particles] — Verlet
    std::vector<glm::vec3> vel; // velocities         [num_particles]
    std::vector<float> invMass; // 0 = pinned         [num_particles]

    void resize(int n) {
        pos.assign(n, {0.f, 0.f, 0.f});
        prevPos.assign(n, {0.f, 0.f, 0.f});
        vel.assign(n, {0.f, 0.f, 0.f});
        invMass.assign(n, 1.f);
    }

    // Reset all particles to a grid layout. Pins are cleared.
    // uDir/vDir define the cloth plane axes (default: XZ = hanging, Z-up).
    // Call state.pin() after this to re-pin desired particles.
    void reset(const ClothModel& model, glm::vec3 origin = {0.f, 0.f, 0.f},
               glm::vec3 uDir = {1.f, 0.f, 0.f},
               glm::vec3 vDir = {0.f, 0.f, 1.f}) {
        resize(model.numParticles());
        float dx = model.width / (model.cols - 1);
        float dv = model.height / (model.rows - 1);
        for (int r = 0; r < model.rows; r++) {
            for (int c = 0; c < model.cols; c++) {
                int i = model.index(r, c);
                pos[i] = origin + uDir * (c * dx) + vDir * (r * dv);
                prevPos[i] = pos[i];
            }
        }
    }

    void pin(int idx) { invMass[idx] = 0.f; }

    int numParticles() const { return static_cast<int>(pos.size()); }
};

} // namespace Sim
} // namespace KE

#endif
