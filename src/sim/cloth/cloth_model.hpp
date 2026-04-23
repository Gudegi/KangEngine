#ifndef _CLOTH_MODEL_HPP_
#define _CLOTH_MODEL_HPP_

#include <glm/glm.hpp>
#include <utility>
#include <vector>

namespace KE {
namespace Sim {

// Static topology — built once, never changes during simulation.
// Reference: Newton's Model pattern.
struct ClothModel {
    int rows = 0;
    int cols = 0;
    float width = 1.f;
    float height = 1.f;

    // Per-particle
    std::vector<glm::vec2> uv; // [rows * cols]

    std::vector<uint32_t> indices; // [2 * (rows-1) * (cols-1) * 3]

    // Constraints — each entry is a (particle_a, particle_b) index pair.
    // restLength[i] matches constraints[i].
    struct Constraint {
        int a, b;
        float restLength;
    };
    std::vector<Constraint> stretches; // horizontal + vertical edges
    std::vector<Constraint> shears;    // diagonal edges
    std::vector<Constraint> bends;     // skip-1 edges (resist bending)

    // Factory: create a flat grid in the XY plane (Z-up).
    // origin: position of the (0,0) corner
    // width/height: total size in meters
    static ClothModel createGrid(int rows, int cols, float width, float height);

    int numParticles() const { return rows * cols; }
    int index(int r, int c) const { return r * cols + c; }
};

} // namespace Sim
} // namespace KE

#endif