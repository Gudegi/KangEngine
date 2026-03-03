#ifndef _GLM_UTILS_HPP_
#define _GLM_UTILS_HPP_
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/quaternion.hpp>
#include "utils/types.hpp"

namespace KE {

glm::vec3 scaleVec3(glm::vec3 vec3, const float s);

// Returns the quaternion that rotates Y-up canonical geometry to the
// given UpAxis. Identity for Y-up, 90-degree X rotation for Z-up.
inline glm::quat upAxisRotation(UpAxis upAxis) {
    if (upAxis == UpAxis::Z) {
        // Rotate +90 degrees around X: Y-up(+Y) -> Z-up(+Z)
        return glm::angleAxis(glm::radians(90.0f), glm::vec3(1, 0, 0));
    }
    if (upAxis == UpAxis::X) {
        // Rotate -90 degrees around Z: Y-up(+Y) -> X-up(+X)
        return glm::angleAxis(glm::radians(-90.0f), glm::vec3(0, 0, 1));
    }
    return glm::quat(1, 0, 0, 0); // identity (Y-up)
}

} // namespace KE

#endif