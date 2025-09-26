#include "glm_utils.hpp"

namespace KE {

glm::vec3 scaleVec3(glm::vec3 vec3, const float s)
{
    vec3.x *= s;
    vec3.y *= s;
    vec3.z *= s;
    return vec3;
}

} // namespace KE