#ifndef _SKELETON_MATH_HPP_
#define _SKELETON_MATH_HPP_

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace KE {
namespace Animation {

// --- Eigen <-> glm conversions ---

inline glm::vec3 toGlm(const Eigen::Vector3f& v) {
    return glm::vec3(v.x(), v.y(), v.z());
}

inline Eigen::Vector3f fromGlm(const glm::vec3& v) {
    return Eigen::Vector3f(v.x, v.y, v.z);
}

// Eigen: (w, x, y, z) internally, constructor Quaternionf(w, x, y, z)
// glm:   (w, x, y, z) internally, constructor quat(w, x, y, z)
inline glm::quat toGlm(const Eigen::Quaternionf& q) {
    return glm::quat(q.w(), q.x(), q.y(), q.z());
}

inline Eigen::Quaternionf fromGlm(const glm::quat& q) {
    return Eigen::Quaternionf(q.w, q.x, q.y, q.z);
}

// --- Transform utilities ---

inline Eigen::Vector3f quatRotate(const Eigen::Quaternionf& q,
                                  const Eigen::Vector3f& v) {
    return q * v;
}

// Convert rotation + translation to a 4x4 model matrix (for shaders)
inline glm::mat4 transformToMat4(const Eigen::Quaternionf& rot,
                                 const Eigen::Vector3f& trans) {
    glm::mat4 m(1.0f);
    m = glm::translate(m, toGlm(trans));
    m *= glm::mat4_cast(toGlm(rot));
    return m;
}

// Convert rotation + translation + scale to a 4x4 model matrix
inline glm::mat4 transformToMat4(const Eigen::Quaternionf& rot,
                                 const Eigen::Vector3f& trans,
                                 const Eigen::Vector3f& scale) {
    glm::mat4 m(1.0f);
    m = glm::translate(m, toGlm(trans));
    m *= glm::mat4_cast(toGlm(rot));
    m = glm::scale(m, toGlm(scale));
    return m;
}

} // namespace Animation
} // namespace KE

#endif
