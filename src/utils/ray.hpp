#ifndef _RAY_HPP_
#define _RAY_HPP_

#include <glm/glm.hpp> // TODO: write own intersect
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/intersect.hpp>

namespace KE {

struct Ray {
    glm::vec3 origin;
    glm::vec3 direction;

    Ray() : origin(0.0f), direction(0.0f, 0.0f, -1.0f) {}

    Ray(const glm::vec3& origin, const glm::vec3& direction)
        : origin(origin), direction(glm::normalize(direction)) {}

    glm::vec3 getPoint(float t) const { return origin + direction * t; }

    bool intersectPlane(const glm::vec3& planeOrigin,
                        const glm::vec3& planeNormal,
                        float& outDistance) const {
        return glm::intersectRayPlane(origin, direction, planeOrigin,
                                      planeNormal, outDistance);
    }

    bool intersectSphere(const glm::vec3& sphereCenter, float sphereRadius,
                         float& outDistance) const {
        return glm::intersectRaySphere(origin, direction, sphereCenter,
                                       sphereRadius * sphereRadius,
                                       outDistance);
    }

    bool intersectTriangle(const glm::vec3& v0, const glm::vec3& v1,
                           const glm::vec3& v2, glm::vec2& outBaryPosition,
                           float& outDistance) const {
        return glm::intersectRayTriangle(origin, direction, v0, v1, v2,
                                         outBaryPosition, outDistance);
    }
};

} // namespace KE

#endif // _KE_RAY_HPP_