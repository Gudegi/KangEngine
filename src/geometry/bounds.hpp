#ifndef _BOUNDS_HPP_
#define _BOUNDS_HPP_

#include "geometry/ray.hpp"

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <vector>

namespace KE {
namespace Geometry {

struct Plane {
    glm::vec3 normal = glm::vec3(0.0f, 1.0f, 0.0f);
    float distance = 0.0f;

    float signedDistance(const glm::vec3& point) const {
        return glm::dot(normal, point) + distance;
    }
};

struct AABB {
    glm::vec3 min = glm::vec3(0.0f);
    glm::vec3 max = glm::vec3(0.0f);

    static AABB empty();

    bool isValid() const {
        return min.x <= max.x && min.y <= max.y && min.z <= max.z;
    }
    glm::vec3 center() const { return (min + max) * 0.5f; }
    glm::vec3 extents() const { return (max - min) * 0.5f; }
    void expand(const glm::vec3& point) {
        min = glm::min(min, point);
        max = glm::max(max, point);
    }
    void expand(const AABB& other) {
        if (!other.isValid())
            return;
        expand(other.min);
        expand(other.max);
    }
};

struct Sphere {
    glm::vec3 center = glm::vec3(0.0f);
    float radius = 0.0f;
};

struct OBB {
    // TODO: impl
};

struct Frustum {
    enum PlaneIndex {
        Left = 0,
        Right,
        Bottom,
        Top,
        Near,
        Far,
        Count,
    };

    Plane planes[Count];

    static Frustum fromViewProjection(const glm::mat4& viewProjection);
};

AABB computeAABB(const std::vector<glm::vec3>& points);
Sphere computeBoundingSphere(const AABB& bounds);
AABB transformAABB(const AABB& bounds, const glm::mat4& transform);
Sphere transformSphere(const Sphere& sphere, const glm::mat4& transform);

bool intersects(const AABB& a, const AABB& b);
bool intersects(const Sphere& a, const Sphere& b);
bool intersects(const Ray& ray, const AABB& bounds, float& outDistance);
bool intersects(const Ray& ray, const AABB& bounds);
bool intersects(const Ray& ray, const Sphere& sphere, float& outDistance);
bool intersects(const Ray& ray, const Sphere& sphere);
bool intersects(const Frustum& frustum, const AABB& bounds);
bool intersects(const Frustum& frustum, const Sphere& sphere);

} // namespace Geometry
} // namespace KE

#endif
