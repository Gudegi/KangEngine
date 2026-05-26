#include "geometry/bounds.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace KE {
namespace Geometry {
namespace {

Plane normalizePlane(const glm::vec4& p) {
    Plane plane;
    const float len = glm::length(glm::vec3(p));
    if (len <= std::numeric_limits<float>::epsilon())
        return plane;

    plane.normal = glm::vec3(p) / len;
    plane.distance = p.w / len;
    return plane;
}

glm::vec4 row(const glm::mat4& m, int r) {
    return glm::vec4(m[0][r], m[1][r], m[2][r], m[3][r]);
}

float maxScaleComponent(const glm::mat4& transform) {
    const glm::vec3 x = glm::vec3(transform[0]);
    const glm::vec3 y = glm::vec3(transform[1]);
    const glm::vec3 z = glm::vec3(transform[2]);
    return std::max({glm::length(x), glm::length(y), glm::length(z)});
}

} // namespace

float Plane::signedDistance(const glm::vec3& point) const {
    return glm::dot(normal, point) + distance;
}

AABB AABB::empty() {
    const float inf = std::numeric_limits<float>::infinity();
    return {glm::vec3(inf), glm::vec3(-inf)};
}

bool AABB::isValid() const {
    return min.x <= max.x && min.y <= max.y && min.z <= max.z;
}

glm::vec3 AABB::center() const { return (min + max) * 0.5f; }

glm::vec3 AABB::extents() const { return (max - min) * 0.5f; }

void AABB::expand(const glm::vec3& point) {
    min = glm::min(min, point);
    max = glm::max(max, point);
}

void AABB::expand(const AABB& other) {
    if (!other.isValid())
        return;
    expand(other.min);
    expand(other.max);
}

Frustum Frustum::fromViewProjection(const glm::mat4& viewProjection) {
    const glm::vec4 r0 = row(viewProjection, 0);
    const glm::vec4 r1 = row(viewProjection, 1);
    const glm::vec4 r2 = row(viewProjection, 2);
    const glm::vec4 r3 = row(viewProjection, 3);

    Frustum frustum;
    frustum.planes[Left] = normalizePlane(r3 + r0);
    frustum.planes[Right] = normalizePlane(r3 - r0);
    frustum.planes[Bottom] = normalizePlane(r3 + r1);
    frustum.planes[Top] = normalizePlane(r3 - r1);
    frustum.planes[Near] = normalizePlane(r3 + r2);
    frustum.planes[Far] = normalizePlane(r3 - r2);
    return frustum;
}

AABB computeAABB(const std::vector<glm::vec3>& points) {
    AABB bounds = AABB::empty();
    for (const glm::vec3& point : points)
        bounds.expand(point);
    return points.empty() ? AABB{} : bounds;
}

Sphere computeBoundingSphere(const AABB& bounds) {
    if (!bounds.isValid())
        return {};

    Sphere sphere;
    sphere.center = bounds.center();
    sphere.radius = glm::length(bounds.extents());
    return sphere;
}

AABB transformAABB(const AABB& bounds, const glm::mat4& transform) {
    if (!bounds.isValid())
        return {};

    const glm::vec3 center = bounds.center();
    const glm::vec3 extents = bounds.extents();
    const glm::vec3 worldCenter =
        glm::vec3(transform * glm::vec4(center, 1.0f));

    glm::mat3 linear(transform);
    for (int c = 0; c < 3; ++c)
        for (int r = 0; r < 3; ++r)
            linear[c][r] = std::abs(linear[c][r]);

    const glm::vec3 worldExtents = linear * extents;
    return {worldCenter - worldExtents, worldCenter + worldExtents};
}

Sphere transformSphere(const Sphere& sphere, const glm::mat4& transform) {
    return {glm::vec3(transform * glm::vec4(sphere.center, 1.0f)),
            sphere.radius * maxScaleComponent(transform)};
}

bool intersects(const AABB& a, const AABB& b) {
    if (!a.isValid() || !b.isValid())
        return false;

    return a.min.x <= b.max.x && a.max.x >= b.min.x && a.min.y <= b.max.y &&
           a.max.y >= b.min.y && a.min.z <= b.max.z && a.max.z >= b.min.z;
}

bool intersects(const Sphere& a, const Sphere& b) {
    const float radius = a.radius + b.radius;
    return glm::dot(a.center - b.center, a.center - b.center) <=
           radius * radius;
}

bool intersects(const Frustum& frustum, const AABB& bounds) {
    if (!bounds.isValid())
        return false;

    for (const Plane& plane : frustum.planes) {
        const glm::vec3 positive(
            plane.normal.x >= 0.0f ? bounds.max.x : bounds.min.x,
            plane.normal.y >= 0.0f ? bounds.max.y : bounds.min.y,
            plane.normal.z >= 0.0f ? bounds.max.z : bounds.min.z);
        if (plane.signedDistance(positive) < 0.0f)
            return false;
    }
    return true;
}

bool intersects(const Frustum& frustum, const Sphere& sphere) {
    for (const Plane& plane : frustum.planes) {
        if (plane.signedDistance(sphere.center) < -sphere.radius)
            return false;
    }
    return true;
}

} // namespace Geometry
} // namespace KE
