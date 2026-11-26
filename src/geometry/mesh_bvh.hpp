#pragma once

#include "geometry/mesh_data.hpp"
#include <cmath>
#include <memory>

namespace KE::Geometry {

// Lazy triangle queries over an immutable, retained source mesh.
class MeshBVH {
  public:
    explicit MeshBVH(std::shared_ptr<const Scene::MeshData> mesh);
    // Tests both triangle sides; distance is measured along a normalized ray.
    bool intersect(const Ray& ray, float& distance) const;
    glm::dvec3 closestPoint(const glm::dvec3& point) const;
    // Closed surfaces only. Ambiguous edge/vertex crossings are retried.
    bool contains(const glm::dvec3& point) const;
    // Preserve world distance under rotated, mirrored, nonuniform transforms.
    bool intersectWorld(const Ray& ray, const glm::mat4& transform,
                        float& distance) const {
        const float determinant = glm::determinant(glm::mat3(transform));
        if (!std::isfinite(determinant) || determinant == 0.0f)
            return false;
        const glm::mat4 inverse = glm::inverse(transform);
        const glm::vec3 direction =
            glm::vec3(inverse * glm::vec4(ray.direction, 0.0f));
        const float scale = glm::length(direction);
        if (!std::isfinite(scale) || scale == 0.0f)
            return false;
        const Ray localRay(glm::vec3(inverse * glm::vec4(ray.origin, 1.0f)),
                           direction);
        if (!intersect(localRay, distance))
            return false;
        distance /= scale;
        return true;
    }

  private:
    struct Node {
        AABB bounds;
        size_t first = 0;
        size_t count = 0;
        size_t left = 0;
        size_t right = 0;
    };

    std::shared_ptr<const Scene::MeshData> _mesh;
    mutable std::vector<size_t> _triangles;
    mutable std::vector<Node> _nodes;
    mutable bool _built = false;

    void build() const;
    size_t buildNode(size_t first, size_t count) const;
    bool intersectNode(size_t node, const Ray& ray, float& distance) const;
    void closestNode(size_t node, const glm::dvec3& point,
                     double& squaredDistance, glm::dvec3& closest) const;
    bool countCrossings(size_t node, const glm::dvec3& origin,
                        const glm::dvec3& direction, size_t& count) const;
};

} // namespace KE::Geometry
