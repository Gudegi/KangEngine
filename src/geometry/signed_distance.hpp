#pragma once

#include "geometry/mesh_bvh.hpp"
#include <algorithm>
#include <map>
#include <stdexcept>

namespace KE::Geometry {

struct SignedDistanceResult {
    double distance;
    glm::dvec3 closest;
    glm::dvec3 gradient;
};

// Owns a snapshot. Input must be a non-self-intersecting solid; edge topology
// checks do not prove absence of surface self-intersections.
class SignedDistanceMesh {
  public:
    explicit SignedDistanceMesh(const Scene::MeshData& source)
        : _mesh(std::make_shared<Scene::MeshData>(source)), _bvh(_mesh) {
        if (_mesh->indices.empty() || _mesh->indices.size() % 3)
            throw std::invalid_argument("Signed distance requires a nonempty closed triangle mesh");
        std::map<std::pair<unsigned, unsigned>, std::pair<int, int>> edges;
        double signedVolume = 0;
        glm::dvec3 reference(0);
        for (size_t i = 0; i < _mesh->indices.size(); i += 3) {
            glm::dvec3 points[3];
            for (size_t corner = 0; corner < 3; ++corner) {
                const auto index = _mesh->indices[i + corner];
                if (index >= _mesh->vertices.size())
                    throw std::invalid_argument("Signed distance triangle index out of range");
                points[corner] = _mesh->vertices[index];
                for (int axis = 0; axis < 3; ++axis)
                    if (!std::isfinite(points[corner][axis]))
                        throw std::invalid_argument("Signed distance requires finite vertices");
                const auto next = _mesh->indices[i + (corner + 1) % 3];
                auto& edge = edges[std::minmax(index, next)];
                ++edge.first;
                edge.second += index < next ? 1 : -1;
            }
            const auto normal = glm::cross(points[1] - points[0], points[2] - points[0]);
            if (glm::dot(normal, normal) <= 1e-30)
                throw std::invalid_argument("Signed distance requires nondegenerate triangles");
            if (i == 0)
                reference = points[0];
            signedVolume += glm::dot(points[0] - reference,
                glm::cross(points[1] - reference, points[2] - reference)) / 6.0;
        }
        for (const auto& entry : edges)
            if (entry.second.first != 2 || entry.second.second != 0)
                throw std::invalid_argument("Signed distance requires a closed, consistently oriented manifold mesh");
        if (std::abs(signedVolume) <= 1e-30)
            throw std::invalid_argument("Signed distance requires a nonzero enclosed volume");
    }

    SignedDistanceResult query(const glm::dvec3& point) const {
        for (int axis = 0; axis < 3; ++axis)
            if (!std::isfinite(point[axis]))
                throw std::invalid_argument("Signed distance query points must be finite");
        const auto closest = _bvh.closestPoint(point);
        double distance = glm::length(point - closest);
        if (distance <= 1e-10)
            return {0, closest, glm::dvec3(0)};
        if (_bvh.contains(point))
            distance = -distance;
        return {distance, closest, (point - closest) / distance};
    }

  private:
    std::shared_ptr<const Scene::MeshData> _mesh;
    MeshBVH _bvh;
};

} // namespace KE::Geometry
