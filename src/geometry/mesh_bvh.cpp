#include "geometry/mesh_bvh.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace KE::Geometry {

MeshBVH::MeshBVH(std::shared_ptr<const Scene::MeshData> mesh)
    : _mesh(std::move(mesh)) {
    if (!_mesh)
        throw std::invalid_argument("MeshBVH requires a source mesh");
}

void MeshBVH::build() const {
    if (_mesh->indices.size() % 3 != 0)
        throw std::invalid_argument("MeshBVH requires triangle indices");
    for (auto index : _mesh->indices) {
        if (index >= _mesh->vertices.size())
            throw std::out_of_range("MeshBVH vertex index is out of range");
        const auto& vertex = _mesh->vertices[index];
        if (!std::isfinite(vertex.x) || !std::isfinite(vertex.y) ||
            !std::isfinite(vertex.z))
            throw std::invalid_argument("MeshBVH requires finite vertices");
    }

    _triangles.resize(_mesh->indices.size() / 3);
    std::iota(_triangles.begin(), _triangles.end(), size_t{0});
    _nodes.clear();
    if (!_triangles.empty()) {
        _nodes.reserve(_triangles.size() / 2 + 1);
        buildNode(0, _triangles.size());
    }
    _built = true;
}

size_t MeshBVH::buildNode(size_t first, size_t count) const {
    const size_t node = _nodes.size();
    _nodes.push_back({AABB::empty(), first, count});
    for (size_t i = first; i < first + count; ++i) {
        const size_t offset = _triangles[i] * 3;
        for (size_t corner = 0; corner < 3; ++corner)
            _nodes[node].bounds.expand(
                _mesh->vertices[_mesh->indices[offset + corner]]);
    }
    if (count <= 8)
        return node;

    const glm::vec3 extent = _nodes[node].bounds.extents();
    int axis = extent.y > extent.x ? 1 : 0;
    if (extent.z > extent[axis])
        axis = 2;
    auto center = [&](size_t triangle) {
        const size_t offset = triangle * 3;
        return _mesh->vertices[_mesh->indices[offset]][axis] / 3.0f +
               _mesh->vertices[_mesh->indices[offset + 1]][axis] / 3.0f +
               _mesh->vertices[_mesh->indices[offset + 2]][axis] / 3.0f;
    };
    const size_t half = count / 2;
    std::nth_element(_triangles.begin() + first,
                     _triangles.begin() + first + half,
                     _triangles.begin() + first + count,
                     [&](size_t a, size_t b) { return center(a) < center(b); });
    const size_t left = buildNode(first, half);
    const size_t right = buildNode(first + half, count - half);
    _nodes[node].left = left;
    _nodes[node].right = right;
    _nodes[node].count = 0;
    return node;
}

bool MeshBVH::intersectNode(size_t node, const Ray& ray,
                            float& distance) const {
    const auto& entry = _nodes[node];
    if (entry.count != 0) {
        bool hit = false;
        for (size_t i = entry.first; i < entry.first + entry.count; ++i) {
            const size_t offset = _triangles[i] * 3;
            glm::vec2 barycentric;
            float candidate;
            if (ray.intersectTriangle(
                    _mesh->vertices[_mesh->indices[offset]],
                    _mesh->vertices[_mesh->indices[offset + 1]],
                    _mesh->vertices[_mesh->indices[offset + 2]], barycentric,
                    candidate) &&
                candidate >= 0.0f && candidate < distance) {
                distance = candidate;
                hit = true;
            }
        }
        return hit;
    }

    size_t nearNode = entry.left;
    size_t farNode = entry.right;
    float nearDistance = std::numeric_limits<float>::infinity();
    float farDistance = std::numeric_limits<float>::infinity();
    float candidate;
    if (intersects(ray, _nodes[nearNode].bounds, candidate))
        nearDistance = candidate;
    if (intersects(ray, _nodes[farNode].bounds, candidate))
        farDistance = candidate;
    if (farDistance < nearDistance) {
        std::swap(nearNode, farNode);
        std::swap(nearDistance, farDistance);
    }
    bool hit =
        nearDistance < distance && intersectNode(nearNode, ray, distance);
    if (farDistance < distance)
        hit = intersectNode(farNode, ray, distance) || hit;
    return hit;
}

bool MeshBVH::intersect(const Ray& ray, float& distance) const {
    if (!_built)
        build();
    distance = std::numeric_limits<float>::infinity();
    float entryDistance;
    return !_nodes.empty() &&
           intersects(ray, _nodes.front().bounds, entryDistance) &&
           intersectNode(0, ray, distance);
}

} // namespace KE::Geometry
