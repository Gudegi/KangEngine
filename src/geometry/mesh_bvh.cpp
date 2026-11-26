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

namespace {

double squaredBoundsDistance(const AABB& bounds, const glm::dvec3& point) {
    const auto nearest = glm::clamp(point, glm::dvec3(bounds.min), glm::dvec3(bounds.max));
    return glm::dot(point - nearest, point - nearest);
}

glm::dvec3 closestTriangle(const glm::dvec3& p, const glm::dvec3& a,
                          const glm::dvec3& b, const glm::dvec3& c) {
    const auto ab = b - a;
    const auto ac = c - a;
    const auto normal = glm::cross(ab, ac);
    const double normalSquared = glm::dot(normal, normal);
    if (normalSquared > 0) {
        const auto projected = p - normal * (glm::dot(p - a, normal) / normalSquared);
        const double u = glm::dot(glm::cross(projected - a, ac), normal) / normalSquared;
        const double v = glm::dot(glm::cross(ab, projected - a), normal) / normalSquared;
        if (u >= 0 && v >= 0 && u + v <= 1)
            return projected;
    }
    auto segment = [&](const glm::dvec3& start, const glm::dvec3& end) {
        const auto edge = end - start;
        const double lengthSquared = glm::dot(edge, edge);
        return start + edge * (lengthSquared > 0
            ? glm::clamp(glm::dot(p - start, edge) / lengthSquared, 0.0, 1.0) : 0.0);
    };
    auto best = segment(a, b);
    for (const auto& candidate : {segment(b, c), segment(c, a)})
        if (glm::dot(p - candidate, p - candidate) < glm::dot(p - best, p - best))
            best = candidate;
    return best;
}

} // namespace

void MeshBVH::closestNode(size_t node, const glm::dvec3& point,
                         double& squaredDistance, glm::dvec3& closest) const {
    const auto& entry = _nodes[node];
    if (squaredBoundsDistance(entry.bounds, point) > squaredDistance)
        return;
    if (entry.count) {
        for (size_t i = entry.first; i < entry.first + entry.count; ++i) {
            const size_t offset = _triangles[i] * 3;
            const auto candidate = closestTriangle(point,
                _mesh->vertices[_mesh->indices[offset]],
                _mesh->vertices[_mesh->indices[offset + 1]],
                _mesh->vertices[_mesh->indices[offset + 2]]);
            const double distance = glm::dot(point - candidate, point - candidate);
            if (distance < squaredDistance) {
                squaredDistance = distance;
                closest = candidate;
            }
        }
        return;
    }
    size_t first = entry.left;
    size_t second = entry.right;
    if (squaredBoundsDistance(_nodes[second].bounds, point) <
        squaredBoundsDistance(_nodes[first].bounds, point))
        std::swap(first, second);
    closestNode(first, point, squaredDistance, closest);
    closestNode(second, point, squaredDistance, closest);
}

glm::dvec3 MeshBVH::closestPoint(const glm::dvec3& point) const {
    if (!_built)
        build();
    if (_nodes.empty())
        throw std::invalid_argument("Closest-point query requires a nonempty triangle mesh");
    double distance = std::numeric_limits<double>::infinity();
    glm::dvec3 closest;
    closestNode(0, point, distance, closest);
    return closest;
}

bool MeshBVH::countCrossings(size_t node, const glm::dvec3& origin,
                             const glm::dvec3& direction, size_t& count) const {
    const auto& entry = _nodes[node];
    const auto t0 = (glm::dvec3(entry.bounds.min) - origin) / direction;
    const auto t1 = (glm::dvec3(entry.bounds.max) - origin) / direction;
    const auto near = glm::min(t0, t1);
    const auto far = glm::max(t0, t1);
    if (std::max({near.x, near.y, near.z, 0.0}) > std::min({far.x, far.y, far.z}))
        return true;
    if (!entry.count)
        return countCrossings(entry.left, origin, direction, count) &&
               countCrossings(entry.right, origin, direction, count);
    for (size_t i = entry.first; i < entry.first + entry.count; ++i) {
        const size_t offset = _triangles[i] * 3;
        const glm::dvec3 a = _mesh->vertices[_mesh->indices[offset]];
        const auto ab = glm::dvec3(_mesh->vertices[_mesh->indices[offset + 1]]) - a;
        const auto ac = glm::dvec3(_mesh->vertices[_mesh->indices[offset + 2]]) - a;
        const auto h = glm::cross(direction, ac);
        const double determinant = glm::dot(ab, h);
        if (std::abs(determinant) < 1e-14 * glm::length(ab) * glm::length(ac))
            continue;
        const auto relative = origin - a;
        const auto q = glm::cross(relative, ab);
        const double u = glm::dot(relative, h) / determinant;
        const double v = glm::dot(direction, q) / determinant;
        const double distance = glm::dot(ac, q) / determinant;
        if (u >= -1e-10 && v >= -1e-10 && u + v <= 1 + 1e-10 && distance >= 0) {
            // Retry a different ray rather than double-count a shared edge.
            if (u < 1e-10 || v < 1e-10 || 1.0 - u - v < 1e-10)
                return false;
            ++count;
        }
    }
    return true;
}

bool MeshBVH::contains(const glm::dvec3& point) const {
    if (!_built)
        build();
    if (_nodes.empty())
        throw std::invalid_argument("Containment requires a nonempty closed mesh");
    const glm::dvec3 directions[] = {{1, .371, .529}, {-.227, 1, .643},
        {.419f, -.193f, 1}, {1, -.713f, -.311f}, {-.617f, .283f, 1}};
    for (const auto& direction : directions) {
        size_t count = 0;
        if (countCrossings(0, point, direction, count))
            return count % 2 != 0;
    }
    throw std::runtime_error("Ambiguous mesh containment; all rays intersect edges or vertices");
}

} // namespace KE::Geometry
