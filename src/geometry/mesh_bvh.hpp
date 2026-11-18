#pragma once

#include "geometry/mesh_data.hpp"
#include <memory>

namespace KE::Geometry {

// Lazy triangle queries over an immutable, retained source mesh.
class MeshBVH {
  public:
    explicit MeshBVH(std::shared_ptr<const Scene::MeshData> mesh);
    // Tests both triangle sides; distance is measured along a normalized ray.
    bool intersect(const Ray& ray, float& distance) const;

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
};

} // namespace KE::Geometry
