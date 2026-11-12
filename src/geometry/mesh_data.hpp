#pragma once

#include "geometry/bounds.hpp"
#include <glm/glm.hpp>
#include <cstddef>
#include <vector>
#include <utility>

namespace KE::Scene {
constexpr std::size_t MaxSkinningBones = 128;

// mesh data, independent of scene and graphics-device ownership.
struct MeshData {
    std::vector<glm::vec3> vertices;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec2> uvs;
    // Tangent.xyz is the tangent direction; tangent.w is bitangent handedness.
    std::vector<glm::vec4> tangents;
    std::vector<unsigned int> indices;

    MeshData() = default;

    MeshData(std::vector<glm::vec3>&& vertices,
             std::vector<glm::vec3>&& normals, std::vector<glm::vec2>&& uvs,
             std::vector<unsigned int>&& indices)
        : vertices(std::move(vertices)), normals(std::move(normals)),
          uvs(std::move(uvs)), indices(std::move(indices)) {}

    // Fill missing normals (flat shading from triangles) and UVs (zeros).
    void fillMissingAttributes() {
        if (normals.empty() && !vertices.empty() && !indices.empty()) {
            normals.resize(vertices.size(), glm::vec3(0.0f));
            for (size_t i = 0; i + 2 < indices.size(); i += 3) {
                const glm::vec3& v0 = vertices[indices[i]];
                const glm::vec3& v1 = vertices[indices[i + 1]];
                const glm::vec3& v2 = vertices[indices[i + 2]];
                glm::vec3 n = glm::normalize(glm::cross(v1 - v0, v2 - v0));
                normals[indices[i]] = n;
                normals[indices[i + 1]] = n;
                normals[indices[i + 2]] = n;
            }
        }
        if (uvs.empty() && !vertices.empty()) {
            uvs.assign(vertices.size(), glm::vec2(0.0f));
        }
    }

    Geometry::AABB computeLocalAABB() const {
        return Geometry::computeAABB(vertices);
    }

    Geometry::Sphere computeLocalSphere() const {
        return Geometry::computeBoundingSphere(computeLocalAABB());
    }
};

// Mesh + skinning payload. use this only when vertex bone attributes are actually needed.
struct SkinnedMeshData {
    MeshData mesh;
    std::vector<glm::ivec4> boneIndices;
    std::vector<glm::vec4> boneWeights;
    std::vector<int> boneNodeIndices;
    std::vector<glm::mat4> inverseBindMatrices;

    SkinnedMeshData() = default;

    SkinnedMeshData(MeshData&& mesh, std::vector<glm::ivec4>&& boneIndices,
                    std::vector<glm::vec4>&& boneWeights)
        : mesh(std::move(mesh)), boneIndices(std::move(boneIndices)),
          boneWeights(std::move(boneWeights)) {}

    bool hasValidVertexSkinning() const {
        return !mesh.vertices.empty() &&
               boneIndices.size() == mesh.vertices.size() &&
               boneWeights.size() == mesh.vertices.size();
    }
};

}
