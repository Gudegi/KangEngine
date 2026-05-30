#ifndef _GEOMETRY_MESH_UTILS_HPP_
#define _GEOMETRY_MESH_UTILS_HPP_

#include "engine/scene/scene_backend.hpp"

#include <glm/geometric.hpp>
#include <mikktspace.h>

#include <cmath>
#include <vector>

namespace KE::Geometry {

namespace Detail {

inline glm::vec3 safeNormalize(const glm::vec3& v, const glm::vec3& fallback) {
    const float len = glm::length(v);
    return len > 1e-8f ? v / len : fallback;
}

inline glm::vec3 fallbackTangentForNormal(const glm::vec3& normal) {
    const glm::vec3 n = safeNormalize(normal, glm::vec3(0.0f, 1.0f, 0.0f));
    const glm::vec3 axis = std::abs(n.y) < 0.9f ? glm::vec3(0.0f, 1.0f, 0.0f)
                                                : glm::vec3(1.0f, 0.0f, 0.0f);
    return safeNormalize(glm::cross(axis, n), glm::vec3(1.0f, 0.0f, 0.0f));
}

struct MikkMeshAdapter {
    Scene::MeshData* mesh = nullptr;
};

inline unsigned int mikkVertexIndex(const Scene::MeshData& mesh, int face,
                                    int vert) {
    return mesh
        .indices[static_cast<size_t>(face) * 3 + static_cast<size_t>(vert)];
}

inline int mikkGetNumFaces(const SMikkTSpaceContext* context) {
    const auto* data =
        static_cast<const MikkMeshAdapter*>(context->m_pUserData);
    return static_cast<int>(data->mesh->indices.size() / 3);
}

inline int mikkGetNumVerticesOfFace(const SMikkTSpaceContext*, const int) {
    return 3;
}

inline void mikkGetPosition(const SMikkTSpaceContext* context, float out[],
                            const int face, const int vert) {
    const auto* data =
        static_cast<const MikkMeshAdapter*>(context->m_pUserData);
    const Scene::MeshData& mesh = *data->mesh;
    const glm::vec3& p = mesh.vertices[mikkVertexIndex(mesh, face, vert)];
    out[0] = p.x;
    out[1] = p.y;
    out[2] = p.z;
}

inline void mikkGetNormal(const SMikkTSpaceContext* context, float out[],
                          const int face, const int vert) {
    const auto* data =
        static_cast<const MikkMeshAdapter*>(context->m_pUserData);
    const Scene::MeshData& mesh = *data->mesh;
    const glm::vec3& n = mesh.normals[mikkVertexIndex(mesh, face, vert)];
    out[0] = n.x;
    out[1] = n.y;
    out[2] = n.z;
}

inline void mikkGetTexCoord(const SMikkTSpaceContext* context, float out[],
                            const int face, const int vert) {
    const auto* data =
        static_cast<const MikkMeshAdapter*>(context->m_pUserData);
    const Scene::MeshData& mesh = *data->mesh;
    const glm::vec2& uv = mesh.uvs[mikkVertexIndex(mesh, face, vert)];
    out[0] = uv.x;
    out[1] = uv.y;
}

inline void mikkSetTSpaceBasic(const SMikkTSpaceContext* context,
                               const float tangent[], const float sign,
                               const int face, const int vert) {
    auto* data = static_cast<MikkMeshAdapter*>(context->m_pUserData);
    Scene::MeshData& mesh = *data->mesh;
    const unsigned int index = mikkVertexIndex(mesh, face, vert);
    mesh.tangents[index] = glm::vec4(tangent[0], tangent[1], tangent[2], sign);
}

} // namespace Detail

inline void orthonormalizeTangents(Scene::MeshData& mesh) {
    if (mesh.tangents.size() != mesh.vertices.size())
        return;

    if (mesh.normals.size() != mesh.vertices.size())
        mesh.fillMissingAttributes();

    for (size_t i = 0; i < mesh.tangents.size(); ++i) {
        const glm::vec3 n =
            Detail::safeNormalize(mesh.normals[i], glm::vec3(0.0f, 1.0f, 0.0f));
        const glm::vec3 t = glm::vec3(mesh.tangents[i]);
        const float handedness = mesh.tangents[i].w < 0.0f ? -1.0f : 1.0f;
        mesh.tangents[i] = glm::vec4(
            Detail::safeNormalize(t - glm::dot(t, n) * n,
                                  Detail::fallbackTangentForNormal(n)),
            handedness);
    }
}

// Generate fallback tangent-space data for triangle meshes. normal mapping
// usable for internal/debug meshes and importers that do not expose tangents
// yet.
inline void computeFallbackTangents(Scene::MeshData& mesh) {
    mesh.tangents.clear();
    if (mesh.vertices.empty() || mesh.normals.size() != mesh.vertices.size() ||
        mesh.uvs.size() != mesh.vertices.size() || mesh.indices.size() < 3) {
        return;
    }

    std::vector<glm::vec3> tangentAccum(mesh.vertices.size(), glm::vec3(0.0f));
    std::vector<glm::vec3> bitangentAccum(mesh.vertices.size(),
                                          glm::vec3(0.0f));
    bool hasValidTriangle = false;

    for (size_t i = 0; i + 2 < mesh.indices.size(); i += 3) {
        const unsigned int i0 = mesh.indices[i];
        const unsigned int i1 = mesh.indices[i + 1];
        const unsigned int i2 = mesh.indices[i + 2];
        if (i0 >= mesh.vertices.size() || i1 >= mesh.vertices.size() ||
            i2 >= mesh.vertices.size()) {
            continue;
        }

        const glm::vec3& p0 = mesh.vertices[i0];
        const glm::vec3& p1 = mesh.vertices[i1];
        const glm::vec3& p2 = mesh.vertices[i2];
        const glm::vec2& uv0 = mesh.uvs[i0];
        const glm::vec2& uv1 = mesh.uvs[i1];
        const glm::vec2& uv2 = mesh.uvs[i2];

        const glm::vec3 edge1 = p1 - p0;
        const glm::vec3 edge2 = p2 - p0;
        const glm::vec2 duv1 = uv1 - uv0;
        const glm::vec2 duv2 = uv2 - uv0;
        const float det = duv1.x * duv2.y - duv2.x * duv1.y;
        if (std::abs(det) < 1e-8f)
            continue;

        const float invDet = 1.0f / det;
        const glm::vec3 tangent = (edge1 * duv2.y - edge2 * duv1.y) * invDet;
        const glm::vec3 bitangent = (edge2 * duv1.x - edge1 * duv2.x) * invDet;

        tangentAccum[i0] += tangent;
        tangentAccum[i1] += tangent;
        tangentAccum[i2] += tangent;
        bitangentAccum[i0] += bitangent;
        bitangentAccum[i1] += bitangent;
        bitangentAccum[i2] += bitangent;
        hasValidTriangle = true;
    }

    if (!hasValidTriangle)
        return;

    mesh.tangents.resize(mesh.vertices.size(), glm::vec4(0.0f));
    for (size_t i = 0; i < mesh.vertices.size(); ++i) {
        const glm::vec3 n =
            Detail::safeNormalize(mesh.normals[i], glm::vec3(0.0f, 1.0f, 0.0f));
        glm::vec3 t = tangentAccum[i];
        if (glm::length(t) < 1e-8f)
            t = Detail::fallbackTangentForNormal(n);

        // Gram-Schmidt: keep tangent orthogonal to the vertex normal.
        t = Detail::safeNormalize(t - glm::dot(t, n) * n,
                                  Detail::fallbackTangentForNormal(n));
        const glm::vec3 b = bitangentAccum[i];
        const float handedness =
            glm::dot(glm::cross(n, t), b) < 0.0f ? -1.0f : 1.0f;
        mesh.tangents[i] = glm::vec4(t, handedness);
    }
}

inline bool computeMikkTangents(Scene::MeshData& mesh) {
    mesh.tangents.clear();
    if (mesh.vertices.empty() || mesh.normals.size() != mesh.vertices.size() ||
        mesh.uvs.size() != mesh.vertices.size() || mesh.indices.size() < 3 ||
        (mesh.indices.size() % 3) != 0) {
        return false;
    }

    for (const unsigned int index : mesh.indices) {
        if (index >= mesh.vertices.size())
            return false;
    }

    mesh.tangents.assign(mesh.vertices.size(), glm::vec4(0.0f));

    Detail::MikkMeshAdapter adapter;
    adapter.mesh = &mesh;

    SMikkTSpaceInterface interface = {};
    interface.m_getNumFaces = Detail::mikkGetNumFaces;
    interface.m_getNumVerticesOfFace = Detail::mikkGetNumVerticesOfFace;
    interface.m_getPosition = Detail::mikkGetPosition;
    interface.m_getNormal = Detail::mikkGetNormal;
    interface.m_getTexCoord = Detail::mikkGetTexCoord;
    interface.m_setTSpaceBasic = Detail::mikkSetTSpaceBasic;
    interface.m_setTSpace = nullptr;

    SMikkTSpaceContext context = {};
    context.m_pInterface = &interface;
    context.m_pUserData = &adapter;

    if (!genTangSpaceDefault(&context)) {
        mesh.tangents.clear();
        return false;
    }

    orthonormalizeTangents(mesh);
    return true;
}

// Generate tangent-space data using MikkTSpace, then fall back to the simple
// averaged method if the mesh is not suitable for MikkTSpace.
inline void computeTangents(Scene::MeshData& mesh) {
    if (!computeMikkTangents(mesh))
        computeFallbackTangents(mesh);
}

} // namespace KE::Geometry

#endif // _GEOMETRY_MESH_UTILS_HPP_
