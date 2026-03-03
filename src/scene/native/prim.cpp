///
/// Scene Prim Implementation
///

#include "prim.hpp"
#include "scene/scene_backend.hpp"
#include "utils/types.hpp"
#include <glm/ext/quaternion_transform.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/ext/scalar_constants.hpp>
#include <glm/fwd.hpp>
#include <glm/trigonometric.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <sstream>
#include "xform_token.hpp"

namespace KE {
namespace Scene {

Prim::Prim(const std::string& name, PrimType type, Prim* parent)
    : _name(name), _type(type), _parent(parent) {
    // Initialize prim path
    if (parent == nullptr) {
        _path = "/"; // root
    } else if (parent->getPath() == "/") {
        _path = "/" + name;
    } else {
        _path = parent->getPath() + "/" + name;
    }
}

Prim* Prim::addChild(const std::string& name, PrimType type) {
    auto child = std::make_unique<Prim>(name, type, this);
    Prim* childPtr = child.get();

    _childrenMap[name] = childPtr;
    _children.push_back(std::move(child));

    return childPtr;
}

Prim* Prim::getChild(const std::string& name) const {
    auto it = _childrenMap.find(name);
    if (it != _childrenMap.end()) {
        return it->second;
    }
    return nullptr;
}

Prim* Prim::getPrimAtPath(const std::string& path) {
    // "/" → 루트
    if (path == "/" || path.empty()) {
        return this;
    }

    // "/World/Cube" 파싱
    std::string pathCopy = path;
    if (pathCopy[0] == '/') {
        pathCopy = pathCopy.substr(1); // "/" 제거
    }

    // "World/Cube" → ["World", "Cube"]
    std::vector<std::string> parts;
    std::stringstream ss(pathCopy);
    std::string part;
    while (std::getline(ss, part, '/')) {
        if (!part.empty()) {
            parts.push_back(part);
        }
    }

    // 순차 탐색
    Prim* current = this;
    for (const auto& name : parts) {
        current = current->getChild(name);
        if (!current) {
            return nullptr; // 경로 없음
        }
    }

    return current;
}

std::vector<Prim*> Prim::getChildren() const {
    std::vector<Prim*> result;
    result.reserve(_children.size());
    for (const auto& child : _children) {
        result.push_back(child.get());
    }
    return result;
}

void Prim::setMeshData(std::shared_ptr<MeshData> data) { _meshData = data; }

std::shared_ptr<MeshData> Prim::getMeshData() const { return _meshData; }

MeshData Prim::createSquareData(float scale) {
    // Scale means the length of one side.
    float half = scale / 2;
    //    v3----- v7
    //   /|      /|
    //  v2------v6|
    //  | |     | |
    //  | v0----|-v4
    //  |/      |/
    //  v1------v5
    //
    std::vector<glm::vec3> positions = {
        // v0, v1, v2, v3
        glm::vec3(-half, -half, -half),
        glm::vec3(-half, -half, half),
        glm::vec3(-half, half, half),
        glm::vec3(-half, half, -half),
        // v4, v5, v6, v7
        glm::vec3(half, -half, -half),
        glm::vec3(half, -half, half),
        glm::vec3(half, half, half),
        glm::vec3(half, half, -half),
        // v0, v1, v5, v4
        glm::vec3(-half, -half, -half),
        glm::vec3(-half, -half, half),
        glm::vec3(half, -half, half),
        glm::vec3(half, -half, -half),
        // v3, v2, v6, v7
        glm::vec3(-half, half, -half),
        glm::vec3(-half, half, half),
        glm::vec3(half, half, half),
        glm::vec3(half, half, -half),
        // v0, v3, v7, v4
        glm::vec3(-half, -half, -half),
        glm::vec3(-half, half, -half),
        glm::vec3(half, half, -half),
        glm::vec3(half, -half, -half),
        // v1, v2, v6, v5
        glm::vec3(-half, -half, half),
        glm::vec3(-half, half, half),
        glm::vec3(half, half, half),
        glm::vec3(half, -half, half),
    }; // positions.size() == 24

    std::vector<glm::vec3> normals = {
        glm::vec3(-1, 0, 0), glm::vec3(-1, 0, 0), glm::vec3(-1, 0, 0),
        glm::vec3(-1, 0, 0), glm::vec3(1, 0, 0),  glm::vec3(1, 0, 0),
        glm::vec3(1, 0, 0),  glm::vec3(1, 0, 0),  glm::vec3(0, -1, 0),
        glm::vec3(0, -1, 0), glm::vec3(0, -1, 0), glm::vec3(0, -1, 0),
        glm::vec3(0, 1, 0),  glm::vec3(0, 1, 0),  glm::vec3(0, 1, 0),
        glm::vec3(0, 1, 0),  glm::vec3(0, 0, -1), glm::vec3(0, 0, -1),
        glm::vec3(0, 0, -1), glm::vec3(0, 0, -1), glm::vec3(0, 0, 1),
        glm::vec3(0, 0, 1),  glm::vec3(0, 0, 1),  glm::vec3(0, 0, 1),
    };

    std::vector<glm::vec2> uvs = {
        glm::vec2(0, 0), glm::vec2(1, 0), glm::vec2(1, 1), glm::vec2(0, 1),
        glm::vec2(0, 0), glm::vec2(1, 0), glm::vec2(1, 1), glm::vec2(0, 1),
        glm::vec2(0, 0), glm::vec2(1, 0), glm::vec2(1, 1), glm::vec2(0, 1),
        glm::vec2(0, 0), glm::vec2(1, 0), glm::vec2(1, 1), glm::vec2(0, 1),
        glm::vec2(0, 0), glm::vec2(1, 0), glm::vec2(1, 1), glm::vec2(0, 1),
        glm::vec2(0, 0), glm::vec2(1, 0), glm::vec2(1, 1), glm::vec2(0, 1),
    };

    std::vector<unsigned int> indices = {
        0,  1,  2,  0,  2,  3,  // left
        4,  6,  5,  4,  7,  6,  // right
        8,  10, 9,  8,  11, 10, // down, v0,v5,v1, v0,v4,v5
        12, 13, 14, 12, 14, 15, // up
        16, 17, 18, 16, 18, 19, // back
        20, 23, 22, 20, 22, 21, // front
    };

    MeshData meshData;
    meshData.vertices = std::move(positions);
    meshData.normals = std::move(normals);
    meshData.uvs = std::move(uvs);
    meshData.indices = std::move(indices);

    return meshData;
}

MeshData Prim::createPlaneData(float scale) {
    return createPlaneData(scale, UpAxis::Y);
}

MeshData Prim::createPlaneData(float scale, UpAxis upAxis) {
    float half = scale / 2;
    //
    // v2 ----- v3
    // |        |
    // |        |
    // v0 ----- v1
    //

    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> normals;

    if (upAxis == UpAxis::Y) {
        // XZ plane, Y-up normal
        positions = {
            glm::vec3(-half, 0, -half),
            glm::vec3(half, 0, -half),
            glm::vec3(-half, 0, half),
            glm::vec3(half, 0, half),
        };
        normals = {
            glm::vec3(0, 1, 0),
            glm::vec3(0, 1, 0),
            glm::vec3(0, 1, 0),
            glm::vec3(0, 1, 0),
        };
    } else if (upAxis == UpAxis::Z) {
        // XY plane, Z-up normal
        positions = {
            glm::vec3(-half, -half, 0),
            glm::vec3(half, -half, 0),
            glm::vec3(-half, half, 0),
            glm::vec3(half, half, 0),
        };
        normals = {
            glm::vec3(0, 0, 1),
            glm::vec3(0, 0, 1),
            glm::vec3(0, 0, 1),
            glm::vec3(0, 0, 1),
        };
    } else { // UpAxis::X
        // YZ plane, X-up normal
        positions = {
            glm::vec3(0, -half, -half),
            glm::vec3(0, half, -half),
            glm::vec3(0, -half, half),
            glm::vec3(0, half, half),
        };
        normals = {
            glm::vec3(1, 0, 0),
            glm::vec3(1, 0, 0),
            glm::vec3(1, 0, 0),
            glm::vec3(1, 0, 0),
        };
    }

    std::vector<glm::vec2> uvs = {
        glm::vec2(0, 0),
        glm::vec2(1, 0),
        glm::vec2(0, 1),
        glm::vec2(1, 1),
    };
    std::vector<unsigned int> indices = {0, 3, 1, 0, 2, 3};

    MeshData meshData;
    meshData.vertices = std::move(positions);
    meshData.normals = std::move(normals);
    meshData.uvs = std::move(uvs);
    meshData.indices = std::move(indices);

    return meshData;
}

MeshData Prim::createSphereData(float radius, int numLongitudes,
                                int numLatitudes) {
    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec2> uvs;
    std::vector<unsigned int> indices;

    int vertexCount = (numLatitudes) * (numLongitudes);
    int indexCount = (numLatitudes - 1) * (numLongitudes - 1) * 6;
    positions.reserve(vertexCount);
    normals.reserve(vertexCount);
    uvs.reserve(vertexCount);
    indices.reserve(indexCount);

    float thetaUnit = (glm::pi<float>() / (numLatitudes - 1));
    float phiUnit = (2 * glm::pi<float>() / (numLongitudes - 1));

    for (int i = 0; i < numLatitudes; i++) {
        float theta = i * thetaUnit;
        float sinTheta = glm::sin(theta);
        float cosTheta = glm::cos(theta);

        for (int j = 0; j < numLongitudes; j++) {
            float phi = j * phiUnit;
            float sinPhi = glm::sin(phi);
            float cosPhi = glm::cos(phi);

            float x = sinTheta * cosPhi;
            float y = sinTheta * sinPhi;
            float z = cosTheta;

            float u =
                static_cast<float>(j) / static_cast<float>(numLongitudes - 1);
            float v =
                static_cast<float>(i) / static_cast<float>(numLatitudes - 1);

            // positions.emplace_back(
            //     glm::vec3(radius * x, radius * y, radius * z));
            // normals.emplace_back(glm::vec3(x, y, z));
            // uvs.emplace_back(glm::vec2(u, v));
            positions.emplace_back(radius * x, radius * y, radius * z);
            normals.emplace_back(x, y, z);
            uvs.emplace_back(u, v);
        }
    }

    for (int i = 0; i < numLatitudes - 1; i++) {
        for (int j = 0; j < numLongitudes - 1; j++) {
            unsigned int first = i * numLongitudes + j;
            unsigned int second = first + numLongitudes;

            // counter clock-wise
            indices.emplace_back(first);
            indices.emplace_back(second);
            indices.emplace_back(first + 1);
            indices.emplace_back(second);
            indices.emplace_back(second + 1);
            indices.emplace_back(first + 1);

            /*
            // clock-wise
            indices.emplace_back(first);
            indices.emplace_back(first + 1);
            indices.emplace_back(second);
            indices.emplace_back(second);
            indices.emplace_back(first + 1);
            indices.emplace_back(second + 1);
            */
        }
    }

    return MeshData(std::move(positions), std::move(normals), std::move(uvs),
                    std::move(indices));
}

// Internal helper: truncated cylinder (topRadius==radius → cylinder,
// topRadius==0 → cone) Ported from references/render_opengl.py
// _create_cylinder_mesh
static MeshData makeCylinderMesh(float radius, float halfLength,
                                 float topRadius, const int perm[3],
                                 int segments) {
    auto P = [&](float x, float y, float z) -> glm::vec3 {
        float v[3] = {x, y, z};
        return {v[perm[0]], v[perm[1]], v[perm[2]]};
    };

    // Side normal slope: -arctan2(topRadius - radius, 2*halfLength)
    // 0 for straight cylinder, positive for cone (normals tilt outward-up)
    float sideSlope = -glm::atan(topRadius - radius, 2.0f * halfLength);
    const float pi2 = glm::two_pi<float>();

    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec2> uvs;
    std::vector<unsigned int> indices;

    // Cap vertices: [0] bottom center, [1] top center
    positions.push_back(P(0.0f, -halfLength, 0.0f));
    normals.push_back(P(0.0f, -1.0f, 0.0f));
    uvs.emplace_back(0.5f, 0.5f);

    positions.push_back(P(0.0f, halfLength, 0.0f));
    normals.push_back(P(0.0f, 1.0f, 0.0f));
    uvs.emplace_back(0.5f, 0.5f);

    for (int j : {-1, 1}) {
        unsigned int ci = (j == 1) ? 1u : 0u;
        float y = j * halfLength;
        float r = (j == -1) ? radius : topRadius;

        for (int i = 0; i < segments; ++i) {
            float theta = pi2 * i / segments;
            float c = glm::cos(theta);
            float s = glm::sin(theta);

            positions.push_back(P(r * c, y, r * s));
            normals.push_back(P(0.0f, (float)j, 0.0f));
            uvs.emplace_back(c * 0.5f + 0.5f, s * 0.5f + 0.5f);

            int cs = ci * segments;
            unsigned int v_curr = 2u + i + cs;
            unsigned int v_next = 2u + (i + 1) % segments + cs;

            if (j == -1) {
                // bottom cap: CCW from -Y (outside)
                indices.push_back(ci);
                indices.push_back(v_curr);
                indices.push_back(v_next);
            } else {
                // top cap: CCW from +Y (outside)
                indices.push_back(v_next);
                indices.push_back(v_curr);
                indices.push_back(ci);
            }
        }
    }

    // Side vertices
    int sideStart = (int)positions.size();
    for (int j : {-1, 1}) {
        float y = j * halfLength;
        float r = (j == -1) ? radius : topRadius;
        float v = ((float)j + 1.0f) / 2.0f;

        for (int i = 0; i < segments; ++i) {
            float theta = pi2 * i / segments;
            float c = glm::cos(theta);
            float s = glm::sin(theta);

            positions.push_back(P(r * c, y, r * s));
            normals.push_back(glm::normalize(P(c, sideSlope, s)));
            float u = (float)i / (float)(segments - 1);
            uvs.emplace_back(u, v);
        }
    }

    for (int i = 0; i < segments; ++i) {
        unsigned int top_i = sideStart + i + segments;
        unsigned int top_next = sideStart + (i + 1) % segments + segments;
        unsigned int bot_i = sideStart + i;
        unsigned int bot_next = sideStart + (i + 1) % segments;

        indices.push_back(top_i);
        indices.push_back(top_next);
        indices.push_back(bot_i);
        indices.push_back(top_next);
        indices.push_back(bot_next);
        indices.push_back(bot_i);
    }

    return MeshData(std::move(positions), std::move(normals), std::move(uvs),
                    std::move(indices));
}

MeshData Prim::createCylinderData(float radius, float length, int segments) {
    return createCylinderData(radius, length, UpAxis::Y, segments);
}

MeshData Prim::createCylinderData(float radius, float length, UpAxis upAxis,
                                  int segments) {
    int perm[3];
    if (upAxis == UpAxis::X) {
        perm[0] = 1;
        perm[1] = 2;
        perm[2] = 0; // long axis → X
    } else if (upAxis == UpAxis::Z) {
        perm[0] = 2;
        perm[1] = 0;
        perm[2] = 1; // long axis → Z
    } else {
        perm[0] = 0;
        perm[1] = 1;
        perm[2] = 2; // long axis → Y (default)
    }
    return makeCylinderMesh(radius, length / 2.0f, radius, perm, segments);
}

MeshData Prim::createArrowData(float baseRadius, float baseHeight,
                               int segments) {
    return createArrowData(baseRadius, baseHeight, UpAxis::Y, -1.0f, -1.0f,
                           segments);
}

MeshData Prim::createArrowData(float baseRadius, float baseHeight,
                               UpAxis upAxis, float capRadius, float capHeight,
                               int segments) {
    if (capRadius < 0.0f)
        capRadius = baseRadius * 1.8f;
    if (capHeight < 0.0f)
        capHeight = baseHeight * 0.18f;

    int perm[3];
    if (upAxis == UpAxis::X) {
        perm[0] = 1;
        perm[1] = 2;
        perm[2] = 0; // long axis → X
    } else if (upAxis == UpAxis::Z) {
        perm[0] = 2;
        perm[1] = 0;
        perm[2] = 1; // long axis → Z
    } else {
        perm[0] = 0;
        perm[1] = 1;
        perm[2] = 2; // long axis → Y (default)
    }

    auto P = [&](float x, float y, float z) -> glm::vec3 {
        float v[3] = {x, y, z};
        return {v[perm[0]], v[perm[1]], v[perm[2]]};
    };

    // Shaft: cylinder centered at origin → shift up so bottom sits at 0
    MeshData shaft = makeCylinderMesh(baseRadius, baseHeight / 2.0f, baseRadius,
                                      perm, segments);
    glm::vec3 shaftOff = P(0.0f, baseHeight / 2.0f, 0.0f);
    for (auto& p : shaft.vertices)
        p += shaftOff;

    // Cone tip: topRadius=0, shift to sit on top of shaft (-epsilon to avoid
    // z-fighting)
    MeshData cone =
        makeCylinderMesh(capRadius, capHeight / 2.0f, 0.0f, perm, segments);
    glm::vec3 coneOff =
        P(0.0f, baseHeight + capHeight / 2.0f - 1e-3f * baseHeight, 0.0f);
    for (auto& p : cone.vertices)
        p += coneOff;

    // Merge shaft + cone
    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec2> uvs;
    std::vector<unsigned int> indices;

    positions.insert(positions.end(), shaft.vertices.begin(),
                     shaft.vertices.end());
    positions.insert(positions.end(), cone.vertices.begin(),
                     cone.vertices.end());
    normals.insert(normals.end(), shaft.normals.begin(), shaft.normals.end());
    normals.insert(normals.end(), cone.normals.begin(), cone.normals.end());
    uvs.insert(uvs.end(), shaft.uvs.begin(), shaft.uvs.end());
    uvs.insert(uvs.end(), cone.uvs.begin(), cone.uvs.end());

    auto baseCount = (unsigned int)shaft.vertices.size();
    indices.insert(indices.end(), shaft.indices.begin(), shaft.indices.end());
    for (auto idx : cone.indices)
        indices.push_back(idx + baseCount);

    return MeshData(std::move(positions), std::move(normals), std::move(uvs),
                    std::move(indices));
}

MeshData Prim::createRectangleData(float xScale, float yScale, float zScale) {
    // Scale means the length of one side.
    float xHalf = xScale / 2;
    float yHalf = yScale / 2;
    float zHalf = zScale / 2;
    //    v3----- v7
    //   /|      /|
    //  v2------v6|
    //  | |     | |
    //  | v0----|-v4
    //  |/      |/
    //  v1------v5
    //
    std::vector<glm::vec3> positions = {
        // v0, v1, v2, v3
        glm::vec3(-xHalf, -yHalf, -zHalf),
        glm::vec3(-xHalf, -yHalf, zHalf),
        glm::vec3(-xHalf, yHalf, zHalf),
        glm::vec3(-xHalf, yHalf, -zHalf),
        // v4, v5, v6, v7
        glm::vec3(xHalf, -yHalf, -zHalf),
        glm::vec3(xHalf, -yHalf, zHalf),
        glm::vec3(xHalf, yHalf, zHalf),
        glm::vec3(xHalf, yHalf, -zHalf),
        // v0, v1, v5, v4
        glm::vec3(-xHalf, -yHalf, -zHalf),
        glm::vec3(-xHalf, -yHalf, zHalf),
        glm::vec3(xHalf, -yHalf, zHalf),
        glm::vec3(xHalf, -yHalf, -zHalf),
        // v3, v2, v6, v7
        glm::vec3(-xHalf, yHalf, -zHalf),
        glm::vec3(-xHalf, yHalf, zHalf),
        glm::vec3(xHalf, yHalf, zHalf),
        glm::vec3(xHalf, yHalf, -zHalf),
        // v0, v3, v7, v4
        glm::vec3(-xHalf, -yHalf, -zHalf),
        glm::vec3(-xHalf, yHalf, -zHalf),
        glm::vec3(xHalf, yHalf, -zHalf),
        glm::vec3(xHalf, -yHalf, -zHalf),
        // v1, v2, v6, v5
        glm::vec3(-xHalf, -yHalf, zHalf),
        glm::vec3(-xHalf, yHalf, zHalf),
        glm::vec3(xHalf, yHalf, zHalf),
        glm::vec3(xHalf, -yHalf, zHalf),
    }; // positions.size() == 24

    std::vector<glm::vec3> normals = {
        glm::vec3(-1, 0, 0), glm::vec3(-1, 0, 0), glm::vec3(-1, 0, 0),
        glm::vec3(-1, 0, 0), glm::vec3(1, 0, 0),  glm::vec3(1, 0, 0),
        glm::vec3(1, 0, 0),  glm::vec3(1, 0, 0),  glm::vec3(0, -1, 0),
        glm::vec3(0, -1, 0), glm::vec3(0, -1, 0), glm::vec3(0, -1, 0),
        glm::vec3(0, 1, 0),  glm::vec3(0, 1, 0),  glm::vec3(0, 1, 0),
        glm::vec3(0, 1, 0),  glm::vec3(0, 0, -1), glm::vec3(0, 0, -1),
        glm::vec3(0, 0, -1), glm::vec3(0, 0, -1), glm::vec3(0, 0, 1),
        glm::vec3(0, 0, 1),  glm::vec3(0, 0, 1),  glm::vec3(0, 0, 1),
    };

    std::vector<glm::vec2> uvs = {
        glm::vec2(0, 0), glm::vec2(1, 0), glm::vec2(1, 1), glm::vec2(0, 1),
        glm::vec2(0, 0), glm::vec2(1, 0), glm::vec2(1, 1), glm::vec2(0, 1),
        glm::vec2(0, 0), glm::vec2(1, 0), glm::vec2(1, 1), glm::vec2(0, 1),
        glm::vec2(0, 0), glm::vec2(1, 0), glm::vec2(1, 1), glm::vec2(0, 1),
        glm::vec2(0, 0), glm::vec2(1, 0), glm::vec2(1, 1), glm::vec2(0, 1),
        glm::vec2(0, 0), glm::vec2(1, 0), glm::vec2(1, 1), glm::vec2(0, 1),
    };

    std::vector<unsigned int> indices = {
        0,  1,  2,  0,  2,  3,  // left
        4,  6,  5,  4,  7,  6,  // right
        8,  10, 9,  8,  11, 10, // down, v0,v5,v1, v0,v4,v5
        12, 13, 14, 12, 14, 15, // up
        16, 17, 18, 16, 18, 19, // back
        20, 23, 22, 20, 22, 21, // front
    };

    MeshData meshData;
    meshData.vertices = std::move(positions);
    meshData.normals = std::move(normals);
    meshData.uvs = std::move(uvs);
    meshData.indices = std::move(indices);

    return meshData;
}

void Prim::traverse(std::function<void(Prim*)> func) {
    func(this);
    for (auto& child : _children) {
        child->traverse(func);
    }
}

glm::mat4 Prim::computeModelMatrix() {
    if (_isDirty) {
        glm::mat4 result(1.0f);

        const std::vector<Token>* orderPtr = &XformTokens::defaultOpOrder;

        if (hasAttribute(XformTokens::opOrder)) {
            std::vector<Token> customOrder;
            const auto& strOrder =
                getAttribute<std::vector<std::string>>(XformTokens::opOrder);
            customOrder.reserve(strOrder.size());
            for (const auto& s : strOrder) {
                customOrder.emplace_back(s);
            }
            orderPtr = &customOrder;
        }

        for (const auto& opToken : *orderPtr) {
            if (!this->hasAttribute(opToken))
                continue;

            XformOpType type = XformTokens::getXformOpType(opToken);
            glm::mat4 opMat(1.0f);

            switch (type) {
            case XformOpType::Translate: {
                auto t = getAttribute<glm::vec3>(opToken);
                opMat = glm::translate(glm::mat4(1.0f), t);
                break;
            }
            case XformOpType::RotateQuat: {
                auto q = getAttribute<glm::quat>(opToken);
                opMat = glm::mat4_cast(q);
                break;
            }
            case XformOpType::RotateXYZ: {
                auto r = getAttribute<glm::vec3>(opToken);
                glm::mat4 rot =
                    glm::rotate(glm::mat4(1.0f), glm::radians(r.x), {1, 0, 0});
                rot = glm::rotate(rot, glm::radians(r.y), {0, 1, 0});
                rot = glm::rotate(rot, glm::radians(r.z), {0, 0, 1});
                opMat = rot;
                break;
            }
            case XformOpType::Scale: {
                auto s = getAttribute<glm::vec3>(opToken);
                opMat = glm::scale(glm::mat4(1.0f), s);
                break;
            }
            default:
                // TODO: Unknown Op 혹은 지원하지 않는 타입 처리
                break;
            }

            result = opMat * result; // T * R * S * vec right when using glm.
            // result = result * opMat; // S * R * T * vec
        }
        _cachedModelMat = result;
        _isDirty = false;
    }
    return _cachedModelMat;
}

} // namespace Scene
} // namespace KE
