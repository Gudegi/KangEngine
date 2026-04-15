///
/// Scene Prim Implementation
///

#include "prim.hpp"
#include "engine/scene/scene_backend.hpp"
#include "utils/types.hpp"
#include <Eigen/Geometry>
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
    _children.emplace_back(std::move(child));

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

    // 순차 탐색
    Prim* current = this;
    size_t start = 0;
    size_t end = pathCopy.find('/');

    while (end != std::string::npos) {
        std::string part = pathCopy.substr(start, end - start);
        if (!part.empty()) {
            current = current->getChild(part);
            if (!current)
                return nullptr;
        }
        start = end + 1;
        end = pathCopy.find('/', start);
    }

    // Last part
    std::string part = pathCopy.substr(start);
    if (!part.empty()) {
        current = current->getChild(part);
    }
    return current;
}

std::vector<Prim*> Prim::getChildren() const {
    std::vector<Prim*> result;
    result.reserve(_children.size());
    for (const auto& child : _children) {
        result.emplace_back(child.get());
    }
    return result;
}

void Prim::setMeshData(std::shared_ptr<MeshData> data) { _meshData = data; }

std::shared_ptr<MeshData> Prim::getMeshData() const { return _meshData; }

MeshData Prim::createSquareData(float scale) { return createCubeData(scale); }

MeshData Prim::createCubeData(float scale) {
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
        // XZ plane, Y-up normal, CCW from +Y with indices {0,1,3,0,3,2}
        // (v1-v0)x(v3-v0) = (2h,0,0)x(2h,0,-2h) -> +Y
        positions = {
            glm::vec3(-half, 0, half),
            glm::vec3(half, 0, half),
            glm::vec3(-half, 0, -half),
            glm::vec3(half, 0, -half),
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
    std::vector<unsigned int> indices = {0, 1, 3, 0, 3, 2};

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
    positions.emplace_back(P(0.0f, -halfLength, 0.0f));
    normals.emplace_back(P(0.0f, -1.0f, 0.0f));
    uvs.emplace_back(0.5f, 0.5f);

    positions.emplace_back(P(0.0f, halfLength, 0.0f));
    normals.emplace_back(P(0.0f, 1.0f, 0.0f));
    uvs.emplace_back(0.5f, 0.5f);

    for (int j : {-1, 1}) {
        unsigned int ci = (j == 1) ? 1u : 0u;
        float y = j * halfLength;
        float r = (j == -1) ? radius : topRadius;

        for (int i = 0; i < segments; ++i) {
            float theta = pi2 * i / segments;
            float c = glm::cos(theta);
            float s = glm::sin(theta);

            positions.emplace_back(P(r * c, y, r * s));
            normals.emplace_back(P(0.0f, (float)j, 0.0f));
            uvs.emplace_back(c * 0.5f + 0.5f, s * 0.5f + 0.5f);

            int cs = ci * segments;
            unsigned int v_curr = 2u + i + cs;
            unsigned int v_next = 2u + (i + 1) % segments + cs;

            if (j == -1) {
                // bottom cap: CCW from -Y (outside)
                indices.emplace_back(ci);
                indices.emplace_back(v_curr);
                indices.emplace_back(v_next);
            } else {
                // top cap: CCW from +Y (outside)
                indices.emplace_back(v_next);
                indices.emplace_back(v_curr);
                indices.emplace_back(ci);
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

            positions.emplace_back(P(r * c, y, r * s));
            normals.emplace_back(glm::normalize(P(c, sideSlope, s)));
            float u = (float)i / (float)(segments - 1);
            uvs.emplace_back(u, v);
        }
    }

    for (int i = 0; i < segments; ++i) {
        unsigned int top_i = sideStart + i + segments;
        unsigned int top_next = sideStart + (i + 1) % segments + segments;
        unsigned int bot_i = sideStart + i;
        unsigned int bot_next = sideStart + (i + 1) % segments;

        indices.emplace_back(top_i);
        indices.emplace_back(top_next);
        indices.emplace_back(bot_i);
        indices.emplace_back(top_next);
        indices.emplace_back(bot_next);
        indices.emplace_back(bot_i);
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

// Helper: merge src into dst, offsetting src indices by dst.vertices.size()
static void mergeMesh(MeshData& dst, MeshData&& src) {
    auto base = static_cast<unsigned int>(dst.vertices.size());
    dst.vertices.insert(dst.vertices.end(), src.vertices.begin(),
                        src.vertices.end());
    dst.normals.insert(dst.normals.end(), src.normals.begin(),
                       src.normals.end());
    dst.uvs.insert(dst.uvs.end(), src.uvs.begin(), src.uvs.end());
    for (auto idx : src.indices)
        dst.indices.emplace_back(idx + base);
}

MeshData Prim::createArrowData(float baseRadius, float baseHeight,
                               UpAxis upAxis, float capRadius, float capHeight,
                               int segments) {
    if (capRadius < 0.0f)
        capRadius = baseRadius * 1.8f;
    if (capHeight < 0.0f)
        capHeight = baseHeight * 0.18f;

    glm::vec3 axisDir;
    if (upAxis == UpAxis::X)
        axisDir = {1, 0, 0};
    else if (upAxis == UpAxis::Z)
        axisDir = {0, 0, 1};
    else
        axisDir = {0, 1, 0};

    // Shaft: cylinder centered at origin → shift so bottom sits at 0
    MeshData shaft =
        createCylinderData(baseRadius, baseHeight, upAxis, segments);
    for (auto& v : shaft.vertices)
        v += axisDir * (baseHeight / 2.0f);

    // Cone tip: shift to sit on top of shaft (small epsilon to avoid
    // z-fighting)
    MeshData cone = createConeData(capRadius, capHeight, upAxis, segments);
    float coneCenter = baseHeight + capHeight / 2.0f - 1e-3f * baseHeight;
    for (auto& v : cone.vertices)
        v += axisDir * coneCenter;

    mergeMesh(shaft, std::move(cone));
    return shaft;
}

MeshData Prim::createCapsuleData(float radius, float height, UpAxis upAxis,
                                 int segments) {
    // rings per hemisphere (at least 4)
    int rings = std::max(4, segments / 4);
    float halfH = height * 0.5f;

    int perm[3];
    if (upAxis == UpAxis::X) {
        perm[0] = 1;
        perm[1] = 2;
        perm[2] = 0;
    } else if (upAxis == UpAxis::Z) {
        perm[0] = 2;
        perm[1] = 0;
        perm[2] = 1;
    } else {
        perm[0] = 0;
        perm[1] = 1;
        perm[2] = 2;
    }

    auto P = [&](float x, float y, float z) -> glm::vec3 {
        float v[3] = {x, y, z};
        return {v[perm[0]], v[perm[1]], v[perm[2]]};
    };

    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec2> uvs;
    std::vector<unsigned int> indices;

    // Layout (internal Y-up):
    //   row 0        : top pole          (theta=0,      yOff=+halfH)
    //   row 1..rings : top hemisphere    (theta→π/2,    yOff=+halfH)
    //   row rings+1  : bottom equator    (theta=π/2,    yOff=-halfH)
    //   row rings+2..2*rings+1: bottom hemisphere→pole (theta→π, yOff=-halfH)
    // The quad strip between row 'rings' and 'rings+1' forms the cylinder wall.
    int totalRings = 2 * rings + 2;
    int vertsPerRing = segments + 1;

    for (int r = 0; r < totalRings; ++r) {
        float theta, yOff;
        if (r <= rings) {
            // top hemisphere: theta 0 → π/2
            theta = glm::half_pi<float>() * r / rings;
            yOff = halfH;
        } else {
            // bottom hemisphere: theta π/2 → π
            int br = r - rings - 1; // 0..rings
            theta = glm::half_pi<float>() + glm::half_pi<float>() * br / rings;
            yOff = -halfH;
        }

        float sinT = glm::sin(theta);
        float cosT = glm::cos(theta);
        float ringY = yOff + radius * cosT;
        float ringR = radius * sinT;

        float vCoord = (float)r / (totalRings - 1);
        for (int s = 0; s <= segments; ++s) {
            float phi = glm::two_pi<float>() * s / segments;
            float cosP = glm::cos(phi);
            float sinP = glm::sin(phi);

            positions.emplace_back(P(ringR * cosP, ringY, ringR * sinP));
            normals.emplace_back(P(sinT * cosP, cosT, sinT * sinP));
            uvs.emplace_back((float)s / segments, vCoord);
        }
    }

    for (int r = 0; r < totalRings - 1; ++r) {
        for (int s = 0; s < segments; ++s) {
            unsigned int a = r * vertsPerRing + s;
            unsigned int b = a + 1;
            unsigned int c = a + vertsPerRing;
            unsigned int d = c + 1;

            // CCW from outside
            indices.emplace_back(a);
            indices.emplace_back(b);
            indices.emplace_back(c);
            indices.emplace_back(b);
            indices.emplace_back(d);
            indices.emplace_back(c);
        }
    }

    return MeshData(std::move(positions), std::move(normals), std::move(uvs),
                    std::move(indices));
}

MeshData Prim::createConeData(float radius, float height, UpAxis upAxis,
                              int segments) {
    int perm[3];
    if (upAxis == UpAxis::X) {
        perm[0] = 1;
        perm[1] = 2;
        perm[2] = 0;
    } else if (upAxis == UpAxis::Z) {
        perm[0] = 2;
        perm[1] = 0;
        perm[2] = 1;
    } else {
        perm[0] = 0;
        perm[1] = 1;
        perm[2] = 2;
    }
    return makeCylinderMesh(radius, height / 2.0f, 0.0f, perm, segments);
}

std::vector<Prim*> Prim::definePoints(SceneBackend* scene,
                                      const std::string& basePath,
                                      const std::vector<glm::vec3>& points,
                                      float radius, glm::vec4 color,
                                      int segments) {
    auto meshData = std::make_shared<MeshData>(
        createSphereData(radius, segments, segments / 2 + 1));

    std::vector<Prim*> result;
    result.reserve(points.size());
    for (size_t i = 0; i < points.size(); ++i) {
        std::string path = basePath + "/" + std::to_string(i);
        auto* prim = scene->definePrim(path, PrimType::Mesh);
        prim->setMeshData(meshData);
        prim->addTranslateOp(points[i]);
        prim->setDisplayColorAlpha(color);
        result.push_back(prim);
    }
    return result;
}

std::vector<Prim*> Prim::defineLines(SceneBackend* scene,
                                     const std::string& basePath,
                                     const std::vector<glm::vec3>& vertices,
                                     const std::vector<unsigned int>& indices,
                                     float radius, glm::vec4 color,
                                     int segments) {

    auto meshData = std::make_shared<MeshData>(
        createCapsuleData(radius, 1.0f, UpAxis::Y, segments));

    std::vector<Prim*> result;
    int segIdx = 0;
    for (size_t i = 0; i + 1 < indices.size(); i += 2) {
        const glm::vec3& a = vertices[indices[i]];
        const glm::vec3& b = vertices[indices[i + 1]];
        glm::vec3 diff = b - a;
        float len = glm::length(diff);
        if (len < 1e-6f)
            continue;

        glm::vec3 dir = diff / len;
        glm::vec3 center = (a + b) * 0.5f;

        // Rotation: Y-axis -> diff direction
        Eigen::Quaternionf eq = Eigen::Quaternionf::FromTwoVectors(
            Eigen::Vector3f::UnitY(), Eigen::Vector3f(dir.x, dir.y, dir.z));
        glm::quat rot(eq.w(), eq.x(), eq.y(), eq.z());

        std::string path = basePath + "/" + std::to_string(segIdx++);
        auto* prim = scene->definePrim(path, PrimType::Mesh);
        prim->setMeshData(meshData);
        prim->addTranslateOp(center);
        prim->addRotateQuaternionOp(rot);
        prim->addScaleOp(
            glm::vec3(1.f, len, 1.f)); // Y-stretch = segment length
        prim->setDisplayColorAlpha(color);
        result.push_back(prim);
    }
    return result;
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

            // Apply operations in reverse order of the list
            // (Pre-multiplication). If opOrder is {Scale, Rotate, Translate},
            // this results in: Result = Translate * (Rotate * (Scale * I)) = T
            // * R * S This produces the standard Local-to-Parent transform.
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
