#include "geometry/primitive_mesh.hpp"

#include <algorithm>
#include <glm/ext/scalar_constants.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/geometric.hpp>
#include <glm/trigonometric.hpp>
#include <vector>

namespace KE {
namespace Geometry {
namespace {

using KE::Scene::MeshData;

// Internal helper: truncated cylinder (topRadius==radius → cylinder,
// topRadius==0 → cone) Ported from references/render_opengl.py
// _create_cylinder_mesh
MeshData makeCylinderMesh(float radius, float halfLength, float topRadius,
                          const int perm[3], int segments) {
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
            normals.emplace_back(P(0.0f, static_cast<float>(j), 0.0f));
            uvs.emplace_back(c * 0.5f + 0.5f, s * 0.5f + 0.5f);

            int cs = ci * segments;
            unsigned int vCurr = 2u + i + cs;
            unsigned int vNext = 2u + (i + 1) % segments + cs;

            if (j == -1) {
                // bottom cap: CCW from -Y (outside)
                indices.emplace_back(ci);
                indices.emplace_back(vCurr);
                indices.emplace_back(vNext);
            } else {
                // top cap: CCW from +Y (outside)
                indices.emplace_back(vNext);
                indices.emplace_back(vCurr);
                indices.emplace_back(ci);
            }
        }
    }

    // Side vertices
    int sideStart = static_cast<int>(positions.size());
    for (int j : {-1, 1}) {
        float y = j * halfLength;
        float r = (j == -1) ? radius : topRadius;
        float v = (static_cast<float>(j) + 1.0f) / 2.0f;

        for (int i = 0; i < segments; ++i) {
            float theta = pi2 * i / segments;
            float c = glm::cos(theta);
            float s = glm::sin(theta);

            positions.emplace_back(P(r * c, y, r * s));
            normals.emplace_back(glm::normalize(P(c, sideSlope, s)));
            float u = static_cast<float>(i) / static_cast<float>(segments - 1);
            uvs.emplace_back(u, v);
        }
    }

    for (int i = 0; i < segments; ++i) {
        unsigned int topI = sideStart + i + segments;
        unsigned int topNext = sideStart + (i + 1) % segments + segments;
        unsigned int botI = sideStart + i;
        unsigned int botNext = sideStart + (i + 1) % segments;

        indices.emplace_back(topI);
        indices.emplace_back(topNext);
        indices.emplace_back(botI);
        indices.emplace_back(topNext);
        indices.emplace_back(botNext);
        indices.emplace_back(botI);
    }

    return MeshData(std::move(positions), std::move(normals), std::move(uvs),
                    std::move(indices));
}

// Helper: merge src into dst, offsetting src indices by dst.vertices.size()
void mergeMesh(MeshData& dst, MeshData&& src) {
    auto base = static_cast<unsigned int>(dst.vertices.size());
    dst.vertices.insert(dst.vertices.end(), src.vertices.begin(),
                        src.vertices.end());
    dst.normals.insert(dst.normals.end(), src.normals.begin(),
                       src.normals.end());
    dst.uvs.insert(dst.uvs.end(), src.uvs.begin(), src.uvs.end());
    for (auto idx : src.indices)
        dst.indices.emplace_back(idx + base);
}

} // namespace

Scene::MeshData createCube(float scale) {
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
        {-1, 0, 0}, {-1, 0, 0}, {-1, 0, 0}, {-1, 0, 0}, {1, 0, 0},  {1, 0, 0},
        {1, 0, 0},  {1, 0, 0},  {0, -1, 0}, {0, -1, 0}, {0, -1, 0}, {0, -1, 0},
        {0, 1, 0},  {0, 1, 0},  {0, 1, 0},  {0, 1, 0},  {0, 0, -1}, {0, 0, -1},
        {0, 0, -1}, {0, 0, -1}, {0, 0, 1},  {0, 0, 1},  {0, 0, 1},  {0, 0, 1},
    };

    std::vector<glm::vec2> uvs = {
        {0, 0}, {1, 0}, {1, 1}, {0, 1}, {0, 0}, {1, 0}, {1, 1}, {0, 1},
        {0, 0}, {1, 0}, {1, 1}, {0, 1}, {0, 0}, {1, 0}, {1, 1}, {0, 1},
        {0, 0}, {1, 0}, {1, 1}, {0, 1}, {0, 0}, {1, 0}, {1, 1}, {0, 1},
    };

    std::vector<unsigned int> indices = {
        0,  1,  2,  0,  2,  3,  // left
        4,  6,  5,  4,  7,  6,  // right
        8,  10, 9,  8,  11, 10, // down, v0,v5,v1, v0,v4,v5
        12, 13, 14, 12, 14, 15, // up
        16, 17, 18, 16, 18, 19, // back
        20, 23, 22, 20, 22, 21, // front
    };

    return MeshData(std::move(positions), std::move(normals), std::move(uvs),
                    std::move(indices));
}

Scene::MeshData createPlane(float scale, UpAxis upAxis) {
    float half = scale / 2.0f;
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
        positions = {{-half, 0, half},
                     {half, 0, half},
                     {-half, 0, -half},
                     {half, 0, -half}};
        normals.assign(4, {0, 1, 0});
    } else if (upAxis == UpAxis::Z) {
        // XY plane, Z-up normal
        positions = {{-half, -half, 0},
                     {half, -half, 0},
                     {-half, half, 0},
                     {half, half, 0}};
        normals.assign(4, {0, 0, 1});
    } else {
        // YZ plane, X-up normal
        positions = {{0, -half, -half},
                     {0, half, -half},
                     {0, -half, half},
                     {0, half, half}};
        normals.assign(4, {1, 0, 0});
    }

    std::vector<glm::vec2> uvs = {
        {0, 0}, {scale, 0}, {0, scale}, {scale, scale}};
    std::vector<unsigned int> indices = {0, 1, 3, 0, 3, 2};

    return MeshData(std::move(positions), std::move(normals), std::move(uvs),
                    std::move(indices));
}

Scene::MeshData createSphere(float radius, int numLongitudes,
                             int numLatitudes) {
    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec2> uvs;
    std::vector<unsigned int> indices;

    positions.reserve(numLatitudes * numLongitudes);
    normals.reserve(numLatitudes * numLongitudes);
    uvs.reserve(numLatitudes * numLongitudes);
    indices.reserve((numLatitudes - 1) * (numLongitudes - 1) * 6);

    float thetaUnit = glm::pi<float>() / (numLatitudes - 1);
    float phiUnit = 2.0f * glm::pi<float>() / (numLongitudes - 1);

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

            float u = static_cast<float>(j) / (numLongitudes - 1);
            float v = static_cast<float>(i) / (numLatitudes - 1);

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

Scene::MeshData createBox(float xScale, float yScale, float zScale) {
    // Scale means the length of one side.
    float xHalf = xScale / 2.0f;
    float yHalf = yScale / 2.0f;
    float zHalf = zScale / 2.0f;

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
        {-xHalf, -yHalf, -zHalf},
        {-xHalf, -yHalf, zHalf},
        {-xHalf, yHalf, zHalf},
        {-xHalf, yHalf, -zHalf},
        // v4, v5, v6, v7
        {xHalf, -yHalf, -zHalf},
        {xHalf, -yHalf, zHalf},
        {xHalf, yHalf, zHalf},
        {xHalf, yHalf, -zHalf},
        // v0, v1, v5, v4
        {-xHalf, -yHalf, -zHalf},
        {-xHalf, -yHalf, zHalf},
        {xHalf, -yHalf, zHalf},
        {xHalf, -yHalf, -zHalf},
        // v3, v2, v6, v7
        {-xHalf, yHalf, -zHalf},
        {-xHalf, yHalf, zHalf},
        {xHalf, yHalf, zHalf},
        {xHalf, yHalf, -zHalf},
        // v0, v3, v7, v4
        {-xHalf, -yHalf, -zHalf},
        {-xHalf, yHalf, -zHalf},
        {xHalf, yHalf, -zHalf},
        {xHalf, -yHalf, -zHalf},
        // v1, v2, v6, v5
        {-xHalf, -yHalf, zHalf},
        {-xHalf, yHalf, zHalf},
        {xHalf, yHalf, zHalf},
        {xHalf, -yHalf, zHalf},
    }; // positions.size() == 24

    std::vector<glm::vec3> normals = {
        {-1, 0, 0}, {-1, 0, 0}, {-1, 0, 0}, {-1, 0, 0}, {1, 0, 0},  {1, 0, 0},
        {1, 0, 0},  {1, 0, 0},  {0, -1, 0}, {0, -1, 0}, {0, -1, 0}, {0, -1, 0},
        {0, 1, 0},  {0, 1, 0},  {0, 1, 0},  {0, 1, 0},  {0, 0, -1}, {0, 0, -1},
        {0, 0, -1}, {0, 0, -1}, {0, 0, 1},  {0, 0, 1},  {0, 0, 1},  {0, 0, 1},
    };

    std::vector<glm::vec2> uvs = {
        {0, 0}, {1, 0}, {1, 1}, {0, 1}, {0, 0}, {1, 0}, {1, 1}, {0, 1},
        {0, 0}, {1, 0}, {1, 1}, {0, 1}, {0, 0}, {1, 0}, {1, 1}, {0, 1},
        {0, 0}, {1, 0}, {1, 1}, {0, 1}, {0, 0}, {1, 0}, {1, 1}, {0, 1},
    };

    std::vector<unsigned int> indices = {
        0,  1,  2,  0,  2,  3,  // left
        4,  6,  5,  4,  7,  6,  // right
        8,  10, 9,  8,  11, 10, // down, v0,v5,v1, v0,v4,v5
        12, 13, 14, 12, 14, 15, // up
        16, 17, 18, 16, 18, 19, // back
        20, 23, 22, 20, 22, 21, // front
    };

    return MeshData(std::move(positions), std::move(normals), std::move(uvs),
                    std::move(indices));
}

Scene::MeshData createCylinder(float radius, float length, UpAxis upAxis,
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

Scene::MeshData createCone(float radius, float height, UpAxis upAxis,
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

Scene::MeshData createArrow(float baseRadius, float baseHeight, UpAxis upAxis,
                            float capRadius, float capHeight, int segments) {
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
    MeshData shaft = createCylinder(baseRadius, baseHeight, upAxis, segments);
    for (auto& v : shaft.vertices)
        v += axisDir * (baseHeight / 2.0f);

    // Cone tip: shift to sit on top of shaft (small epsilon to avoid
    // z-fighting)
    MeshData cone = createCone(capRadius, capHeight, upAxis, segments);
    float coneCenter = baseHeight + capHeight / 2.0f - 1e-3f * baseHeight;
    for (auto& v : cone.vertices)
        v += axisDir * coneCenter;

    mergeMesh(shaft, std::move(cone));
    return shaft;
}

Scene::MeshData createCapsule(float radius, float height, UpAxis upAxis,
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

    int totalRings = 2 * rings + 2;
    int vertsPerRing = segments + 1;

    // Layout (internal Y-up):
    //   row 0        : top pole          (theta=0,      yOff=+halfH)
    //   row 1..rings : top hemisphere    (theta→π/2,    yOff=+halfH)
    //   row rings+1  : bottom equator    (theta=π/2,    yOff=-halfH)
    //   row rings+2..2*rings+1: bottom hemisphere→pole (theta→π, yOff=-halfH)
    // The quad strip between row 'rings' and 'rings+1' forms the cylinder wall.
    for (int r = 0; r < totalRings; ++r) {
        float theta, yOff;
        if (r <= rings) {
            // top hemisphere: theta 0 → π/2
            theta = glm::half_pi<float>() * r / rings;
            yOff = halfH;
        } else {
            // bottom hemisphere: theta π/2 → π
            int br = r - rings - 1;
            theta = glm::half_pi<float>() + glm::half_pi<float>() * br / rings;
            yOff = -halfH;
        }

        float sinT = glm::sin(theta);
        float cosT = glm::cos(theta);
        float ringY = yOff + radius * cosT;
        float ringR = radius * sinT;

        float vCoord = static_cast<float>(r) / (totalRings - 1);
        for (int s = 0; s <= segments; ++s) {
            float phi = glm::two_pi<float>() * s / segments;
            float cosP = glm::cos(phi);
            float sinP = glm::sin(phi);

            positions.emplace_back(P(ringR * cosP, ringY, ringR * sinP));
            normals.emplace_back(P(sinT * cosP, cosT, sinT * sinP));
            uvs.emplace_back(static_cast<float>(s) / segments, vCoord);
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

} // namespace Geometry
} // namespace KE
