#define TINYOBJLOADER_IMPLEMENTATION
#include "mesh_loader.hpp"
#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>

namespace KE {
namespace Asset {

namespace {

bool isBinaryStl(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open())
        return false;

    // Read first 80 bytes (header), then check if "solid" prefix
    // is followed by valid ASCII STL content
    char header[80];
    file.read(header, 80);
    if (!file.good())
        return false;

    // Read triangle count
    uint32_t triCount = 0;
    file.read(reinterpret_cast<char*>(&triCount), sizeof(triCount));
    if (!file.good())
        return false;

    // Check if file size matches binary STL format
    // Binary: 80 header + 4 count + (50 bytes per triangle)
    file.seekg(0, std::ios::end);
    auto fileSize = file.tellg();
    auto expectedSize = static_cast<std::streampos>(84 + triCount * 50);

    return fileSize == expectedSize;
}

Scene::MeshData loadBinaryStl(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open STL file: " << path << std::endl;
        return {};
    }

    // Skip 80-byte header
    file.seekg(80);

    uint32_t triCount = 0;
    file.read(reinterpret_cast<char*>(&triCount), sizeof(triCount));

    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec2> uvs;
    std::vector<unsigned int> indices;

    positions.reserve(triCount * 3);
    normals.reserve(triCount * 3);
    indices.reserve(triCount * 3);

    for (uint32_t i = 0; i < triCount; i++) {
        float normal[3];
        float v1[3], v2[3], v3[3];
        uint16_t attrByteCount;

        file.read(reinterpret_cast<char*>(normal), 12);
        file.read(reinterpret_cast<char*>(v1), 12);
        file.read(reinterpret_cast<char*>(v2), 12);
        file.read(reinterpret_cast<char*>(v3), 12);
        file.read(reinterpret_cast<char*>(&attrByteCount),
                  sizeof(attrByteCount));

        if (!file.good()) {
            std::cerr << "STL read error at triangle " << i << std::endl;
            break;
        }

        glm::vec3 n(normal[0], normal[1], normal[2]);

        unsigned int baseIdx = static_cast<unsigned int>(positions.size());

        positions.emplace_back(v1[0], v1[1], v1[2]);
        positions.emplace_back(v2[0], v2[1], v2[2]);
        positions.emplace_back(v3[0], v3[1], v3[2]);

        normals.push_back(n);
        normals.push_back(n);
        normals.push_back(n);

        indices.push_back(baseIdx);
        indices.push_back(baseIdx + 1);
        indices.push_back(baseIdx + 2);
    }

    Scene::MeshData meshData;
    meshData.vertices = std::move(positions);
    meshData.normals = std::move(normals);
    meshData.uvs = std::move(uvs);
    meshData.indices = std::move(indices);
    return meshData;
}

Scene::MeshData loadAsciiStl(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Failed to open STL file: " << path << std::endl;
        return {};
    }

    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec2> uvs;
    std::vector<unsigned int> indices;

    std::string line;
    glm::vec3 currentNormal(0.0f, 1.0f, 0.0f);

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string keyword;
        iss >> keyword;

        if (keyword == "facet") {
            std::string normalStr;
            iss >> normalStr; // "normal"
            float nx, ny, nz;
            iss >> nx >> ny >> nz;
            currentNormal = glm::vec3(nx, ny, nz);
        } else if (keyword == "vertex") {
            float vx, vy, vz;
            iss >> vx >> vy >> vz;

            unsigned int idx = static_cast<unsigned int>(positions.size());
            positions.emplace_back(vx, vy, vz);
            normals.push_back(currentNormal);
            indices.push_back(idx);
        }
    }

    Scene::MeshData meshData;
    meshData.vertices = std::move(positions);
    meshData.normals = std::move(normals);
    meshData.uvs = std::move(uvs);
    meshData.indices = std::move(indices);
    return meshData;
}

} // anonymous namespace

Scene::MeshData loadStl(const std::string& path) {
    if (isBinaryStl(path)) {
        return loadBinaryStl(path);
    }
    return loadAsciiStl(path);
}

Scene::MeshData loadObj(
    std::string path, std::optional<tinyobj::ObjReaderConfig> render_config) {

    tinyobj::ObjReader reader;
    tinyobj::ObjReaderConfig config;

    if (render_config.has_value()) {
        config = render_config.value();
    } else {
        config.mtl_search_path = "./"; // Path to material files
    }

    if (!reader.ParseFromFile(path, config)) {
        if (!reader.Error().empty()) {
            std::cerr << "TinyObjReader: " << reader.Error();
        }
        exit(1);
    }

    if (!reader.Warning().empty()) {
        std::cout << "TinyObjReader: " << reader.Warning();
    }

    auto& attrib = reader.GetAttrib();
    auto& shapes = reader.GetShapes();
    auto& materials = reader.GetMaterials();

    std::vector<glm::vec3> positions = {};
    std::vector<glm::vec3> normals = {};
    std::vector<glm::vec2> uvs = {};
    std::vector<unsigned int> indices = {};

    // Loop over shapes
    for (size_t s = 0; s < shapes.size(); s++) {
        // Loop over faces(polygon)
        size_t index_offset = 0;
        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
            size_t fv = size_t(shapes[s].mesh.num_face_vertices[f]);

            // Loop over vertices in the face.
            for (size_t v = 0; v < fv; v++) {
                // access to vertex
                tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
                tinyobj::real_t vx =
                    attrib.vertices[3 * size_t(idx.vertex_index) + 0];
                tinyobj::real_t vy =
                    attrib.vertices[3 * size_t(idx.vertex_index) + 1];
                tinyobj::real_t vz =
                    attrib.vertices[3 * size_t(idx.vertex_index) + 2];

                positions.push_back(glm::vec3(vx, vy, vz));

                // Normal (default if not present)
                if (idx.normal_index >= 0) {
                    tinyobj::real_t nx =
                        attrib.normals[3 * size_t(idx.normal_index) + 0];
                    tinyobj::real_t ny =
                        attrib.normals[3 * size_t(idx.normal_index) + 1];
                    tinyobj::real_t nz =
                        attrib.normals[3 * size_t(idx.normal_index) + 2];
                    normals.push_back(glm::vec3(nx, ny, nz));
                } else {
                    normals.push_back(glm::vec3(0.0f, 1.0f, 0.0f));
                }

                // UV (default if not present)
                if (idx.texcoord_index >= 0) {
                    tinyobj::real_t tx =
                        attrib.texcoords[2 * size_t(idx.texcoord_index) + 0];
                    tinyobj::real_t ty =
                        attrib.texcoords[2 * size_t(idx.texcoord_index) + 1];
                    uvs.push_back(glm::vec2(tx, ty));
                } else {
                    uvs.push_back(glm::vec2(0.0f, 0.0f));
                }

                // Optional: vertex colors
                // tinyobj::real_t red   =
                // attrib.colors[3*size_t(idx.vertex_index)+0];
                // tinyobj::real_t green =
                // attrib.colors[3*size_t(idx.vertex_index)+1];
                // tinyobj::real_t blue  =
                // attrib.colors[3*size_t(idx.vertex_index)+2];

                // Index - current vertex index
                // indices.push_back((unsigned
                // int)size_t(idx.texcoord_index)); // wrong
                indices.push_back(
                    static_cast<unsigned int>(positions.size() - 1));
            }
            index_offset += fv;
        }
    }

    Scene::MeshData meshData;
    meshData.vertices = positions;
    meshData.normals = normals;
    meshData.uvs = uvs;
    meshData.indices = indices;

    return meshData;
}

} // namespace Asset
} // namespace KE
