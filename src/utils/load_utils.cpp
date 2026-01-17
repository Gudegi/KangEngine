#define TINYOBJLOADER_IMPLEMENTATION
#include "load_utils.hpp"

namespace KE {

KE::Scene::MeshData
loadObj(std::string path,
        std::optional<tinyobj::ObjReaderConfig> render_config) {

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

} // namespace KE