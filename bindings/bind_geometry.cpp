#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "geometry/primitive_mesh.hpp"
#include "py_array_view.hpp"

namespace py = pybind11;

void bind_geometry(py::module& m) {
    py::module scene = m.attr("scene");
    py::module geometry = m.attr("geometry");

    // MeshData struct (with shared_ptr holder for set_mesh_data compatibility)
    py::class_<KE::Scene::MeshData, std::shared_ptr<KE::Scene::MeshData>>(
        scene, "MeshData",
        "Static mesh payload with vertex attributes and triangle indices.")
        .def(py::init<>(), "Create empty mesh data.")
        .def_static(
            "from_arrays",
            [](const FloatArray& vertices, const IntArray& indices) {
                auto points = vec3ArrayView(vertices, "vertices");
                auto indexInfo = indices.request();
                if (indexInfo.size % 3)
                    throw py::value_error("indices must contain triangles");
                auto mesh = std::make_shared<KE::Scene::MeshData>();
                mesh->vertices.reserve(points.count);
                for (size_t i = 0; i < points.count; ++i) {
                    const float* p = points.data + 3 * i;
                    if (!std::isfinite(p[0]) || !std::isfinite(p[1]) ||
                        !std::isfinite(p[2]))
                        throw py::value_error("vertices must be finite");
                    mesh->vertices.emplace_back(p[0], p[1], p[2]);
                }
                const int* data = static_cast<const int*>(indexInfo.ptr);
                for (py::ssize_t i = 0; i < indexInfo.size; ++i) {
                    if (data[i] < 0 || size_t(data[i]) >= points.count)
                        throw py::value_error("mesh index out of range");
                    mesh->indices.push_back(static_cast<unsigned int>(data[i]));
                }
                mesh->fillMissingAttributes();
                return mesh;
            },
            py::arg("vertices"), py::arg("indices"),
            "Create a mesh from CPU vertex [N,3] and triangle index arrays.")
        .def_readwrite("vertices", &KE::Scene::MeshData::vertices,
                       "Vertex positions.")
        .def_readwrite("normals", &KE::Scene::MeshData::normals,
                       "Vertex normals.")
        .def_readwrite("uvs", &KE::Scene::MeshData::uvs,
                       "Vertex texture coordinates.")
        .def_readwrite("indices", &KE::Scene::MeshData::indices,
                       "Triangle index buffer.");

    auto meshResult = [](KE::Scene::MeshData data) {
        return std::make_shared<KE::Scene::MeshData>(std::move(data));
    };
    geometry.def(
        "create_cube_data",
        [meshResult](float scale) {
            return meshResult(KE::Geometry::createCube(scale));
        },
        py::arg("scale") = 1.0f, "Create a cube mesh payload.");
    geometry.def(
        "create_plane_data",
        [meshResult](float scale, KE::UpAxis upAxis) {
            return meshResult(KE::Geometry::createPlane(scale, upAxis));
        },
        py::arg("scale"), py::arg("up_axis") = KE::UpAxis::Y,
        "Create a plane mesh payload.");
    geometry.def(
        "create_sphere_data",
        [meshResult](float radius, int longitudes, int latitudes) {
            return meshResult(
                KE::Geometry::createSphere(radius, longitudes, latitudes));
        },
        py::arg("radius"), py::arg("num_longitudes"), py::arg("num_latitudes"),
        "Create a UV sphere mesh payload.");
    geometry.def(
        "create_box_data",
        [meshResult](float x, float y, float z) {
            return meshResult(KE::Geometry::createBox(x, y, z));
        },
        py::arg("x_scale"), py::arg("y_scale"), py::arg("z_scale"),
        "Create a box mesh payload with independent axis lengths.");
    geometry.def(
        "create_cylinder_data",
        [meshResult](float radius, float length, KE::UpAxis upAxis,
                     int segments) {
            return meshResult(
                KE::Geometry::createCylinder(radius, length, upAxis, segments));
        },
        py::arg("radius"), py::arg("length"),
        py::arg("up_axis") = KE::UpAxis::Y, py::arg("segments") = 32,
        "Create a cylinder mesh payload.");
    geometry.def(
        "create_arrow_data",
        [meshResult](float baseRadius, float baseHeight, KE::UpAxis upAxis,
                     float capRadius, float capHeight, int segments) {
            return meshResult(KE::Geometry::createArrow(baseRadius, baseHeight,
                                                        upAxis, capRadius,
                                                        capHeight, segments));
        },
        py::arg("base_radius"), py::arg("base_height"),
        py::arg("up_axis") = KE::UpAxis::Y, py::arg("cap_radius") = -1.0f,
        py::arg("cap_height") = -1.0f, py::arg("segments") = 32,
        "Create an arrow mesh payload.");
    geometry.def(
        "create_capsule_data",
        [meshResult](float radius, float height, KE::UpAxis upAxis,
                     int segments) {
            return meshResult(
                KE::Geometry::createCapsule(radius, height, upAxis, segments));
        },
        py::arg("radius"), py::arg("height"),
        py::arg("up_axis") = KE::UpAxis::Y, py::arg("segments") = 32,
        "Create a capsule mesh payload.");
    geometry.def(
        "create_cone_data",
        [meshResult](float radius, float height, KE::UpAxis upAxis,
                     int segments) {
            return meshResult(
                KE::Geometry::createCone(radius, height, upAxis, segments));
        },
        py::arg("radius"), py::arg("height"),
        py::arg("up_axis") = KE::UpAxis::Y, py::arg("segments") = 32,
        "Create a cone mesh payload.");

    py::class_<KE::Scene::SkinnedMeshData,
               std::shared_ptr<KE::Scene::SkinnedMeshData>>(
        scene, "SkinnedMeshData",
        "Mesh payload with skinning attributes for skeletal animation.")
        .def(py::init<>(), "Create empty skinned mesh data.")
        .def(
            py::init([](std::shared_ptr<KE::Scene::MeshData> mesh,
                        const IntArray& boneIndices,
                        const FloatArray& boneWeights,
                        std::vector<int> boneNodeIndices,
                        const FloatArray& inverseBindMatrices) {
                if (!mesh)
                    throw py::value_error("mesh is None");
                auto result = std::make_shared<KE::Scene::SkinnedMeshData>();
                result->mesh = *mesh;
                result->boneIndices = intVec4Array(boneIndices, "bone_indices");
                result->boneWeights = vec4Array(boneWeights, "bone_weights");
                result->boneNodeIndices = std::move(boneNodeIndices);
                result->inverseBindMatrices = mat4RowMajorArray(
                    inverseBindMatrices, "inverse_bind_matrices");
                const size_t vertexCount = result->mesh.vertices.size();
                if (result->boneIndices.size() != vertexCount ||
                    result->boneWeights.size() != vertexCount) {
                    throw py::value_error("bone_indices and bone_weights must "
                                          "match mesh vertices");
                }
                if (result->boneNodeIndices.size() !=
                    result->inverseBindMatrices.size()) {
                    throw py::value_error("bone_node_indices and "
                                          "inverse_bind_matrices must match");
                }
                const int boneCount =
                    static_cast<int>(result->inverseBindMatrices.size());
                for (size_t vertex = 0; vertex < vertexCount; ++vertex) {
                    const glm::ivec4 indices = result->boneIndices[vertex];
                    const glm::vec4 weights = result->boneWeights[vertex];
                    for (int slot = 0; slot < 4; ++slot) {
                        if (weights[slot] > 0.0f &&
                            (indices[slot] < 0 || indices[slot] >= boneCount)) {
                            throw py::value_error("weighted bone index is "
                                                  "outside the bone palette");
                        }
                    }
                }
                return result;
            }),
            py::arg("mesh"), py::arg("bone_indices"), py::arg("bone_weights"),
            py::arg("bone_node_indices"), py::arg("inverse_bind_matrices"),
            "Create a complete skinned mesh payload from array data.")
        .def_readwrite("mesh", &KE::Scene::SkinnedMeshData::mesh,
                       "Base static mesh data.")
        .def_readwrite("bone_node_indices",
                       &KE::Scene::SkinnedMeshData::boneNodeIndices,
                       "Skeleton node index for each bound bone.")
        .def(
            "has_valid_vertex_skinning",
            &KE::Scene::SkinnedMeshData::hasValidVertexSkinning,
            "Return true when vertex bone data matches the mesh vertex count.");
}
