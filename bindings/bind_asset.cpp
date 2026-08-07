///
/// Asset Python Bindings
/// Asset importer and loader result bindings.
///

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include "py_array_view.hpp"

#include "asset/articulation_desc.hpp"
#include "asset/bvh_loader.hpp"
#include "asset/fbx_loader.hpp"
#include "asset/heightmap_loader.hpp"
#include "asset/mesh_loader.hpp"
#include "asset/mjcf_loader.hpp"
#include "asset/usd_loader.hpp"
#include "animation/skeleton_math.hpp"
#include "animation/skeleton_motion.hpp"
#include "animation/skeleton_state.hpp"
#include "animation/skeleton_tree.hpp"
#include "animation/skinning.hpp"
#include "bridge/articulation_visual_bridge.hpp"
#include "bridge/skeletal_visual_bridge.hpp"
#include "engine/core/app/app.hpp"
#include "engine/scene/native/prim.hpp"
#include "engine/scene/scene_backend.hpp"

namespace py = pybind11;
using namespace KE;
using namespace KE::Asset;
using namespace KE::Animation;
using namespace KE::Bridge;

namespace {

Eigen::Vector3f eigenVec3FromArray(const FloatArray& array, const char* name) {
    auto view = vec3ArrayView(array, name);
    if (view.count != 1)
        throw py::value_error(std::string(name) + " expected shape [3]");
    return Eigen::Vector3f(view.data[0], view.data[1], view.data[2]);
}

Eigen::Quaternionf eigenQuatXyzwFromArray(const FloatArray& array,
                                          const char* name) {
    auto view = vec4ArrayView(array, name);
    if (view.count != 1)
        throw py::value_error(std::string(name) + " expected shape [4]");
    return Eigen::Quaternionf(view.data[3], view.data[0], view.data[1],
                              view.data[2]);
}

std::vector<Eigen::Quaternionf> eigenQuatXyzwArray(const FloatArray& array,
                                                   const char* name) {
    auto view = vec4ArrayView(array, name);
    std::vector<Eigen::Quaternionf> result;
    result.reserve(view.count);
    for (size_t i = 0; i < view.count; ++i) {
        const float* q = view.data + i * 4;
        result.emplace_back(q[3], q[0], q[1], q[2]);
    }
    return result;
}

py::array_t<int>
intArrayFromVec4Vector(const std::vector<std::array<int, 4>>& values) {
    py::array_t<int> array(
        {static_cast<py::ssize_t>(values.size()), py::ssize_t(4)});
    auto view = array.mutable_unchecked<2>();
    for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(values.size()); ++i) {
        for (py::ssize_t j = 0; j < 4; ++j)
            view(i, j) = values[static_cast<size_t>(i)][static_cast<size_t>(j)];
    }
    return array;
}

py::array_t<int> intArrayFromVec4Vector(const std::vector<glm::ivec4>& values) {
    py::array_t<int> array(
        {static_cast<py::ssize_t>(values.size()), py::ssize_t(4)});
    auto view = array.mutable_unchecked<2>();
    for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(values.size()); ++i) {
        const glm::ivec4& v = values[static_cast<size_t>(i)];
        view(i, 0) = v.x;
        view(i, 1) = v.y;
        view(i, 2) = v.z;
        view(i, 3) = v.w;
    }
    return array;
}

py::array_t<float>
floatArrayFromVec4Vector(const std::vector<std::array<float, 4>>& values) {
    py::array_t<float> array(
        {static_cast<py::ssize_t>(values.size()), py::ssize_t(4)});
    auto view = array.mutable_unchecked<2>();
    for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(values.size()); ++i) {
        for (py::ssize_t j = 0; j < 4; ++j) {
            view(i, j) = values[static_cast<size_t>(i)][static_cast<size_t>(j)];
        }
    }
    return array;
}

py::array_t<float>
floatArrayFromVec4Vector(const std::vector<glm::vec4>& values) {
    py::array_t<float> array(
        {static_cast<py::ssize_t>(values.size()), py::ssize_t(4)});
    auto view = array.mutable_unchecked<2>();
    for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(values.size()); ++i) {
        const glm::vec4& v = values[static_cast<size_t>(i)];
        view(i, 0) = v.x;
        view(i, 1) = v.y;
        view(i, 2) = v.z;
        view(i, 3) = v.w;
    }
    return array;
}

py::array_t<float>
floatArrayFromVec3Vector(const std::vector<glm::vec3>& values) {
    py::array_t<float> array(
        {static_cast<py::ssize_t>(values.size()), py::ssize_t(3)});
    auto view = array.mutable_unchecked<2>();
    for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(values.size()); ++i) {
        const glm::vec3& v = values[static_cast<size_t>(i)];
        view(i, 0) = v.x;
        view(i, 1) = v.y;
        view(i, 2) = v.z;
    }
    return array;
}

py::array_t<float>
floatArrayFromVec3Vector(const std::vector<Eigen::Vector3f>& values) {
    py::array_t<float> array(
        {static_cast<py::ssize_t>(values.size()), py::ssize_t(3)});
    auto view = array.mutable_unchecked<2>();
    for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(values.size()); ++i) {
        const Eigen::Vector3f& v = values[static_cast<size_t>(i)];
        view(i, 0) = v.x();
        view(i, 1) = v.y();
        view(i, 2) = v.z();
    }
    return array;
}

py::array_t<float>
floatArrayFromMat4Vector(const std::vector<std::array<float, 16>>& values) {
    py::array_t<float> array({static_cast<py::ssize_t>(values.size()),
                              py::ssize_t(4), py::ssize_t(4)});
    auto view = array.mutable_unchecked<3>();
    for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(values.size()); ++i) {
        for (py::ssize_t row = 0; row < 4; ++row) {
            for (py::ssize_t col = 0; col < 4; ++col) {
                view(i, row, col) = values[static_cast<size_t>(i)]
                                          [static_cast<size_t>(row * 4 + col)];
            }
        }
    }
    return array;
}

py::array_t<float>
floatArrayFromMat4Vector(const std::vector<glm::mat4>& values) {
    py::array_t<float> array({static_cast<py::ssize_t>(values.size()),
                              py::ssize_t(4), py::ssize_t(4)});
    auto view = array.mutable_unchecked<3>();
    for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(values.size()); ++i) {
        const glm::mat4& m = values[static_cast<size_t>(i)];
        for (py::ssize_t row = 0; row < 4; ++row) {
            for (py::ssize_t col = 0; col < 4; ++col)
                view(i, row, col) =
                    m[static_cast<size_t>(col)][static_cast<size_t>(row)];
        }
    }
    return array;
}

py::array_t<float>
floatArrayFromMat4Vector(const std::vector<Eigen::Matrix4f>& values) {
    py::array_t<float> array({static_cast<py::ssize_t>(values.size()),
                              py::ssize_t(4), py::ssize_t(4)});
    auto view = array.mutable_unchecked<3>();
    for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(values.size()); ++i) {
        const Eigen::Matrix4f& m = values[static_cast<size_t>(i)];
        for (py::ssize_t row = 0; row < 4; ++row) {
            for (py::ssize_t col = 0; col < 4; ++col)
                view(i, row, col) =
                    m(static_cast<int>(row), static_cast<int>(col));
        }
    }
    return array;
}

void writeMat4VectorToPy(const std::vector<Eigen::Matrix4f>& values,
                         const FloatArray& output, const char* name) {
    py::buffer_info info = output.request();
    if (info.ndim != 3 || info.shape[1] != 4 || info.shape[2] != 4)
        throw py::value_error(std::string(name) + " expected shape [N, 4, 4]");
    if (static_cast<size_t>(info.shape[0]) != values.size())
        throw py::value_error(std::string(name) + " first dimension mismatch");

    auto* out = static_cast<float*>(info.ptr);
    for (size_t i = 0; i < values.size(); ++i) {
        const Eigen::Matrix4f& m = values[i];
        float* dst = out + i * 16;
        for (int row = 0; row < 4; ++row) {
            for (int col = 0; col < 4; ++col)
                dst[row * 4 + col] = m(row, col);
        }
    }
}

std::vector<std::array<int, 4>> intVec4ArrayFromPy(py::array_t<int> array,
                                                   const char* name) {
    py::buffer_info info = array.request();
    if (info.ndim != 2 || info.shape[1] != 4)
        throw py::value_error(std::string(name) + " expected shape [N, 4]");
    const auto* p = static_cast<const int*>(info.ptr);
    std::vector<std::array<int, 4>> result;
    result.reserve(static_cast<size_t>(info.shape[0]));
    for (py::ssize_t i = 0; i < info.shape[0]; ++i) {
        result.push_back(
            {p[i * 4 + 0], p[i * 4 + 1], p[i * 4 + 2], p[i * 4 + 3]});
    }
    return result;
}

std::vector<std::array<float, 4>> floatVec4ArrayFromPy(const FloatArray& array,
                                                       const char* name) {
    auto view = vec4ArrayView(array, name);
    std::vector<std::array<float, 4>> result;
    result.reserve(view.count);
    for (size_t i = 0; i < view.count; ++i) {
        const float* p = view.data + i * 4;
        result.push_back({p[0], p[1], p[2], p[3]});
    }
    return result;
}

std::vector<Eigen::Vector3f> eigenVec3ArrayFromPy(const FloatArray& array,
                                                  const char* name) {
    auto view = vec3ArrayView(array, name);
    std::vector<Eigen::Vector3f> result;
    result.reserve(view.count);
    for (size_t i = 0; i < view.count; ++i) {
        const float* p = view.data + i * 3;
        result.emplace_back(p[0], p[1], p[2]);
    }
    return result;
}

std::vector<int> intVectorFromPy(py::array_t<int> array, const char* name) {
    py::buffer_info info = array.request();
    if (info.ndim != 1)
        throw py::value_error(std::string(name) + " expected shape [N]");
    const auto* p = static_cast<const int*>(info.ptr);
    return std::vector<int>(p, p + info.shape[0]);
}

std::vector<Eigen::Matrix4f> eigenMat4ArrayFromPy(const FloatArray& array,
                                                  const char* name) {
    auto view = mat4ArrayView(array, name);
    std::vector<Eigen::Matrix4f> result;
    result.reserve(view.count);
    for (size_t i = 0; i < view.count; ++i) {
        Eigen::Matrix4f m;
        const float* p = view.data + i * 16;
        for (int row = 0; row < 4; ++row) {
            for (int col = 0; col < 4; ++col)
                m(row, col) = p[row * 4 + col];
        }
        result.push_back(m);
    }
    return result;
}

} // namespace

void bind_asset(py::module& m) {
    py::module asset =
        m.def_submodule("asset", "Asset importers and loader result types.");

    py::class_<ObjMaterialInfo>(
        asset, "ObjMaterialInfo",
        "Material metadata imported from an OBJ/MTL pair.")
        .def_readonly("name", &ObjMaterialInfo::name)
        .def_readonly("ambient_color", &ObjMaterialInfo::ambientColor)
        .def_readonly("diffuse_color", &ObjMaterialInfo::diffuseColor)
        .def_readonly("specular_color", &ObjMaterialInfo::specularColor)
        .def_readonly("shininess", &ObjMaterialInfo::shininess)
        .def_readonly("diffuse_texture_path",
                      &ObjMaterialInfo::diffuseTexturePath)
        .def_readonly("specular_texture_path",
                      &ObjMaterialInfo::specularTexturePath)
        .def_readonly("alpha_texture_path", &ObjMaterialInfo::alphaTexturePath)
        .def_readonly("normal_texture_path",
                      &ObjMaterialInfo::normalTexturePath)
        .def_readonly("has_diffuse_texture",
                      &ObjMaterialInfo::hasDiffuseTexture)
        .def_readonly("has_specular_texture",
                      &ObjMaterialInfo::hasSpecularTexture)
        .def_readonly("has_alpha_texture", &ObjMaterialInfo::hasAlphaTexture)
        .def_readonly("has_normal_texture", &ObjMaterialInfo::hasNormalTexture);

    py::class_<ObjMeshSubsetInfo>(
        asset, "ObjMeshSubsetInfo",
        "OBJ submesh containing faces that share one material.")
        .def_readonly("name", &ObjMeshSubsetInfo::name)
        .def_readonly("material_index", &ObjMeshSubsetInfo::materialIndex)
        .def_property_readonly(
            "mesh_data",
            [](const ObjMeshSubsetInfo& self) {
                return std::make_shared<KE::Scene::MeshData>(self.meshData);
            },
            "Static mesh payload for this material subset.");

    py::class_<ObjMeshInfo>(
        asset, "ObjMeshInfo",
        "OBJ mesh payload plus material metadata imported from MTL.")
        .def_property_readonly(
            "mesh_data",
            [](const ObjMeshInfo& self) {
                return std::make_shared<KE::Scene::MeshData>(self.meshData);
            },
            "Static mesh payload.")
        .def_readonly("materials", &ObjMeshInfo::materials)
        .def_readonly("subsets", &ObjMeshInfo::subsets)
        .def_readonly("primary_material_index",
                      &ObjMeshInfo::primaryMaterialIndex)
        .def_property_readonly(
            "material_count",
            [](const ObjMeshInfo& self) { return self.materials.size(); })
        .def_property_readonly("subset_count", [](const ObjMeshInfo& self) {
            return self.subsets.size();
        });

    asset.def(
        "load_obj",
        [](const std::string& path) {
            return std::make_shared<KE::Scene::MeshData>(
                KE::Asset::loadObj(path));
        },
        py::arg("path"),
        "Load an OBJ file and return scene.MeshData. MTL is parsed internally; "
        "use load_obj_with_materials() when material metadata is needed.");

    asset.def(
        "load_obj_with_materials",
        [](const std::string& path) {
            return KE::Asset::loadObjWithMaterials(path);
        },
        py::arg("path"),
        "Load an OBJ file and return mesh data plus MTL material metadata.");

    asset.def(
        "load_stl",
        [](const std::string& path) {
            return std::make_shared<KE::Scene::MeshData>(
                KE::Asset::loadStl(path));
        },
        py::arg("path"), "Load an STL file and return scene.MeshData.");

    asset.def(
        "load_heightmap_terrain",
        [](const std::string& path, KE::UpAxis upAxis, float horizontalScale,
           float heightScale, float heightOffset, int sampleStride) {
            KE::Asset::HeightmapTerrainOptions options;
            options.upAxis = upAxis;
            options.horizontalScale = horizontalScale;
            options.heightScale = heightScale;
            options.heightOffset = heightOffset;
            options.sampleStride = sampleStride;
            return std::make_shared<KE::Scene::MeshData>(
                KE::Asset::HeightmapLoader::loadHeightMapTerrain(path,
                                                                 options));
        },
        py::arg("path"), py::arg("up_axis") = KE::UpAxis::Y,
        py::arg("horizontal_scale") = 1.0f, py::arg("height_scale") = 64.0f,
        py::arg("height_offset") = -16.0f, py::arg("sample_stride") = 1,
        "Load a grayscale/RGB heightmap image and build a terrain MeshData.");

    asset.def(
        "height_field_to_mesh",
        [](const FloatArray& heights, KE::UpAxis upAxis, float horizontalScale,
           bool center) {
            py::buffer_info info = heights.request();
            if (info.ndim != 2)
                throw py::value_error(
                    "height_field_to_mesh expected shape [rows, cols]");
            const int rows = static_cast<int>(info.shape[0]);
            const int cols = static_cast<int>(info.shape[1]);
            if (rows < 2 || cols < 2)
                throw py::value_error("height_field_to_mesh requires at least "
                                      "a 2x2 height field");

            KE::Asset::HeightFieldMeshOptions options;
            options.upAxis = upAxis;
            options.horizontalScale = horizontalScale;
            options.center = center;
            return std::make_shared<KE::Scene::MeshData>(
                KE::Asset::heightFieldToMesh(
                    static_cast<const float*>(info.ptr), rows, cols, options));
        },
        py::arg("heights"), py::arg("up_axis") = KE::UpAxis::Y,
        py::arg("horizontal_scale") = 1.0f, py::arg("center") = true,
        "Convert a 2D float height field array [rows, cols] into "
        "scene.MeshData.");
    py::class_<ImportDiagnostics>(
        asset, "ImportDiagnostics",
        "Warnings collected while importing an asset.")
        .def_readonly("warnings", &ImportDiagnostics::warnings,
                      "Human-readable importer warnings.");

    py::class_<MJCFImportResult>(asset, "MJCFImportResult",
                                 "Parsed MJCF articulation plus diagnostics.")
        .def_readonly("articulation", &MJCFImportResult::articulation,
                      "Imported articulation description.")
        .def_readonly("diagnostics", &MJCFImportResult::diagnostics,
                      "Importer diagnostics.");

    py::class_<MJCFLoader>(asset, "MJCFLoader",
                           "Loader for MJCF/XML articulation descriptions.")
        .def_static("parse", &MJCFLoader::parse, py::arg("mjcf_path"),
                    py::arg("scale") = 1.0f, py::arg("order") = "DFS",
                    "Parse MJCF and return an articulation description with diagnostics.")
        .def_static("load", &MJCFLoader::load, py::arg("mjcf_path"),
                    py::arg("scale") = 1.0f, py::arg("order") = "DFS",
                    "Load MJCF and return an articulation description.");

    py::class_<BVHImportResult>(asset, "BVHImportResult",
                                "Parsed BVH skeleton motion plus diagnostics.")
        .def_readonly("motion", &BVHImportResult::motion,
                      "Imported skeleton motion.")
        .def_readonly("diagnostics", &BVHImportResult::diagnostics,
                      "Importer diagnostics.")
        .def_readonly("frame_count", &BVHImportResult::frameCount,
                      "Number of motion frames.")
        .def_readonly("frame_time", &BVHImportResult::frameTime,
                      "Seconds per source frame.")
        .def_readonly("frame_rate", &BVHImportResult::frameRate,
                      "Source frame rate in Hz.");

    py::class_<BVHLoader>(asset, "BVHLoader",
                          "Loader for BVH skeleton and motion files.")
        .def_static("load_skeleton", &BVHLoader::loadSkeleton,
                    py::arg("bvh_path"), py::arg("scale") = 1.0f,
                    "Load only the skeleton hierarchy from a BVH file.")
        .def_static("load_motion", &BVHLoader::loadMotion, py::arg("bvh_path"),
                    py::arg("scale") = 1.0f,
                    "Load skeleton motion from a BVH file.")
        .def_static("parse", &BVHLoader::parse, py::arg("bvh_path"),
                    py::arg("scale") = 1.0f,
                    "Parse BVH and return motion plus diagnostics.");

    py::class_<FBXAnimationClipInfo>(
        asset, "FBXAnimationClipInfo",
        "Animation clip metadata discovered in an FBX file.")
        .def_readonly("name", &FBXAnimationClipInfo::name, "Clip name.")
        .def_readonly("start_time", &FBXAnimationClipInfo::startTime,
                      "Clip start time in seconds.")
        .def_readonly("end_time", &FBXAnimationClipInfo::endTime,
                      "Clip end time in seconds.")
        .def_readonly("frame_rate", &FBXAnimationClipInfo::frameRate,
                      "Clip source frame rate in Hz.");

    py::class_<FBXMaterialInfo>(asset, "FBXMaterialInfo",
                                "Material metadata imported from an FBX file.")
        .def_readonly("name", &FBXMaterialInfo::name, "Material name.")
        .def_property_readonly(
            "diffuse_color",
            [](const FBXMaterialInfo& self) {
                return py::make_tuple(
                    self.diffuseColor[0], self.diffuseColor[1],
                    self.diffuseColor[2], self.diffuseColor[3]);
            },
            "Diffuse color as RGBA.")
        .def_readonly("diffuse_texture_path",
                      &FBXMaterialInfo::diffuseTexturePath,
                      "Resolved diffuse texture path.")
        .def_readonly("normal_texture_path",
                      &FBXMaterialInfo::normalTexturePath,
                      "Resolved normal texture path.")
        .def_readonly("has_diffuse_texture",
                      &FBXMaterialInfo::hasDiffuseTexture,
                      "Whether a diffuse texture is referenced.")
        .def_readonly("has_embedded_diffuse_texture",
                      &FBXMaterialInfo::hasEmbeddedDiffuseTexture,
                      "Whether the diffuse texture was embedded in the FBX.")
        .def_readonly("has_normal_texture", &FBXMaterialInfo::hasNormalTexture,
                      "Whether a normal texture is referenced.")
        .def_readonly("has_embedded_normal_texture",
                      &FBXMaterialInfo::hasEmbeddedNormalTexture,
                      "Whether the normal texture was embedded in the FBX.");

    py::class_<FBXMeshMetadata, std::shared_ptr<FBXMeshMetadata>>(
        asset, "FBXMeshMetadata",
        "Summary metadata for a mesh imported from FBX.")
        .def_readonly("name", &FBXMeshMetadata::name, "Mesh name.")
        .def_readonly("vertex_count", &FBXMeshMetadata::vertexCount,
                      "Number of vertices.")
        .def_readonly("index_count", &FBXMeshMetadata::indexCount,
                      "Number of indices.")
        .def_readonly("material_count", &FBXMeshMetadata::materialCount,
                      "Number of assigned materials.")
        .def_readonly("primary_material_index",
                      &FBXMeshMetadata::primaryMaterialIndex,
                      "Primary material index.")
        .def_readonly("has_normals", &FBXMeshMetadata::hasNormals,
                      "Whether normals are present.")
        .def_readonly("has_uvs", &FBXMeshMetadata::hasUVs,
                      "Whether UV coordinates are present.")
        .def_readonly("has_skin", &FBXMeshMetadata::hasSkin,
                      "Whether skinning data is present.")
        .def_readonly("skin_cluster_names", &FBXMeshMetadata::skinClusterNames,
                      "Names of skin clusters affecting this mesh.")
        .def_readonly("materials", &FBXMeshMetadata::materials,
                      "Material metadata assigned to this mesh.");

    py::class_<FBXStaticMeshInfo, std::shared_ptr<FBXStaticMeshInfo>>(
        asset, "FBXMeshInfo", "Static mesh payload imported from an FBX file.")
        .def_property_readonly(
            "metadata",
            [](const FBXStaticMeshInfo& self) { return self.metadata; })
        .def_property_readonly(
            "name",
            [](const FBXStaticMeshInfo& self) { return self.metadata.name; })
        .def_property_readonly("vertex_count",
                               [](const FBXStaticMeshInfo& self) {
                                   return self.metadata.vertexCount;
                               })
        .def_property_readonly("index_count",
                               [](const FBXStaticMeshInfo& self) {
                                   return self.metadata.indexCount;
                               })
        .def_property_readonly("material_count",
                               [](const FBXStaticMeshInfo& self) {
                                   return self.metadata.materialCount;
                               })
        .def_property_readonly("primary_material_index",
                               [](const FBXStaticMeshInfo& self) {
                                   return self.metadata.primaryMaterialIndex;
                               })
        .def_property_readonly("has_normals",
                               [](const FBXStaticMeshInfo& self) {
                                   return self.metadata.hasNormals;
                               })
        .def_property_readonly(
            "has_uvs",
            [](const FBXStaticMeshInfo& self) { return self.metadata.hasUVs; })
        .def_property_readonly(
            "has_skin",
            [](const FBXStaticMeshInfo& self) { return self.metadata.hasSkin; })
        .def_property_readonly("skin_cluster_names",
                               [](const FBXStaticMeshInfo& self) {
                                   return self.metadata.skinClusterNames;
                               })
        .def_property_readonly("materials",
                               [](const FBXStaticMeshInfo& self) {
                                   return self.metadata.materials;
                               })
        .def_property_readonly("mesh_data",
                               [](std::shared_ptr<FBXStaticMeshInfo> self) {
                                   return std::shared_ptr<KE::Scene::MeshData>(
                                       self, &self->meshData);
                               });

    py::class_<FBXDebug::SkinClusterInfo>(asset, "FBXSkinClusterInfo")
        .def_readonly("mesh_name", &FBXDebug::SkinClusterInfo::meshName)
        .def_readonly("cluster_name", &FBXDebug::SkinClusterInfo::clusterName)
        .def_readonly("link_name", &FBXDebug::SkinClusterInfo::linkName)
        .def_readonly("weight_count", &FBXDebug::SkinClusterInfo::weightCount)
        .def_readonly("index_count", &FBXDebug::SkinClusterInfo::indexCount)
        .def_readonly("min_index", &FBXDebug::SkinClusterInfo::minIndex)
        .def_readonly("max_index", &FBXDebug::SkinClusterInfo::maxIndex)
        .def_readonly("min_weight", &FBXDebug::SkinClusterInfo::minWeight)
        .def_readonly("max_weight", &FBXDebug::SkinClusterInfo::maxWeight)
        .def_readonly("weight_sum", &FBXDebug::SkinClusterInfo::weightSum)
        .def_readonly("transform_translation",
                      &FBXDebug::SkinClusterInfo::transformTranslation)
        .def_readonly("transform_link_translation",
                      &FBXDebug::SkinClusterInfo::transformLinkTranslation);

    py::class_<FBXSkinnedMeshInfo, std::shared_ptr<FBXSkinnedMeshInfo>>(
        asset, "FBXSkinnedMeshInfo",
        "Skinned mesh payload imported from an FBX file.")
        .def_property_readonly(
            "metadata",
            [](const FBXSkinnedMeshInfo& self) { return self.metadata; })
        .def_property_readonly(
            "name",
            [](const FBXSkinnedMeshInfo& self) { return self.metadata.name; })
        .def_property_readonly("vertex_count",
                               [](const FBXSkinnedMeshInfo& self) {
                                   return self.metadata.vertexCount;
                               })
        .def_property_readonly("index_count",
                               [](const FBXSkinnedMeshInfo& self) {
                                   return self.metadata.indexCount;
                               })
        .def_property_readonly("material_count",
                               [](const FBXSkinnedMeshInfo& self) {
                                   return self.metadata.materialCount;
                               })
        .def_property_readonly("primary_material_index",
                               [](const FBXSkinnedMeshInfo& self) {
                                   return self.metadata.primaryMaterialIndex;
                               })
        .def_property_readonly("materials",
                               [](const FBXSkinnedMeshInfo& self) {
                                   return self.metadata.materials;
                               })
        .def_property_readonly("has_normals",
                               [](const FBXSkinnedMeshInfo& self) {
                                   return self.metadata.hasNormals;
                               })
        .def_property_readonly(
            "has_uvs",
            [](const FBXSkinnedMeshInfo& self) { return self.metadata.hasUVs; })
        .def_property_readonly("has_skin",
                               [](const FBXSkinnedMeshInfo& self) {
                                   return self.metadata.hasSkin;
                               })
        .def_property_readonly("skin_cluster_names",
                               [](const FBXSkinnedMeshInfo& self) {
                                   return self.metadata.skinClusterNames;
                               })
        .def_property_readonly("mesh_data",
                               [](std::shared_ptr<FBXSkinnedMeshInfo> self) {
                                   return std::shared_ptr<KE::Scene::MeshData>(
                                       self, &self->skinnedMeshData.mesh);
                               })
        .def_property_readonly(
            "skinned_mesh_data",
            [](std::shared_ptr<FBXSkinnedMeshInfo> self) {
                return std::shared_ptr<KE::Scene::SkinnedMeshData>(
                    self, &self->skinnedMeshData);
            })
        .def_property_readonly("vertices",
                               [](const FBXSkinnedMeshInfo& self) {
                                   return floatArrayFromVec3Vector(
                                       self.skinnedMeshData.mesh.vertices);
                               })
        .def_property_readonly("normals",
                               [](const FBXSkinnedMeshInfo& self) {
                                   return floatArrayFromVec3Vector(
                                       self.skinnedMeshData.mesh.normals);
                               })
        .def_property_readonly("bone_indices",
                               [](const FBXSkinnedMeshInfo& self) {
                                   return intArrayFromVec4Vector(
                                       self.skinnedMeshData.boneIndices);
                               })
        .def_property_readonly("bone_weights",
                               [](const FBXSkinnedMeshInfo& self) {
                                   return floatArrayFromVec4Vector(
                                       self.skinnedMeshData.boneWeights);
                               })
        .def_readonly("bone_names", &FBXSkinnedMeshInfo::boneNames)
        .def_property_readonly("bone_node_indices",
                               [](const FBXSkinnedMeshInfo& self) {
                                   return self.skinnedMeshData.boneNodeIndices;
                               })
        .def_property_readonly("bind_bone_matrices",
                               [](const FBXSkinnedMeshInfo& self) {
                                   return floatArrayFromMat4Vector(
                                       self.bindBoneMatrices);
                               })
        .def_property_readonly(
            "inverse_bind_matrices",
            [](const FBXSkinnedMeshInfo& self) {
                return floatArrayFromMat4Vector(
                    self.skinnedMeshData.inverseBindMatrices);
            })
        .def_property_readonly("bind_mesh_matrices",
                               [](const FBXSkinnedMeshInfo& self) {
                                   return floatArrayFromMat4Vector(
                                       self.bindMeshMatrices);
                               })
        .def_readonly("overweight_vertex_count",
                      &FBXSkinnedMeshInfo::overweightVertexCount)
        .def_readonly("unweighted_vertex_count",
                      &FBXSkinnedMeshInfo::unweightedVertexCount)
        .def_readonly("mismatched_cluster_count",
                      &FBXSkinnedMeshInfo::mismatchedClusterCount);

    py::class_<FBXCharacterData>(
        asset, "FBXCharacterData",
        "FBX character import result with motion and skinned meshes.")
        .def_readonly("motion", &FBXCharacterData::motion,
                      "Imported skeleton motion.")
        .def_property_readonly(
            "skinned_meshes",
            [](const FBXCharacterData& self) {
                py::list result;
                for (const auto& mesh : self.skinnedMeshes) {
                    result.append(std::make_shared<FBXSkinnedMeshInfo>(mesh));
                }
                return result;
            },
            "Imported skinned meshes.");

    py::class_<FBXImportResult>(asset, "FBXImportResult",
                                "Parsed FBX character, clips, and diagnostics.")
        .def_readonly("character", &FBXImportResult::character,
                      "Imported FBX character data.")
        .def_property_readonly(
            "motion",
            [](const FBXImportResult& self) { return self.character.motion; },
            "Imported skeleton motion.")
        .def_property_readonly(
            "skinned_meshes",
            [](const FBXImportResult& self) {
                py::list result;
                for (const auto& mesh : self.character.skinnedMeshes) {
                    result.append(std::make_shared<FBXSkinnedMeshInfo>(mesh));
                }
                return result;
            },
            "Imported skinned meshes.")
        .def_readonly("clips", &FBXImportResult::clips,
                      "Animation clips discovered in the FBX.")
        .def_readonly("diagnostics", &FBXImportResult::diagnostics,
                      "Importer diagnostics.");

    py::class_<FBXLoader>(asset, "FBXLoader",
                          "Loader for FBX skeletons, motions, and meshes.")
        .def_static("load_skeleton", &FBXLoader::loadSkeleton,
                    py::arg("fbx_path"), py::arg("scale") = 0.01f,
                    "Load only the skeleton hierarchy from an FBX file.")
        .def_static("load_animation_clip_infos",
                    &FBXLoader::loadAnimationClipInfos, py::arg("fbx_path"),
                    "List animation clips available in an FBX file.")
        .def_static("load_motion", &FBXLoader::loadMotion, py::arg("fbx_path"),
                    py::arg("clip_index") = -1, py::arg("fps") = -1.0f,
                    py::arg("scale") = 0.01f,
                    "Load skeleton motion from an FBX file.")
        .def_static(
            "load_meshes",
            [](const std::string& fbxPath, float scale) {
                py::list result;
                for (auto& mesh : FBXLoader::loadMeshes(fbxPath, scale)) {
                    result.append(
                        std::make_shared<FBXStaticMeshInfo>(std::move(mesh)));
                }
                return result;
            },
            py::arg("fbx_path"), py::arg("scale") = 0.01f,
            "Load static meshes from an FBX file.")
        .def_static(
            "parse", &FBXLoader::parse, py::arg("fbx_path"),
            py::arg("clip_index") = -1, py::arg("fps") = -1.0f,
            py::arg("scale") = 0.01f,
            "Parse an FBX file and return character data plus diagnostics.")
        .def_static("parse_with_bind", &FBXLoader::parseWithBind,
                    py::arg("motion_fbx_path"), py::arg("bind_fbx_path"),
                    py::arg("clip_index") = -1, py::arg("fps") = -1.0f,
                    py::arg("scale") = 0.01f,
                    "Parse FBX motion with a separate bind-pose FBX file.")
        .def_static("load_character", &FBXLoader::loadCharacter,
                    py::arg("fbx_path"), py::arg("clip_index") = -1,
                    py::arg("fps") = -1.0f, py::arg("scale") = 0.01f,
                    "Load FBX character data without diagnostics.")
        .def_static("load_character_with_bind",
                    &FBXLoader::loadCharacterWithBind,
                    py::arg("motion_fbx_path"), py::arg("bind_fbx_path"),
                    py::arg("clip_index") = -1, py::arg("fps") = -1.0f,
                    py::arg("scale") = 0.01f,
                    "Load FBX character data using a separate bind-pose file.")
        .def_static(
            "load_skinned_meshes",
            [](const std::string& fbxPath, float scale) {
                py::list result;
                for (auto& mesh :
                     FBXLoader::loadSkinnedMeshes(fbxPath, scale)) {
                    result.append(
                        std::make_shared<FBXSkinnedMeshInfo>(std::move(mesh)));
                }
                return result;
            },
            py::arg("fbx_path"), py::arg("scale") = 0.01f,
            "Load skinned meshes from an FBX file.");

    py::module_ fbxDebug = asset.def_submodule(
        "FBXDebug", "FBX debug and importer inspection helpers.");
    fbxDebug.def("load_skin_cluster_infos", &FBXDebug::loadSkinClusterInfos,
                 py::arg("fbx_path"), py::arg("scale") = 0.01f,
                 "Load FBX skin cluster diagnostics.");

    py::class_<USDMeshInfo, std::shared_ptr<USDMeshInfo>>(
        asset, "USDMeshInfo", "Mesh payload imported from a USD scene.")
        .def_readonly("prim_path", &USDMeshInfo::primPath, "USD prim path.")
        .def_readonly("name", &USDMeshInfo::name, "Mesh name.")
        .def_readonly("material_path", &USDMeshInfo::materialPath,
                      "USD material path if found.")
        .def_readonly("diffuse_texture_path", &USDMeshInfo::diffuseTexturePath,
                      "Resolved diffuse texture path.")
        .def_readonly("normal_texture_path", &USDMeshInfo::normalTexturePath,
                      "Resolved normal texture path.")
        .def_property_readonly("mesh_data",
                               [](std::shared_ptr<USDMeshInfo> self) {
                                   return std::shared_ptr<KE::Scene::MeshData>(
                                       self, &self->meshData);
                               })
        .def_property_readonly("vertex_count",
                               [](const USDMeshInfo& self) {
                                   return self.meshData.vertices.size();
                               })
        .def_property_readonly("index_count", [](const USDMeshInfo& self) {
            return self.meshData.indices.size();
        });

    py::class_<USDImportResult>(asset, "USDImportResult",
                                "Parsed USD meshes plus diagnostics.")
        .def_property_readonly(
            "meshes",
            [](const USDImportResult& self) {
                py::list result;
                for (const auto& mesh : self.meshes) {
                    result.append(std::make_shared<USDMeshInfo>(mesh));
                }
                return result;
            },
            "Meshes imported from the USD scene.")
        .def_readonly("diagnostics", &USDImportResult::diagnostics,
                      "Importer diagnostics.");

    py::class_<USDLoader>(asset, "USDLoader", "Loader for USD mesh scenes.")
        .def_static("parse", &USDLoader::parse, py::arg("usd_path"),
                    py::arg("scale") = 1.0f,
                    "Parse USD and return meshes plus diagnostics.")
        .def_static(
            "load_meshes",
            [](const std::string& usdPath, float scale) {
                py::list result;
                for (auto& mesh : USDLoader::loadMeshes(usdPath, scale)) {
                    result.append(
                        std::make_shared<USDMeshInfo>(std::move(mesh)));
                }
                return result;
            },
            py::arg("usd_path"), py::arg("scale") = 1.0f,
            "Load mesh payloads from a USD file.");
}
