///
/// Animation System Python Bindings
/// Skeleton animation bindings plus asset importer bindings.
///

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include "py_array_view.hpp"

#include "animation/character_description.hpp"
#include "asset/bvh_loader.hpp"
#include "asset/fbx_loader.hpp"
#include "asset/mesh_loader.hpp"
#include "asset/mjcf_loader.hpp"
#include "asset/usd_loader.hpp"
#include "animation/skeleton_math.hpp"
#include "animation/skeleton_motion.hpp"
#include "animation/skeleton_state.hpp"
#include "animation/skeleton_tree.hpp"
#include "animation/skinning.hpp"
#include "bridge/skeleton_bridge.hpp"
#include "bridge/skeleton_visual_bridge.hpp"
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

void bind_animation(py::module& m) {
    py::module anim = m.def_submodule(
        "animation", "Skeleton animation, skinning, and visualization APIs.");
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
        .def_readonly("normal_texture_path",
                      &ObjMaterialInfo::normalTexturePath)
        .def_readonly("has_diffuse_texture",
                      &ObjMaterialInfo::hasDiffuseTexture)
        .def_readonly("has_specular_texture",
                      &ObjMaterialInfo::hasSpecularTexture)
        .def_readonly("has_normal_texture",
                      &ObjMaterialInfo::hasNormalTexture);

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
        .def_property_readonly("material_count",
                               [](const ObjMeshInfo& self) {
                                   return self.materials.size();
                               })
        .def_property_readonly("subset_count",
                               [](const ObjMeshInfo& self) {
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
        [](const std::string& path) { return KE::Asset::loadObjWithMaterials(path); },
        py::arg("path"),
        "Load an OBJ file and return mesh data plus MTL material metadata.");

    asset.def(
        "load_stl",
        [](const std::string& path) {
            return std::make_shared<KE::Scene::MeshData>(
                KE::Asset::loadStl(path));
        },
        py::arg("path"), "Load an STL file and return scene.MeshData.");

    anim.def(
        "cpu_skin",
        [](const FloatArray& bindPositions, const FloatArray& bindNormals,
           py::array_t<int> boneIndices, const FloatArray& boneWeights,
           py::array_t<int> boneNodeIndices, const FloatArray& inverseBinds,
           const FloatArray& currentGlobalMatrices) {
            std::vector<Eigen::Vector3f> positions =
                eigenVec3ArrayFromPy(bindPositions, "bind_positions");
            std::vector<Eigen::Vector3f> normals =
                eigenVec3ArrayFromPy(bindNormals, "bind_normals");
            std::vector<std::array<int, 4>> indices =
                intVec4ArrayFromPy(boneIndices, "bone_indices");
            std::vector<std::array<float, 4>> weights =
                floatVec4ArrayFromPy(boneWeights, "bone_weights");
            std::vector<int> nodeIndices =
                intVectorFromPy(boneNodeIndices, "bone_node_indices");
            std::vector<Eigen::Matrix4f> inv =
                eigenMat4ArrayFromPy(inverseBinds, "inverse_bind_matrices");
            std::vector<Eigen::Matrix4f> globals = eigenMat4ArrayFromPy(
                currentGlobalMatrices, "current_global_matrices");

            CPUSkinningInput input;
            input.bindPositions = &positions;
            input.bindNormals = &normals;
            input.boneIndices = &indices;
            input.boneWeights = &weights;
            input.boneNodeIndices = &nodeIndices;
            input.inverseBindMatrices = &inv;
            input.currentGlobalMatrices = &globals;
            CPUSkinningResult result = Skinning::cpuSkin(input);

            py::dict out;
            out["positions"] = floatArrayFromVec3Vector(result.positions);
            out["normals"] = floatArrayFromVec3Vector(result.normals);
            return out;
        },
        py::arg("bind_positions"), py::arg("bind_normals"),
        py::arg("bone_indices"), py::arg("bone_weights"),
        py::arg("bone_node_indices"), py::arg("inverse_bind_matrices"),
        py::arg("current_global_matrices"),
        "CPU-skin bind-space positions and normals with current bone "
        "matrices.");

    anim.def(
        "compute_skinning_matrices",
        [](py::array_t<int> boneNodeIndices, const FloatArray& inverseBinds,
           const FloatArray& currentGlobalMatrices) {
            std::vector<int> nodeIndices =
                intVectorFromPy(boneNodeIndices, "bone_node_indices");
            std::vector<Eigen::Matrix4f> inv =
                eigenMat4ArrayFromPy(inverseBinds, "inverse_bind_matrices");
            std::vector<Eigen::Matrix4f> globals = eigenMat4ArrayFromPy(
                currentGlobalMatrices, "current_global_matrices");
            return floatArrayFromMat4Vector(
                Skinning::computeSkinningMatrices(nodeIndices, inv, globals));
        },
        py::arg("bone_node_indices"), py::arg("inverse_bind_matrices"),
        py::arg("current_global_matrices"),
        "Compute skinning matrices from skeleton globals and inverse binds.");

    anim.def(
        "compute_skinning_matrices_into",
        [](py::array_t<int> boneNodeIndices, const FloatArray& inverseBinds,
           const FloatArray& currentGlobalMatrices, const FloatArray& output) {
            std::vector<int> nodeIndices =
                intVectorFromPy(boneNodeIndices, "bone_node_indices");
            std::vector<Eigen::Matrix4f> inv =
                eigenMat4ArrayFromPy(inverseBinds, "inverse_bind_matrices");
            std::vector<Eigen::Matrix4f> globals = eigenMat4ArrayFromPy(
                currentGlobalMatrices, "current_global_matrices");
            std::vector<Eigen::Matrix4f> matrices;
            Skinning::computeSkinningMatricesInto(nodeIndices, inv, globals,
                                                  matrices);
            writeMat4VectorToPy(matrices, output, "output");
            return output;
        },
        py::arg("bone_node_indices"), py::arg("inverse_bind_matrices"),
        py::arg("current_global_matrices"), py::arg("output"),
        "Compute skinning matrices into an existing output array.");

    // Joint (from character_description.hpp)
    py::class_<Joint>(anim, "Joint",
                      "Joint metadata imported from robot/character assets.")
        .def_readonly("name", &Joint::name, "Joint name.")
        .def_readonly("lo_limit", &Joint::loLimit, "Lower joint limit.")
        .def_readonly("hi_limit", &Joint::hiLimit, "Upper joint limit.")
        .def_property_readonly(
            "axis", [](const Joint& j) { return toGlm(j.axis); },
            "Joint axis.");

    py::enum_<Site::Type>(anim, "SiteType", "MJCF site geometry type.")
        .value("Sphere", Site::Type::Sphere)
        .value("Capsule", Site::Type::Capsule)
        .value("Box", Site::Type::Box)
        .export_values();

    // Site
    py::class_<Site>(anim, "Site",
                     "Imported MJCF site attached to a character body.")
        .def_readonly("type", &Site::type, "Site geometry type.")
        .def_readonly("name", &Site::name, "Site name.")
        .def_readonly("body_index", &Site::bodyIndex,
                      "Index of the body this site belongs to.")
        .def_property_readonly(
            "pos", [](const Site& s) { return toGlm(s.pos); },
            "Local site position.")
        .def_property_readonly(
            "quat", [](const Site& s) { return toGlm(s.quat); },
            "Local site orientation.")
        .def_property_readonly(
            "size", [](const Site& s) { return toGlm(s.size); },
            "Site size parameters.")
        .def_property_readonly(
            "rgba",
            [](const Site& s) {
                return glm::vec4(s.rgba.x(), s.rgba.y(), s.rgba.z(),
                                 s.rgba.w());
            },
            "Site display color.")
        .def_readonly("has_zaxis", &Site::hasZAxis,
                      "Whether this site has an explicit z-axis.")
        .def_property_readonly(
            "zaxis", [](const Site& s) { return toGlm(s.zaxis); },
            "Explicit site z-axis if present.");

    // MeshInfo
    py::class_<MeshInfo>(anim, "MeshInfo",
                         "Visual mesh reference imported from an MJCF asset.")
        .def_readonly("body_name", &MeshInfo::bodyName, "Owning body name.")
        .def_readonly("mesh_file", &MeshInfo::meshFile, "Mesh file path.")
        .def_readonly("body_index", &MeshInfo::bodyIndex, "Owning body index.")
        .def_property_readonly(
            "pos", [](const MeshInfo& m) { return toGlm(m.pos); },
            "Local mesh position.")
        .def_property_readonly(
            "quat",
            [](const MeshInfo& m) {
                return glm::quat(m.quat.w(), m.quat.x(), m.quat.y(),
                                 m.quat.z());
            },
            "Local mesh orientation.")
        .def_property_readonly(
            "rgba",
            [](const MeshInfo& m) {
                return glm::vec4(m.rgba.x(), m.rgba.y(), m.rgba.z(),
                                 m.rgba.w());
            },
            "Mesh display color.");

    py::enum_<CollisionGeom::Type>(
        anim, "CollisionGeomType",
        "Collision geometry type imported from character assets.")
        .value("Capsule", CollisionGeom::Type::Capsule)
        .value("Cylinder", CollisionGeom::Type::Cylinder)
        .value("Sphere", CollisionGeom::Type::Sphere)
        .value("Box", CollisionGeom::Type::Box)
        .export_values();

    py::class_<CollisionGeom>(anim, "CollisionGeom",
                              "Collision geometry attached to a body.")
        .def_readonly("type", &CollisionGeom::type, "Collision geometry type.")
        .def_property_readonly(
            "pos", [](const CollisionGeom& g) { return toGlm(g.pos); },
            "Local collision position.")
        .def_property_readonly(
            "quat", [](const CollisionGeom& g) { return toGlm(g.quat); },
            "Local collision orientation.")
        .def_property_readonly(
            "size",
            [](const CollisionGeom& g) {
                return std::vector<float>{g.size[0], g.size[1], g.size[2]};
            },
            "Collision size parameters.")
        .def_readonly("has_from_to", &CollisionGeom::hasFromTo,
                      "Whether capsule-style from/to endpoints are present.")
        .def_property_readonly(
            "from_pos", [](const CollisionGeom& g) { return toGlm(g.from); },
            "Collision endpoint start position.")
        .def_property_readonly(
            "to_pos", [](const CollisionGeom& g) { return toGlm(g.to); },
            "Collision endpoint end position.")
        .def_readonly("friction", &CollisionGeom::friction,
                      "Imported friction value.")
        .def_readonly("condim", &CollisionGeom::condim,
                      "Imported contact dimensionality.")
        .def_readonly("margin", &CollisionGeom::margin,
                      "Imported collision margin.");

    // CharacterData — aggregate returned by asset importers.
    py::class_<CharacterData>(anim, "CharacterData",
                              "Imported character description with skeleton, "
                              "meshes, joints, and sites.")
        .def_readonly("skeleton_tree", &CharacterData::skeletonTree,
                      "Imported skeleton hierarchy.")
        .def_readonly("mesh_infos", &CharacterData::meshInfos,
                      "Visual mesh references.")
        .def_readonly("mesh_dir", &CharacterData::meshDir,
                      "Directory used to resolve mesh files.")
        .def_readonly("sites", &CharacterData::sites, "Imported site markers.")
        // joints: return dict[int, list[Joint]]
        .def_property_readonly(
            "joints",
            [](const CharacterData& d) {
                py::dict result;
                for (const auto& [idx, jvec] : d.joints)
                    result[py::int_(idx)] = jvec;
                return result;
            },
            "Joint metadata keyed by body index.")
        .def_property_readonly(
            "collision_geoms",
            [](const CharacterData& d) {
                py::dict result;
                for (const auto& [idx, geoms] : d.collisionGeoms)
                    result[py::int_(idx)] = geoms;
                return result;
            },
            "Collision geometry keyed by body index.");

    py::class_<ImportDiagnostics>(
        asset, "ImportDiagnostics",
        "Warnings collected while importing an asset.")
        .def_readonly("warnings", &ImportDiagnostics::warnings,
                      "Human-readable importer warnings.");

    py::class_<MJCFImportResult>(asset, "MJCFImportResult",
                                 "Parsed MJCF character plus diagnostics.")
        .def_readonly("character", &MJCFImportResult::character,
                      "Imported character data.")
        .def_readonly("diagnostics", &MJCFImportResult::diagnostics,
                      "Importer diagnostics.");

    py::class_<MJCFLoader>(asset, "MJCFLoader",
                           "Loader for MJCF/XML robot character assets.")
        .def_static("parse", &MJCFLoader::parse, py::arg("mjcf_path"),
                    py::arg("scale") = 1.0f, py::arg("order") = "DFS",
                    "Parse MJCF and return character data with diagnostics.")
        .def_static("load", &MJCFLoader::load, py::arg("mjcf_path"),
                    py::arg("scale") = 1.0f, py::arg("order") = "DFS",
                    "Load MJCF and return character data.");

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

    // SkeletonTree (read-only after construction)
    py::class_<SkeletonTree, std::shared_ptr<SkeletonTree>>(
        anim, "SkeletonTree",
        "Read-only skeleton hierarchy with local bind transforms.")
        .def("num_joints", &SkeletonTree::numJoints,
             "Return the number of joints/nodes.")
        .def("node_name", &SkeletonTree::nodeName, py::arg("index"),
             "Return a joint name by index.")
        .def("parent_index", &SkeletonTree::parentIndex, py::arg("index"),
             "Return a joint's parent index, or -1 for the root.")
        .def(
            "local_translation",
            [](const SkeletonTree& self, int i) {
                return toGlm(self.localTranslation(i));
            },
            py::arg("index"), "Return local bind translation for a joint.")
        .def(
            "local_rotation",
            [](const SkeletonTree& self, int i) {
                return toGlm(self.localRotation(i));
            },
            py::arg("index"), "Return local bind rotation for a joint.")
        .def("node_names", &SkeletonTree::nodeNames, "Return all joint names.")
        .def("parent_indices", &SkeletonTree::parentIndices,
             "Return parent index for each joint.")
        .def("print", &SkeletonTree::print,
             "Print the skeleton hierarchy for debugging.");

    py::class_<SkeletonMotion>(
        anim, "SkeletonMotion",
        "Sampled skeleton animation clip with root motion and local rotations.")
        .def("num_frames", &SkeletonMotion::numFrames,
             "Return the number of sampled frames.")
        .def("num_joints", &SkeletonMotion::numJoints,
             "Return the number of joints.")
        .def("fps", &SkeletonMotion::fps, "Return the clip frame rate in Hz.")
        .def("duration", &SkeletonMotion::duration,
             "Return the clip duration in seconds.")
        .def("motion_name", &SkeletonMotion::motionName,
             "Return the motion/clip name.")
        .def(
            "node_names",
            [](const SkeletonMotion& self) {
                return self.skeletonTree().nodeNames();
            },
            "Return skeleton joint names.")
        .def(
            "parent_indices",
            [](const SkeletonMotion& self) {
                return self.skeletonTree().parentIndices();
            },
            "Return skeleton parent indices.")
        .def("frame", &SkeletonMotion::frame, py::arg("frame_index"),
             "Return a SkeletonState for a frame.")
        .def("sample", &SkeletonMotion::sample, py::arg("time"),
             py::arg("loop") = true,
             "Sample a SkeletonState at time in seconds.")
        .def(
            "root_translation",
            [](const SkeletonMotion& self, int frame) {
                return toGlm(self.rootTranslation(frame));
            },
            py::arg("frame"), "Return root translation for a frame.")
        .def(
            "local_rotation",
            [](const SkeletonMotion& self, int frame, int joint) {
                return toGlm(self.localRotation(frame, joint));
            },
            py::arg("frame"), py::arg("joint"),
            "Return local joint rotation for a frame.")
        .def("root_translations_flat", &SkeletonMotion::rootTranslationsFlat,
             "Return root translations as a flat float array.")
        .def("local_rotations_wxyz_flat",
             &SkeletonMotion::localRotationsWxyzFlat,
             "Return local rotations as a flat WXYZ float array.");

    // Transform (FK result)
    py::class_<Transform>(anim, "Transform",
                          "Forward-kinematics transform result.")
        .def_property_readonly(
            "rotation", [](const Transform& t) { return toGlm(t.rotation); },
            "World or local rotation.")
        .def_property_readonly(
            "translation",
            [](const Transform& t) { return toGlm(t.translation); },
            "World or local translation.");

    // SkeletonState
    py::class_<SkeletonState>(anim, "SkeletonState",
                              "Pose state for a SkeletonTree, stored as root "
                              "translation and rotations.")
        .def_static(
            "from_rotation_and_root_translation",
            [](std::shared_ptr<SkeletonTree> tree, const FloatArray& rotations,
               const FloatArray& rootTranslation, bool isLocal) {
                auto rot = eigenQuatXyzwArray(rotations, "rotations");
                if (static_cast<int>(rot.size()) != tree->numJoints()) {
                    throw py::value_error(
                        "rotations must have shape [num_joints, 4]");
                }
                return SkeletonState::fromRotationAndRootTranslation(
                    tree, rot,
                    eigenVec3FromArray(rootTranslation, "root_translation"),
                    isLocal);
            },
            py::arg("tree"), py::arg("rotations"), py::arg("root_translation"),
            py::arg("is_local") = true,
            "Create a pose from rotations and root translation.")
        .def("num_joints", &SkeletonState::numJoints,
             "Return the number of joints.")
        .def("is_local", &SkeletonState::isLocal,
             "Return true if rotations are stored in local space.")
        .def("compute_global_transforms",
             &SkeletonState::computeGlobalTransforms,
             "Compute global FK transforms for all joints.")
        .def(
            "compute_global_matrices",
            [](const SkeletonState& self) {
                const auto transforms = self.computeGlobalTransforms();
                py::array_t<float> array(
                    {static_cast<py::ssize_t>(transforms.size()),
                     py::ssize_t(4), py::ssize_t(4)});
                auto view = array.mutable_unchecked<3>();
                for (py::ssize_t i = 0;
                     i < static_cast<py::ssize_t>(transforms.size()); ++i) {
                    const auto& t = transforms[static_cast<size_t>(i)];
                    const Eigen::Matrix3f r =
                        t.rotation.normalized().toRotationMatrix();
                    for (py::ssize_t row = 0; row < 4; ++row) {
                        for (py::ssize_t col = 0; col < 4; ++col)
                            view(i, row, col) = (row == col) ? 1.0f : 0.0f;
                    }
                    for (py::ssize_t row = 0; row < 3; ++row) {
                        for (py::ssize_t col = 0; col < 3; ++col) {
                            view(i, row, col) =
                                r(static_cast<int>(row), static_cast<int>(col));
                        }
                    }
                    view(i, 0, 3) = t.translation.x();
                    view(i, 1, 3) = t.translation.y();
                    view(i, 2, 3) = t.translation.z();
                }
                return array;
            },
            "Compute global FK matrices for all joints.")
        .def(
            "compute_global_positions",
            [](const SkeletonState& self) {
                auto positions = self.computeGlobalPositions();
                std::vector<glm::vec3> result;
                result.reserve(positions.size());
                for (const auto& p : positions) {
                    result.push_back(toGlm(p));
                }
                return result;
            },
            "Compute global joint positions.")
        .def(
            "rotation",
            [](const SkeletonState& self, int i) {
                return toGlm(self.rotation(i));
            },
            py::arg("index"), "Return a joint rotation.")
        .def(
            "set_rotation",
            [](SkeletonState& self, int i, const FloatArray& q) {
                self.setRotation(i, eigenQuatXyzwFromArray(q, "rotation"));
            },
            py::arg("index"), py::arg("rotation"),
            "Set a joint rotation from an XYZW array.")
        .def(
            "set_rotation",
            [](SkeletonState& self, int i, const glm::quat& q) {
                self.setRotation(i, fromGlm(q));
            },
            py::arg("index"), py::arg("rotation"),
            "Set a joint rotation from a quaternion.")
        .def(
            "root_translation",
            [](const SkeletonState& self) {
                return toGlm(self.rootTranslation());
            },
            "Return the root translation.")
        .def(
            "set_root_translation",
            [](SkeletonState& self, const FloatArray& t) {
                self.setRootTranslation(
                    eigenVec3FromArray(t, "root_translation"));
            },
            py::arg("translation"), "Set the root translation from an array.")
        .def(
            "set_root_translation",
            [](SkeletonState& self, const glm::vec3& t) {
                self.setRootTranslation(fromGlm(t));
            },
            py::arg("translation"), "Set the root translation from a vec3.")
        .def("print_global_positions", &SkeletonState::printGlobalPositions,
             "Print global joint positions for debugging.");

    // SkeletonBridge
    py::class_<SkeletonBridge>(
        anim, "SkeletonBridge",
        "Bridge that maps a skeleton pose onto scene prim transforms.")
        .def_static(
            "from_mjcf",
            [](const std::string& mjcfPath, Scene::SceneBackend* scene,
               const std::string& primBasePath, float scale,
               const std::string& order, const std::string& meshAssetBasePath) {
                return SkeletonBridge::fromMJCF(mjcfPath, scene, primBasePath,
                                                scale, order,
                                                meshAssetBasePath);
            },
            py::arg("mjcf_path"), py::arg("scene"),
            py::arg("prim_base_path") = "/robot", py::arg("scale") = 1.0f,
            py::arg("order") = "DFS", py::arg("mesh_asset_base_path") = "",
            "Create a skeleton bridge and scene prims from an MJCF file.")
        .def("apply_pose", &SkeletonBridge::applyPose,
             "Apply the bridge's current SkeletonState to scene prims.")
        .def(
            "set_joint_rotation",
            [](SkeletonBridge& self, int idx, const FloatArray& q) {
                self.setJointRotation(idx,
                                      eigenQuatXyzwFromArray(q, "rotation"));
            },
            py::arg("index"), py::arg("rotation"),
            "Set a joint rotation from an XYZW array.")
        .def(
            "set_joint_rotation",
            [](SkeletonBridge& self, int idx, const glm::quat& q) {
                self.setJointRotation(idx, fromGlm(q));
            },
            py::arg("index"), py::arg("rotation"),
            "Set a joint rotation from a quaternion.")
        .def(
            "set_root_translation",
            [](SkeletonBridge& self, const FloatArray& t) {
                self.setRootTranslation(
                    eigenVec3FromArray(t, "root_translation"));
            },
            py::arg("translation"), "Set the root translation from an array.")
        .def(
            "set_root_translation",
            [](SkeletonBridge& self, const glm::vec3& t) {
                self.setRootTranslation(fromGlm(t));
            },
            py::arg("translation"), "Set the root translation from a vec3.")
        .def("reset_to_zero_pose", &SkeletonBridge::resetToZeroPose,
             "Reset all joint rotations and root translation to zero pose.")
        .def(
            "skeleton",
            [](SkeletonBridge& self) -> const SkeletonTree& {
                return self.fk().skeleton();
            },
            py::return_value_policy::reference_internal,
            "Return the bridged skeleton tree.")
        .def(
            "state",
            [](SkeletonBridge& self) -> SkeletonState& {
                return self.fk().state();
            },
            py::return_value_policy::reference_internal,
            "Return the mutable current skeleton state.")
        .def("body_prim", &SkeletonBridge::bodyPrim, py::arg("index"),
             py::return_value_policy::reference,
             "Return the scene prim for a body index.")
        .def("body_prims", &SkeletonBridge::bodyPrims,
             py::return_value_policy::reference_internal,
             "Return all body scene prims.")
        .def("num_bodies", &SkeletonBridge::numBodies,
             "Return the number of bridged bodies.");

    py::class_<SkeletonBridgeAsset>(
        anim, "SkeletonBridgeAsset",
        "Reusable skeleton bridge asset that can instantiate scene prims.")
        .def_static("from_mjcf", &SkeletonBridgeAsset::fromMJCF,
                    py::arg("mjcf_path"), py::arg("scale") = 1.0f,
                    py::arg("order") = "DFS",
                    "Load reusable bridge asset data from an MJCF file.")
        .def("define_mesh_assets", &SkeletonBridgeAsset::defineMeshAssets,
             py::arg("scene"), py::arg("mesh_asset_base_path"),
             "Define shared mesh asset prims in a scene.")
        .def("instantiate", &SkeletonBridgeAsset::instantiate, py::arg("scene"),
             py::arg("prim_base_path") = "/robot",
             py::arg("mesh_asset_base_path") = "",
             "Instantiate this asset into a scene.")
        .def("num_bodies", &SkeletonBridgeAsset::numBodies,
             "Return the number of bodies in this asset.");

    py::class_<SkeletonVisualConfig>(
        anim, "SkeletonVisualConfig",
        "Style settings for SkeletonVisualBridge line/joint rendering.")
        .def(py::init<>(), "Create default skeleton visual settings.")
        .def_readwrite("bone_color", &SkeletonVisualConfig::boneColor,
                       "RGBA color for bones.")
        .def_readwrite("joint_color", &SkeletonVisualConfig::jointColor,
                       "RGBA color for joints.")
        .def_readwrite("bone_radius", &SkeletonVisualConfig::boneRadius,
                       "Radius used for bone line geometry.")
        .def_readwrite("joint_radius", &SkeletonVisualConfig::jointRadius,
                       "Radius used for joint point geometry.")
        .def_readwrite("segments", &SkeletonVisualConfig::segments,
                       "Segment count for generated round geometry.")
        .def_readwrite("visible", &SkeletonVisualConfig::visible,
                       "Initial visibility state.")
        .def_readwrite("show_joints", &SkeletonVisualConfig::showJoints,
                       "Whether joint markers should be visible.");

    py::class_<SkeletonVisualBridge>(
        anim, "SkeletonVisualBridge",
        "Instanced line/point renderer for skeleton poses and motion clips.")
        .def(py::init<>(), "Create an empty skeleton visual bridge.")
        .def_static(
            "define",
            [](App* app, Backend::Shader* shader, const std::string& basePath,
               const SkeletonState& state, const SkeletonVisualConfig& config) {
                return SkeletonVisualBridge::define(app, shader, basePath,
                                                    state, config);
            },
            py::arg("app"), py::arg("shader"), py::arg("base_path"),
            py::arg("state"), py::arg("config") = SkeletonVisualConfig{},
            "Create skeleton visuals from a SkeletonState.")
        .def_static(
            "define",
            [](App* app, Backend::Shader* shader, const std::string& basePath,
               const SkeletonMotion& motion, float time, bool loop,
               const SkeletonVisualConfig& config) {
                return SkeletonVisualBridge::define(app, shader, basePath,
                                                    motion, time, loop, config);
            },
            py::arg("app"), py::arg("shader"), py::arg("base_path"),
            py::arg("motion"), py::arg("time") = 0.0f, py::arg("loop") = true,
            py::arg("config") = SkeletonVisualConfig{},
            "Create skeleton visuals by sampling a SkeletonMotion.")
        .def("apply_state", &SkeletonVisualBridge::applyState, py::arg("state"),
             "Update visuals from a SkeletonState.")
        .def("apply_motion", &SkeletonVisualBridge::applyMotion,
             py::arg("motion"), py::arg("time"), py::arg("loop") = true,
             "Update visuals by sampling a SkeletonMotion.")
        .def("set_visible", &SkeletonVisualBridge::setVisible,
             py::arg("visible"), "Set visibility for all skeleton visuals.")
        .def("set_show_joints", &SkeletonVisualBridge::setShowJoints,
             py::arg("show_joints"), "Show or hide joint markers.")
        .def("bone_handle", &SkeletonVisualBridge::boneHandle,
             "Return the renderable handle used for bones.")
        .def("joint_handle", &SkeletonVisualBridge::jointHandle,
             "Return the renderable handle used for joints.");
}
