///
/// Animation System Python Bindings
/// MJCFLoader, CharacterData, SkeletonTree, SkeletonState, SkeletonBridge
///

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include "py_array_view.hpp"

#include "animation/character_description.hpp"
#include "asset/fbx_loader.hpp"
#include "asset/mjcf_loader.hpp"
#include "animation/skeleton_math.hpp"
#include "animation/skeleton_motion.hpp"
#include "animation/skeleton_state.hpp"
#include "animation/skeleton_tree.hpp"
#include "animation/skinning.hpp"
#include "bridge/skeleton_bridge.hpp"
#include "engine/scene/native/prim.hpp"
#include "engine/scene/scene_backend.hpp"

namespace py = pybind11;
using namespace KE;
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
            view(i, j) =
                values[static_cast<size_t>(i)][static_cast<size_t>(j)];
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
    py::array_t<float> array(
        {static_cast<py::ssize_t>(values.size()), py::ssize_t(4),
         py::ssize_t(4)});
    auto view = array.mutable_unchecked<3>();
    for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(values.size()); ++i) {
        for (py::ssize_t row = 0; row < 4; ++row) {
            for (py::ssize_t col = 0; col < 4; ++col) {
                view(i, row, col) =
                    values[static_cast<size_t>(i)]
                          [static_cast<size_t>(row * 4 + col)];
            }
        }
    }
    return array;
}

py::array_t<float> floatArrayFromMat4Vector(const std::vector<glm::mat4>& values) {
    py::array_t<float> array(
        {static_cast<py::ssize_t>(values.size()), py::ssize_t(4),
         py::ssize_t(4)});
    auto view = array.mutable_unchecked<3>();
    for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(values.size()); ++i) {
        const glm::mat4& m = values[static_cast<size_t>(i)];
        for (py::ssize_t row = 0; row < 4; ++row) {
            for (py::ssize_t col = 0; col < 4; ++col)
                view(i, row, col) = m[static_cast<size_t>(col)]
                                     [static_cast<size_t>(row)];
        }
    }
    return array;
}

py::array_t<float>
floatArrayFromMat4Vector(const std::vector<Eigen::Matrix4f>& values) {
    py::array_t<float> array(
        {static_cast<py::ssize_t>(values.size()), py::ssize_t(4),
         py::ssize_t(4)});
    auto view = array.mutable_unchecked<3>();
    for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(values.size()); ++i) {
        const Eigen::Matrix4f& m = values[static_cast<size_t>(i)];
        for (py::ssize_t row = 0; row < 4; ++row) {
            for (py::ssize_t col = 0; col < 4; ++col)
                view(i, row, col) = m(static_cast<int>(row),
                                      static_cast<int>(col));
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

std::vector<std::array<float, 4>>
floatVec4ArrayFromPy(const FloatArray& array, const char* name) {
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
    py::module anim = m.def_submodule("animation", "Skeleton animation system");

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
            std::vector<Eigen::Matrix4f> globals =
                eigenMat4ArrayFromPy(currentGlobalMatrices,
                                     "current_global_matrices");

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
        py::arg("current_global_matrices"));

    anim.def(
        "compute_skinning_matrices",
        [](py::array_t<int> boneNodeIndices, const FloatArray& inverseBinds,
           const FloatArray& currentGlobalMatrices) {
            std::vector<int> nodeIndices =
                intVectorFromPy(boneNodeIndices, "bone_node_indices");
            std::vector<Eigen::Matrix4f> inv =
                eigenMat4ArrayFromPy(inverseBinds, "inverse_bind_matrices");
            std::vector<Eigen::Matrix4f> globals =
                eigenMat4ArrayFromPy(currentGlobalMatrices,
                                     "current_global_matrices");
            return floatArrayFromMat4Vector(Skinning::computeSkinningMatrices(
                nodeIndices, inv, globals));
        },
        py::arg("bone_node_indices"), py::arg("inverse_bind_matrices"),
        py::arg("current_global_matrices"));

    anim.def(
        "compute_skinning_matrices_into",
        [](py::array_t<int> boneNodeIndices, const FloatArray& inverseBinds,
           const FloatArray& currentGlobalMatrices, const FloatArray& output) {
            std::vector<int> nodeIndices =
                intVectorFromPy(boneNodeIndices, "bone_node_indices");
            std::vector<Eigen::Matrix4f> inv =
                eigenMat4ArrayFromPy(inverseBinds, "inverse_bind_matrices");
            std::vector<Eigen::Matrix4f> globals =
                eigenMat4ArrayFromPy(currentGlobalMatrices,
                                     "current_global_matrices");
            std::vector<Eigen::Matrix4f> matrices;
            Skinning::computeSkinningMatricesInto(nodeIndices, inv, globals,
                                                  matrices);
            writeMat4VectorToPy(matrices, output, "output");
            return output;
        },
        py::arg("bone_node_indices"), py::arg("inverse_bind_matrices"),
        py::arg("current_global_matrices"), py::arg("output"));

    // Joint (from character_description.hpp)
    py::class_<Joint>(anim, "Joint")
        .def_readonly("name", &Joint::name)
        .def_readonly("lo_limit", &Joint::loLimit)
        .def_readonly("hi_limit", &Joint::hiLimit)
        .def_property_readonly("axis",
                               [](const Joint& j) { return toGlm(j.axis); });

    // MeshInfo
    py::class_<MeshInfo>(anim, "MeshInfo")
        .def_readonly("body_name", &MeshInfo::bodyName)
        .def_readonly("mesh_file", &MeshInfo::meshFile)
        .def_readonly("body_index", &MeshInfo::bodyIndex)
        .def_property_readonly("pos",
                               [](const MeshInfo& m) { return toGlm(m.pos); })
        .def_property_readonly("quat",
                               [](const MeshInfo& m) {
                                   return glm::quat(m.quat.w(), m.quat.x(),
                                                    m.quat.y(), m.quat.z());
                               })
        .def_property_readonly("rgba", [](const MeshInfo& m) {
            return glm::vec4(m.rgba.x(), m.rgba.y(), m.rgba.z(), m.rgba.w());
        });

    py::enum_<CollisionGeom::Type>(anim, "CollisionGeomType")
        .value("Capsule", CollisionGeom::Type::Capsule)
        .value("Cylinder", CollisionGeom::Type::Cylinder)
        .value("Sphere", CollisionGeom::Type::Sphere)
        .value("Box", CollisionGeom::Type::Box)
        .export_values();

    py::class_<CollisionGeom>(anim, "CollisionGeom")
        .def_readonly("type", &CollisionGeom::type)
        .def_property_readonly(
            "pos", [](const CollisionGeom& g) { return toGlm(g.pos); })
        .def_property_readonly(
            "quat", [](const CollisionGeom& g) { return toGlm(g.quat); })
        .def_property_readonly("size",
                               [](const CollisionGeom& g) {
                                   return std::vector<float>{
                                       g.size[0], g.size[1], g.size[2]};
                               })
        .def_readonly("has_from_to", &CollisionGeom::hasFromTo)
        .def_property_readonly(
            "from_pos", [](const CollisionGeom& g) { return toGlm(g.from); })
        .def_property_readonly(
            "to_pos", [](const CollisionGeom& g) { return toGlm(g.to); })
        .def_readonly("friction", &CollisionGeom::friction)
        .def_readonly("condim", &CollisionGeom::condim)
        .def_readonly("margin", &CollisionGeom::margin);

    // CharacterData — aggregate returned by MJCFLoader::load()
    py::class_<CharacterData>(anim, "CharacterData")
        .def_readonly("skeleton_tree", &CharacterData::skeletonTree)
        .def_readonly("mesh_infos", &CharacterData::meshInfos)
        .def_readonly("mesh_dir", &CharacterData::meshDir)
        // joints: return dict[int, list[Joint]]
        .def_property_readonly("joints",
                               [](const CharacterData& d) {
                                   py::dict result;
                                   for (const auto& [idx, jvec] : d.joints)
                                       result[py::int_(idx)] = jvec;
                                   return result;
                               })
        .def_property_readonly("collision_geoms", [](const CharacterData& d) {
            py::dict result;
            for (const auto& [idx, geoms] : d.collisionGeoms)
                result[py::int_(idx)] = geoms;
            return result;
        });

    // MJCFLoader
    py::class_<MJCFLoader>(anim, "MJCFLoader")
        .def_static("load", &MJCFLoader::load, py::arg("mjcf_path"),
                    py::arg("scale") = 1.0f, py::arg("order") = "DFS");

    py::class_<FBXAnimationClipInfo>(anim, "FBXAnimationClipInfo")
        .def_readonly("name", &FBXAnimationClipInfo::name)
        .def_readonly("start_time", &FBXAnimationClipInfo::startTime)
        .def_readonly("end_time", &FBXAnimationClipInfo::endTime)
        .def_readonly("frame_rate", &FBXAnimationClipInfo::frameRate);

    py::class_<FBXMaterialInfo>(anim, "FBXMaterialInfo")
        .def_readonly("name", &FBXMaterialInfo::name)
        .def_property_readonly("diffuse_color",
                               [](const FBXMaterialInfo& self) {
                                   return py::make_tuple(
                                       self.diffuseColor[0],
                                       self.diffuseColor[1],
                                       self.diffuseColor[2],
                                       self.diffuseColor[3]);
                               })
        .def_readonly("diffuse_texture_path",
                      &FBXMaterialInfo::diffuseTexturePath)
        .def_readonly("has_diffuse_texture",
                      &FBXMaterialInfo::hasDiffuseTexture)
        .def_readonly("has_embedded_diffuse_texture",
                      &FBXMaterialInfo::hasEmbeddedDiffuseTexture);

    py::class_<FBXMeshMetadata, std::shared_ptr<FBXMeshMetadata>>(
        anim, "FBXMeshMetadata")
        .def_readonly("name", &FBXMeshMetadata::name)
        .def_readonly("vertex_count", &FBXMeshMetadata::vertexCount)
        .def_readonly("index_count", &FBXMeshMetadata::indexCount)
        .def_readonly("material_count", &FBXMeshMetadata::materialCount)
        .def_readonly("primary_material_index",
                      &FBXMeshMetadata::primaryMaterialIndex)
        .def_readonly("has_normals", &FBXMeshMetadata::hasNormals)
        .def_readonly("has_uvs", &FBXMeshMetadata::hasUVs)
        .def_readonly("has_skin", &FBXMeshMetadata::hasSkin)
        .def_readonly("skin_cluster_names",
                      &FBXMeshMetadata::skinClusterNames)
        .def_readonly("materials", &FBXMeshMetadata::materials);

    py::class_<FBXMeshInfo, std::shared_ptr<FBXMeshInfo>, FBXMeshMetadata>(
        anim, "FBXMeshInfo")
        .def_property_readonly("mesh_data",
                               [](std::shared_ptr<FBXMeshInfo> self) {
                                   return std::shared_ptr<KE::Scene::MeshData>(
                                       self, &self->meshData);
                               });

    py::class_<FBXSkinClusterInfo>(anim, "FBXSkinClusterInfo")
        .def_readonly("mesh_name", &FBXSkinClusterInfo::meshName)
        .def_readonly("cluster_name", &FBXSkinClusterInfo::clusterName)
        .def_readonly("link_name", &FBXSkinClusterInfo::linkName)
        .def_readonly("weight_count", &FBXSkinClusterInfo::weightCount)
        .def_readonly("index_count", &FBXSkinClusterInfo::indexCount)
        .def_readonly("min_index", &FBXSkinClusterInfo::minIndex)
        .def_readonly("max_index", &FBXSkinClusterInfo::maxIndex)
        .def_readonly("min_weight", &FBXSkinClusterInfo::minWeight)
        .def_readonly("max_weight", &FBXSkinClusterInfo::maxWeight)
        .def_readonly("weight_sum", &FBXSkinClusterInfo::weightSum)
        .def_readonly("transform_translation",
                      &FBXSkinClusterInfo::transformTranslation)
        .def_readonly("transform_link_translation",
                      &FBXSkinClusterInfo::transformLinkTranslation);

    py::class_<FBXSkinnedMeshInfo, std::shared_ptr<FBXSkinnedMeshInfo>>(
        anim, "FBXSkinnedMeshInfo")
        .def_property_readonly("metadata",
                               [](const FBXSkinnedMeshInfo& self) {
                                   return self.mesh;
                               })
        .def_property_readonly(
            "name", [](const FBXSkinnedMeshInfo& self) { return self.mesh.name; })
        .def_property_readonly("vertex_count",
                               [](const FBXSkinnedMeshInfo& self) {
                                   return self.mesh.vertexCount;
                               })
        .def_property_readonly("index_count",
                               [](const FBXSkinnedMeshInfo& self) {
                                   return self.mesh.indexCount;
                               })
        .def_property_readonly("material_count",
                               [](const FBXSkinnedMeshInfo& self) {
                                   return self.mesh.materialCount;
                               })
        .def_property_readonly("primary_material_index",
                               [](const FBXSkinnedMeshInfo& self) {
                                   return self.mesh.primaryMaterialIndex;
                               })
        .def_property_readonly("materials",
                               [](const FBXSkinnedMeshInfo& self) {
                                   return self.mesh.materials;
                               })
        .def_property_readonly("has_normals",
                               [](const FBXSkinnedMeshInfo& self) {
                                   return self.mesh.hasNormals;
                               })
        .def_property_readonly("has_uvs",
                               [](const FBXSkinnedMeshInfo& self) {
                                   return self.mesh.hasUVs;
                               })
        .def_property_readonly("has_skin",
                               [](const FBXSkinnedMeshInfo& self) {
                                   return self.mesh.hasSkin;
                               })
        .def_property_readonly("skin_cluster_names",
                               [](const FBXSkinnedMeshInfo& self) {
                                   return self.mesh.skinClusterNames;
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
        .def_property_readonly("inverse_bind_matrices",
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

    // FBXLoader
    py::class_<FBXLoader>(anim, "FBXLoader")
        .def_static("load_skeleton", &FBXLoader::loadSkeleton,
                    py::arg("fbx_path"), py::arg("scale") = 0.01f)
        .def_static("load_animation_clip_infos",
                    &FBXLoader::loadAnimationClipInfos, py::arg("fbx_path"))
        .def_static("load_motion", &FBXLoader::loadMotion,
                    py::arg("fbx_path"), py::arg("clip_index") = 0,
                    py::arg("fps") = 30.0f, py::arg("scale") = 0.01f)
        .def_static(
            "load_meshes",
            [](const std::string& fbxPath, float scale) {
                py::list result;
                for (auto& mesh : FBXLoader::loadMeshes(fbxPath, scale)) {
                    result.append(std::make_shared<FBXMeshInfo>(
                        std::move(mesh)));
                }
                return result;
            },
            py::arg("fbx_path"), py::arg("scale") = 0.01f)
        .def_static("load_skin_cluster_infos",
                    &FBXLoader::loadSkinClusterInfos, py::arg("fbx_path"),
                    py::arg("scale") = 0.01f)
        .def_static(
            "load_skinned_meshes",
            [](const std::string& fbxPath, float scale) {
                py::list result;
                for (auto& mesh :
                     FBXLoader::loadSkinnedMeshes(fbxPath, scale)) {
                    result.append(std::make_shared<FBXSkinnedMeshInfo>(
                        std::move(mesh)));
                }
                return result;
            },
            py::arg("fbx_path"), py::arg("scale") = 0.01f);

    // SkeletonTree (read-only after construction)
    py::class_<SkeletonTree, std::shared_ptr<SkeletonTree>>(anim,
                                                            "SkeletonTree")
        .def("num_joints", &SkeletonTree::numJoints)
        .def("node_name", &SkeletonTree::nodeName)
        .def("parent_index", &SkeletonTree::parentIndex)
        .def("local_translation",
             [](const SkeletonTree& self, int i) {
                 return toGlm(self.localTranslation(i));
             })
        .def("local_rotation",
             [](const SkeletonTree& self, int i) {
                 return toGlm(self.localRotation(i));
             })
        .def("node_names", &SkeletonTree::nodeNames)
        .def("parent_indices", &SkeletonTree::parentIndices)
        .def("print", &SkeletonTree::print);

    py::class_<SkeletonMotion>(anim, "SkeletonMotion")
        .def("num_frames", &SkeletonMotion::numFrames)
        .def("num_joints", &SkeletonMotion::numJoints)
        .def("fps", &SkeletonMotion::fps)
        .def("duration", &SkeletonMotion::duration)
        .def("motion_name", &SkeletonMotion::motionName)
        .def("node_names",
             [](const SkeletonMotion& self) {
                 return self.skeletonTree().nodeNames();
             })
        .def("parent_indices",
             [](const SkeletonMotion& self) {
                 return self.skeletonTree().parentIndices();
             })
        .def("frame", &SkeletonMotion::frame)
        .def("sample", &SkeletonMotion::sample, py::arg("time"),
             py::arg("loop") = true)
        .def("root_translation",
             [](const SkeletonMotion& self, int frame) {
                 return toGlm(self.rootTranslation(frame));
             })
        .def("local_rotation",
             [](const SkeletonMotion& self, int frame, int joint) {
                 return toGlm(self.localRotation(frame, joint));
             })
        .def("root_translations_flat",
             &SkeletonMotion::rootTranslationsFlat)
        .def("local_rotations_wxyz_flat",
             &SkeletonMotion::localRotationsWxyzFlat);

    // Transform (FK result)
    py::class_<Transform>(anim, "Transform")
        .def_property_readonly(
            "rotation", [](const Transform& t) { return toGlm(t.rotation); })
        .def_property_readonly("translation", [](const Transform& t) {
            return toGlm(t.translation);
        });

    // SkeletonState
    py::class_<SkeletonState>(anim, "SkeletonState")
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
            py::arg("is_local") = true)
        .def("num_joints", &SkeletonState::numJoints)
        .def("is_local", &SkeletonState::isLocal)
        .def("compute_global_transforms",
             &SkeletonState::computeGlobalTransforms)
        .def("compute_global_matrices",
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
                                 r(static_cast<int>(row),
                                   static_cast<int>(col));
                         }
                     }
                     view(i, 0, 3) = t.translation.x();
                     view(i, 1, 3) = t.translation.y();
                     view(i, 2, 3) = t.translation.z();
                 }
                 return array;
             })
        .def("compute_global_positions",
             [](const SkeletonState& self) {
                 auto positions = self.computeGlobalPositions();
                 std::vector<glm::vec3> result;
                 result.reserve(positions.size());
                 for (const auto& p : positions) {
                     result.push_back(toGlm(p));
                 }
                 return result;
             })
        .def("rotation", [](const SkeletonState& self,
                            int i) { return toGlm(self.rotation(i)); })
        .def("set_rotation",
             [](SkeletonState& self, int i, const FloatArray& q) {
                 self.setRotation(i, eigenQuatXyzwFromArray(q, "rotation"));
             })
        .def("set_rotation",
             [](SkeletonState& self, int i, const glm::quat& q) {
                 self.setRotation(i, fromGlm(q));
             })
        .def("root_translation",
             [](const SkeletonState& self) {
                 return toGlm(self.rootTranslation());
             })
        .def("set_root_translation",
             [](SkeletonState& self, const FloatArray& t) {
                 self.setRootTranslation(
                     eigenVec3FromArray(t, "root_translation"));
             })
        .def("set_root_translation",
             [](SkeletonState& self, const glm::vec3& t) {
                 self.setRootTranslation(fromGlm(t));
             })
        .def("print_global_positions", &SkeletonState::printGlobalPositions);

    // SkeletonBridge
    py::class_<SkeletonBridge>(anim, "SkeletonBridge")
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
            py::arg("order") = "DFS", py::arg("mesh_asset_base_path") = "")
        .def("apply_pose", &SkeletonBridge::applyPose)
        .def("set_joint_rotation",
             [](SkeletonBridge& self, int idx, const FloatArray& q) {
                 self.setJointRotation(idx,
                                       eigenQuatXyzwFromArray(q, "rotation"));
             })
        .def("set_joint_rotation",
             [](SkeletonBridge& self, int idx, const glm::quat& q) {
                 self.setJointRotation(idx, fromGlm(q));
             })
        .def("set_root_translation",
             [](SkeletonBridge& self, const FloatArray& t) {
                 self.setRootTranslation(
                     eigenVec3FromArray(t, "root_translation"));
             })
        .def("set_root_translation",
             [](SkeletonBridge& self, const glm::vec3& t) {
                 self.setRootTranslation(fromGlm(t));
             })
        .def("reset_to_zero_pose", &SkeletonBridge::resetToZeroPose)
        .def(
            "skeleton",
            [](SkeletonBridge& self) -> const SkeletonTree& {
                return self.fk().skeleton();
            },
            py::return_value_policy::reference_internal)
        .def(
            "state",
            [](SkeletonBridge& self) -> SkeletonState& {
                return self.fk().state();
            },
            py::return_value_policy::reference_internal)
        .def("body_prim", &SkeletonBridge::bodyPrim,
             py::return_value_policy::reference)
        .def("body_prims", &SkeletonBridge::bodyPrims,
             py::return_value_policy::reference_internal)
        .def("num_bodies", &SkeletonBridge::numBodies);

    py::class_<SkeletonBridgeAsset>(anim, "SkeletonBridgeAsset")
        .def_static("from_mjcf", &SkeletonBridgeAsset::fromMJCF,
                    py::arg("mjcf_path"), py::arg("scale") = 1.0f,
                    py::arg("order") = "DFS")
        .def("define_mesh_assets", &SkeletonBridgeAsset::defineMeshAssets,
             py::arg("scene"), py::arg("mesh_asset_base_path"))
        .def("instantiate", &SkeletonBridgeAsset::instantiate, py::arg("scene"),
             py::arg("prim_base_path") = "/robot",
             py::arg("mesh_asset_base_path") = "")
        .def("num_bodies", &SkeletonBridgeAsset::numBodies);
}
