///
/// Animation System Python Bindings
/// Skeleton animation bindings.
///

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include "py_array_view.hpp"

#include "character/character_description.hpp"
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
using namespace KE::Animation;
using namespace KE::Character;
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

    // ArticulationVisual / ArticulationVisualAsset
    py::class_<ArticulationVisualBridge>(
        anim, "ArticulationVisual",
        "Viewer-side articulated rigid-link visual object.")
        .def_static(
            "from_mjcf",
            [](const std::string& mjcfPath, Scene::SceneBackend* scene,
               const std::string& primBasePath, float scale,
               const std::string& order, const std::string& meshAssetBasePath) {
                return ArticulationVisualBridge::fromMJCF(mjcfPath, scene, primBasePath,
                                                scale, order,
                                                meshAssetBasePath);
            },
            py::arg("mjcf_path"), py::arg("scene"),
            py::arg("prim_base_path") = "/robot", py::arg("scale") = 1.0f,
            py::arg("order") = "DFS", py::arg("mesh_asset_base_path") = "",
            "Create an articulation visual and scene prims from an MJCF file.")
        .def("apply_pose", &ArticulationVisualBridge::applyPose,
             "Apply the current skeleton/body state to scene prims.")
        .def(
            "set_joint_rotation",
            [](ArticulationVisualBridge& self, int idx, const FloatArray& q) {
                self.setJointRotation(idx,
                                      eigenQuatXyzwFromArray(q, "rotation"));
            },
            py::arg("index"), py::arg("rotation"),
            "Set a joint rotation from an XYZW array.")
        .def(
            "set_joint_rotation",
            [](ArticulationVisualBridge& self, int idx, const glm::quat& q) {
                self.setJointRotation(idx, fromGlm(q));
            },
            py::arg("index"), py::arg("rotation"),
            "Set a joint rotation from a quaternion.")
        .def(
            "set_root_translation",
            [](ArticulationVisualBridge& self, const FloatArray& t) {
                self.setRootTranslation(
                    eigenVec3FromArray(t, "root_translation"));
            },
            py::arg("translation"), "Set the root translation from an array.")
        .def(
            "set_root_translation",
            [](ArticulationVisualBridge& self, const glm::vec3& t) {
                self.setRootTranslation(fromGlm(t));
            },
            py::arg("translation"), "Set the root translation from a vec3.")
        .def("reset_to_zero_pose", &ArticulationVisualBridge::resetToZeroPose,
             "Reset all joint rotations and root translation to zero pose.")
        .def(
            "skeleton",
            [](ArticulationVisualBridge& self) -> const SkeletonTree& {
                return self.fk().skeleton();
            },
            py::return_value_policy::reference_internal,
            "Return the bridged skeleton tree.")
        .def(
            "state",
            [](ArticulationVisualBridge& self) -> SkeletonState& {
                return self.fk().state();
            },
            py::return_value_policy::reference_internal,
            "Return the mutable current skeleton state.")
        .def("body_prim", &ArticulationVisualBridge::bodyPrim, py::arg("index"),
             py::return_value_policy::reference,
             "Return the scene prim for a body index.")
        .def("body_prims", &ArticulationVisualBridge::bodyPrims,
             py::return_value_policy::reference_internal,
             "Return all body scene prims.")
        .def("render_prims", &ArticulationVisualBridge::renderPrims,
             py::return_value_policy::reference_internal,
             "Return actual renderable mesh prims.")
        .def("render_prim_body_indices", &ArticulationVisualBridge::renderPrimBodyIndices,
             py::return_value_policy::reference_internal,
             "Return body index for each render prim.")
        .def("num_bodies", &ArticulationVisualBridge::numBodies,
             "Return the number of bridged bodies.");

    py::class_<ArticulationVisualBridgeAsset>(
        anim, "ArticulationVisualAsset",
        "Reusable articulated rigid-link visual asset that can instantiate scene prims.")
        .def_static("from_mjcf", &ArticulationVisualBridgeAsset::fromMJCF,
                    py::arg("mjcf_path"), py::arg("scale") = 1.0f,
                    py::arg("order") = "DFS",
                    "Load reusable bridge asset data from an MJCF file.")
        .def("define_mesh_assets", &ArticulationVisualBridgeAsset::defineMeshAssets,
             py::arg("scene"), py::arg("mesh_asset_base_path"),
             py::arg("split_visual_geoms") = false,
             "Define shared mesh asset prims in a scene.")
        .def("instantiate", &ArticulationVisualBridgeAsset::instantiate, py::arg("scene"),
             py::arg("prim_base_path") = "/robot",
             py::arg("mesh_asset_base_path") = "",
             py::arg("split_visual_geoms") = false,
             "Instantiate this asset into a scene.")
        .def("num_bodies", &ArticulationVisualBridgeAsset::numBodies,
             "Return the number of bodies in this asset.");

    py::class_<SkeletalVisualConfig>(
        anim, "SkeletalVisualConfig",
        "Style settings for SkeletalVisual line/joint rendering.")
        .def(py::init<>(), "Create default skeleton visual settings.")
        .def_readwrite("bone_color", &SkeletalVisualConfig::boneColor,
                       "RGBA color for bones.")
        .def_readwrite("joint_color", &SkeletalVisualConfig::jointColor,
                       "RGBA color for joints.")
        .def_readwrite("bone_radius", &SkeletalVisualConfig::boneRadius,
                       "Radius used for bone line geometry.")
        .def_readwrite("joint_radius", &SkeletalVisualConfig::jointRadius,
                       "Radius used for joint point geometry.")
        .def_readwrite("segments", &SkeletalVisualConfig::segments,
                       "Segment count for generated round geometry.")
        .def_readwrite("visible", &SkeletalVisualConfig::visible,
                       "Initial visibility state.")
        .def_readwrite("show_joints", &SkeletalVisualConfig::showJoints,
                       "Whether joint markers should be visible.");

    py::class_<SkeletalVisualBridge>(
        anim, "SkeletalVisual",
        "Instanced line/point renderer for skeleton poses and motion clips.")
        .def(py::init<>(), "Create an empty skeletal visual.")
        .def_static(
            "define",
            [](App* app, Backend::Shader* shader, const std::string& basePath,
               const SkeletonState& state, const SkeletalVisualConfig& config) {
                return SkeletalVisualBridge::define(app, shader, basePath,
                                                    state, config);
            },
            py::arg("app"), py::arg("shader"), py::arg("base_path"),
            py::arg("state"), py::arg("config") = SkeletalVisualConfig{},
            "Create skeleton visuals from a SkeletonState.")
        .def_static(
            "define",
            [](App* app, Backend::Shader* shader, const std::string& basePath,
               const SkeletonMotion& motion, float time, bool loop,
               const SkeletalVisualConfig& config) {
                return SkeletalVisualBridge::define(app, shader, basePath,
                                                    motion, time, loop, config);
            },
            py::arg("app"), py::arg("shader"), py::arg("base_path"),
            py::arg("motion"), py::arg("time") = 0.0f, py::arg("loop") = true,
            py::arg("config") = SkeletalVisualConfig{},
            "Create skeleton visuals by sampling a SkeletonMotion.")
        .def("apply_state", &SkeletalVisualBridge::applyState, py::arg("state"),
             "Update visuals from a SkeletonState.")
        .def("apply_motion", &SkeletalVisualBridge::applyMotion,
             py::arg("motion"), py::arg("time"), py::arg("loop") = true,
             "Update visuals by sampling a SkeletonMotion.")
        .def("set_visible", &SkeletalVisualBridge::setVisible,
             py::arg("visible"), "Set visibility for all skeleton visuals.")
        .def("set_show_joints", &SkeletalVisualBridge::setShowJoints,
             py::arg("show_joints"), "Show or hide joint markers.")
        .def("bone_handle", &SkeletalVisualBridge::boneHandle,
             "Return the renderable handle used for bones.")
        .def("joint_handle", &SkeletalVisualBridge::jointHandle,
             "Return the renderable handle used for joints.");
}
