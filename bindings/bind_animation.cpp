///
/// Animation System Python Bindings
/// Skeleton animation bindings.
///

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/typing.h>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <algorithm>
#include <iterator>
#include "py_array_view.hpp"

#include "asset/articulation_desc.hpp"
#include "animation/skeleton_math.hpp"
#include "animation/full_body_ik.hpp"
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

std::vector<Eigen::Quaternionf> eigenQuatWxyzArray(const FloatArray& array,
                                                   const char* name) {
    auto view = vec4ArrayView(array, name);
    std::vector<Eigen::Quaternionf> result;
    result.reserve(view.count);
    for (size_t i = 0; i < view.count; ++i) {
        const float* q = view.data + i * 4;
        result.emplace_back(q[0], q[1], q[2], q[3]);
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

py::array_t<float> motionVec3Array(const std::vector<Eigen::Vector3f>& values,
                                   int frames, int joints) {
    py::array_t<float> output({static_cast<py::ssize_t>(frames),
                               static_cast<py::ssize_t>(joints),
                               py::ssize_t(3)});
    auto view = output.mutable_unchecked<3>();
    for (int frame = 0; frame < frames; ++frame) {
        for (int joint = 0; joint < joints; ++joint) {
            const auto& value =
                values[static_cast<size_t>(frame * joints + joint)];
            view(frame, joint, 0) = value.x();
            view(frame, joint, 1) = value.y();
            view(frame, joint, 2) = value.z();
        }
    }
    return output;
}

py::array_t<float> motionGlobalMatrices(const SkeletonMotion& motion) {
    const auto transforms = motion.globalTransforms();
    py::array_t<float> output({static_cast<py::ssize_t>(motion.numFrames()),
                               static_cast<py::ssize_t>(motion.numJoints()),
                               py::ssize_t(4), py::ssize_t(4)});
    auto view = output.mutable_unchecked<4>();
    for (int frame = 0; frame < motion.numFrames(); ++frame) {
        for (int joint = 0; joint < motion.numJoints(); ++joint) {
            const Transform& transform = transforms[static_cast<size_t>(
                frame * motion.numJoints() + joint)];
            const Eigen::Matrix3f rotation =
                transform.rotation.toRotationMatrix();
            for (int row = 0; row < 4; ++row) {
                for (int col = 0; col < 4; ++col) {
                    float value = row == col ? 1.0f : 0.0f;
                    if (row < 3 && col < 3)
                        value = rotation(row, col);
                    else if (row < 3 && col == 3)
                        value = transform.translation[row];
                    view(frame, joint, row, col) = value;
                }
            }
        }
    }
    return output;
}

py::array_t<float> motionGlobalRotations(const SkeletonMotion& motion) {
    const auto transforms = motion.globalTransforms();
    py::array_t<float> output({static_cast<py::ssize_t>(motion.numFrames()),
                               static_cast<py::ssize_t>(motion.numJoints()),
                               py::ssize_t(4)});
    auto view = output.mutable_unchecked<3>();
    for (int frame = 0; frame < motion.numFrames(); ++frame) {
        for (int joint = 0; joint < motion.numJoints(); ++joint) {
            const auto& rotation =
                transforms[static_cast<size_t>(frame * motion.numJoints() +
                                               joint)]
                    .rotation;
            view(frame, joint, 0) = rotation.w();
            view(frame, joint, 1) = rotation.x();
            view(frame, joint, 2) = rotation.y();
            view(frame, joint, 3) = rotation.z();
        }
    }
    return output;
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
                intVec4ArrayFromPy(boneIndices, "skin_bone_indices");
            std::vector<std::array<float, 4>> weights =
                floatVec4ArrayFromPy(boneWeights, "skin_bone_weights");
            std::vector<int> nodeIndices =
                intVectorFromPy(boneNodeIndices, "skin_bone_node_indices");
            std::vector<Eigen::Matrix4f> inv =
                eigenMat4ArrayFromPy(inverseBinds, "inverse_bind_matrices");
            std::vector<Eigen::Matrix4f> globals = eigenMat4ArrayFromPy(
                currentGlobalMatrices, "skeleton_global_matrices");

            CPUSkinningInput input;
            input.bindPositions = &positions;
            input.bindNormals = &normals;
            input.boneIndices = &indices;
            input.boneWeights = &weights;
            input.boneNodeIndices = &nodeIndices;
            input.inverseBindMatrices = &inv;
            input.currentGlobalMatrices = &globals;
            CPUSkinningResult result = Skinning::cpuSkin(input);

            py::typing::Dict<py::str, py::array_t<float>> out;
            out["positions"] = floatArrayFromVec3Vector(result.positions);
            out["normals"] = floatArrayFromVec3Vector(result.normals);
            return out;
        },
        py::arg("bind_positions"), py::arg("bind_normals"),
        py::arg("skin_bone_indices"), py::arg("skin_bone_weights"),
        py::arg("skin_bone_node_indices"), py::arg("inverse_bind_matrices"),
        py::arg("skeleton_global_matrices"),
        "CPU-skin bind-space positions and normals with current bone "
        "matrices.");

    anim.def(
        "compute_skinning_matrices",
        [](py::array_t<int> boneNodeIndices, const FloatArray& inverseBinds,
           const FloatArray& currentGlobalMatrices) {
            std::vector<int> nodeIndices =
                intVectorFromPy(boneNodeIndices, "skin_bone_node_indices");
            std::vector<Eigen::Matrix4f> inv =
                eigenMat4ArrayFromPy(inverseBinds, "inverse_bind_matrices");
            std::vector<Eigen::Matrix4f> globals = eigenMat4ArrayFromPy(
                currentGlobalMatrices, "skeleton_global_matrices");
            return floatArrayFromMat4Vector(
                Skinning::computeSkinningMatrices(nodeIndices, inv, globals));
        },
        py::arg("skin_bone_node_indices"), py::arg("inverse_bind_matrices"),
        py::arg("skeleton_global_matrices"),
        "Compute skinning matrices from skeleton globals and inverse binds.");

    anim.def(
        "compute_skinning_matrices_into",
        [](py::array_t<int> boneNodeIndices, const FloatArray& inverseBinds,
           const FloatArray& currentGlobalMatrices, const FloatArray& output) {
            std::vector<int> nodeIndices =
                intVectorFromPy(boneNodeIndices, "skin_bone_node_indices");
            std::vector<Eigen::Matrix4f> inv =
                eigenMat4ArrayFromPy(inverseBinds, "inverse_bind_matrices");
            std::vector<Eigen::Matrix4f> globals = eigenMat4ArrayFromPy(
                currentGlobalMatrices, "skeleton_global_matrices");
            std::vector<Eigen::Matrix4f> matrices;
            Skinning::computeSkinningMatricesInto(nodeIndices, inv, globals,
                                                  matrices);
            writeMat4VectorToPy(matrices, output, "output");
            return output;
        },
        py::arg("skin_bone_node_indices"), py::arg("inverse_bind_matrices"),
        py::arg("skeleton_global_matrices"), py::arg("output"),
        "Compute skinning matrices into an existing output array.");

    // SkeletonTree (read-only after construction)
    py::class_<SkeletonTree, std::shared_ptr<SkeletonTree>>(
        anim, "SkeletonTree",
        "Read-only skeleton hierarchy with local bind transforms.")
        .def(py::init([](std::vector<std::string> names,
                         std::vector<int> parents,
                         const FloatArray& translations,
                         const FloatArray& rotationsWxyz) {
                 auto translationView =
                     vec3ArrayView(translations, "local_translations");
                 auto rotationValues =
                     eigenQuatWxyzArray(rotationsWxyz, "local_rotations_wxyz");
                 if (translationView.count != names.size() ||
                     rotationValues.size() != names.size() ||
                     parents.size() != names.size()) {
                     throw py::value_error(
                         "names, parents, local_translations, and "
                         "local_rotations "
                         "must have the same joint count");
                 }
                 std::vector<Eigen::Vector3f> translationValues;
                 translationValues.reserve(translationView.count);
                 for (size_t i = 0; i < translationView.count; ++i) {
                     const float* value = translationView.data + i * 3;
                     translationValues.emplace_back(value[0], value[1],
                                                    value[2]);
                 }
                 return std::make_shared<SkeletonTree>(
                     std::move(names), std::move(parents),
                     std::move(translationValues), std::move(rotationValues),
                     std::vector<int>(translationView.count, 0));
             }),
             py::arg("names"), py::arg("parents"),
             py::arg("local_translations"), py::arg("local_rotations_wxyz"),
             "Create a skeleton from NumPy translations and WXYZ bind "
             "rotations.")
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
        .def_static(
            "from_arrays",
            [](const SkeletonTree& tree, const FloatArray& rootTranslations,
               const FloatArray& localRotationsWxyz, float fps,
               std::string motionName) {
                py::buffer_info rootInfo = rootTranslations.request();
                py::buffer_info rotationInfo = localRotationsWxyz.request();
                if (rootInfo.ndim != 2 || rootInfo.shape[1] != 3) {
                    throw py::value_error(
                        "root_translations expected shape [frames, 3]");
                }
                if (rotationInfo.ndim != 3 ||
                    rotationInfo.shape[0] != rootInfo.shape[0] ||
                    rotationInfo.shape[1] != tree.numJoints() ||
                    rotationInfo.shape[2] != 4) {
                    throw py::value_error("local_rotations_wxyz expected shape "
                                          "[frames, num_joints, 4]");
                }
                const auto* root = static_cast<const float*>(rootInfo.ptr);
                const auto* rotations =
                    static_cast<const float*>(rotationInfo.ptr);
                return SkeletonMotion(
                    std::make_shared<SkeletonTree>(tree), fps,
                    std::move(motionName),
                    std::vector<float>(root, root + rootInfo.size),
                    std::vector<float>(rotations,
                                       rotations + rotationInfo.size));
            },
            py::arg("skeleton_tree"), py::arg("root_translations"),
            py::arg("local_rotations_wxyz"), py::arg("fps"),
            py::arg("motion_name") = "Motion",
            "Create a motion from root translations [F, 3] and WXYZ local "
            "rotations [F, J, 4].")
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
        .def_property_readonly(
            "skeleton_tree",
            [](const SkeletonMotion& self) {
                return std::const_pointer_cast<SkeletonTree>(
                    self.skeletonTreePtr());
            },
            "Return the motion's read-only skeleton hierarchy.")
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
             "Return local rotations as a flat WXYZ float array.")
        .def(
            "root_translations",
            [](const SkeletonMotion& self) {
                py::array_t<float> output(
                    {static_cast<py::ssize_t>(self.numFrames()),
                     py::ssize_t(3)});
                const auto& values = self.rootTranslationsFlat();
                if (!values.empty()) {
                    std::memcpy(output.mutable_data(), values.data(),
                                values.size() * sizeof(float));
                }
                return output;
            },
            "Return root translations as float32 [F, 3].")
        .def(
            "local_rotations_wxyz",
            [](const SkeletonMotion& self) {
                py::array_t<float> output(
                    {static_cast<py::ssize_t>(self.numFrames()),
                     static_cast<py::ssize_t>(self.numJoints()),
                     py::ssize_t(4)});
                const auto& values = self.localRotationsWxyzFlat();
                if (!values.empty()) {
                    std::memcpy(output.mutable_data(), values.data(),
                                values.size() * sizeof(float));
                }
                return output;
            },
            "Return WXYZ local rotations as float32 [F, J, 4].")
        .def("global_matrices", &motionGlobalMatrices,
             "Compute global transforms as float32 [F, J, 4, 4].")
        .def(
            "global_positions",
            [](const SkeletonMotion& self) {
                return motionVec3Array(self.globalPositions(), self.numFrames(),
                                       self.numJoints());
            },
            "Compute global joint positions as float32 [F, J, 3].")
        .def("global_rotations_wxyz", &motionGlobalRotations,
             "Compute WXYZ global joint rotations as float32 [F, J, 4].")
        .def(
            "root_linear_velocities",
            [](const SkeletonMotion& self) {
                return floatArrayFromVec3Vector(self.rootLinearVelocities());
            },
            "Compute root linear velocities as float32 [F, 3].")
        .def(
            "root_linear_accelerations",
            [](const SkeletonMotion& self) {
                return floatArrayFromVec3Vector(self.rootLinearAccelerations());
            },
            "Compute root linear accelerations as float32 [F, 3].")
        .def(
            "global_linear_velocities",
            [](const SkeletonMotion& self) {
                return motionVec3Array(self.globalLinearVelocities(),
                                       self.numFrames(), self.numJoints());
            },
            "Compute global joint linear velocities as float32 [F, J, 3].")
        .def(
            "global_angular_velocities",
            [](const SkeletonMotion& self) {
                return motionVec3Array(self.globalAngularVelocities(),
                                       self.numFrames(), self.numJoints());
            },
            "Compute global joint angular velocities as float32 [F, J, 3].")
        .def(
            "global_linear_accelerations",
            [](const SkeletonMotion& self) {
                return motionVec3Array(self.globalLinearAccelerations(),
                                       self.numFrames(), self.numJoints());
            },
            "Compute global joint linear accelerations as float32 [F, J, 3].")
        .def(
            "global_angular_accelerations",
            [](const SkeletonMotion& self) {
                return motionVec3Array(self.globalAngularAccelerations(),
                                       self.numFrames(), self.numJoints());
            },
            "Compute global joint angular accelerations as float32 [F, J, 3].");

    anim.def(
        "solve_full_body_ik_batch",
        [](const SkeletonMotion& motion, const FloatArray& targets,
           py::array_t<int, py::array::c_style | py::array::forcecast>
               effectors,
           const FloatArray& offsets,
           py::array_t<int, py::array::c_style | py::array::forcecast>
               controlJoints,
           const FloatArray& controlAxes, int maxIterations) {
            const auto targetInfo = targets.request();
            const auto effectorInfo = effectors.request();
            const auto offsetInfo = offsets.request();
            const auto jointInfo = controlJoints.request();
            const auto axisInfo = controlAxes.request();
            if (targetInfo.ndim != 3 ||
                targetInfo.shape[0] != motion.numFrames() ||
                targetInfo.shape[2] != 3)
                throw py::value_error(
                    "targets expected shape [frames, effectors, 3]");
            const py::ssize_t count = targetInfo.shape[1];
            if (effectorInfo.ndim != 1 || effectorInfo.shape[0] != count ||
                offsetInfo.ndim != 2 || offsetInfo.shape[0] != count ||
                offsetInfo.shape[1] != 3)
                throw py::value_error(
                    "effector arrays have inconsistent shapes");
            if (jointInfo.ndim != 1 || axisInfo.ndim != 2 ||
                axisInfo.shape[0] != jointInfo.shape[0] ||
                axisInfo.shape[1] != 3)
                throw py::value_error(
                    "controls expected shapes [C] and [C, 3]");

            const float* targetData = static_cast<const float*>(targetInfo.ptr);
            const int* effectorData = static_cast<const int*>(effectorInfo.ptr);
            const float* offsetData = static_cast<const float*>(offsetInfo.ptr);
            const int* jointData = static_cast<const int*>(jointInfo.ptr);
            const float* axisData = static_cast<const float*>(axisInfo.ptr);
            std::vector<float> targetValues(targetData,
                                            targetData + targetInfo.size);
            std::vector<IKEffector> effectorValues;
            effectorValues.reserve(static_cast<size_t>(count));
            for (py::ssize_t i = 0; i < count; ++i)
                effectorValues.push_back(
                    {effectorData[i],
                     Eigen::Vector3f(offsetData[i * 3], offsetData[i * 3 + 1],
                                     offsetData[i * 3 + 2])});
            std::vector<IKJointControl> controls;
            controls.reserve(static_cast<size_t>(jointInfo.size));
            for (py::ssize_t i = 0; i < jointInfo.size; ++i) {
                const int joint = jointData[i];
                auto existing =
                    std::find_if(controls.begin(), controls.end(),
                                 [joint](const IKJointControl& control) {
                                     return control.joint == joint;
                                 });
                if (existing == controls.end()) {
                    controls.push_back({joint, {}});
                    existing = std::prev(controls.end());
                }
                existing->axes.emplace_back(
                    axisData[i * 3], axisData[i * 3 + 1], axisData[i * 3 + 2]);
            }

            FullBodyIKConfig config;
            config.maxIterations = maxIterations;

            FullBodyIKBatchResult solved;
            {
                py::gil_scoped_release release;
                solved = solveFullBodyIKBatch(motion, targetValues,
                                              effectorValues, controls, config);
            }
            py::array_t<float> positions(
                {static_cast<py::ssize_t>(motion.numFrames()),
                 static_cast<py::ssize_t>(motion.numJoints()), py::ssize_t(3)});
            py::array_t<float> errors(
                {static_cast<py::ssize_t>(motion.numFrames()), count});
            py::array_t<int> iterations(
                static_cast<py::ssize_t>(motion.numFrames()));
            std::memcpy(positions.mutable_data(), solved.bodyPositions.data(),
                        solved.bodyPositions.size() * sizeof(float));
            std::memcpy(errors.mutable_data(), solved.finalErrors.data(),
                        solved.finalErrors.size() * sizeof(float));
            std::memcpy(iterations.mutable_data(), solved.iterations.data(),
                        solved.iterations.size() * sizeof(int));
            return py::make_tuple(std::move(solved.motion),
                                  std::move(positions), std::move(errors),
                                  std::move(iterations));
        },
        py::arg("motion"), py::arg("targets"), py::arg("effector_joints"),
        py::arg("effector_offsets"), py::arg("control_joints"),
        py::arg("control_axes"), py::arg("max_iterations") = 0,
        "Solve full-body IK for all frames in a motion batch.");

    anim.def(
        "solve_full_body_ik",
        [](const SkeletonState& state, const FloatArray& targets,
           py::array_t<int, py::array::c_style | py::array::forcecast>
               effectors,
           const FloatArray& offsets,
           py::array_t<int, py::array::c_style | py::array::forcecast>
               controlJoints,
           const FloatArray& controlAxes, int maxIterations) {
            const auto targetInfo = targets.request();
            const auto effectorInfo = effectors.request();
            const auto offsetInfo = offsets.request();
            const auto jointInfo = controlJoints.request();
            const auto axisInfo = controlAxes.request();
            if (targetInfo.ndim != 2 || targetInfo.shape[1] != 3)
                throw py::value_error("targets expected shape [effectors, 3]");
            const py::ssize_t count = targetInfo.shape[0];
            if (effectorInfo.ndim != 1 || effectorInfo.shape[0] != count ||
                offsetInfo.ndim != 2 || offsetInfo.shape[0] != count ||
                offsetInfo.shape[1] != 3)
                throw py::value_error(
                    "effector arrays have inconsistent shapes");
            if (jointInfo.ndim != 1 || axisInfo.ndim != 2 ||
                axisInfo.shape[0] != jointInfo.shape[0] ||
                axisInfo.shape[1] != 3)
                throw py::value_error(
                    "controls expected shapes [C] and [C, 3]");

            const auto* targetData = static_cast<const float*>(targetInfo.ptr);
            const auto* effectorData =
                static_cast<const int*>(effectorInfo.ptr);
            const auto* offsetData = static_cast<const float*>(offsetInfo.ptr);
            const auto* jointData = static_cast<const int*>(jointInfo.ptr);
            const auto* axisData = static_cast<const float*>(axisInfo.ptr);
            std::vector<Eigen::Vector3f> targetValues;
            std::vector<IKEffector> effectorValues;
            targetValues.reserve(static_cast<size_t>(count));
            effectorValues.reserve(static_cast<size_t>(count));
            for (py::ssize_t i = 0; i < count; ++i) {
                targetValues.emplace_back(targetData[i * 3],
                                          targetData[i * 3 + 1],
                                          targetData[i * 3 + 2]);
                effectorValues.push_back(
                    {effectorData[i],
                     Eigen::Vector3f(offsetData[i * 3], offsetData[i * 3 + 1],
                                     offsetData[i * 3 + 2])});
            }
            std::vector<IKJointControl> controls;
            controls.reserve(static_cast<size_t>(jointInfo.size));
            for (py::ssize_t i = 0; i < jointInfo.size; ++i) {
                const int joint = jointData[i];
                auto existing =
                    std::find_if(controls.begin(), controls.end(),
                                 [joint](const IKJointControl& control) {
                                     return control.joint == joint;
                                 });
                if (existing == controls.end()) {
                    controls.push_back({joint, {}});
                    existing = std::prev(controls.end());
                }
                existing->axes.emplace_back(
                    axisData[i * 3], axisData[i * 3 + 1], axisData[i * 3 + 2]);
            }
            FullBodyIKConfig config;
            config.maxIterations = maxIterations;
            FullBodyIKResult solved;
            {
                py::gil_scoped_release release;
                solved = solveFullBodyIK(state, targetValues, effectorValues,
                                         controls, config);
            }
            py::array_t<float> positions(
                {static_cast<py::ssize_t>(state.numJoints()), py::ssize_t(3)});
            py::array_t<float> errors(count);
            std::memcpy(positions.mutable_data(), solved.bodyPositions.data(),
                        solved.bodyPositions.size() * sizeof(float));
            std::memcpy(errors.mutable_data(), solved.finalErrors.data(),
                        solved.finalErrors.size() * sizeof(float));
            return py::make_tuple(std::move(solved.state), std::move(positions),
                                  std::move(errors), solved.iterations);
        },
        py::arg("state"), py::arg("targets"), py::arg("effector_joints"),
        py::arg("effector_offsets"), py::arg("control_joints"),
        py::arg("control_axes"), py::arg("max_iterations") = 0,
        "Solve full-body IK for one skeleton state.");

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
            [](std::shared_ptr<SkeletonTree> tree,
               const FloatArray& rotationsWxyz,
               const FloatArray& rootTranslation, bool isLocal) {
                auto rot = eigenQuatWxyzArray(rotationsWxyz, "rotations_wxyz");
                if (static_cast<int>(rot.size()) != tree->numJoints()) {
                    throw py::value_error(
                        "rotations must have shape [num_joints, 4]");
                }
                return SkeletonState::fromRotationAndRootTranslation(
                    tree, rot,
                    eigenVec3FromArray(rootTranslation, "root_translation"),
                    isLocal);
            },
            py::arg("tree"), py::arg("rotations_wxyz"),
            py::arg("root_translation"), py::arg("is_local") = true,
            "Create a pose from WXYZ rotations and root translation.")
        .def("num_joints", &SkeletonState::numJoints,
             "Return the number of joints.")
        .def_property_readonly(
            "skeleton_tree",
            [](const SkeletonState& self) {
                return std::const_pointer_cast<SkeletonTree>(
                    self.skeletonTreePtr());
            },
            "Return the state's read-only skeleton hierarchy.")
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
               const std::string& order, const std::string& meshAssetBasePath,
               const std::string& targetCoordinateSystem) {
                return ArticulationVisualBridge::fromMJCF(
                    mjcfPath, scene, primBasePath, scale, order,
                    meshAssetBasePath,
                    Utils::coordinateSystemFromString(targetCoordinateSystem));
            },
            py::arg("mjcf_path"), py::arg("scene"), py::arg("path") = "/robot",
            py::arg("scale") = 1.0f, py::arg("order") = "DFS",
            py::arg("mesh_asset_base_path") = "",
            py::arg("target_coordinate_system") = "z_up_x_forward",
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
        .def("render_prim_body_indices",
             &ArticulationVisualBridge::renderPrimBodyIndices,
             py::return_value_policy::reference_internal,
             "Return body index for each render prim.")
        .def("num_bodies", &ArticulationVisualBridge::numBodies,
             "Return the number of bridged bodies.");

    py::class_<ArticulationVisualBridgeAsset>(
        anim, "ArticulationVisualAsset",
        "Reusable articulated rigid-link visual asset that can instantiate "
        "scene prims.")
        .def_static(
            "from_mjcf",
            [](const std::string& path, float scale, const std::string& order,
               const std::string& targetCoordinateSystem) {
                return ArticulationVisualBridgeAsset::fromMJCF(
                    path, scale, order,
                    Utils::coordinateSystemFromString(targetCoordinateSystem));
            },
            py::arg("mjcf_path"), py::arg("scale") = 1.0f,
            py::arg("order") = "DFS",
            py::arg("target_coordinate_system") = "z_up_x_forward",
            "Load reusable bridge asset data from an MJCF file.")
        .def("define_mesh_assets",
             &ArticulationVisualBridgeAsset::defineMeshAssets, py::arg("scene"),
             py::arg("mesh_asset_base_path"),
             py::arg("split_visual_geoms") = false,
             "Define shared mesh asset prims in a scene.")
        .def("instantiate", &ArticulationVisualBridgeAsset::instantiate,
             py::arg("scene"), py::arg("path") = "/robot",
             py::arg("mesh_asset_base_path") = "",
             py::arg("split_visual_geoms") = false,
             "Instantiate this asset into a scene.")
        .def("num_bodies", &ArticulationVisualBridgeAsset::numBodies,
             "Return the number of bodies in this asset.");

    py::class_<SkeletalVisualConfig>(
        anim, "SkeletalVisualConfig",
        "Style settings for SkeletalVisual line/joint rendering.")
        .def(py::init([](const std::optional<glm::vec4>& boneColor,
                         const std::optional<glm::vec4>& jointColor,
                         float boneRadius, float jointRadius, int segments,
                         bool visible, bool showJoints) {
                 SkeletalVisualConfig config;
                 if (boneColor)
                     config.boneColor = *boneColor;
                 if (jointColor)
                     config.jointColor = *jointColor;
                 config.boneRadius = boneRadius;
                 config.jointRadius = jointRadius;
                 config.segments = segments;
                 config.visible = visible;
                 config.showJoints = showJoints;
                 return config;
             }),
             py::kw_only(), py::arg("bone_color") = py::none(),
             py::arg("joint_color") = py::none(),
             py::arg("bone_radius") = SkeletalVisualConfig{}.boneRadius,
             py::arg("joint_radius") = SkeletalVisualConfig{}.jointRadius,
             py::arg("segments") = SkeletalVisualConfig{}.segments,
             py::arg("visible") = SkeletalVisualConfig{}.visible,
             py::arg("show_joints") = SkeletalVisualConfig{}.showJoints,
             "Create skeleton visual settings from keyword fields.")
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
                       "Whether joint markers should be visible.")
        .def("__repr__", [](const SkeletalVisualConfig& config) {
            return py::str(
                       "SkeletalVisualConfig(bone_color=({:g}, {:g}, {:g}, "
                       "{:g}), joint_color=({:g}, {:g}, {:g}, {:g}), "
                       "bone_radius={:g}, joint_radius={:g}, segments={!r}, "
                       "visible={!r}, show_joints={!r})")
                .attr("format")(
                    config.boneColor.x, config.boneColor.y, config.boneColor.z,
                    config.boneColor.w, config.jointColor.x,
                    config.jointColor.y, config.jointColor.z,
                    config.jointColor.w, config.boneRadius, config.jointRadius,
                    config.segments, config.visible, config.showJoints);
        });

    py::class_<SkeletalVisualBridge>(
        anim, "SkeletalVisual",
        "Instanced line/point renderer for skeleton poses and motion clips.")
        .def(py::init<>(), "Create an empty skeletal visual.")
        .def_static(
            "define",
            [](App* app, Material* material, const std::string& basePath,
               const SkeletonState& state, const SkeletalVisualConfig& config) {
                return SkeletalVisualBridge::define(app, material, basePath,
                                                    state, config);
            },
            py::arg("app"), py::arg("material"), py::arg("path"),
            py::arg("state"), py::arg("config") = SkeletalVisualConfig{},
            "Create skeleton visuals from a SkeletonState.")
        .def_static(
            "define",
            [](App* app, Material* material, const std::string& basePath,
               const SkeletonMotion& motion, float time, bool loop,
               const SkeletalVisualConfig& config) {
                return SkeletalVisualBridge::define(app, material, basePath,
                                                    motion, time, loop, config);
            },
            py::arg("app"), py::arg("material"), py::arg("path"),
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
        .def("remove", &SkeletalVisualBridge::remove,
             "Remove all authored skeleton visual scene objects.")
        .def("bone_handle", &SkeletalVisualBridge::boneHandle,
             "Return the renderable handle used for bones.")
        .def("joint_handle", &SkeletalVisualBridge::jointHandle,
             "Return the renderable handle used for joints.");
}
