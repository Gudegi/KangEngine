///
/// Scene Backend Python Bindings
/// With USD Stage access for Python OpenUSD integration
///

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include "engine/scene/scene_backend.hpp"
#include "engine/core/app/app.hpp"
#include "engine/graphics/backend/base/graphics_device.hpp"
#include "engine/scene/debug_draw.hpp"
#include "engine/scene/native/prim.hpp"
#include "engine/scene/native/token.hpp"
#include "py_array_view.hpp"

#ifdef KANGENGINE_USE_USD
#include "engine/scene/usd/usd_scene.hpp"
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usd/prim.h>
#endif

namespace py = pybind11;


void bind_scene(py::module& m) {
    py::module scene =
        m.def_submodule("scene",
                        "Scene graph, prim, mesh, and debug drawing APIs.");

    // Token class
    py::class_<KE::Scene::Token>(
        scene, "Token",
        "Interned-style attribute key used by scene prims.")
        .def(py::init<>(), "Create an empty token.")
        .def(py::init<const std::string&>(), py::arg("value"),
             "Create a token from a string.")
        .def("id", &KE::Scene::Token::id,
             "Return the token's stable numeric id.")
        .def("str", &KE::Scene::Token::str,
             "Return the token string.")
        .def("__eq__", &KE::Scene::Token::operator==)
        .def("__ne__", &KE::Scene::Token::operator!=)
        .def("__repr__", [](const KE::Scene::Token& t) {
            return "Token('" + t.str() + "')";
        });

    // PrimType enum
    py::enum_<KE::Scene::PrimType>(
        scene, "PrimType",
        "Scene prim categories used by KangEngine's scene graph.")
        .value("Root", KE::Scene::PrimType::Root)
        .value("Xform", KE::Scene::PrimType::Xform)
        .value("Mesh", KE::Scene::PrimType::Mesh)
        .value("MeshInstance", KE::Scene::PrimType::MeshInstance)
        .value("Camera", KE::Scene::PrimType::Camera)
        .value("Light", KE::Scene::PrimType::Light)
        .export_values();

    py::enum_<KE::Scene::ManipulationPolicy>(
        scene, "ManipulationPolicy",
        "How viewport manipulation resolves from a picked prim.")
        .value("Inherit", KE::Scene::ManipulationPolicy::Inherit)
        .value("Self", KE::Scene::ManipulationPolicy::Self)
        .value("Parent", KE::Scene::ManipulationPolicy::Parent)
        .value("Root", KE::Scene::ManipulationPolicy::Root)
        .value("Disabled", KE::Scene::ManipulationPolicy::Disabled)
        .export_values();

    // Prim class
    py::class_<KE::Scene::Prim>(
        scene, "Prim",
        "USD-like scene graph node with hierarchy, transform, and mesh data.")
        .def(py::init<const std::string&, KE::Scene::PrimType,
                      KE::Scene::Prim*>(),
             py::arg("name"), py::arg("type"), py::arg("parent") = nullptr,
             "Create a prim with an optional parent.")
        // Getters
        .def("get_name", &KE::Scene::Prim::getName,
             "Return this prim's local name.")
        .def("get_path", &KE::Scene::Prim::getPath,
             "Return this prim's absolute scene path.")
        .def("get_type", &KE::Scene::Prim::getType,
             "Return this prim's type.")
        .def("get_parent", &KE::Scene::Prim::getParent,
             py::return_value_policy::reference,
             "Return this prim's parent, or None for the root.")
        // Hierarchy
        .def("add_child", &KE::Scene::Prim::addChild, py::arg("name"),
             py::arg("type"), py::return_value_policy::reference,
             "Create and return a child prim.")
        .def("get_child", &KE::Scene::Prim::getChild, py::arg("name"),
             py::return_value_policy::reference,
             "Return a direct child by name.")
        .def("get_prim_at_path", &KE::Scene::Prim::getPrimAtPath,
             py::arg("path"), py::return_value_policy::reference,
             "Find a descendant prim by absolute path.")
        .def("get_children", &KE::Scene::Prim::getChildren,
             "Return direct child prims.")
        // Mesh data
        .def("set_mesh_data", &KE::Scene::Prim::setMeshData,
             py::arg("mesh_data"), "Attach mesh data to this prim.")
        .def("get_mesh_data", &KE::Scene::Prim::getMeshData,
             "Return mesh data attached directly to this prim.")
        .def("set_mesh_source_path", &KE::Scene::Prim::setMeshSourcePath,
             py::arg("path"), "Set a source path for mesh instancing.")
        .def("get_mesh_source_path", &KE::Scene::Prim::getMeshSourcePath,
             "Return the mesh source path for mesh instances.")
        .def("resolve_mesh_data", &KE::Scene::Prim::resolveMeshData,
             "Return direct mesh data or resolved source mesh data.")
        // Static mesh creation (returns shared_ptr for set_mesh_data
        // compatibility)
        .def_static("create_square_data",
                    [](float scale) {
                        return std::make_shared<KE::Scene::MeshData>(
                            KE::Scene::Prim::createSquareData(scale));
                    },
                    py::arg("scale") = 1.0f,
                    "Create square mesh data.")
        .def_static(
            "create_plane_data",
            [](float scale, KE::UpAxis upAxis) {
                return std::make_shared<KE::Scene::MeshData>(
                    KE::Scene::Prim::createPlaneData(scale, upAxis));
            },
            py::arg("scale"), py::arg("up_axis") = KE::UpAxis::Y,
            "Create plane mesh data.")
        .def_static(
            "create_sphere_data",
            [](float radius, int numLongitudes, int numLatitudes) {
                return std::make_shared<KE::Scene::MeshData>(
                    KE::Scene::Prim::createSphereData(radius, numLongitudes,
                                                      numLatitudes));
            },
            py::arg("radius"), py::arg("num_longitudes"),
            py::arg("num_latitudes"), "Create sphere mesh data.")
        .def_static(
            "create_rectangle_data",
            [](float xScale, float yScale, float zScale) {
                return std::make_shared<KE::Scene::MeshData>(
                    KE::Scene::Prim::createRectangleData(xScale, yScale,
                                                         zScale));
            },
            py::arg("x_scale"), py::arg("y_scale"), py::arg("z_scale"),
            "Create box/rectangle mesh data.")
        .def_static(
            "create_cylinder_data",
            [](float radius, float length, KE::UpAxis upAxis, int segments) {
                return std::make_shared<KE::Scene::MeshData>(
                    KE::Scene::Prim::createCylinderData(radius, length, upAxis,
                                                        segments));
            },
            py::arg("radius"), py::arg("length"),
            py::arg("up_axis") = KE::UpAxis::Y, py::arg("segments") = 32,
            "Create cylinder mesh data.")
        .def_static(
            "create_capsule_data",
            [](float radius, float height, KE::UpAxis upAxis, int segments) {
                return std::make_shared<KE::Scene::MeshData>(
                    KE::Scene::Prim::createCapsuleData(radius, height, upAxis,
                                                       segments));
            },
            py::arg("radius"), py::arg("height"),
            py::arg("up_axis") = KE::UpAxis::Y, py::arg("segments") = 32,
            "Create capsule mesh data.")
        .def_static("define_manipulation_group",
                    &KE::Scene::Prim::defineManipulationGroup,
                    py::arg("scene"), py::arg("path"),
                    py::return_value_policy::reference,
                    "Create an Xform prim that manipulates as a group.")
        // Transform ops
        .def("set_local_translation", &KE::Scene::Prim::setLocalTranslation,
             py::arg("translation"), "Set local translation.")
        .def("set_local_scale", &KE::Scene::Prim::setLocalScale,
             py::arg("scale"), "Set local scale.")
        .def("set_local_rotation", &KE::Scene::Prim::setLocalRotation,
             py::arg("rotation"), "Set local quaternion rotation.")
        .def("set_local_matrix", &KE::Scene::Prim::setLocalMatrix,
             py::arg("matrix"), "Set local transform matrix.")
        .def("set_world_translation", &KE::Scene::Prim::setWorldTranslation,
             py::arg("translation"), "Set world translation.")
        .def("set_world_rotation", &KE::Scene::Prim::setWorldRotation,
             py::arg("rotation"), "Set world quaternion rotation.")
        .def("set_world_matrix", &KE::Scene::Prim::setWorldMatrix,
             py::arg("matrix"), "Set world transform matrix.")
        .def("add_translate_op", &KE::Scene::Prim::addTranslateOp,
             py::arg("translation"), "Compatibility alias for local translation.")
        .def("add_scale_op", &KE::Scene::Prim::addScaleOp, py::arg("scale"),
             "Compatibility alias for local scale.")
        .def("add_rotate_quaternion_op",
             &KE::Scene::Prim::addRotateQuaternionOp, py::arg("rotation"),
             "Compatibility alias for local quaternion rotation.")
        // Display color
        .def("set_display_color", &KE::Scene::Prim::setDisplayColor,
             py::arg("color"), "Set RGB display color metadata.")
        .def("get_display_color", &KE::Scene::Prim::getDisplayColor,
             "Return RGB display color metadata if present.")
        .def("set_display_color_alpha", &KE::Scene::Prim::setDisplayColorAlpha,
             py::arg("color"), "Set RGBA display color metadata.")
        .def("get_display_color_alpha", &KE::Scene::Prim::getDisplayColorAlpha,
             "Return RGBA display color metadata if present.")
        // Model matrix
        .def("compute_model_matrix", &KE::Scene::Prim::computeModelMatrix,
             "Compute this prim's model matrix.")
        .def("compute_local_matrix", &KE::Scene::Prim::computeLocalMatrix,
             "Compute this prim's local transform matrix.")
        .def("compute_world_matrix", &KE::Scene::Prim::computeWorldMatrix,
             "Compute this prim's world transform matrix.")
        .def("is_visible", &KE::Scene::Prim::isVisible,
             "Return local visibility state.")
        .def("set_visible", &KE::Scene::Prim::setVisible, py::arg("visible"),
             "Set local visibility state.")
        .def("is_visible_in_hierarchy",
             &KE::Scene::Prim::isVisibleInHierarchy,
             "Return visibility after parent hierarchy is considered.")
        .def("is_active", &KE::Scene::Prim::isActive,
             "Return local active state.")
        .def("set_active", &KE::Scene::Prim::setActive, py::arg("active"),
             "Set local active state.")
        .def("is_active_in_hierarchy", &KE::Scene::Prim::isActiveInHierarchy,
             "Return active state after parent hierarchy is considered.")
        .def("get_manipulation_policy",
             &KE::Scene::Prim::getManipulationPolicy,
             "Return this prim's manipulation policy.")
        .def("set_manipulation_policy",
             &KE::Scene::Prim::setManipulationPolicy, py::arg("policy"),
             "Set this prim's manipulation policy.")
        .def("resolve_manipulation_target",
             py::overload_cast<>(&KE::Scene::Prim::resolveManipulationTarget),
             py::return_value_policy::reference,
             "Resolve which prim should be manipulated from this prim.")
        // setAttribute with specific types
        .def("set_attribute_vec3",
             [](KE::Scene::Prim& self, const std::string& name,
                const glm::vec3& value) { self.setAttribute(name, value); },
             py::arg("name"), py::arg("value"), "Set a vec3 attribute.")
        .def("set_attribute_vec4",
             [](KE::Scene::Prim& self, const std::string& name,
                const glm::vec4& value) { self.setAttribute(name, value); },
             py::arg("name"), py::arg("value"), "Set a vec4 attribute.")
        .def("set_attribute_quat",
             [](KE::Scene::Prim& self, const std::string& name,
                const glm::quat& value) { self.setAttribute(name, value); },
             py::arg("name"), py::arg("value"),
             "Set a quaternion attribute.")
        .def("set_attribute_float",
             [](KE::Scene::Prim& self, const std::string& name, float value) {
                 self.setAttribute(name, value);
             },
             py::arg("name"), py::arg("value"), "Set a float attribute.")
        .def("set_attribute_int",
             [](KE::Scene::Prim& self, const std::string& name, int value) {
                 self.setAttribute(name, value);
             },
             py::arg("name"), py::arg("value"), "Set an integer attribute.")
        .def("set_attribute_string",
             [](KE::Scene::Prim& self, const std::string& name,
                const std::string& value) { self.setAttribute(name, value); },
             py::arg("name"), py::arg("value"), "Set a string attribute.")
        // getAttribute with specific types
        .def("get_attribute_vec3",
             [](KE::Scene::Prim& self, const std::string& name) {
                 return self.getAttribute<glm::vec3>(name);
             },
             py::arg("name"), "Get a vec3 attribute.")
        .def("get_attribute_vec4",
             [](KE::Scene::Prim& self, const std::string& name) {
                 return self.getAttribute<glm::vec4>(name);
             },
             py::arg("name"), "Get a vec4 attribute.")
        .def("get_attribute_quat",
             [](KE::Scene::Prim& self, const std::string& name) {
                 return self.getAttribute<glm::quat>(name);
             },
             py::arg("name"), "Get a quaternion attribute.")
        .def("get_attribute_float",
             [](KE::Scene::Prim& self, const std::string& name) {
                 return self.getAttribute<float>(name);
             },
             py::arg("name"), "Get a float attribute.")
        .def("get_attribute_int",
             [](KE::Scene::Prim& self, const std::string& name) {
                 return self.getAttribute<int>(name);
             },
             py::arg("name"), "Get an integer attribute.")
        .def("get_attribute_string",
             [](KE::Scene::Prim& self, const std::string& name) {
                 return self.getAttribute<std::string>(name);
             },
             py::arg("name"), "Get a string attribute.")
        .def("has_attribute", py::overload_cast<const std::string&>(
                                  &KE::Scene::Prim::hasAttribute, py::const_),
             py::arg("name"), "Return true when an attribute exists.")
        .def("traverse", &KE::Scene::Prim::traverse, py::arg("callback"),
             "Traverse this prim subtree and call callback for each prim.");

    // BackendType enum
    py::enum_<KE::Scene::BackendType>(
        scene, "BackendType",
        "Scene backend implementation type.")
        .value("Native", KE::Scene::BackendType::Native)
        .value("USD", KE::Scene::BackendType::USD)
        .export_values();

    // MeshData struct (with shared_ptr holder for set_mesh_data compatibility)
    py::class_<KE::Scene::MeshData, std::shared_ptr<KE::Scene::MeshData>>(
        scene, "MeshData",
        "Static mesh payload with vertex attributes and triangle indices.")
        .def(py::init<>(), "Create empty mesh data.")
        .def_readwrite("vertices", &KE::Scene::MeshData::vertices,
                       "Vertex positions.")
        .def_readwrite("normals", &KE::Scene::MeshData::normals,
                       "Vertex normals.")
        .def_readwrite("uvs", &KE::Scene::MeshData::uvs,
                       "Vertex texture coordinates.")
        .def_readwrite("indices", &KE::Scene::MeshData::indices,
                       "Triangle index buffer.");

    py::class_<KE::Scene::SkinnedMeshData,
               std::shared_ptr<KE::Scene::SkinnedMeshData>>(
        scene, "SkinnedMeshData",
        "Mesh payload with skinning attributes for skeletal animation.")
        .def(py::init<>(), "Create empty skinned mesh data.")
        .def_readwrite("mesh", &KE::Scene::SkinnedMeshData::mesh,
                       "Base static mesh data.")
        .def_readwrite("bone_node_indices",
                       &KE::Scene::SkinnedMeshData::boneNodeIndices,
                       "Skeleton node index for each bound bone.")
        .def("has_valid_vertex_skinning",
             &KE::Scene::SkinnedMeshData::hasValidVertexSkinning,
             "Return true when vertex bone data matches the mesh vertex count.");

    py::class_<KE::Scene::DebugDraw>(
        scene, "DebugDraw",
        "Helpers for creating debug lines, arrows, and coordinate axes.")
        .def_static(
            "log_lines",
            [](KE::App* app, KE::Backend::Shader* shader,
               const std::string& path, const FloatArray& starts,
               const FloatArray& ends, const FloatArray& colors, float radius,
               int segments) {
                auto s = vec3ArrayView(starts, "starts");
                auto e = vec3ArrayView(ends, "ends");
                auto c = vec4ArrayView(colors, "colors");
                if (s.count != e.count) {
                    throw py::value_error(
                        "starts and ends must have the same length");
                }
                return KE::Scene::DebugDraw::logLines(
                    app, shader, path, s.data, e.data, c.data, s.count,
                    c.count, radius, segments);
            },
            py::arg("app"), py::arg("shader"), py::arg("path"),
            py::arg("starts"), py::arg("ends"), py::arg("colors"),
            py::arg("radius") = 0.005f, py::arg("segments") = 8,
            "Create instanced debug line geometry from numpy arrays.")
        .def_static(
            "log_lines",
            [](KE::App* app, KE::Backend::Shader* shader,
               const std::string& path, const std::vector<glm::vec3>& starts,
               const std::vector<glm::vec3>& ends,
               const std::vector<glm::vec4>& colors, float radius,
               int segments) {
                return KE::Scene::DebugDraw::logLines(
                    app, shader, path, starts, ends, colors, radius, segments);
            },
            py::arg("app"), py::arg("shader"), py::arg("path"),
            py::arg("starts"), py::arg("ends"), py::arg("colors"),
            py::arg("radius") = 0.005f, py::arg("segments") = 8,
            "Create instanced debug line geometry.")
        .def_static(
            "update_lines",
            [](KE::App* app, uint32_t handle, const FloatArray& starts,
               const FloatArray& ends, const FloatArray& colors) {
                auto s = vec3ArrayView(starts, "starts");
                auto e = vec3ArrayView(ends, "ends");
                auto c = vec4ArrayView(colors, "colors");
                if (s.count != e.count) {
                    throw py::value_error(
                        "starts and ends must have the same length");
                }
                KE::Scene::DebugDraw::updateLines(app, handle, s.data, e.data,
                                                  c.data, s.count, c.count);
            },
            py::arg("app"), py::arg("handle"), py::arg("starts"),
            py::arg("ends"), py::arg("colors"),
            "Update existing debug line geometry from numpy arrays.")
        .def_static(
            "update_lines",
            [](KE::App* app, uint32_t handle,
               const std::vector<glm::vec3>& starts,
               const std::vector<glm::vec3>& ends,
               const std::vector<glm::vec4>& colors) {
                KE::Scene::DebugDraw::updateLines(app, handle, starts, ends,
                                                  colors);
            },
            py::arg("app"), py::arg("handle"), py::arg("starts"),
            py::arg("ends"), py::arg("colors"),
            "Update existing debug line geometry.")
        .def_static(
            "log_arrows",
            [](KE::App* app, KE::Backend::Shader* shader,
               const std::string& path, const FloatArray& starts,
               const FloatArray& ends, const FloatArray& colors, float radius,
               int segments) {
                auto s = vec3ArrayView(starts, "starts");
                auto e = vec3ArrayView(ends, "ends");
                auto c = vec4ArrayView(colors, "colors");
                if (s.count != e.count) {
                    throw py::value_error(
                        "starts and ends must have the same length");
                }
                return KE::Scene::DebugDraw::logArrows(
                    app, shader, path, s.data, e.data, c.data, s.count,
                    c.count, radius, segments);
            },
            py::arg("app"), py::arg("shader"), py::arg("path"),
            py::arg("starts"), py::arg("ends"), py::arg("colors"),
            py::arg("radius") = 0.02f, py::arg("segments") = 12,
            "Create instanced debug arrow geometry from numpy arrays.")
        .def_static(
            "log_arrows",
            [](KE::App* app, KE::Backend::Shader* shader,
               const std::string& path, const std::vector<glm::vec3>& starts,
               const std::vector<glm::vec3>& ends,
               const std::vector<glm::vec4>& colors, float radius,
               int segments) {
                return KE::Scene::DebugDraw::logArrows(
                    app, shader, path, starts, ends, colors, radius, segments);
            },
            py::arg("app"), py::arg("shader"), py::arg("path"),
            py::arg("starts"), py::arg("ends"), py::arg("colors"),
            py::arg("radius") = 0.02f, py::arg("segments") = 12,
            "Create instanced debug arrow geometry.")
        .def_static(
            "update_arrows",
            [](KE::App* app, uint32_t handle, const FloatArray& starts,
               const FloatArray& ends, const FloatArray& colors) {
                auto s = vec3ArrayView(starts, "starts");
                auto e = vec3ArrayView(ends, "ends");
                auto c = vec4ArrayView(colors, "colors");
                if (s.count != e.count) {
                    throw py::value_error(
                        "starts and ends must have the same length");
                }
                KE::Scene::DebugDraw::updateArrows(app, handle, s.data, e.data,
                                                   c.data, s.count, c.count);
            },
            py::arg("app"), py::arg("handle"), py::arg("starts"),
            py::arg("ends"), py::arg("colors"),
            "Update existing debug arrow geometry from numpy arrays.")
        .def_static(
            "update_arrows",
            [](KE::App* app, uint32_t handle,
               const std::vector<glm::vec3>& starts,
               const std::vector<glm::vec3>& ends,
               const std::vector<glm::vec4>& colors) {
                KE::Scene::DebugDraw::updateArrows(app, handle, starts, ends,
                                                   colors);
            },
            py::arg("app"), py::arg("handle"), py::arg("starts"),
            py::arg("ends"), py::arg("colors"),
            "Update existing debug arrow geometry.")
        .def_static(
            "log_coordinate_axes",
            [](KE::App* app, KE::Backend::Shader* shader,
               const std::string& path, glm::vec3 origin, glm::quat orientation,
               float length, float radius, int segments) {
                return KE::Scene::DebugDraw::logCoordinateAxes(
                    app, shader, path, origin, orientation, length, radius,
                    segments);
            },
            py::arg("app"), py::arg("shader"), py::arg("path"),
            py::arg("origin"), py::arg("orientation"), py::arg("length") = 1.0f,
            py::arg("radius") = 0.005f, py::arg("segments") = 8,
            "Create instanced XYZ coordinate axes.")
        .def_static(
            "log_scene_lines",
            [](KE::Scene::SceneBackend* sceneBackend,
               const std::string& basePath, const FloatArray& starts,
               const FloatArray& ends, const FloatArray& colors, float radius,
               int segments) {
                return KE::Scene::DebugDraw::logLines(
                    sceneBackend, basePath, vec3Array(starts, "starts"),
                    vec3Array(ends, "ends"), vec4Array(colors, "colors"),
                    radius, segments);
            },
            py::arg("scene"), py::arg("base_path"), py::arg("starts"),
            py::arg("ends"), py::arg("colors"), py::arg("radius") = 0.005f,
            py::arg("segments") = 8, py::return_value_policy::reference,
            "Create scene-backed line prims from numpy arrays.")
        .def_static(
            "log_scene_lines",
            [](KE::Scene::SceneBackend* sceneBackend,
               const std::string& basePath,
               const std::vector<glm::vec3>& starts,
               const std::vector<glm::vec3>& ends,
               const std::vector<glm::vec4>& colors, float radius,
               int segments) {
                return KE::Scene::DebugDraw::logLines(sceneBackend, basePath,
                                                      starts, ends, colors,
                                                      radius, segments);
            },
            py::arg("scene"), py::arg("base_path"), py::arg("starts"),
            py::arg("ends"), py::arg("colors"), py::arg("radius") = 0.005f,
            py::arg("segments") = 8, py::return_value_policy::reference,
            "Create scene-backed line prims.")
        .def_static(
            "log_scene_arrows",
            [](KE::Scene::SceneBackend* sceneBackend,
               const std::string& basePath, const FloatArray& starts,
               const FloatArray& ends, const FloatArray& colors, float radius,
               int segments) {
                return KE::Scene::DebugDraw::logArrows(
                    sceneBackend, basePath, vec3Array(starts, "starts"),
                    vec3Array(ends, "ends"), vec4Array(colors, "colors"),
                    radius, segments);
            },
            py::arg("scene"), py::arg("base_path"), py::arg("starts"),
            py::arg("ends"), py::arg("colors"), py::arg("radius") = 0.02f,
            py::arg("segments") = 12, py::return_value_policy::reference,
            "Create scene-backed arrow prims from numpy arrays.")
        .def_static(
            "log_scene_arrows",
            [](KE::Scene::SceneBackend* sceneBackend,
               const std::string& basePath,
               const std::vector<glm::vec3>& starts,
               const std::vector<glm::vec3>& ends,
               const std::vector<glm::vec4>& colors, float radius,
               int segments) {
                return KE::Scene::DebugDraw::logArrows(sceneBackend, basePath,
                                                       starts, ends, colors,
                                                       radius, segments);
            },
            py::arg("scene"), py::arg("base_path"), py::arg("starts"),
            py::arg("ends"), py::arg("colors"), py::arg("radius") = 0.02f,
            py::arg("segments") = 12, py::return_value_policy::reference,
            "Create scene-backed arrow prims.")
        .def_static(
            "log_scene_coordinate_axes",
            [](KE::Scene::SceneBackend* sceneBackend,
               const std::string& basePath, glm::vec3 origin,
               glm::quat orientation, float length, float radius,
               int segments) {
                return KE::Scene::DebugDraw::logCoordinateAxes(
                    sceneBackend, basePath, origin, orientation, length, radius,
                    segments);
            },
            py::arg("scene"), py::arg("base_path"), py::arg("origin"),
            py::arg("orientation"), py::arg("length") = 1.0f,
            py::arg("radius") = 0.005f, py::arg("segments") = 8,
            py::return_value_policy::reference,
            "Create scene-backed XYZ coordinate axes.");

    // SceneBackend interface
    py::class_<KE::Scene::SceneBackend>(
        scene, "SceneBackend",
        "Abstract scene backend interface for native and USD scenes.")
        .def("get_backend_type", &KE::Scene::SceneBackend::getBackendType,
             "Return the backend implementation type.")
        .def("load_scene", &KE::Scene::SceneBackend::loadScene,
             py::arg("path"), "Load a scene from a file.")
        .def("save_scene", &KE::Scene::SceneBackend::saveScene,
             py::arg("path"), "Save a scene to a file.")
        .def("load_mesh", &KE::Scene::SceneBackend::loadMesh,
             py::arg("prim_path"), "Load mesh data for a prim path.")
        .def("list_meshes", &KE::Scene::SceneBackend::listMeshes,
             "Return mesh prim paths known by this scene.")
        .def("define_prim", &KE::Scene::SceneBackend::definePrim,
             py::arg("path"), py::arg("type"),
             py::return_value_policy::reference,
             "Define and return a prim at a scene path.")
        .def("get_root_prim", &KE::Scene::SceneBackend::getRootPrim,
             py::return_value_policy::reference,
             "Return the scene root prim.");

#ifdef KANGENGINE_USE_USD
    // USDScene with direct USD Stage access!
    py::class_<KE::Scene::USDScene, KE::Scene::SceneBackend>(scene, "USDScene")
        .def(py::init<>(), "Create a new USD Scene")

        // Backend interface
        .def("get_backend_type", &KE::Scene::USDScene::getBackendType)
        .def("load_scene", &KE::Scene::USDScene::loadScene)
        .def("save_scene", &KE::Scene::USDScene::saveScene)
        .def("load_mesh", &KE::Scene::USDScene::loadMesh)
        .def("list_meshes", &KE::Scene::USDScene::listMeshes)

        // USD-specific API
        .def("create_new", &KE::Scene::USDScene::createNew,
             "Create new in-memory USD stage")
        // Note: get_prim, create_* methods return UsdPrim which requires
        // custom type casters. For now, use them for side effects only.
        .def(
            "create_xform",
            [](KE::Scene::USDScene& self, const std::string& path) {
                self.createXform(path);
            },
            "Create Xform (transform/group) at path")
        .def(
            "create_mesh",
            [](KE::Scene::USDScene& self, const std::string& path) {
                self.createMesh(path);
            },
            "Create Mesh at path")
        .def(
            "create_sphere",
            [](KE::Scene::USDScene& self, const std::string& path,
               double radius) { self.createSphere(path, radius); },
            "Create Sphere at path", py::arg("path"), py::arg("radius") = 1.0)
        .def(
            "create_cube",
            [](KE::Scene::USDScene& self, const std::string& path,
               double size) { self.createCube(path, size); },
            "Create Cube at path", py::arg("path"), py::arg("size") = 1.0)
        .def("print_hierarchy", &KE::Scene::USDScene::printHierarchy,
             "Print scene hierarchy (debug)");

    // NOTE: get_stage() would require custom USD type casters to work.
    // For now, use file-based workflow:
    //   1. Create scene with Python USD → Save to file
    //   2. Load file in KangEngine → Render
    //   3. Export from KangEngine → Edit with Python USD
    // This is actually the recommended workflow for production pipelines!
#endif

    // Factory
    scene.def("create_backend", &KE::Scene::SceneFactory::createBackend,
              "Create a scene backend", py::arg("type"));

    // Helper function
    scene.def(
        "load_mesh_from_usd",
        [](const std::string& usd_path,
           const std::string& prim_path) -> KE::Scene::MeshData {
#ifdef KANGENGINE_USE_USD
            auto backend = KE::Scene::SceneFactory::createBackend(
                KE::Scene::BackendType::USD);
            backend->loadScene(usd_path);
            return backend->loadMesh(prim_path);
#else
            throw std::runtime_error("USD support not compiled");
#endif
        },
        py::arg("usd_path"), py::arg("prim_path"),
        "Load mesh from USD file (helper function)");

    // USD 지원 확인
    scene.def("has_usd_support", []() -> bool {
#ifdef KANGENGINE_USE_USD
        return true;
#else
        return false;
#endif
    });
}
