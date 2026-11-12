// Scene registration and debug drawing require graphics support.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "engine/core/app/app.hpp"
#include "engine/scene/debug_draw.hpp"
#include "engine/scene/component/scene_render_system.hpp"
#include "engine/scene/component/render_component.hpp"
#include "py_array_view.hpp"

namespace py = pybind11;

void bind_scene_render(py::module& m) {
    py::module scene = m.attr("scene");

    py::class_<KE::Scene::SceneRenderSystem>(
        scene, "SceneRenderSystem",
        "Registry connecting scene RenderComponents to renderer resources. "
        "This hides raw renderable handles for normal scene use.")
        .def_property_readonly(
            "registration_count",
            &KE::Scene::SceneRenderSystem::registrationCount,
            "Return the number of registered render components.")
        .def("is_registered", &KE::Scene::SceneRenderSystem::isRegistered,
             py::arg("component"),
             "Return whether a render component owns renderer resources.")
        .def("_handle", &KE::Scene::SceneRenderSystem::handle,
             py::arg("component"),
             "Return the renderer batch handle for internal bridges; "
             "unregistered components return InvalidHandle.")
        .def("shares_batch", &KE::Scene::SceneRenderSystem::sharesBatch,
             py::arg("first"), py::arg("second"),
             "Return whether two components share one renderer batch without "
             "exposing the raw handle.")
        .def("set_double_sided", &KE::Scene::SceneRenderSystem::setDoubleSided,
             py::arg("component"), py::arg("enabled") = true,
             "Set double-sided rendering through component registration.")
        .def("set_casts_shadow", &KE::Scene::SceneRenderSystem::setCastsShadow,
             py::arg("component"), py::arg("enabled") = true,
             "Set shadow casting through component registration.")
        .def("set_alpha_mode", &KE::Scene::SceneRenderSystem::setAlphaMode,
             py::arg("component"), py::arg("mode"), py::arg("cutoff") = 0.5f,
             "Set alpha rendering through component registration.")
        .def("set_material", &KE::Scene::SceneRenderSystem::setMaterial,
             py::arg("component"), py::arg("material"),
             "Replace a registered SceneGraph renderable's material and move "
             "it to the matching renderer batch. ExternalBuffer renderables "
             "reject dynamic material replacement.")
        .def(
            "set_texture",
            [](KE::Scene::SceneRenderSystem& self,
               KE::Scene::RenderComponent& component,
               KE::Backend::Texture* texture, KE::TextureRole role) {
                self.setTexture(component, texture, role);
            },
            py::arg("component"), py::arg("texture"), py::arg("role"),
            "Set a texture by material role through component registration.")
        .def(
            "set_texture",
            [](KE::Scene::SceneRenderSystem& self,
               KE::Scene::RenderComponent& component,
               KE::Backend::Texture* texture,
               int slot) { self.setTexture(component, texture, slot); },
            py::arg("component"), py::arg("texture"), py::arg("slot") = 0,
            "Set a texture by raw renderer slot through component "
            "registration.")
        .def("set_external_buffer",
             &KE::Scene::SceneRenderSystem::setExternalBuffer,
             py::arg("component"), py::arg("descriptor"),
             "Attach an external transform buffer through registration.")
        .def(
            "set_instance_colors",
            [](KE::Scene::SceneRenderSystem& self,
               KE::Scene::RenderComponent& component,
               const FloatArray& colors) {
                self.setInstanceColors(component, vec4Array(colors, "colors"));
            },
            py::arg("component"), py::arg("colors"),
            "Set ExternalBuffer RGBA multipliers without replacing transforms. "
            "Accept one color or one per instance; empty resets to white.")
        .def(
            "update_instances",
            [](KE::Scene::SceneRenderSystem& self,
               KE::Scene::RenderComponent& component,
               const FloatArray& transforms, py::object colors) {
                auto transformVec = mat4Array(transforms, "transforms");
                std::vector<glm::vec4> colorVec;
                const std::vector<glm::vec4>* colorPtr = nullptr;
                if (!colors.is_none()) {
                    auto colorArray = colors.cast<FloatArray>();
                    colorVec = vec4Array(colorArray, "colors");
                    if (!colorVec.empty() && colorVec.size() != 1 &&
                        colorVec.size() != transformVec.size()) {
                        throw py::value_error(
                            "colors must be empty, length 1, or match "
                            "transforms length");
                    }
                    colorPtr = &colorVec;
                }
                self.updateInstances(component, transformVec, colorPtr);
            },
            py::arg("component"), py::arg("transforms"),
            py::arg("colors") = py::none(),
            "Update instanced transforms and optional colors through "
            "component registration.")
        .def(
            "update_geometry",
            [](KE::Scene::SceneRenderSystem& self,
               KE::Scene::RenderComponent& component,
               const FloatArray& positions, py::object normals) {
                auto positionVec = vec3Array(positions, "positions");
                std::vector<glm::vec3> normalVec;
                if (!normals.is_none()) {
                    auto normalArray = normals.cast<FloatArray>();
                    normalVec = vec3Array(normalArray, "normals");
                    if (normalVec.size() != positionVec.size()) {
                        throw py::value_error(
                            "normals must match positions length");
                    }
                }
                self.updateGeometry(component, positionVec, normalVec);
            },
            py::arg("component"), py::arg("positions"),
            py::arg("normals") = py::none(),
            "Update dynamic vertex positions and optional normals through "
            "component registration.")
        .def(
            "update_skinning",
            [](KE::Scene::SceneRenderSystem& self,
               KE::Scene::RenderComponent& component,
               const FloatArray& boneMatrices) {
                self.updateSkinning(
                    component,
                    mat4RowMajorArray(boneMatrices, "bone_matrices"));
            },
            py::arg("component"), py::arg("bone_matrices"),
            "Update skinned bone matrices through component registration.");

    py::class_<KE::Scene::DebugDraw>(
        scene, "DebugDraw",
        "Helpers for creating debug lines, arrows, and coordinate axes.")
        .def_static(
            "make_arrow_transform",
            [](const glm::vec3& start, const glm::vec3& end) -> py::object {
                glm::mat4 transform(1.0f);
                if (!KE::Scene::DebugDraw::makeArrowTransform(start, end,
                                                              transform))
                    return py::none();
                return py::cast(transform);
            },
            py::arg("start"), py::arg("end"),
            "Return the arrow transform, or None for a zero-length arrow.")
        .def_static(
            "make_line_transform",
            [](const glm::vec3& start, const glm::vec3& end) -> py::object {
                glm::mat4 transform(1.0f);
                if (!KE::Scene::DebugDraw::makeLineTransform(start, end,
                                                             transform))
                    return py::none();
                return py::cast(transform);
            },
            py::arg("start"), py::arg("end"),
            "Return the centered line transform, or None for a zero-length "
            "line.")
        .def_static(
            "log_component_lines",
            [](KE::App* app, KE::Material* material, const std::string& path,
               const FloatArray& starts, const FloatArray& ends,
               py::object colors, float radius, int segments) {
                auto s = vec3Array(starts, "starts");
                auto e = vec3Array(ends, "ends");
                std::vector<glm::vec4> c;
                if (!colors.is_none())
                    c = vec4Array(colors.cast<FloatArray>(), "colors");
                if (s.size() != e.size()) {
                    throw py::value_error(
                        "starts and ends must have the same length");
                }
                if (!c.empty() && c.size() != 1 && c.size() != s.size()) {
                    throw py::value_error(
                        "colors must be empty, length 1, or match starts "
                        "length");
                }
                return KE::Scene::DebugDraw::logLineComponent(
                    app, material, path, s, e, c, radius, segments);
            },
            py::arg("app"), py::arg("material"), py::arg("path"),
            py::arg("starts"), py::arg("ends"), py::arg("colors") = py::none(),
            py::arg("radius") = 0.005f, py::arg("segments") = 8,
            "Create instanced debug line geometry and return its "
            "RenderComponent.")
        .def_static(
            "update_component_lines",
            [](KE::App* app, KE::Scene::RenderComponent& component,
               const FloatArray& starts, const FloatArray& ends,
               py::object colors) {
                auto s = vec3Array(starts, "starts");
                auto e = vec3Array(ends, "ends");
                std::vector<glm::vec4> c;
                if (!colors.is_none())
                    c = vec4Array(colors.cast<FloatArray>(), "colors");
                if (s.size() != e.size()) {
                    throw py::value_error(
                        "starts and ends must have the same length");
                }
                if (!c.empty() && c.size() != 1 && c.size() != s.size()) {
                    throw py::value_error(
                        "colors must be empty, length 1, or match starts "
                        "length");
                }
                KE::Scene::DebugDraw::updateLines(app, component, s, e, c);
            },
            py::arg("app"), py::arg("component"), py::arg("starts"),
            py::arg("ends"), py::arg("colors") = py::none(),
            "Update component-backed debug line geometry.")
        .def_static(
            "log_component_arrows",
            [](KE::App* app, KE::Material* material, const std::string& path,
               const FloatArray& starts, const FloatArray& ends,
               py::object colors, float radius, int segments) {
                auto s = vec3Array(starts, "starts");
                auto e = vec3Array(ends, "ends");
                std::vector<glm::vec4> c;
                if (!colors.is_none())
                    c = vec4Array(colors.cast<FloatArray>(), "colors");
                if (s.size() != e.size()) {
                    throw py::value_error(
                        "starts and ends must have the same length");
                }
                if (!c.empty() && c.size() != 1 && c.size() != s.size()) {
                    throw py::value_error(
                        "colors must be empty, length 1, or match starts "
                        "length");
                }
                return KE::Scene::DebugDraw::logArrowComponent(
                    app, material, path, s, e, c, radius, segments);
            },
            py::arg("app"), py::arg("material"), py::arg("path"),
            py::arg("starts"), py::arg("ends"), py::arg("colors") = py::none(),
            py::arg("radius") = 0.02f, py::arg("segments") = 12,
            "Create instanced debug arrow geometry and return its "
            "RenderComponent.")
        .def_static(
            "update_component_arrows",
            [](KE::App* app, KE::Scene::RenderComponent& component,
               const FloatArray& starts, const FloatArray& ends,
               py::object colors) {
                auto s = vec3Array(starts, "starts");
                auto e = vec3Array(ends, "ends");
                std::vector<glm::vec4> c;
                if (!colors.is_none())
                    c = vec4Array(colors.cast<FloatArray>(), "colors");
                if (s.size() != e.size()) {
                    throw py::value_error(
                        "starts and ends must have the same length");
                }
                if (!c.empty() && c.size() != 1 && c.size() != s.size()) {
                    throw py::value_error(
                        "colors must be empty, length 1, or match starts "
                        "length");
                }
                KE::Scene::DebugDraw::updateArrows(app, component, s, e, c);
            },
            py::arg("app"), py::arg("component"), py::arg("starts"),
            py::arg("ends"), py::arg("colors") = py::none(),
            "Update component-backed debug arrow geometry.")
        .def_static(
            "log_coordinate_axes",
            [](KE::App* app, KE::Material* material, const std::string& path,
               glm::vec3 origin, glm::quat orientation, float length,
               float radius, int segments) {
                return KE::Scene::DebugDraw::logCoordinateAxes(
                    app, material, path, origin, orientation, length, radius,
                    segments);
            },
            py::arg("app"), py::arg("material"), py::arg("path"),
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
            py::arg("scene"), py::arg("path"), py::arg("starts"),
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
            py::arg("scene"), py::arg("path"), py::arg("starts"),
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
            py::arg("scene"), py::arg("path"), py::arg("starts"),
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
            py::arg("scene"), py::arg("path"), py::arg("starts"),
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
            py::arg("scene"), py::arg("path"), py::arg("origin"),
            py::arg("orientation"), py::arg("length") = 1.0f,
            py::arg("radius") = 0.005f, py::arg("segments") = 8,
            py::return_value_policy::reference,
            "Create scene-backed XYZ coordinate axes.");
}
