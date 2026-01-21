///
/// Scene Backend Python Bindings
/// With USD Stage access for Python OpenUSD integration
///

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include "scene/scene_backend.hpp"
#include "scene/native/prim.hpp"
#include "scene/native/token.hpp"

#ifdef KANGENGINE_USE_USD
#include "scene/usd/usd_scene.hpp"
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usd/prim.h>
#endif

namespace py = pybind11;

void bind_scene(py::module& m) {
    py::module scene = m.def_submodule("scene", "Scene loading backends");

    // Token class
    py::class_<KE::Scene::Token>(scene, "Token")
        .def(py::init<>())
        .def(py::init<const std::string&>())
        .def("id", &KE::Scene::Token::id)
        .def("str", &KE::Scene::Token::str)
        .def("__eq__", &KE::Scene::Token::operator==)
        .def("__ne__", &KE::Scene::Token::operator!=)
        .def("__repr__", [](const KE::Scene::Token& t) {
            return "Token('" + t.str() + "')";
        });

    // PrimType enum
    py::enum_<KE::Scene::PrimType>(scene, "PrimType")
        .value("Root", KE::Scene::PrimType::Root)
        .value("Xform", KE::Scene::PrimType::Xform)
        .value("Mesh", KE::Scene::PrimType::Mesh)
        .value("Camera", KE::Scene::PrimType::Camera)
        .value("Light", KE::Scene::PrimType::Light)
        .export_values();

    // Prim class
    py::class_<KE::Scene::Prim, std::shared_ptr<KE::Scene::Prim>>(scene, "Prim")
        .def(py::init<const std::string&, KE::Scene::PrimType,
                      KE::Scene::Prim*>(),
             py::arg("name"), py::arg("type"), py::arg("parent") = nullptr)
        // Getters
        .def("get_name", &KE::Scene::Prim::getName)
        .def("get_path", &KE::Scene::Prim::getPath)
        .def("get_type", &KE::Scene::Prim::getType)
        .def("get_parent", &KE::Scene::Prim::getParent,
             py::return_value_policy::reference)
        // Hierarchy
        .def("add_child", &KE::Scene::Prim::addChild,
             py::return_value_policy::reference)
        .def("get_child", &KE::Scene::Prim::getChild,
             py::return_value_policy::reference)
        .def("get_prim_at_path", &KE::Scene::Prim::getPrimAtPath,
             py::return_value_policy::reference)
        .def("get_children", &KE::Scene::Prim::getChildren)
        // Mesh data
        .def("set_mesh_data", &KE::Scene::Prim::setMeshData)
        .def("get_mesh_data", &KE::Scene::Prim::getMeshData)
        // Static mesh creation (returns shared_ptr for set_mesh_data
        // compatibility)
        .def_static("create_square_data",
                    [](float scale) {
                        return std::make_shared<KE::Scene::MeshData>(
                            KE::Scene::Prim::createSquareData(scale));
                    })
        .def_static("create_plane_data",
                    [](float scale) {
                        return std::make_shared<KE::Scene::MeshData>(
                            KE::Scene::Prim::createPlaneData(scale));
                    })
        .def_static(
            "create_sphere_data",
            [](float radius, int numLongitudes, int numLatitudes) {
                return std::make_shared<KE::Scene::MeshData>(
                    KE::Scene::Prim::createSphereData(radius, numLongitudes,
                                                      numLatitudes));
            },
            py::arg("radius"), py::arg("num_longitudes"),
            py::arg("num_latitudes"))
        // Transform ops
        .def("add_translate_op", &KE::Scene::Prim::addTranslateOp)
        .def("add_scale_op", &KE::Scene::Prim::addScaleOp)
        .def("add_rotate_quaternion_op",
             &KE::Scene::Prim::addRotateQuaternionOp)
        // Display color
        .def("set_display_color", &KE::Scene::Prim::setDisplayColor)
        .def("get_display_color", &KE::Scene::Prim::getDisplayColor)
        .def("set_display_color_alpha", &KE::Scene::Prim::setDisplayColorAlpha)
        .def("get_display_color_alpha", &KE::Scene::Prim::getDisplayColorAlpha)
        // Model matrix
        .def("compute_model_matrix", &KE::Scene::Prim::computeModelMatrix)
        // setAttribute with specific types
        .def("set_attribute_vec3",
             [](KE::Scene::Prim& self, const std::string& name,
                const glm::vec3& value) { self.setAttribute(name, value); })
        .def("set_attribute_vec4",
             [](KE::Scene::Prim& self, const std::string& name,
                const glm::vec4& value) { self.setAttribute(name, value); })
        .def("set_attribute_quat",
             [](KE::Scene::Prim& self, const std::string& name,
                const glm::quat& value) { self.setAttribute(name, value); })
        .def("set_attribute_float",
             [](KE::Scene::Prim& self, const std::string& name, float value) {
                 self.setAttribute(name, value);
             })
        .def("set_attribute_int",
             [](KE::Scene::Prim& self, const std::string& name, int value) {
                 self.setAttribute(name, value);
             })
        .def("set_attribute_string",
             [](KE::Scene::Prim& self, const std::string& name,
                const std::string& value) { self.setAttribute(name, value); })
        // getAttribute with specific types
        .def("get_attribute_vec3",
             [](KE::Scene::Prim& self, const std::string& name) {
                 return self.getAttribute<glm::vec3>(name);
             })
        .def("get_attribute_vec4",
             [](KE::Scene::Prim& self, const std::string& name) {
                 return self.getAttribute<glm::vec4>(name);
             })
        .def("get_attribute_quat",
             [](KE::Scene::Prim& self, const std::string& name) {
                 return self.getAttribute<glm::quat>(name);
             })
        .def("get_attribute_float",
             [](KE::Scene::Prim& self, const std::string& name) {
                 return self.getAttribute<float>(name);
             })
        .def("get_attribute_int",
             [](KE::Scene::Prim& self, const std::string& name) {
                 return self.getAttribute<int>(name);
             })
        .def("get_attribute_string",
             [](KE::Scene::Prim& self, const std::string& name) {
                 return self.getAttribute<std::string>(name);
             })
        .def("has_attribute", py::overload_cast<const std::string&>(
                                  &KE::Scene::Prim::hasAttribute, py::const_))
        .def("traverse", &KE::Scene::Prim::traverse);

    // BackendType enum
    py::enum_<KE::Scene::BackendType>(scene, "BackendType")
        .value("Native", KE::Scene::BackendType::Native)
        .value("USD", KE::Scene::BackendType::USD)
        .export_values();

    // MeshData struct (with shared_ptr holder for set_mesh_data compatibility)
    py::class_<KE::Scene::MeshData, std::shared_ptr<KE::Scene::MeshData>>(
        scene, "MeshData")
        .def(py::init<>())
        .def_readwrite("vertices", &KE::Scene::MeshData::vertices)
        .def_readwrite("normals", &KE::Scene::MeshData::normals)
        .def_readwrite("uvs", &KE::Scene::MeshData::uvs)
        .def_readwrite("indices", &KE::Scene::MeshData::indices);

    // SceneBackend interface
    py::class_<KE::Scene::SceneBackend>(scene, "SceneBackend")
        .def("get_backend_type", &KE::Scene::SceneBackend::getBackendType)
        .def("load_scene", &KE::Scene::SceneBackend::loadScene)
        .def("save_scene", &KE::Scene::SceneBackend::saveScene)
        .def("load_mesh", &KE::Scene::SceneBackend::loadMesh)
        .def("list_meshes", &KE::Scene::SceneBackend::listMeshes);

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
