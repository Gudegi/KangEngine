///
/// Scene Backend Python Bindings
/// With USD Stage access for Python OpenUSD integration
///

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "scene/scene_backend.hpp"

#ifdef KANGENGINE_USE_USD
#include "scene/usd/usd_scene.hpp"
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usd/prim.h>
#endif

namespace py = pybind11;

void bind_scene(py::module& m) {
    py::module scene = m.def_submodule("scene", "Scene loading backends");

    // BackendType enum
    py::enum_<KE::Scene::BackendType>(scene, "BackendType")
        .value("Native", KE::Scene::BackendType::Native)
        .value("USD", KE::Scene::BackendType::USD)
        .export_values();

    // MeshData struct
    py::class_<KE::Scene::MeshData>(scene, "MeshData")
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
        .def("create_xform", [](KE::Scene::USDScene& self, const std::string& path) {
            self.createXform(path);
        }, "Create Xform (transform/group) at path")
        .def("create_mesh", [](KE::Scene::USDScene& self, const std::string& path) {
            self.createMesh(path);
        }, "Create Mesh at path")
        .def("create_sphere", [](KE::Scene::USDScene& self, const std::string& path, double radius) {
            self.createSphere(path, radius);
        }, "Create Sphere at path",
             py::arg("path"), py::arg("radius") = 1.0)
        .def("create_cube", [](KE::Scene::USDScene& self, const std::string& path, double size) {
            self.createCube(path, size);
        }, "Create Cube at path",
             py::arg("path"), py::arg("size") = 1.0)
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
              "Create a scene backend",
              py::arg("type"));

    // Helper function 
    scene.def("load_mesh_from_usd",
        [](const std::string& usd_path, const std::string& prim_path) -> KE::Scene::MeshData {
#ifdef KANGENGINE_USE_USD
            auto backend = KE::Scene::SceneFactory::createBackend(KE::Scene::BackendType::USD);
            backend->loadScene(usd_path);
            return backend->loadMesh(prim_path);
#else
            throw std::runtime_error("USD support not compiled");
#endif
        },
        py::arg("usd_path"),
        py::arg("prim_path"),
        "Load mesh from USD file (helper function)"
    );

    // USD 지원 확인
    scene.def("has_usd_support", []() -> bool {
#ifdef KANGENGINE_USE_USD
        return true;
#else
        return false;
#endif
    });
}
