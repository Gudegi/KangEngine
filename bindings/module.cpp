// Define the native module entry point and cross-domain registration order.
#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_core(py::module&);
void bind_sim_buffer(py::module&);
void bind_render(py::module&);
void bind_material(py::module&);
void bind_geometry(py::module&);
void bind_animation(py::module&);
void bind_scene(py::module&);
void bind_asset(py::module&);
void bind_articulation_desc(py::module&);
void bind_sim_visual(py::module&);
void bind_scene_render(py::module&);
void bind_animation_visual(py::module&);
void bind_imgui(py::module&);
void bind_keys(py::module&);
void bind_viewer(py::module&);
#ifdef KANGENGINE_USE_PHYSX
void bind_physics(py::module&);
void bind_physics_gpu(py::module&);
#endif

PYBIND11_MODULE(_kangengine, m) {
    m.doc() = "KangEngine Python bindings";

    // Preserve public paths: geometry binds mesh payloads under scene.
    m.def_submodule("scene",
                    "Scene graph, prim, mesh, and debug drawing APIs.");
    m.def_submodule("geometry",
                    "Geometry algorithms and procedural mesh factories.");

    // Render descriptors require buffer views; materials require Texture.
    bind_core(m);
    bind_sim_buffer(m);
    bind_render(m);
    bind_material(m);

    bind_geometry(m);
    bind_animation(m);
    bind_scene(m);
    bind_asset(m);
    bind_articulation_desc(m);
#ifdef KANGENGINE_USE_PHYSX
    bind_physics(m);
    bind_physics_gpu(m);
#endif

    // Register scene/animation visual return types before App methods.
    bind_scene_render(m);
    bind_animation_visual(m);
    bind_imgui(m);
    bind_keys(m);
    bind_viewer(m);
    // Simulation visuals require physics and articulation visual types.
    bind_sim_visual(m);
}
