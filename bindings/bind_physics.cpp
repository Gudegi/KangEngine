///
/// Physics Python Bindings
/// PhysicsWorld, Articulation, PhysicsBridge  (PhysX only)
///

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#ifdef KANGENGINE_USE_PHYSX
#include "animation/character_description.hpp"
#include "bridge/physics_bridge.hpp"
#include "bridge/skeleton_bridge.hpp"
#include "engine/core/app/app.hpp"
#include "engine/scene/scene_backend.hpp"
#include "physics/articulation.hpp"
#include "physics/physics.hpp"
#endif

namespace py = pybind11;

void bind_physics(py::module& m) {
#ifdef KANGENGINE_USE_PHYSX
    using namespace KE;
    using namespace KE::Animation;
    using namespace KE::Bridge;

    // PhysicsConfig
    py::class_<PhysicsConfig>(m, "PhysicsConfig")
        .def(py::init<>())
        .def_static("y_up", &PhysicsConfig::yUp)
        .def_static("z_up", &PhysicsConfig::zUp)
        .def_readwrite("dt", &PhysicsConfig::dt);

    // PhysicsWorld (non-copyable, non-movable — Python must keep it alive)
    py::class_<PhysicsWorld>(m, "PhysicsWorld")
        .def(py::init<PhysicsConfig>(), py::arg("config") = PhysicsConfig{})
        .def("step", &PhysicsWorld::step)
        .def("add_default_ground", &PhysicsWorld::addDefaultGround)
        .def("num_body_actors", &PhysicsWorld::numBodyActors)
        .def("set_dt", &PhysicsWorld::setDt);

    // ArticulationConfig
    py::class_<ArticulationConfig>(m, "ArticulationConfig")
        .def(py::init<>())
        .def_static("fixed_base", &ArticulationConfig::fixedBase)
        .def_static("free_base", &ArticulationConfig::freeBase)
        .def_readwrite("fix_base", &ArticulationConfig::fixBase)
        .def_readwrite("disable_self_collision",
                       &ArticulationConfig::disableSelfCollision)
        .def_readwrite("solver_iterations",
                       &ArticulationConfig::solverIterations)
        .def_readwrite("root_linear_damping",
                       &ArticulationConfig::rootLinearDamping)
        .def_readwrite("root_angular_damping",
                       &ArticulationConfig::rootAngularDamping)
        .def_readwrite("link_linear_damping",
                       &ArticulationConfig::linkLinearDamping)
        .def_readwrite("link_angular_damping",
                       &ArticulationConfig::linkAngularDamping)
        .def_readwrite("max_angular_velocity",
                       &ArticulationConfig::maxAngularVelocity)
        .def_readwrite("enable_ccd", &ArticulationConfig::enableCCD);

    // Articulation (non-copyable)
    py::class_<Articulation>(m, "Articulation")
        .def(py::init<>())
        .def_static(
            "build",
            [](PhysicsWorld& physics, const CharacterData& data,
               const ArticulationConfig& cfg) {
                return Articulation::build(physics, data.skeletonTree,
                                           data.collisionGeoms, data.joints,
                                           data.inertials, cfg);
            },
            py::arg("physics"), py::arg("data"),
            py::arg("cfg") = ArticulationConfig{})
        .def("num_links", &Articulation::numLinks)
        .def("num_dofs", &Articulation::numDofs)
        .def("set_drive_targets", &Articulation::setDriveTargets,
             py::arg("targets"), py::arg("kp"), py::arg("kd"))
        .def(
            "reset_root",
            [](Articulation& self, const glm::vec3& pos, const glm::quat& rot) {
                self.resetRoot(PxTransform(PxVec3(pos.x, pos.y, pos.z),
                                           PxQuat(rot.x, rot.y, rot.z, rot.w)));
            },
            py::arg("pos") = glm::vec3(0.f),
            py::arg("rot") = glm::quat(1.f, 0.f, 0.f, 0.f))
        // joints: dict[int, list[Joint]]
        .def("joints", [](const Articulation& self) {
            py::dict result;
            for (const auto& [idx, jvec] : self.joints())
                result[py::int_(idx)] = jvec;
            return result;
        });

    // PhysicsBridge
    py::class_<PhysicsBridge>(m, "PhysicsBridge")
        .def(py::init<App*>(), py::arg("app") = nullptr)
        .def("add", &PhysicsBridge::add, py::arg("artic"),
             py::arg("skel_bridge"))
        .def("sync", &PhysicsBridge::sync)
        .def("set_collision_visible", &PhysicsBridge::setCollisionVisible)
        .def(
            "add_collision_visuals",
            [](PhysicsBridge& self, const Articulation& artic,
               Scene::SceneBackend* scene, const std::string& basePath,
               bool visibleByDefault) {
                return self.addCollisionVisuals(artic, scene, basePath,
                                                visibleByDefault);
            },
            py::arg("artic"), py::arg("scene"),
            py::arg("base_path") = "/collision",
            py::arg("visible_by_default") = false,
            py::return_value_policy::reference);
#endif
}
