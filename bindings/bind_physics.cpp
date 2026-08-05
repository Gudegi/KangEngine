///
/// Physics Python Bindings
/// PhysicsWorld, Articulation, PhysicsBridge  (PhysX only)
///

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include "py_array_view.hpp"

#ifdef KANGENGINE_USE_PHYSX
#include "asset/heightmap_loader.hpp"
#include "character/character_description.hpp"
#include "bridge/physics_bridge.hpp"
#include "bridge/articulation_visual_bridge.hpp"
#include "engine/core/app/app.hpp"
#include "engine/scene/scene_backend.hpp"
#include "physics/articulation.hpp"
#include "physics/physics_material.hpp"
#include "physics/physics.hpp"
#include <extensions/PxRigidBodyExt.h>
#endif

namespace py = pybind11;

#ifdef KANGENGINE_USE_PHYSX
namespace {

KE::Physics::PhysicsMaterialDesc physicsMaterialFromPy(py::handle obj,
                                                       const char* name) {
    if (py::isinstance<KE::Physics::PhysicsMaterialDesc>(obj))
        return obj.cast<KE::Physics::PhysicsMaterialDesc>();

    if (auto values = fixedFloatArray<3>(obj, name)) {
        return KE::Physics::PhysicsMaterialDesc{(*values)[0], (*values)[1],
                                                (*values)[2]};
    }

    if (py::isinstance<py::sequence>(obj) && py::len(obj) == 3) {
        auto seq = obj.cast<py::sequence>();
        return KE::Physics::PhysicsMaterialDesc{
            seq[0].cast<float>(), seq[1].cast<float>(), seq[2].cast<float>()};
    }

    throw py::value_error(std::string(name) +
                          " expects PhysicsMaterialDesc or 3 values "
                          "[static_friction, dynamic_friction, restitution]");
}

} // namespace
#endif

void bind_physics(py::module& m) {
#ifdef KANGENGINE_USE_PHYSX
    using namespace KE;
    using namespace KE::Character;
    using namespace KE::Physics;
    using namespace KE::Bridge;

    py::module physics = m.def_submodule(
        "physics", "PhysX world, articulation, and collision material APIs.");

    py::class_<PhysicsGpuDynamicsConfig>(
        physics, "PhysicsGpuDynamicsConfig",
        "PhysX GPU dynamics buffer capacities used during scene creation.")
        .def(py::init([](uint64_t temp_buffer_capacity,
                         uint32_t max_rigid_contact_count,
                         uint32_t max_rigid_patch_count, uint32_t heap_capacity,
                         uint32_t found_lost_pairs_capacity,
                         uint32_t found_lost_aggregate_pairs_capacity,
                         uint32_t total_aggregate_pairs_capacity,
                         uint32_t collision_stack_size,
                         uint32_t max_num_partitions) {
                 PhysicsGpuDynamicsConfig config;
                 config.tempBufferCapacity = temp_buffer_capacity;
                 config.maxRigidContactCount = max_rigid_contact_count;
                 config.maxRigidPatchCount = max_rigid_patch_count;
                 config.heapCapacity = heap_capacity;
                 config.foundLostPairsCapacity = found_lost_pairs_capacity;
                 config.foundLostAggregatePairsCapacity =
                     found_lost_aggregate_pairs_capacity;
                 config.totalAggregatePairsCapacity =
                     total_aggregate_pairs_capacity;
                 config.collisionStackSize = collision_stack_size;
                 config.maxNumPartitions = max_num_partitions;
                 return config;
             }),
             py::kw_only(),
             py::arg("temp_buffer_capacity") =
                 PhysicsGpuDynamicsConfig{}.tempBufferCapacity,
             py::arg("max_rigid_contact_count") =
                 PhysicsGpuDynamicsConfig{}.maxRigidContactCount,
             py::arg("max_rigid_patch_count") =
                 PhysicsGpuDynamicsConfig{}.maxRigidPatchCount,
             py::arg("heap_capacity") = PhysicsGpuDynamicsConfig{}.heapCapacity,
             py::arg("found_lost_pairs_capacity") =
                 PhysicsGpuDynamicsConfig{}.foundLostPairsCapacity,
             py::arg("found_lost_aggregate_pairs_capacity") =
                 PhysicsGpuDynamicsConfig{}.foundLostAggregatePairsCapacity,
             py::arg("total_aggregate_pairs_capacity") =
                 PhysicsGpuDynamicsConfig{}.totalAggregatePairsCapacity,
             py::arg("collision_stack_size") =
                 PhysicsGpuDynamicsConfig{}.collisionStackSize,
             py::arg("max_num_partitions") =
                 PhysicsGpuDynamicsConfig{}.maxNumPartitions,
             "Create GPU dynamics capacities from keyword arguments.")
        .def_readwrite("temp_buffer_capacity",
                       &PhysicsGpuDynamicsConfig::tempBufferCapacity,
                       "Temporary GPU buffer capacity in bytes.")
        .def_readwrite("max_rigid_contact_count",
                       &PhysicsGpuDynamicsConfig::maxRigidContactCount,
                       "Maximum rigid contact count.")
        .def_readwrite("max_rigid_patch_count",
                       &PhysicsGpuDynamicsConfig::maxRigidPatchCount,
                       "Maximum rigid contact patch count.")
        .def_readwrite("heap_capacity", &PhysicsGpuDynamicsConfig::heapCapacity,
                       "GPU dynamics heap capacity in bytes.")
        .def_readwrite("found_lost_pairs_capacity",
                       &PhysicsGpuDynamicsConfig::foundLostPairsCapacity,
                       "Capacity for found and lost rigid pairs.")
        .def_readwrite(
            "found_lost_aggregate_pairs_capacity",
            &PhysicsGpuDynamicsConfig::foundLostAggregatePairsCapacity,
            "Capacity for found and lost aggregate pairs.")
        .def_readwrite("total_aggregate_pairs_capacity",
                       &PhysicsGpuDynamicsConfig::totalAggregatePairsCapacity,
                       "Capacity for all aggregate pairs.")
        .def_readwrite("collision_stack_size",
                       &PhysicsGpuDynamicsConfig::collisionStackSize,
                       "GPU collision stack size in bytes.")
        .def_readwrite("max_num_partitions",
                       &PhysicsGpuDynamicsConfig::maxNumPartitions,
                       "Maximum GPU dynamics partition count.")
        .def("__repr__", [](const PhysicsGpuDynamicsConfig& config) {
            return py::str(
                       "PhysicsGpuDynamicsConfig("
                       "temp_buffer_capacity={!r}, "
                       "max_rigid_contact_count={!r}, "
                       "max_rigid_patch_count={!r}, heap_capacity={!r}, "
                       "found_lost_pairs_capacity={!r}, "
                       "found_lost_aggregate_pairs_capacity={!r}, "
                       "total_aggregate_pairs_capacity={!r}, "
                       "collision_stack_size={!r}, max_num_partitions={!r})")
                .attr("format")(
                    config.tempBufferCapacity, config.maxRigidContactCount,
                    config.maxRigidPatchCount, config.heapCapacity,
                    config.foundLostPairsCapacity,
                    config.foundLostAggregatePairsCapacity,
                    config.totalAggregatePairsCapacity,
                    config.collisionStackSize, config.maxNumPartitions);
        });

    // PhysicsConfig
    py::class_<PhysicsConfig>(
        physics, "PhysicsConfig",
        "PhysX world configuration including timestep, up axis, and reporting.")
        .def(py::init([](float dt, int solver_type, float static_friction,
                         float dynamic_friction, float restitution,
                         bool enable_gpu,
                         const PhysicsGpuDynamicsConfig& gpu_dynamics,
                         bool enable_contact_reports,
                         bool enable_body_accelerations,
                         uint32_t cpu_dispatcher_threads,
                         float bounce_threshold_velocity,
                         float friction_offset_threshold,
                         float friction_correlation_distance,
                         bool enable_stabilization) {
                 PhysicsConfig config;
                 config.dt = dt;
                 if (solver_type == 0) {
                     config.solverType = PxSolverType::ePGS;
                 } else if (solver_type == 1) {
                     config.solverType = PxSolverType::eTGS;
                 } else {
                     throw py::value_error(
                         "solver_type must be 0 (PGS) or 1 (TGS)");
                 }
                 config.friction[0] = static_friction;
                 config.friction[1] = dynamic_friction;
                 config.friction[2] = restitution;
                 config.enableGPU = enable_gpu;
                 config.gpuDynamics = gpu_dynamics;
                 config.enableContactReports = enable_contact_reports;
                 config.enableBodyAccelerations = enable_body_accelerations;
                 config.cpuDispatcherThreads = cpu_dispatcher_threads;
                 config.bounceThresholdVelocity = bounce_threshold_velocity;
                 config.frictionOffsetThreshold = friction_offset_threshold;
                 config.frictionCorrelationDistance =
                     friction_correlation_distance;
                 config.enableStabilization = enable_stabilization;
                 return config;
             }),
             py::kw_only(), py::arg_v("dt", PhysicsConfig{}.dt, "1.0 / 60.0"),
             py::arg("solver_type") = 1,
             py::arg("static_friction") = PhysicsConfig{}.friction[0],
             py::arg("dynamic_friction") = PhysicsConfig{}.friction[1],
             py::arg("restitution") = PhysicsConfig{}.friction[2],
             py::arg("enable_gpu") = PhysicsConfig{}.enableGPU,
             py::arg_v("gpu_dynamics", PhysicsConfig{}.gpuDynamics,
                       "PhysicsGpuDynamicsConfig()"),
             py::arg("enable_contact_reports") =
                 PhysicsConfig{}.enableContactReports,
             py::arg("enable_body_accelerations") =
                 PhysicsConfig{}.enableBodyAccelerations,
             py::arg("cpu_dispatcher_threads") =
                 PhysicsConfig{}.cpuDispatcherThreads,
             py::arg("bounce_threshold_velocity") =
                 PhysicsConfig{}.bounceThresholdVelocity,
             py::arg("friction_offset_threshold") =
                 PhysicsConfig{}.frictionOffsetThreshold,
             py::arg("friction_correlation_distance") =
                 PhysicsConfig{}.frictionCorrelationDistance,
             py::arg("enable_stabilization") =
                 PhysicsConfig{}.enableStabilization,
             "Create physics configuration from keyword arguments.")
        .def_static("y_up", &PhysicsConfig::yUp,
                    "Create configuration for a Y-up world.")
        .def_static("z_up", &PhysicsConfig::zUp,
                    "Create configuration for a Z-up world.")
        .def_readwrite("dt", &PhysicsConfig::dt,
                       "Simulation timestep in seconds.")
        .def_property(
            "static_friction",
            [](const PhysicsConfig& c) { return c.friction[0]; },
            [](PhysicsConfig& c, float value) { c.friction[0] = value; },
            "Default material static friction.")
        .def_property(
            "dynamic_friction",
            [](const PhysicsConfig& c) { return c.friction[1]; },
            [](PhysicsConfig& c, float value) { c.friction[1] = value; },
            "Default material dynamic friction.")
        .def_property(
            "restitution", [](const PhysicsConfig& c) { return c.friction[2]; },
            [](PhysicsConfig& c, float value) { c.friction[2] = value; },
            "Default material restitution.")
        .def_readwrite("enable_gpu", &PhysicsConfig::enableGPU,
                       "Enable PhysX GPU features when available.")
        .def_readwrite("gpu_dynamics", &PhysicsConfig::gpuDynamics,
                       "GPU dynamics memory capacities used at scene creation.")
        .def_readwrite("enable_contact_reports",
                       &PhysicsConfig::enableContactReports,
                       "Enable contact collection during simulation.")
        .def_readwrite("enable_body_accelerations",
                       &PhysicsConfig::enableBodyAccelerations,
                       "Enable PhysX rigid-body acceleration state.")
        .def_readwrite("cpu_dispatcher_threads",
                       &PhysicsConfig::cpuDispatcherThreads,
                       "Worker threads used by the PhysX CPU dispatcher.")
        .def_readwrite("bounce_threshold_velocity",
                       &PhysicsConfig::bounceThresholdVelocity,
                       "Relative speed below which contacts do not bounce.")
        .def_readwrite("friction_offset_threshold",
                       &PhysicsConfig::frictionOffsetThreshold,
                       "Contact separation threshold for friction anchors.")
        .def_readwrite("friction_correlation_distance",
                       &PhysicsConfig::frictionCorrelationDistance,
                       "Distance used to correlate friction patches.")
        .def_readwrite("enable_stabilization",
                       &PhysicsConfig::enableStabilization,
                       "Enable the PhysX scene stabilization pass.")
        .def_property(
            "solver_type",
            [](const PhysicsConfig& c) {
                return c.solverType == PxSolverType::ePGS ? 0 : 1;
            },
            [](PhysicsConfig& c, int value) {
                if (value == 0) {
                    c.solverType = PxSolverType::ePGS;
                } else if (value == 1) {
                    c.solverType = PxSolverType::eTGS;
                } else {
                    throw py::value_error(
                        "solver_type must be 0 (PGS) or 1 (TGS)");
                }
            },
            "Solver type: 0 for PGS, 1 for TGS.")
        .def("__repr__", [](const PhysicsConfig& config) {
            const int solver_type =
                config.solverType == PxSolverType::ePGS ? 0 : 1;
            return py::str("PhysicsConfig(dt={:g}, solver_type={!r}, "
                           "static_friction={:g}, dynamic_friction={:g}, "
                           "restitution={:g}, enable_gpu={!r}, "
                           "gpu_dynamics={!r}, enable_contact_reports={!r}, "
                           "enable_body_accelerations={!r}, "
                           "cpu_dispatcher_threads={!r}, "
                           "bounce_threshold_velocity={:g}, "
                           "friction_offset_threshold={:g}, "
                           "friction_correlation_distance={:g}, "
                           "enable_stabilization={!r})")
                .attr("format")(
                    config.dt, solver_type, config.friction[0],
                    config.friction[1], config.friction[2], config.enableGPU,
                    config.gpuDynamics, config.enableContactReports,
                    config.enableBodyAccelerations,
                    config.cpuDispatcherThreads, config.bounceThresholdVelocity,
                    config.frictionOffsetThreshold,
                    config.frictionCorrelationDistance,
                    config.enableStabilization);
        });

    py::class_<PhysicsMaterialDesc>(
        physics, "PhysicsMaterialDesc",
        "PhysX material factors used by collision shapes.")
        .def(py::init<>(), "Create default material [1, 1, 0].")
        .def(py::init<float, float, float>(), py::arg("static_friction") = 1.f,
             py::arg("dynamic_friction") = 1.f, py::arg("restitution") = 0.f,
             "Create a material from scalar PhysX material factors.")
        .def(py::init([](py::handle values) {
                 return physicsMaterialFromPy(values, "PhysicsMaterialDesc");
             }),
             py::arg("values"),
             "Create from [static_friction, dynamic_friction, restitution]. "
             "Accepts list/tuple, NumPy array, or CPU torch tensor.")
        .def_readwrite("static_friction", &PhysicsMaterialDesc::staticFriction,
                       "PhysX static friction coefficient.")
        .def_readwrite("dynamic_friction",
                       &PhysicsMaterialDesc::dynamicFriction,
                       "PhysX dynamic friction coefficient.")
        .def_readwrite("restitution", &PhysicsMaterialDesc::restitution,
                       "PhysX restitution coefficient.")
        .def(
            "as_tuple",
            [](const PhysicsMaterialDesc& material) {
                return py::make_tuple(material.staticFriction,
                                      material.dynamicFriction,
                                      material.restitution);
            },
            "Return (static_friction, dynamic_friction, restitution).")
        .def("__repr__", [](const PhysicsMaterialDesc& material) {
            return "PhysicsMaterialDesc(static_friction=" +
                   std::to_string(material.staticFriction) +
                   ", dynamic_friction=" +
                   std::to_string(material.dynamicFriction) +
                   ", restitution=" + std::to_string(material.restitution) +
                   ")";
        });

    py::class_<CollisionMaterialOverride>(
        physics, "CollisionMaterialOverride",
        "Collision material override matched by body/geom name or index. "
        "Later overrides win.")
        .def(py::init<>(), "Create an empty/global override descriptor.")
        .def(py::init([](py::handle material) {
                 CollisionMaterialOverride entry;
                 entry.material = physicsMaterialFromPy(material, "material");
                 return entry;
             }),
             py::arg("material"),
             "Create a global material override for every collision geom.")
        .def_static(
            "all_geoms",
            [](py::handle material) {
                CollisionMaterialOverride entry;
                entry.material = physicsMaterialFromPy(material, "material");
                return entry;
            },
            py::arg("material"),
            "Override every collision geom in the built actor/articulation.")
        .def_static(
            "for_body",
            [](const std::string& bodyName, py::handle material) {
                CollisionMaterialOverride entry;
                entry.bodyName = bodyName;
                entry.material = physicsMaterialFromPy(material, "material");
                return entry;
            },
            py::arg("body_name"), py::arg("material"),
            "Override every collision geom on a named body.")
        .def_static(
            "for_geom",
            [](const std::string& bodyName, const std::string& geomName,
               py::handle material) {
                CollisionMaterialOverride entry;
                entry.bodyName = bodyName;
                entry.geomName = geomName;
                entry.material = physicsMaterialFromPy(material, "material");
                return entry;
            },
            py::arg("body_name"), py::arg("geom_name"), py::arg("material"),
            "Override one named collision geom on a named body.")
        .def_static(
            "for_indices",
            [](int bodyIndex, int geomIndex, py::handle material) {
                CollisionMaterialOverride entry;
                entry.bodyIndex = bodyIndex;
                entry.geomIndex = geomIndex;
                entry.material = physicsMaterialFromPy(material, "material");
                return entry;
            },
            py::arg("body_index"), py::arg("geom_index"), py::arg("material"),
            "Override by imported body index and collision geom index.")
        .def_readwrite("body_index", &CollisionMaterialOverride::bodyIndex,
                       "Matched body index, or -1 for name/all matching.")
        .def_readwrite("body_name", &CollisionMaterialOverride::bodyName,
                       "Matched body name, or empty for all bodies.")
        .def_readwrite("geom_index", &CollisionMaterialOverride::geomIndex,
                       "Matched geom index inside the body, or -1.")
        .def_readwrite("geom_name", &CollisionMaterialOverride::geomName,
                       "Matched geom name, or empty for all geoms.")
        .def_readwrite("material", &CollisionMaterialOverride::material,
                       "Override material.")
        .def("__repr__", [](const CollisionMaterialOverride& entry) {
            return "CollisionMaterialOverride(body_index=" +
                   std::to_string(entry.bodyIndex) + ", body_name='" +
                   entry.bodyName +
                   "', geom_index=" + std::to_string(entry.geomIndex) +
                   ", geom_name='" + entry.geomName + "', material=" +
                   py::repr(py::cast(entry.material)).cast<std::string>() + ")";
        });

    physics.def(
        "mjcf_friction_to_physx",
        [](const std::vector<float>& friction) {
            return mjcfFrictionToPhysX(friction);
        },
        py::arg("friction"),
        "Map MJCF geom friction values to KangEngine's PhysX material "
        "descriptor.");

    py::class_<ContactPoint>(physics, "ContactPoint",
                             "Contact point reported by the PhysX world.")
        .def_readonly("position", &ContactPoint::position,
                      "World-space contact position.")
        .def_readonly("normal", &ContactPoint::normal,
                      "World-space contact normal.")
        .def_readonly("impulse", &ContactPoint::impulse, "Contact impulse.")
        .def_readonly("separation", &ContactPoint::separation,
                      "Contact separation distance.");

    py::class_<PxRigidDynamic, std::unique_ptr<PxRigidDynamic, py::nodelete>>(
        m, "RigidDynamic", "PhysX dynamic rigid body owned by a PhysicsWorld.")
        .def(
            "get_root_position",
            [](const PxRigidDynamic& self) {
                PxTransform pose = self.getGlobalPose();
                return floatArrayFromVector({pose.p.x, pose.p.y, pose.p.z});
            },
            "Return root position as [x, y, z].")
        .def(
            "get_root_rotation",
            [](const PxRigidDynamic& self) {
                PxTransform pose = self.getGlobalPose();
                return floatArrayFromVector(
                    {pose.q.x, pose.q.y, pose.q.z, pose.q.w});
            },
            "Return root rotation as XYZW quaternion.")
        .def(
            "get_root_linear_velocity",
            [](const PxRigidDynamic& self) {
                PxVec3 v = self.getLinearVelocity();
                return floatArrayFromVector({v.x, v.y, v.z});
            },
            "Return root linear velocity.")
        .def(
            "get_root_angular_velocity",
            [](const PxRigidDynamic& self) {
                PxVec3 v = self.getAngularVelocity();
                return floatArrayFromVector({v.x, v.y, v.z});
            },
            "Return root angular velocity.")
        .def("get_mass", &PxRigidDynamic::getMass, "Return rigid body mass.")
        .def("get_inverse_mass", &PxRigidDynamic::getInvMass,
             "Return inverse rigid body mass.")
        .def(
            "get_local_com_position",
            [](const PxRigidDynamic& self) {
                const PxVec3 position = self.getCMassLocalPose().p;
                return floatArrayFromVector(
                    {position.x, position.y, position.z});
            },
            "Return the center-of-mass position in the actor frame.")
        .def(
            "get_local_com_rotation",
            [](const PxRigidDynamic& self) {
                const PxQuat rotation = self.getCMassLocalPose().q;
                return floatArrayFromVector(
                    {rotation.x, rotation.y, rotation.z, rotation.w});
            },
            "Return the center-of-mass frame rotation as an XYZW quaternion.")
        .def(
            "get_mass_space_inertia",
            [](const PxRigidDynamic& self) {
                const PxVec3 inertia = self.getMassSpaceInertiaTensor();
                return floatArrayFromVector({inertia.x, inertia.y, inertia.z});
            },
            "Return the diagonal inertia tensor in the center-of-mass frame.")
        .def(
            "get_mass_space_inverse_inertia",
            [](const PxRigidDynamic& self) {
                const PxVec3 inertia = self.getMassSpaceInvInertiaTensor();
                return floatArrayFromVector({inertia.x, inertia.y, inertia.z});
            },
            "Return inverse diagonal inertia in the center-of-mass frame.")
        .def(
            "set_kinematic",
            [](PxRigidDynamic& self, bool enabled) {
                self.setRigidBodyFlag(PxRigidBodyFlag::eKINEMATIC, enabled);
            },
            py::arg("enabled") = true,
            "Enable or disable kinematic rigid-body behavior.")
        .def("release", &PxRigidDynamic::release,
             "Release the underlying PhysX actor.")
        .def(
            "set_root_state",
            [](PxRigidDynamic& self, const FloatArray& pos,
               const FloatArray& rot_xyzw, const FloatArray& linear_velocity,
               const FloatArray& angular_velocity) {
                auto p = vec3ArrayView(pos, "pos");
                auto q = vec4ArrayView(rot_xyzw, "rot_xyzw");
                auto lv = vec3ArrayView(linear_velocity, "linear_velocity");
                auto av = vec3ArrayView(angular_velocity, "angular_velocity");
                if (p.count != 1 || q.count != 1 || lv.count != 1 ||
                    av.count != 1) {
                    throw py::value_error(
                        "set_root_state expects single pos[3], rot_xyzw[4], "
                        "linear_velocity[3], angular_velocity[3]");
                }
                self.setGlobalPose(PxTransform(
                    PxVec3(p.data[0], p.data[1], p.data[2]),
                    PxQuat(q.data[0], q.data[1], q.data[2], q.data[3])));
                self.setLinearVelocity(
                    PxVec3(lv.data[0], lv.data[1], lv.data[2]));
                self.setAngularVelocity(
                    PxVec3(av.data[0], av.data[1], av.data[2]));
                self.wakeUp();
            },
            py::arg("pos"), py::arg("rot_xyzw"), py::arg("linear_velocity"),
            py::arg("angular_velocity"),
            "Set root pose and velocities from numpy arrays.")
        .def(
            "set_root_state",
            [](PxRigidDynamic& self, const std::vector<float>& pos,
               const std::vector<float>& rot_xyzw,
               const std::vector<float>& linear_velocity,
               const std::vector<float>& angular_velocity) {
                if (pos.size() != 3 || rot_xyzw.size() != 4 ||
                    linear_velocity.size() != 3 ||
                    angular_velocity.size() != 3) {
                    throw std::runtime_error(
                        "set_root_state expects pos[3], rot_xyzw[4], "
                        "linear_velocity[3], angular_velocity[3]");
                }
                self.setGlobalPose(
                    PxTransform(PxVec3(pos[0], pos[1], pos[2]),
                                PxQuat(rot_xyzw[0], rot_xyzw[1], rot_xyzw[2],
                                       rot_xyzw[3])));
                self.setLinearVelocity(PxVec3(linear_velocity[0],
                                              linear_velocity[1],
                                              linear_velocity[2]));
                self.setAngularVelocity(PxVec3(angular_velocity[0],
                                               angular_velocity[1],
                                               angular_velocity[2]));
                self.wakeUp();
            },
            py::arg("pos"), py::arg("rot_xyzw"),
            py::arg("linear_velocity") = std::vector<float>{0.f, 0.f, 0.f},
            py::arg("angular_velocity") = std::vector<float>{0.f, 0.f, 0.f},
            "Set root pose and velocities from Python sequences.")
        .def(
            "add_force",
            [](PxRigidDynamic& self, const FloatArray& force) {
                auto f = vec3ArrayView(force, "force");
                if (f.count != 1)
                    throw py::value_error("add_force expects force[3]");
                self.addForce(PxVec3(f.data[0], f.data[1], f.data[2]));
            },
            py::arg("force"), "Apply a world-space force from a numpy array.")
        .def(
            "add_force",
            [](PxRigidDynamic& self, const std::vector<float>& force) {
                if (force.size() != 3)
                    throw std::runtime_error("add_force expects force[3]");
                self.addForce(PxVec3(force[0], force[1], force[2]));
            },
            py::arg("force"), "Apply a world-space force.")
        .def(
            "add_force_at_position",
            [](PxRigidDynamic& self, const FloatArray& force,
               const FloatArray& position) {
                auto f = vec3ArrayView(force, "force");
                auto p = vec3ArrayView(position, "position");
                if (f.count != 1 || p.count != 1) {
                    throw py::value_error(
                        "add_force_at_position expects force[3] and "
                        "position[3]");
                }
                PxRigidBodyExt::addForceAtPos(
                    self, PxVec3(f.data[0], f.data[1], f.data[2]),
                    PxVec3(p.data[0], p.data[1], p.data[2]));
            },
            py::arg("force"), py::arg("position"),
            "Apply a world-space force at a world-space position.")
        .def(
            "add_force_at_position",
            [](PxRigidDynamic& self, const std::vector<float>& force,
               const std::vector<float>& position) {
                if (force.size() != 3 || position.size() != 3) {
                    throw std::runtime_error(
                        "add_force_at_position expects force[3] and "
                        "position[3]");
                }
                PxRigidBodyExt::addForceAtPos(
                    self, PxVec3(force[0], force[1], force[2]),
                    PxVec3(position[0], position[1], position[2]));
            },
            py::arg("force"), py::arg("position"),
            "Apply a world-space force at a world-space position.");

    py::class_<PxRigidStatic, std::unique_ptr<PxRigidStatic, py::nodelete>>(
        m, "RigidStatic", "PhysX static rigid body owned by a PhysicsWorld.")
        .def("get_root_position",
             [](const PxRigidStatic& self) {
                 const PxTransform pose = self.getGlobalPose();
                 return floatArrayFromVector({pose.p.x, pose.p.y, pose.p.z});
             })
        .def("get_root_rotation",
             [](const PxRigidStatic& self) {
                 const PxTransform pose = self.getGlobalPose();
                 return floatArrayFromVector(
                     {pose.q.x, pose.q.y, pose.q.z, pose.q.w});
             })
        .def(
            "set_root_pose",
            [](PxRigidStatic& self, const std::vector<float>& pos,
               const std::vector<float>& rotXyzw) {
                if (pos.size() != 3 || rotXyzw.size() != 4)
                    throw py::value_error(
                        "set_root_pose expects pos[3], rot_xyzw[4]");
                self.setGlobalPose(PxTransform(
                    PxVec3(pos[0], pos[1], pos[2]),
                    PxQuat(rotXyzw[0], rotXyzw[1], rotXyzw[2], rotXyzw[3])));
            },
            py::arg("pos"), py::arg("rot_xyzw"))
        .def("release", &PxRigidStatic::release);

    // PhysicsWorld (non-copyable, non-movable — Python must keep it alive)
    py::class_<PhysicsWorld>(
        physics, "PhysicsWorld",
        "PhysX simulation world for rigid bodies and articulations.")
        .def(py::init<PhysicsConfig>(), py::arg("config") = PhysicsConfig{},
             "Create a physics world from configuration.")
        .def("step", &PhysicsWorld::step,
             "Advance simulation by one configured timestep.")
        .def("add_default_ground", &PhysicsWorld::addDefaultGround,
             "Add a default static ground plane.")
        .def("clear_ground_actors", &PhysicsWorld::clearGroundActors,
             "Remove all ground actors from the scene.")
        .def("num_ground_actors", &PhysicsWorld::numGroundActors,
             "Return the number of ground actors.")
        .def("num_cached_materials", &PhysicsWorld::numCachedMaterials,
             "Return the number of non-default cached PhysX materials.")
        .def("num_body_actors", &PhysicsWorld::numBodyActors,
             "Return the number of dynamic body actors.")
        .def("num_contacts", &PhysicsWorld::numContacts,
             "Return the number of collected contact points.")
        .def("get_contacts", &PhysicsWorld::getContacts,
             py::return_value_policy::reference_internal,
             "Return contact points collected during the last step.")
        .def("clear_contacts", &PhysicsWorld::clearContacts,
             "Clear collected contact points.")
        .def(
            "add_static_box",
            [](PhysicsWorld& self, const std::vector<float>& halfExtents,
               const std::vector<float>& pos, const std::vector<float>& rotXyzw,
               bool registerAsGround) {
                if (halfExtents.size() != 3 || pos.size() != 3 ||
                    rotXyzw.size() != 4) {
                    throw py::value_error(
                        "add_static_box expects half_extents[3], pos[3], "
                        "and rot_xyzw[4]");
                }
                self.createStaticBox(
                    glm::vec3(halfExtents[0], halfExtents[1], halfExtents[2]),
                    glm::vec3(pos[0], pos[1], pos[2]),
                    glm::quat(rotXyzw[3], rotXyzw[0], rotXyzw[1], rotXyzw[2]),
                    registerAsGround);
            },
            py::arg("half_extents"), py::arg("pos"),
            py::arg("rot_xyzw") = std::vector<float>{0.f, 0.f, 0.f, 1.f},
            py::arg("register_as_ground") = true,
            "Add a static box collider, useful for ramps and simple terrain.")
        .def(
            "add_heightfield",
            [](PhysicsWorld& self, const FloatArray& heights, int rows,
               int cols, float horizontalScale, UpAxis upAxis, bool center,
               bool registerAsGround, const PhysicsMaterialDesc& material) {
                auto h = floatVectorView(heights, "heights");
                if (rows < 2 || cols < 2)
                    throw py::value_error(
                        "add_heightfield expects rows >= 2 and cols >= 2");
                if (h.count != static_cast<size_t>(rows * cols))
                    throw py::value_error(
                        "add_heightfield expects heights.size == rows * cols");
                return self.createStaticHeightField(
                           h.data, rows, cols, horizontalScale, material,
                           upAxis, center, registerAsGround) != nullptr;
            },
            py::arg("heights"), py::arg("rows"), py::arg("cols"),
            py::arg("horizontal_scale") = 1.0f, py::arg("up_axis") = UpAxis::Y,
            py::arg("center") = true, py::arg("register_as_ground") = true,
            py::arg("material") = PhysicsMaterialDesc{},
            "Create a static PhysX heightfield collider from a row-major "
            "float height grid. The shape is registered as ground by default.")
        .def(
            "add_heightmap_collision",
            [](PhysicsWorld& self, const std::string& path, UpAxis upAxis,
               float horizontalScale, float heightScale, float heightOffset,
               int sampleStride, bool center, bool registerAsGround,
               const PhysicsMaterialDesc& material) {
                Asset::HeightmapTerrainOptions options;
                options.upAxis = upAxis;
                options.horizontalScale = horizontalScale;
                options.heightScale = heightScale;
                options.heightOffset = heightOffset;
                options.sampleStride = sampleStride;
                Asset::HeightFieldData field =
                    Asset::HeightmapLoader::loadHeightField(path, options);
                if (field.heights.empty())
                    return false;
                return self.createStaticHeightField(
                           field.heights.data(), field.rows, field.cols,
                           field.horizontalScale, material, upAxis, center,
                           registerAsGround) != nullptr;
            },
            py::arg("path"), py::arg("up_axis") = UpAxis::Y,
            py::arg("horizontal_scale") = 1.0f, py::arg("height_scale") = 64.0f,
            py::arg("height_offset") = -16.0f, py::arg("sample_stride") = 1,
            py::arg("center") = true, py::arg("register_as_ground") = true,
            py::arg("material") = PhysicsMaterialDesc{},
            "Load a heightmap image and create a static PhysX heightfield "
            "collider using the same sampling options as the render terrain.")
        .def(
            "create_dynamic_box",
            [](PhysicsWorld& self, const std::vector<float>& halfExtents,
               const std::vector<float>& pos, const std::vector<float>& rotXyzw,
               float density) {
                if (halfExtents.size() != 3 || pos.size() != 3 ||
                    rotXyzw.size() != 4) {
                    throw py::value_error(
                        "create_dynamic_box expects half_extents[3], pos[3], "
                        "and rot_xyzw[4]");
                }
                return self.createDynamicBox(
                    glm::vec3(halfExtents[0], halfExtents[1], halfExtents[2]),
                    glm::vec3(pos[0], pos[1], pos[2]),
                    glm::quat(rotXyzw[3], rotXyzw[0], rotXyzw[1], rotXyzw[2]),
                    density);
            },
            py::arg("half_extents"), py::arg("pos"),
            py::arg("rot_xyzw") = std::vector<float>{0.f, 0.f, 0.f, 1.f},
            py::arg("density") = 1.0f, py::return_value_policy::reference,
            "Create a dynamic box for low-level physics tests and tools.")
        .def(
            "create_dynamic_sphere",
            [](PhysicsWorld& self, float radius, const std::vector<float>& pos,
               const std::vector<float>& rotXyzw, float density) {
                if (pos.size() != 3 || rotXyzw.size() != 4) {
                    throw py::value_error(
                        "create_dynamic_sphere expects pos[3] and "
                        "rot_xyzw[4]");
                }
                return self.createDynamicSphere(
                    radius, glm::vec3(pos[0], pos[1], pos[2]),
                    glm::quat(rotXyzw[3], rotXyzw[0], rotXyzw[1], rotXyzw[2]),
                    density);
            },
            py::arg("radius"), py::arg("pos"),
            py::arg("rot_xyzw") = std::vector<float>{0.f, 0.f, 0.f, 1.f},
            py::arg("density") = 1.0f, py::return_value_policy::reference,
            "Create a dynamic sphere for low-level physics tests and tools.")
        .def(
            "get_contact_forces",
            [](const PhysicsWorld& self, const Articulation& articulation,
               bool groundOnly) {
                return floatArrayFromVector(
                    self.getContactForcesFlat(articulation, groundOnly));
            },
            py::arg("articulation"), py::arg("ground_only") = false,
            "Return per-link contact forces for an articulation.")
        .def(
            "get_ground_contact_forces",
            [](const PhysicsWorld& self, const Articulation& articulation) {
                return floatArrayFromVector(
                    self.getGroundContactForcesFlat(articulation));
            },
            py::arg("articulation"),
            "Return per-link ground contact forces for an articulation.")
        .def(
            "get_rigid_contact_force",
            [](const PhysicsWorld& self, const PxRigidDynamic& rigid,
               bool groundOnly) {
                return floatArrayFromVector(
                    self.getRigidContactForceFlat(rigid, groundOnly));
            },
            py::arg("rigid"), py::arg("ground_only") = false,
            "Return net contact force on a rigid body.")
        .def(
            "get_rigid_ground_contact_force",
            [](const PhysicsWorld& self, const PxRigidDynamic& rigid) {
                return floatArrayFromVector(
                    self.getRigidGroundContactForceFlat(rigid));
            },
            py::arg("rigid"),
            "Return net ground contact force on a rigid body.")
        .def("set_rigid_collision_material",
             &PhysicsWorld::setRigidCollisionMaterial, py::arg("rigid"),
             py::arg("material"),
             "Replace all collision shape materials on an existing rigid body. "
             "Returns the number of updated shapes.")
        .def("set_rigid_collision_material_overrides",
             &PhysicsWorld::setRigidCollisionMaterialOverrides,
             py::arg("rigid"), py::arg("data"), py::arg("material_overrides"),
             "Apply named/indexed material overrides to an existing rigid body "
             "created from character data. Returns updated shape count.")
        .def(
            "create_dynamic_rigid",
            [](PhysicsWorld& self, const CharacterData& data,
               const FloatArray& pos, const FloatArray& rot_xyzw, float density,
               PxU32 collisionGroup, float contactOffset, float restOffset,
               const std::vector<CollisionMaterialOverride>&
                   materialOverrides) {
                auto p = vec3ArrayView(pos, "pos");
                auto q = vec4ArrayView(rot_xyzw, "rot_xyzw");
                if (p.count != 1 || q.count != 1) {
                    throw py::value_error(
                        "create_dynamic_rigid expects pos[3], rot_xyzw[4]");
                }
                return self.createDynamicRigid(
                    data, glm::vec3(p.data[0], p.data[1], p.data[2]),
                    glm::quat(q.data[3], q.data[0], q.data[1], q.data[2]),
                    density, collisionGroup, contactOffset, restOffset,
                    materialOverrides);
            },
            py::arg("data"), py::arg("pos"), py::arg("rot_xyzw"),
            py::arg("density") = 1.0f, py::arg("collision_group") = 0,
            py::arg("contact_offset") = 0.02f, py::arg("rest_offset") = 0.0f,
            py::arg("material_overrides") =
                std::vector<CollisionMaterialOverride>{},
            py::return_value_policy::reference,
            "Create a dynamic rigid body from imported character data.")
        .def(
            "create_static_rigid",
            [](PhysicsWorld& self, const CharacterData& data,
               const std::vector<float>& pos,
               const std::vector<float>& rot_xyzw, PxU32 collisionGroup,
               float contactOffset, float restOffset,
               const std::vector<CollisionMaterialOverride>&
                   materialOverrides) {
                if (pos.size() != 3 || rot_xyzw.size() != 4)
                    throw py::value_error(
                        "create_static_rigid expects pos[3], rot_xyzw[4]");
                return self.createStaticRigid(
                    data, glm::vec3(pos[0], pos[1], pos[2]),
                    glm::quat(rot_xyzw[3], rot_xyzw[0], rot_xyzw[1],
                              rot_xyzw[2]),
                    collisionGroup, contactOffset, restOffset,
                    materialOverrides);
            },
            py::arg("data"), py::arg("pos"),
            py::arg("rot_xyzw") = std::vector<float>{0.f, 0.f, 0.f, 1.f},
            py::arg("collision_group") = 0, py::arg("contact_offset") = 0.02f,
            py::arg("rest_offset") = 0.0f,
            py::arg("material_overrides") =
                std::vector<CollisionMaterialOverride>{},
            py::return_value_policy::reference,
            "Create a static rigid body from imported character data.")
        .def(
            "create_dynamic_rigid",
            [](PhysicsWorld& self, const CharacterData& data,
               const std::vector<float>& pos,
               const std::vector<float>& rot_xyzw, float density,
               PxU32 collisionGroup, float contactOffset, float restOffset,
               const std::vector<CollisionMaterialOverride>&
                   materialOverrides) {
                if (pos.size() != 3 || rot_xyzw.size() != 4) {
                    throw std::runtime_error(
                        "create_dynamic_rigid expects pos[3], rot_xyzw[4]");
                }
                return self.createDynamicRigid(
                    data, glm::vec3(pos[0], pos[1], pos[2]),
                    glm::quat(rot_xyzw[3], rot_xyzw[0], rot_xyzw[1],
                              rot_xyzw[2]),
                    density, collisionGroup, contactOffset, restOffset,
                    materialOverrides);
            },
            py::arg("data"), py::arg("pos"),
            py::arg("rot_xyzw") = std::vector<float>{0.f, 0.f, 0.f, 1.f},
            py::arg("density") = 1.0f, py::arg("collision_group") = 0,
            py::arg("contact_offset") = 0.02f, py::arg("rest_offset") = 0.0f,
            py::arg("material_overrides") =
                std::vector<CollisionMaterialOverride>{},
            py::return_value_policy::reference,
            "Create a dynamic rigid body from imported character data.")
        .def("set_dt", &PhysicsWorld::setDt, py::arg("dt"),
             "Set simulation timestep in seconds.");

    // ArticulationConfig
    py::class_<ArticulationConfig>(physics, "ArticulationConfig",
                                   "PhysX articulation construction settings.")
        .def(
            py::init([](bool fix_base, bool disable_self_collision,
                        bool use_aggregate, int solver_position_iteration_count,
                        int solver_velocity_iteration_count,
                        uint32_t collision_group, float root_linear_damping,
                        float root_angular_damping, float link_linear_damping,
                        float link_angular_damping, float max_angular_velocity,
                        float max_depenetration_velocity, float sleep_threshold,
                        float stabilization_threshold,
                        bool enable_gyroscopic_forces, float contact_offset,
                        float rest_offset,
                        const std::vector<CollisionMaterialOverride>&
                            material_overrides,
                        bool enable_ccd) {
                ArticulationConfig config;
                config.fixBase = fix_base;
                config.disableSelfCollision = disable_self_collision;
                config.useAggregate = use_aggregate;
                config.solverPositionIterations =
                    solver_position_iteration_count;
                config.solverVelocityIterations =
                    solver_velocity_iteration_count;
                config.collisionGroup = collision_group;
                config.rootLinearDamping = root_linear_damping;
                config.rootAngularDamping = root_angular_damping;
                config.linkLinearDamping = link_linear_damping;
                config.linkAngularDamping = link_angular_damping;
                config.maxAngularVelocity = max_angular_velocity;
                config.maxDepenetrationVelocity = max_depenetration_velocity;
                config.sleepThreshold = sleep_threshold;
                config.stabilizationThreshold = stabilization_threshold;
                config.enableGyroscopicForces = enable_gyroscopic_forces;
                config.contactOffset = contact_offset;
                config.restOffset = rest_offset;
                config.materialOverrides = material_overrides;
                config.enableCCD = enable_ccd;
                return config;
            }),
            py::kw_only(), py::arg("fix_base") = ArticulationConfig{}.fixBase,
            py::arg("disable_self_collision") =
                ArticulationConfig{}.disableSelfCollision,
            py::arg("use_aggregate") = ArticulationConfig{}.useAggregate,
            py::arg("solver_position_iteration_count") =
                ArticulationConfig{}.solverPositionIterations,
            py::arg("solver_velocity_iteration_count") =
                ArticulationConfig{}.solverVelocityIterations,
            py::arg("collision_group") = ArticulationConfig{}.collisionGroup,
            py::arg("root_linear_damping") =
                ArticulationConfig{}.rootLinearDamping,
            py::arg("root_angular_damping") =
                ArticulationConfig{}.rootAngularDamping,
            py::arg("link_linear_damping") =
                ArticulationConfig{}.linkLinearDamping,
            py::arg("link_angular_damping") =
                ArticulationConfig{}.linkAngularDamping,
            py::arg("max_angular_velocity") =
                ArticulationConfig{}.maxAngularVelocity,
            py::arg("max_depenetration_velocity") =
                ArticulationConfig{}.maxDepenetrationVelocity,
            py::arg("sleep_threshold") = ArticulationConfig{}.sleepThreshold,
            py::arg("stabilization_threshold") =
                ArticulationConfig{}.stabilizationThreshold,
            py::arg("enable_gyroscopic_forces") =
                ArticulationConfig{}.enableGyroscopicForces,
            py::arg("contact_offset") = ArticulationConfig{}.contactOffset,
            py::arg("rest_offset") = ArticulationConfig{}.restOffset,
            py::arg("material_overrides") =
                ArticulationConfig{}.materialOverrides,
            py::arg("enable_ccd") = ArticulationConfig{}.enableCCD,
            "Create articulation configuration from keyword arguments.")
        .def_static("fixed_base", &ArticulationConfig::fixedBase,
                    "Create configuration for a fixed-base articulation.")
        .def_static("free_base", &ArticulationConfig::freeBase,
                    "Create configuration for a free-base articulation.")
        .def_readwrite("fix_base", &ArticulationConfig::fixBase,
                       "Whether the root body is fixed.")
        .def_readwrite("disable_self_collision",
                       &ArticulationConfig::disableSelfCollision,
                       "Disable self collision between articulation links.")
        .def_readwrite(
            "use_aggregate", &ArticulationConfig::useAggregate,
            "Group articulation links into one PhysX broadphase aggregate.")
        .def_property(
            "solver_iterations",
            [](const ArticulationConfig& c) {
                return c.solverPositionIterations;
            },
            [](ArticulationConfig& c, int value) {
                c.solverPositionIterations = value;
            },
            "Compatibility alias for solver_position_iteration_count.")
        .def_readwrite("solver_position_iteration_count",
                       &ArticulationConfig::solverPositionIterations,
                       "Position solver iteration count.")
        .def_readwrite("solver_velocity_iteration_count",
                       &ArticulationConfig::solverVelocityIterations,
                       "Velocity solver iteration count.")
        .def_readwrite("collision_group", &ArticulationConfig::collisionGroup,
                       "Collision group bit used for created actors.")
        .def_readwrite("root_linear_damping",
                       &ArticulationConfig::rootLinearDamping,
                       "Linear damping for the root link.")
        .def_readwrite("root_angular_damping",
                       &ArticulationConfig::rootAngularDamping,
                       "Angular damping for the root link.")
        .def_readwrite("link_linear_damping",
                       &ArticulationConfig::linkLinearDamping,
                       "Linear damping for child links.")
        .def_readwrite("link_angular_damping",
                       &ArticulationConfig::linkAngularDamping,
                       "Angular damping for child links.")
        .def_readwrite("max_angular_velocity",
                       &ArticulationConfig::maxAngularVelocity,
                       "Maximum angular velocity for links.")
        .def_readwrite("max_depenetration_velocity",
                       &ArticulationConfig::maxDepenetrationVelocity,
                       "Maximum depenetration velocity for links.")
        .def_readwrite("sleep_threshold", &ArticulationConfig::sleepThreshold,
                       "Articulation sleep threshold.")
        .def_readwrite("stabilization_threshold",
                       &ArticulationConfig::stabilizationThreshold,
                       "Articulation stabilization threshold.")
        .def_readwrite("enable_gyroscopic_forces",
                       &ArticulationConfig::enableGyroscopicForces,
                       "Enable gyroscopic forces on articulation links.")
        .def_readwrite("contact_offset", &ArticulationConfig::contactOffset,
                       "PhysX contact offset.")
        .def_readwrite("rest_offset", &ArticulationConfig::restOffset,
                       "PhysX rest offset.")
        .def_readwrite("material_overrides",
                       &ArticulationConfig::materialOverrides,
                       "Build-time collision material overrides.")
        .def(
            "add_material_override",
            [](ArticulationConfig& self,
               const CollisionMaterialOverride& entry) -> ArticulationConfig& {
                self.materialOverrides.push_back(entry);
                return self;
            },
            py::arg("override"), py::return_value_policy::reference_internal,
            "Append a build-time collision material override. Later overrides "
            "win.")
        .def(
            "clear_material_overrides",
            [](ArticulationConfig& self) -> ArticulationConfig& {
                self.materialOverrides.clear();
                return self;
            },
            py::return_value_policy::reference_internal,
            "Remove all collision material overrides.")
        .def_readwrite("enable_ccd", &ArticulationConfig::enableCCD,
                       "Enable continuous collision detection.")
        .def("__repr__", [](const ArticulationConfig& config) {
            return py::str(
                       "ArticulationConfig(fix_base={!r}, "
                       "disable_self_collision={!r}, use_aggregate={!r}, "
                       "solver_position_iteration_count={!r}, "
                       "solver_velocity_iteration_count={!r}, "
                       "collision_group={!r}, root_linear_damping={:g}, "
                       "root_angular_damping={:g}, link_linear_damping={:g}, "
                       "link_angular_damping={:g}, "
                       "max_angular_velocity={:g}, "
                       "max_depenetration_velocity={:g}, "
                       "sleep_threshold={:g}, "
                       "stabilization_threshold={:g}, "
                       "enable_gyroscopic_forces={!r}, contact_offset={:g}, "
                       "rest_offset={:g}, material_overrides={!r}, "
                       "enable_ccd={!r})")
                .attr("format")(
                    config.fixBase, config.disableSelfCollision,
                    config.useAggregate, config.solverPositionIterations,
                    config.solverVelocityIterations, config.collisionGroup,
                    config.rootLinearDamping, config.rootAngularDamping,
                    config.linkLinearDamping, config.linkAngularDamping,
                    config.maxAngularVelocity, config.maxDepenetrationVelocity,
                    config.sleepThreshold, config.stabilizationThreshold,
                    config.enableGyroscopicForces, config.contactOffset,
                    config.restOffset, config.materialOverrides,
                    config.enableCCD);
        });

    py::class_<ArticulationTemplate, std::shared_ptr<ArticulationTemplate>>(
        physics, "ArticulationTemplate",
        "Immutable articulation metadata shared by many PhysX instances.")
        .def_static(
            "create",
            [](const CharacterData& data, const ArticulationConfig& cfg) {
                return ArticulationTemplate::create(
                    data.skeletonTree, data.collisionGeoms, data.joints,
                    data.inertials, cfg);
            },
            py::arg("data"), py::arg("cfg") = ArticulationConfig{},
            "Precompute shared skeleton, rest transforms, collision metadata, "
            "and DOF metadata.")
        .def("num_links", &ArticulationTemplate::numLinks)
        .def("num_dofs", &ArticulationTemplate::numDofs)
        .def_property_readonly("body_names", &ArticulationTemplate::bodyNames);

    // Articulation (non-copyable)
    py::class_<Articulation>(physics, "Articulation",
                             "PhysX articulated character or robot built from "
                             "imported character data.")
        .def(py::init<>(), "Create an empty articulation handle.")
        .def_static(
            "build",
            [](PhysicsWorld& physics, const CharacterData& data,
               const ArticulationConfig& cfg) {
                return Articulation::build(physics, data.skeletonTree,
                                           data.collisionGeoms, data.joints,
                                           data.inertials, cfg);
            },
            py::arg("physics"), py::arg("data"),
            py::arg("cfg") = ArticulationConfig{}, py::keep_alive<0, 1>(),
            "Build an articulation in a PhysicsWorld from character data.")
        .def_static(
            "build_from_template",
            [](PhysicsWorld& physics,
               std::shared_ptr<ArticulationTemplate> template_,
               const ArticulationConfig& cfg) {
                return Articulation::build(physics, std::move(template_), cfg);
            },
            py::arg("physics"), py::arg("template"),
            py::arg("cfg") = ArticulationConfig{}, py::keep_alive<0, 1>(),
            "Build one PhysX articulation from shared immutable metadata.")
        .def_property_readonly(
            "template", &Articulation::articulationTemplate,
            "Shared immutable template used to build this instance.")
        .def("num_links", &Articulation::numLinks,
             "Return the number of links.")
        .def("num_dofs", &Articulation::numDofs,
             "Return the number of controllable DOFs.")
        .def("release", &Articulation::release,
             "Release the underlying PhysX articulation.")
        .def(
            "get_root_position",
            [](const Articulation& self) {
                return floatArrayFromVector(self.getRootPositionFlat());
            },
            "Return root position as [x, y, z].")
        .def(
            "get_root_rotation",
            [](const Articulation& self) {
                return floatArrayFromVector(self.getRootRotationFlat());
            },
            "Return root rotation as XYZW quaternion.")
        .def(
            "get_root_linear_velocity",
            [](const Articulation& self) {
                return floatArrayFromVector(self.getRootLinearVelocityFlat());
            },
            "Return root linear velocity.")
        .def(
            "get_root_angular_velocity",
            [](const Articulation& self) {
                return floatArrayFromVector(self.getRootAngularVelocityFlat());
            },
            "Return root angular velocity.")
        .def(
            "get_link_positions",
            [](const Articulation& self) {
                return floatArrayFromVector(self.getLinkPositionsFlat());
            },
            "Return flat per-link positions.")
        .def(
            "get_link_rotations",
            [](const Articulation& self) {
                return floatArrayFromVector(self.getLinkRotationsFlat());
            },
            "Return flat per-link XYZW rotations.")
        .def(
            "get_link_linear_velocities",
            [](const Articulation& self) {
                return floatArrayFromVector(self.getLinkLinearVelocitiesFlat());
            },
            "Return flat per-link linear velocities.")
        .def(
            "get_link_angular_velocities",
            [](const Articulation& self) {
                return floatArrayFromVector(
                    self.getLinkAngularVelocitiesFlat());
            },
            "Return flat per-link angular velocities.")
        .def("get_link_indices", &Articulation::getLinkIndices,
             "Return PhysX low-level link indices in visual link order.")
        .def(
            "get_dof_positions",
            [](const Articulation& self) {
                return floatArrayFromVector(self.getDofPositions());
            },
            "Return DOF positions.")
        .def(
            "get_dof_velocities",
            [](const Articulation& self) {
                return floatArrayFromVector(self.getDofVelocities());
            },
            "Return DOF velocities.")
        .def(
            "get_dof_forces",
            [](const Articulation& self) {
                return floatArrayFromVector(self.getDofForces());
            },
            "Return measured/applied DOF forces.")
        .def("get_dof_names", &Articulation::getDofNames, "Return DOF names.")
        .def("get_dof_gpu_indices", &Articulation::getDofGpuIndices,
             "Return logical DOF -> PhysX low-level GPU DOF indices.")
        .def(
            "get_dof_limits",
            [](const Articulation& self) {
                return floatArrayFromVec2Vector(self.getDofLimits());
            },
            "Return DOF lower/upper limits.")
        .def(
            "get_dof_effort_limits",
            [](const Articulation& self) {
                return floatArrayFromVector(self.getDofEffortLimits());
            },
            "Return imported DOF effort limits.")
        .def(
            "get_link_masses",
            [](const Articulation& self) {
                return floatArrayFromVector(self.getLinkMasses());
            },
            "Return per-link masses.")
        .def("calc_mass", &Articulation::calcMass,
             "Return total articulation mass.")
        .def(
            "set_drive_targets",
            [](Articulation& self, const FloatArray& targets, float kp,
               float kd) {
                self.setDriveTargets(floatVectorArray(targets, "targets"), kp,
                                     kd);
            },
            py::arg("targets"), py::arg("kp"), py::arg("kd"),
            "Set position drive targets with uniform PD gains.")
        .def("set_drive_targets",
             static_cast<void (Articulation::*)(const std::vector<float>&,
                                                float, float)>(
                 &Articulation::setDriveTargets),
             py::arg("targets"), py::arg("kp"), py::arg("kd"))
        .def(
            "set_drive_targets",
            [](Articulation& self, const FloatArray& targets) {
                self.setDriveTargets(floatVectorArray(targets, "targets"));
            },
            py::arg("targets"), "Set position drive targets.")
        .def("set_drive_targets",
             static_cast<void (Articulation::*)(const std::vector<float>&)>(
                 &Articulation::setDriveTargets),
             py::arg("targets"))
        .def(
            "set_drive_velocity_targets",
            [](Articulation& self, const FloatArray& targets) {
                self.setDriveVelocityTargets(
                    floatVectorArray(targets, "targets"));
            },
            py::arg("targets"), "Set velocity drive targets.")
        .def("set_drive_velocity_targets",
             &Articulation::setDriveVelocityTargets, py::arg("targets"),
             "Set velocity drive targets.")
        .def(
            "set_kps",
            [](Articulation& self, const FloatArray& kps) {
                self.setKPs(floatVectorArray(kps, "kps"));
            },
            py::arg("kps"), "Set per-DOF proportional gains.")
        .def("set_kps", &Articulation::setKPs, py::arg("kps"),
             "Set per-DOF proportional gains.")
        .def(
            "set_kds",
            [](Articulation& self, const FloatArray& kds) {
                self.setKDs(floatVectorArray(kds, "kds"));
            },
            py::arg("kds"), "Set per-DOF derivative gains.")
        .def("set_kds", &Articulation::setKDs, py::arg("kds"),
             "Set per-DOF derivative gains.")
        .def(
            "set_effort_limits",
            [](Articulation& self, const FloatArray& effortLimits) {
                self.setEffortLimits(
                    floatVectorArray(effortLimits, "effort_limits"));
            },
            py::arg("effort_limits"), "Set per-DOF effort limits.")
        .def("set_effort_limits", &Articulation::setEffortLimits,
             py::arg("effort_limits"), "Set per-DOF effort limits.")
        .def(
            "set_joint_forces",
            [](Articulation& self, const FloatArray& forces) {
                self.setJointForces(floatVectorArray(forces, "forces"));
            },
            py::arg("forces"), "Apply per-DOF joint forces.")
        .def("set_joint_forces", &Articulation::setJointForces,
             py::arg("forces"), "Apply per-DOF joint forces.")
        .def("set_collision_material", &Articulation::setCollisionMaterial,
             py::arg("physics"), py::arg("material"),
             "Replace all collision shape materials on this articulation at "
             "runtime. Returns the number of updated shapes.")
        .def("set_collision_material_overrides",
             &Articulation::setCollisionMaterialOverrides, py::arg("physics"),
             py::arg("material_overrides"),
             "Apply named/indexed collision material overrides at runtime. "
             "Returns the number of updated shapes.")
        .def(
            "add_link_force",
            [](Articulation& self, int linkIndex, const FloatArray& force) {
                auto f = vec3ArrayView(force, "force");
                if (f.count != 1)
                    throw py::value_error("add_link_force expects force[3]");
                self.addLinkForce(linkIndex,
                                  PxVec3(f.data[0], f.data[1], f.data[2]));
            },
            py::arg("link_index"), py::arg("force"),
            "Apply a world-space force to a link.")
        .def(
            "add_link_force",
            [](Articulation& self, int linkIndex,
               const std::vector<float>& force) {
                if (force.size() != 3) {
                    throw std::runtime_error("add_link_force expects force[3]");
                }
                self.addLinkForce(linkIndex,
                                  PxVec3(force[0], force[1], force[2]));
            },
            py::arg("link_index"), py::arg("force"),
            "Apply a world-space force to a link.")
        .def(
            "add_link_force_at_position",
            [](Articulation& self, int linkIndex, const FloatArray& force,
               const FloatArray& position) {
                auto f = vec3ArrayView(force, "force");
                auto p = vec3ArrayView(position, "position");
                if (f.count != 1 || p.count != 1) {
                    throw py::value_error(
                        "add_link_force_at_position expects force[3] and "
                        "position[3]");
                }
                self.addLinkForceAtPosition(
                    linkIndex, PxVec3(f.data[0], f.data[1], f.data[2]),
                    PxVec3(p.data[0], p.data[1], p.data[2]));
            },
            py::arg("link_index"), py::arg("force"), py::arg("position"),
            "Apply a force to a link at a world-space position.")
        .def(
            "add_link_force_at_position",
            [](Articulation& self, int linkIndex,
               const std::vector<float>& force,
               const std::vector<float>& position) {
                if (force.size() != 3 || position.size() != 3) {
                    throw std::runtime_error(
                        "add_link_force_at_position expects force[3] and "
                        "position[3]");
                }
                self.addLinkForceAtPosition(
                    linkIndex, PxVec3(force[0], force[1], force[2]),
                    PxVec3(position[0], position[1], position[2]));
            },
            py::arg("link_index"), py::arg("force"), py::arg("position"),
            "Apply a force to a link at a world-space position.")
        .def("get_kps", &Articulation::getKPs,
             "Return per-DOF proportional gains.")
        .def("get_kds", &Articulation::getKDs,
             "Return per-DOF derivative gains.")
        .def(
            "get_effort_limits",
            [](const Articulation& self) {
                return floatArrayFromVector(self.getEffortLimits());
            },
            "Return current per-DOF effort limits.")
        .def(
            "reset_root",
            [](Articulation& self, const glm::vec3& pos, const glm::quat& rot) {
                self.resetRoot(PxTransform(PxVec3(pos.x, pos.y, pos.z),
                                           PxQuat(rot.x, rot.y, rot.z, rot.w)));
            },
            py::arg("pos") = glm::vec3(0.f),
            py::arg("rot") = glm::quat(1.f, 0.f, 0.f, 0.f), "Reset root pose.")
        .def(
            "set_root_state",
            [](Articulation& self, const FloatArray& pos,
               const FloatArray& rot_xyzw, const FloatArray& linear_velocity,
               const FloatArray& angular_velocity) {
                auto p = vec3ArrayView(pos, "pos");
                auto q = vec4ArrayView(rot_xyzw, "rot_xyzw");
                auto lv = vec3ArrayView(linear_velocity, "linear_velocity");
                auto av = vec3ArrayView(angular_velocity, "angular_velocity");
                if (p.count != 1 || q.count != 1 || lv.count != 1 ||
                    av.count != 1) {
                    throw py::value_error(
                        "set_root_state expects single pos[3], rot_xyzw[4], "
                        "linear_velocity[3], angular_velocity[3]");
                }
                self.setRootState(
                    PxTransform(
                        PxVec3(p.data[0], p.data[1], p.data[2]),
                        PxQuat(q.data[0], q.data[1], q.data[2], q.data[3])),
                    PxVec3(lv.data[0], lv.data[1], lv.data[2]),
                    PxVec3(av.data[0], av.data[1], av.data[2]));
            },
            py::arg("pos"), py::arg("rot_xyzw"), py::arg("linear_velocity"),
            py::arg("angular_velocity"),
            "Set root pose and velocities from numpy arrays.")
        .def(
            "set_root_state",
            [](Articulation& self, const std::vector<float>& pos,
               const std::vector<float>& rot_xyzw,
               const std::vector<float>& linear_velocity,
               const std::vector<float>& angular_velocity) {
                if (pos.size() != 3 || rot_xyzw.size() != 4 ||
                    linear_velocity.size() != 3 ||
                    angular_velocity.size() != 3) {
                    throw std::runtime_error(
                        "set_root_state expects pos[3], rot_xyzw[4], "
                        "linear_velocity[3], angular_velocity[3]");
                }

                self.setRootState(PxTransform(PxVec3(pos[0], pos[1], pos[2]),
                                              PxQuat(rot_xyzw[0], rot_xyzw[1],
                                                     rot_xyzw[2], rot_xyzw[3])),
                                  PxVec3(linear_velocity[0], linear_velocity[1],
                                         linear_velocity[2]),
                                  PxVec3(angular_velocity[0],
                                         angular_velocity[1],
                                         angular_velocity[2]));
            },
            py::arg("pos"), py::arg("rot_xyzw"),
            py::arg("linear_velocity") = std::vector<float>{0.f, 0.f, 0.f},
            py::arg("angular_velocity") = std::vector<float>{0.f, 0.f, 0.f},
            "Set root pose and velocities from Python sequences.")
        .def(
            "set_dof_state",
            [](Articulation& self, const FloatArray& positions,
               const FloatArray& velocities) {
                self.setDofState(floatVectorArray(positions, "positions"),
                                 floatVectorArray(velocities, "velocities"));
            },
            py::arg("positions"), py::arg("velocities"),
            "Set DOF positions and velocities from numpy arrays.")
        .def("set_dof_state", &Articulation::setDofState, py::arg("positions"),
             py::arg("velocities"), "Set DOF positions and velocities.")
        // joints: dict[int, list[JointDesc]]
        .def(
            "joints",
            [](const Articulation& self) {
                py::dict result;
                for (const auto& [idx, jvec] : self.joints())
                    result[py::int_(idx)] = jvec;
                return result;
            },
            "Return imported joint metadata keyed by body index.");

    // PhysicsBridge
    py::class_<PhysicsBridge>(
        physics, "PhysicsBridge",
        "Syncs PhysX articulation state into KangEngine scene/render visuals.")
        .def(py::init<>(), "Create a scene-graph physics bridge.")
        .def("add", &PhysicsBridge::add, py::arg("artic"),
             py::arg("skel_bridge"), py::keep_alive<1, 2>(),
             py::keep_alive<1, 3>(),
             "Connect an articulation to an ArticulationVisualBridge.")
        .def("sync", &PhysicsBridge::sync,
             "Copy latest physics transforms into connected visuals.")
        .def("set_collision_visible", &PhysicsBridge::setCollisionVisible,
             py::arg("visible"), "Show or hide collision visual prims.")
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
            py::return_value_policy::reference, py::keep_alive<1, 2>(),
            py::keep_alive<1, 3>(),
            "Create scene prims that visualize articulation collision shapes.");
#endif
}
