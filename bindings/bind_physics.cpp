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
#include "animation/character_description.hpp"
#include "bridge/physics_bridge.hpp"
#include "bridge/skeleton_bridge.hpp"
#include "engine/core/app/app.hpp"
#include "engine/scene/scene_backend.hpp"
#include "physics/articulation.hpp"
#include "physics/physics.hpp"
#include <extensions/PxRigidBodyExt.h>
#endif

namespace py = pybind11;

void bind_physics(py::module& m) {
#ifdef KANGENGINE_USE_PHYSX
    using namespace KE;
    using namespace KE::Animation;
    using namespace KE::Bridge;

    py::class_<PhysicsGpuDynamicsConfig>(
        m, "PhysicsGpuDynamicsConfig",
        "PhysX GPU dynamics buffer capacities used during scene creation.")
        .def(py::init<>())
        .def_readwrite("temp_buffer_capacity",
                       &PhysicsGpuDynamicsConfig::tempBufferCapacity)
        .def_readwrite("max_rigid_contact_count",
                       &PhysicsGpuDynamicsConfig::maxRigidContactCount)
        .def_readwrite("max_rigid_patch_count",
                       &PhysicsGpuDynamicsConfig::maxRigidPatchCount)
        .def_readwrite("heap_capacity",
                       &PhysicsGpuDynamicsConfig::heapCapacity)
        .def_readwrite("found_lost_pairs_capacity",
                       &PhysicsGpuDynamicsConfig::foundLostPairsCapacity)
        .def_readwrite("collision_stack_size",
                       &PhysicsGpuDynamicsConfig::collisionStackSize);

    // PhysicsConfig
    py::class_<PhysicsConfig>(
        m, "PhysicsConfig",
        "PhysX world configuration including timestep, up axis, and reporting.")
        .def(py::init<>(), "Create default physics configuration.")
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
            "restitution",
            [](const PhysicsConfig& c) { return c.friction[2]; },
            [](PhysicsConfig& c, float value) { c.friction[2] = value; },
            "Default material restitution.")
        .def_readwrite("enable_gpu", &PhysicsConfig::enableGPU,
                       "Enable PhysX GPU features when available.")
        .def_readwrite("gpu_dynamics", &PhysicsConfig::gpuDynamics,
                       "GPU dynamics memory capacities used at scene creation.")
        .def_readwrite("enable_contact_reports",
                       &PhysicsConfig::enableContactReports,
                       "Enable contact collection during simulation.");

    py::class_<ContactPoint>(m, "ContactPoint",
                             "Contact point reported by the PhysX world.")
        .def_readonly("position", &ContactPoint::position,
                      "World-space contact position.")
        .def_readonly("normal", &ContactPoint::normal,
                      "World-space contact normal.")
        .def_readonly("impulse", &ContactPoint::impulse,
                      "Contact impulse.")
        .def_readonly("separation", &ContactPoint::separation,
                      "Contact separation distance.");

    py::class_<PxRigidDynamic, std::unique_ptr<PxRigidDynamic, py::nodelete>>(
        m, "RigidDynamic",
        "PhysX dynamic rigid body owned by a PhysicsWorld.")
        .def("get_root_position",
             [](const PxRigidDynamic& self) {
                 PxTransform pose = self.getGlobalPose();
                 return floatArrayFromVector({pose.p.x, pose.p.y, pose.p.z});
             },
             "Return root position as [x, y, z].")
        .def("get_root_rotation",
             [](const PxRigidDynamic& self) {
                 PxTransform pose = self.getGlobalPose();
                 return floatArrayFromVector(
                     {pose.q.x, pose.q.y, pose.q.z, pose.q.w});
             },
             "Return root rotation as XYZW quaternion.")
        .def("get_root_linear_velocity",
             [](const PxRigidDynamic& self) {
                 PxVec3 v = self.getLinearVelocity();
                 return floatArrayFromVector({v.x, v.y, v.z});
             },
             "Return root linear velocity.")
        .def("get_root_angular_velocity",
             [](const PxRigidDynamic& self) {
                 PxVec3 v = self.getAngularVelocity();
                 return floatArrayFromVector({v.x, v.y, v.z});
             },
             "Return root angular velocity.")
        .def("get_mass", &PxRigidDynamic::getMass,
             "Return rigid body mass.")
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
        .def("add_force",
             [](PxRigidDynamic& self, const FloatArray& force) {
                 auto f = vec3ArrayView(force, "force");
                 if (f.count != 1)
                     throw py::value_error("add_force expects force[3]");
                 self.addForce(PxVec3(f.data[0], f.data[1], f.data[2]));
             },
             py::arg("force"), "Apply a world-space force from a numpy array.")
        .def("add_force",
             [](PxRigidDynamic& self, const std::vector<float>& force) {
                 if (force.size() != 3)
                     throw std::runtime_error("add_force expects force[3]");
                 self.addForce(PxVec3(force[0], force[1], force[2]));
             },
             py::arg("force"), "Apply a world-space force.")
        .def("add_force_at_position",
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
        .def("add_force_at_position",
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

    // PhysicsWorld (non-copyable, non-movable — Python must keep it alive)
    py::class_<PhysicsWorld>(
        m, "PhysicsWorld",
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
            "create_dynamic_box",
            [](PhysicsWorld& self, const std::vector<float>& halfExtents,
               const std::vector<float>& pos,
               const std::vector<float>& rotXyzw, float density) {
                if (halfExtents.size() != 3 || pos.size() != 3 ||
                    rotXyzw.size() != 4) {
                    throw py::value_error(
                        "create_dynamic_box expects half_extents[3], pos[3], "
                        "and rot_xyzw[4]");
                }
                return self.createDynamicBox(
                    glm::vec3(halfExtents[0], halfExtents[1], halfExtents[2]),
                    glm::vec3(pos[0], pos[1], pos[2]),
                    glm::quat(rotXyzw[3], rotXyzw[0], rotXyzw[1],
                              rotXyzw[2]),
                    density);
            },
            py::arg("half_extents"), py::arg("pos"),
            py::arg("rot_xyzw") = std::vector<float>{0.f, 0.f, 0.f, 1.f},
            py::arg("density") = 1.0f,
            py::return_value_policy::reference,
            "Create a dynamic box for low-level physics tests and tools.")
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
        .def(
            "create_dynamic_rigid",
            [](PhysicsWorld& self, const CharacterData& data,
               const FloatArray& pos, const FloatArray& rot_xyzw, float density,
               PxU32 collisionGroup, float contactOffset, float restOffset) {
                auto p = vec3ArrayView(pos, "pos");
                auto q = vec4ArrayView(rot_xyzw, "rot_xyzw");
                if (p.count != 1 || q.count != 1) {
                    throw py::value_error(
                        "create_dynamic_rigid expects pos[3], rot_xyzw[4]");
                }
                return self.createDynamicRigid(
                    data, glm::vec3(p.data[0], p.data[1], p.data[2]),
                    glm::quat(q.data[3], q.data[0], q.data[1], q.data[2]),
                    density, collisionGroup, contactOffset, restOffset);
            },
            py::arg("data"), py::arg("pos"), py::arg("rot_xyzw"),
            py::arg("density") = 1.0f, py::arg("collision_group") = 0,
            py::arg("contact_offset") = 0.02f, py::arg("rest_offset") = 0.0f,
            py::return_value_policy::reference,
            "Create a dynamic rigid body from imported character data.")
        .def(
            "create_dynamic_rigid",
            [](PhysicsWorld& self, const CharacterData& data,
               const std::vector<float>& pos,
               const std::vector<float>& rot_xyzw, float density,
               PxU32 collisionGroup, float contactOffset, float restOffset) {
                if (pos.size() != 3 || rot_xyzw.size() != 4) {
                    throw std::runtime_error(
                        "create_dynamic_rigid expects pos[3], rot_xyzw[4]");
                }
                return self.createDynamicRigid(
                    data, glm::vec3(pos[0], pos[1], pos[2]),
                    glm::quat(rot_xyzw[3], rot_xyzw[0], rot_xyzw[1],
                              rot_xyzw[2]),
                    density, collisionGroup, contactOffset, restOffset);
            },
            py::arg("data"), py::arg("pos"),
            py::arg("rot_xyzw") = std::vector<float>{0.f, 0.f, 0.f, 1.f},
            py::arg("density") = 1.0f, py::arg("collision_group") = 0,
            py::arg("contact_offset") = 0.02f, py::arg("rest_offset") = 0.0f,
            py::return_value_policy::reference,
            "Create a dynamic rigid body from imported character data.")
        .def("set_dt", &PhysicsWorld::setDt, py::arg("dt"),
             "Set simulation timestep in seconds.");

    // ArticulationConfig
    py::class_<ArticulationConfig>(
        m, "ArticulationConfig",
        "PhysX articulation construction settings.")
        .def(py::init<>(), "Create default articulation configuration.")
        .def_static("fixed_base", &ArticulationConfig::fixedBase,
                    "Create configuration for a fixed-base articulation.")
        .def_static("free_base", &ArticulationConfig::freeBase,
                    "Create configuration for a free-base articulation.")
        .def_readwrite("fix_base", &ArticulationConfig::fixBase,
                       "Whether the root body is fixed.")
        .def_readwrite("disable_self_collision",
                       &ArticulationConfig::disableSelfCollision,
                       "Disable self collision between articulation links.")
        .def_readwrite("solver_iterations",
                       &ArticulationConfig::solverIterations,
                       "Solver iteration count.")
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
        .def_readwrite("contact_offset", &ArticulationConfig::contactOffset,
                       "PhysX contact offset.")
        .def_readwrite("rest_offset", &ArticulationConfig::restOffset,
                       "PhysX rest offset.")
        .def_readwrite("enable_ccd", &ArticulationConfig::enableCCD,
                       "Enable continuous collision detection.");

    // Articulation (non-copyable)
    py::class_<Articulation>(
        m, "Articulation",
        "PhysX articulated character or robot built from imported character data.")
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
            py::arg("cfg") = ArticulationConfig{},
            "Build an articulation in a PhysicsWorld from character data.")
        .def("num_links", &Articulation::numLinks,
             "Return the number of links.")
        .def("num_dofs", &Articulation::numDofs,
             "Return the number of controllable DOFs.")
        .def("release", &Articulation::release,
             "Release the underlying PhysX articulation.")
        .def("get_root_position",
             [](const Articulation& self) {
                 return floatArrayFromVector(self.getRootPositionFlat());
             },
             "Return root position as [x, y, z].")
        .def("get_root_rotation",
             [](const Articulation& self) {
                 return floatArrayFromVector(self.getRootRotationFlat());
             },
             "Return root rotation as XYZW quaternion.")
        .def("get_root_linear_velocity",
             [](const Articulation& self) {
                 return floatArrayFromVector(self.getRootLinearVelocityFlat());
             },
             "Return root linear velocity.")
        .def("get_root_angular_velocity",
             [](const Articulation& self) {
                 return floatArrayFromVector(self.getRootAngularVelocityFlat());
             },
             "Return root angular velocity.")
        .def("get_link_positions",
             [](const Articulation& self) {
                 return floatArrayFromVector(self.getLinkPositionsFlat());
             },
             "Return flat per-link positions.")
        .def("get_link_rotations",
             [](const Articulation& self) {
                 return floatArrayFromVector(self.getLinkRotationsFlat());
             },
             "Return flat per-link XYZW rotations.")
        .def("get_link_linear_velocities",
             [](const Articulation& self) {
                 return floatArrayFromVector(
                     self.getLinkLinearVelocitiesFlat());
             },
             "Return flat per-link linear velocities.")
        .def("get_link_angular_velocities",
             [](const Articulation& self) {
                 return floatArrayFromVector(
                     self.getLinkAngularVelocitiesFlat());
             },
             "Return flat per-link angular velocities.")
        .def("get_link_indices", &Articulation::getLinkIndices,
             "Return PhysX low-level link indices in visual link order.")
        .def("get_dof_positions",
             [](const Articulation& self) {
                 return floatArrayFromVector(self.getDofPositions());
             },
             "Return DOF positions.")
        .def("get_dof_velocities",
             [](const Articulation& self) {
                 return floatArrayFromVector(self.getDofVelocities());
             },
             "Return DOF velocities.")
        .def("get_dof_forces",
             [](const Articulation& self) {
                 return floatArrayFromVector(self.getDofForces());
             },
             "Return measured/applied DOF forces.")
        .def("get_dof_names", &Articulation::getDofNames,
             "Return DOF names.")
        .def("get_dof_limits",
             [](const Articulation& self) {
                 return floatArrayFromVec2Vector(self.getDofLimits());
             },
             "Return DOF lower/upper limits.")
        .def("get_dof_effort_limits",
             [](const Articulation& self) {
                 return floatArrayFromVector(self.getDofEffortLimits());
             },
             "Return imported DOF effort limits.")
        .def("get_link_masses",
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
        .def("get_effort_limits",
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
            py::arg("rot") = glm::quat(1.f, 0.f, 0.f, 0.f),
            "Reset root pose.")
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
        // joints: dict[int, list[Joint]]
        .def("joints", [](const Articulation& self) {
            py::dict result;
            for (const auto& [idx, jvec] : self.joints())
                result[py::int_(idx)] = jvec;
            return result;
        }, "Return imported joint metadata keyed by body index.");

    // PhysicsBridge
    py::class_<PhysicsBridge>(
        m, "PhysicsBridge",
        "Syncs PhysX articulation state into KangEngine scene/render visuals.")
        .def(py::init<>(), "Create a scene-graph physics bridge.")
        .def("add", &PhysicsBridge::add, py::arg("artic"),
             py::arg("skel_bridge"),
             "Connect an articulation to a SkeletonBridge.")
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
            py::return_value_policy::reference,
            "Create scene prims that visualize articulation collision shapes.");
#endif
}
