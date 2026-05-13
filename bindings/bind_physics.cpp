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
        .def_readwrite("dt", &PhysicsConfig::dt)
        .def_readwrite("enable_contact_reports",
                       &PhysicsConfig::enableContactReports);

    py::class_<ContactPoint>(m, "ContactPoint")
        .def_readonly("position", &ContactPoint::position)
        .def_readonly("normal", &ContactPoint::normal)
        .def_readonly("impulse", &ContactPoint::impulse)
        .def_readonly("separation", &ContactPoint::separation);

    py::class_<PxRigidDynamic, std::unique_ptr<PxRigidDynamic, py::nodelete>>(
        m, "RigidDynamic")
        .def("get_root_position", [](const PxRigidDynamic& self) {
            PxTransform pose = self.getGlobalPose();
            return floatArrayFromVector({pose.p.x, pose.p.y, pose.p.z});
        })
        .def("get_root_rotation", [](const PxRigidDynamic& self) {
            PxTransform pose = self.getGlobalPose();
            return floatArrayFromVector(
                {pose.q.x, pose.q.y, pose.q.z, pose.q.w});
        })
        .def("get_root_linear_velocity", [](const PxRigidDynamic& self) {
            PxVec3 v = self.getLinearVelocity();
            return floatArrayFromVector({v.x, v.y, v.z});
        })
        .def("get_root_angular_velocity", [](const PxRigidDynamic& self) {
            PxVec3 v = self.getAngularVelocity();
            return floatArrayFromVector({v.x, v.y, v.z});
        })
        .def("get_mass", &PxRigidDynamic::getMass)
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
            py::arg("pos"), py::arg("rot_xyzw"),
            py::arg("linear_velocity"), py::arg("angular_velocity"))
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
                self.setGlobalPose(PxTransform(
                    PxVec3(pos[0], pos[1], pos[2]),
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
            py::arg("angular_velocity") = std::vector<float>{0.f, 0.f, 0.f})
        .def("add_force",
             [](PxRigidDynamic& self, const FloatArray& force) {
                 auto f = vec3ArrayView(force, "force");
                 if (f.count != 1)
                     throw py::value_error("add_force expects force[3]");
                 self.addForce(PxVec3(f.data[0], f.data[1], f.data[2]));
             })
        .def("add_force",
             [](PxRigidDynamic& self, const std::vector<float>& force) {
                 if (force.size() != 3)
                     throw std::runtime_error("add_force expects force[3]");
                 self.addForce(PxVec3(force[0], force[1], force[2]));
             });

    // PhysicsWorld (non-copyable, non-movable — Python must keep it alive)
    py::class_<PhysicsWorld>(m, "PhysicsWorld")
        .def(py::init<PhysicsConfig>(), py::arg("config") = PhysicsConfig{})
        .def("step", &PhysicsWorld::step)
        .def("add_default_ground", &PhysicsWorld::addDefaultGround)
        .def("clear_ground_actors", &PhysicsWorld::clearGroundActors)
        .def("num_ground_actors", &PhysicsWorld::numGroundActors)
        .def("num_body_actors", &PhysicsWorld::numBodyActors)
        .def("num_contacts", &PhysicsWorld::numContacts)
        .def("get_contacts", &PhysicsWorld::getContacts,
             py::return_value_policy::reference_internal)
        .def("clear_contacts", &PhysicsWorld::clearContacts)
        .def("get_contact_forces",
             [](const PhysicsWorld& self, const Articulation& articulation,
                bool groundOnly) {
                 return floatArrayFromVector(
                     self.getContactForcesFlat(articulation, groundOnly));
             },
             py::arg("articulation"), py::arg("ground_only") = false)
        .def("get_ground_contact_forces",
             [](const PhysicsWorld& self, const Articulation& articulation) {
                 return floatArrayFromVector(
                     self.getGroundContactForcesFlat(articulation));
             },
             py::arg("articulation"))
        .def("get_rigid_contact_force",
             [](const PhysicsWorld& self, const PxRigidDynamic& rigid,
                bool groundOnly) {
                 return floatArrayFromVector(
                     self.getRigidContactForceFlat(rigid, groundOnly));
             },
             py::arg("rigid"),
             py::arg("ground_only") = false)
        .def("get_rigid_ground_contact_force",
             [](const PhysicsWorld& self, const PxRigidDynamic& rigid) {
                 return floatArrayFromVector(
                     self.getRigidGroundContactForceFlat(rigid));
             },
             py::arg("rigid"))
        .def(
            "create_dynamic_rigid",
            [](PhysicsWorld& self, const CharacterData& data,
               const FloatArray& pos, const FloatArray& rot_xyzw,
               float density, PxU32 collisionGroup, float contactOffset,
               float restOffset) {
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
            py::arg("contact_offset") = 0.02f,
            py::arg("rest_offset") = 0.0f,
            py::return_value_policy::reference)
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
            py::arg("contact_offset") = 0.02f,
            py::arg("rest_offset") = 0.0f,
            py::return_value_policy::reference)
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
        .def_readwrite("collision_group", &ArticulationConfig::collisionGroup)
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
        .def_readwrite("contact_offset", &ArticulationConfig::contactOffset)
        .def_readwrite("rest_offset", &ArticulationConfig::restOffset)
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
        .def("release", &Articulation::release)
        .def("get_root_position",
             [](const Articulation& self) {
                 return floatArrayFromVector(self.getRootPositionFlat());
             })
        .def("get_root_rotation",
             [](const Articulation& self) {
                 return floatArrayFromVector(self.getRootRotationFlat());
             })
        .def("get_root_linear_velocity",
             [](const Articulation& self) {
                 return floatArrayFromVector(self.getRootLinearVelocityFlat());
             })
        .def("get_root_angular_velocity",
             [](const Articulation& self) {
                 return floatArrayFromVector(self.getRootAngularVelocityFlat());
             })
        .def("get_link_positions",
             [](const Articulation& self) {
                 return floatArrayFromVector(self.getLinkPositionsFlat());
             })
        .def("get_link_rotations",
             [](const Articulation& self) {
                 return floatArrayFromVector(self.getLinkRotationsFlat());
             })
        .def("get_link_linear_velocities",
             [](const Articulation& self) {
                 return floatArrayFromVector(
                     self.getLinkLinearVelocitiesFlat());
             })
        .def("get_link_angular_velocities",
             [](const Articulation& self) {
                 return floatArrayFromVector(
                     self.getLinkAngularVelocitiesFlat());
             })
        .def("get_dof_positions",
             [](const Articulation& self) {
                 return floatArrayFromVector(self.getDofPositions());
             })
        .def("get_dof_velocities",
             [](const Articulation& self) {
                 return floatArrayFromVector(self.getDofVelocities());
             })
        .def("get_dof_forces",
             [](const Articulation& self) {
                 return floatArrayFromVector(self.getDofForces());
             })
        .def("get_dof_names", &Articulation::getDofNames)
        .def("get_dof_limits",
             [](const Articulation& self) {
                 return floatArrayFromVec2Vector(self.getDofLimits());
             })
        .def("get_dof_effort_limits",
             [](const Articulation& self) {
                 return floatArrayFromVector(self.getDofEffortLimits());
             })
        .def("get_link_masses",
             [](const Articulation& self) {
                 return floatArrayFromVector(self.getLinkMasses());
             })
        .def("calc_mass", &Articulation::calcMass)
        .def("set_drive_targets",
             [](Articulation& self, const FloatArray& targets, float kp,
                float kd) {
                 self.setDriveTargets(floatVectorArray(targets, "targets"), kp,
                                      kd);
             },
             py::arg("targets"), py::arg("kp"), py::arg("kd"))
        .def("set_drive_targets",
             static_cast<void (Articulation::*)(const std::vector<float>&,
                                                float, float)>(
                 &Articulation::setDriveTargets),
             py::arg("targets"), py::arg("kp"), py::arg("kd"))
        .def("set_drive_targets",
             [](Articulation& self, const FloatArray& targets) {
                 self.setDriveTargets(floatVectorArray(targets, "targets"));
             },
             py::arg("targets"))
        .def("set_drive_targets",
             static_cast<void (Articulation::*)(const std::vector<float>&)>(
                 &Articulation::setDriveTargets),
             py::arg("targets"))
        .def("set_drive_velocity_targets",
             [](Articulation& self, const FloatArray& targets) {
                 self.setDriveVelocityTargets(
                     floatVectorArray(targets, "targets"));
             },
             py::arg("targets"))
        .def("set_drive_velocity_targets",
             &Articulation::setDriveVelocityTargets, py::arg("targets"))
        .def("set_kps",
             [](Articulation& self, const FloatArray& kps) {
                 self.setKPs(floatVectorArray(kps, "kps"));
             },
             py::arg("kps"))
        .def("set_kps", &Articulation::setKPs, py::arg("kps"))
        .def("set_kds",
             [](Articulation& self, const FloatArray& kds) {
                 self.setKDs(floatVectorArray(kds, "kds"));
             },
             py::arg("kds"))
        .def("set_kds", &Articulation::setKDs, py::arg("kds"))
        .def("set_effort_limits",
             [](Articulation& self, const FloatArray& effortLimits) {
                 self.setEffortLimits(
                     floatVectorArray(effortLimits, "effort_limits"));
             },
             py::arg("effort_limits"))
        .def("set_effort_limits", &Articulation::setEffortLimits,
             py::arg("effort_limits"))
        .def("set_joint_forces",
             [](Articulation& self, const FloatArray& forces) {
                 self.setJointForces(floatVectorArray(forces, "forces"));
             },
             py::arg("forces"))
        .def("set_joint_forces", &Articulation::setJointForces,
             py::arg("forces"))
        .def(
            "add_link_force",
            [](Articulation& self, int linkIndex, const FloatArray& force) {
                auto f = vec3ArrayView(force, "force");
                if (f.count != 1)
                    throw py::value_error("add_link_force expects force[3]");
                self.addLinkForce(linkIndex,
                                  PxVec3(f.data[0], f.data[1], f.data[2]));
            },
            py::arg("link_index"), py::arg("force"))
        .def(
            "add_link_force",
            [](Articulation& self, int linkIndex,
               const std::vector<float>& force) {
                if (force.size() != 3) {
                    throw std::runtime_error(
                        "add_link_force expects force[3]");
                }
                self.addLinkForce(
                    linkIndex,
                    PxVec3(force[0], force[1], force[2]));
            },
            py::arg("link_index"), py::arg("force"))
        .def("get_kps", &Articulation::getKPs)
        .def("get_kds", &Articulation::getKDs)
        .def("get_effort_limits",
             [](const Articulation& self) {
                 return floatArrayFromVector(self.getEffortLimits());
             })
        .def(
            "reset_root",
            [](Articulation& self, const glm::vec3& pos, const glm::quat& rot) {
                self.resetRoot(PxTransform(PxVec3(pos.x, pos.y, pos.z),
                                           PxQuat(rot.x, rot.y, rot.z, rot.w)));
            },
            py::arg("pos") = glm::vec3(0.f),
            py::arg("rot") = glm::quat(1.f, 0.f, 0.f, 0.f))
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
                    PxTransform(PxVec3(p.data[0], p.data[1], p.data[2]),
                                PxQuat(q.data[0], q.data[1], q.data[2],
                                       q.data[3])),
                    PxVec3(lv.data[0], lv.data[1], lv.data[2]),
                    PxVec3(av.data[0], av.data[1], av.data[2]));
            },
            py::arg("pos"), py::arg("rot_xyzw"),
            py::arg("linear_velocity"), py::arg("angular_velocity"))
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
            py::arg("angular_velocity") = std::vector<float>{0.f, 0.f, 0.f})
        .def("set_dof_state",
             [](Articulation& self, const FloatArray& positions,
                const FloatArray& velocities) {
                 self.setDofState(floatVectorArray(positions, "positions"),
                                  floatVectorArray(velocities, "velocities"));
             },
             py::arg("positions"), py::arg("velocities"))
        .def("set_dof_state", &Articulation::setDofState, py::arg("positions"),
             py::arg("velocities"))
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
        .def("add_instanced",
             static_cast<void (PhysicsBridge::*)(
                 const Articulation&, const std::vector<MeshHandle>&)>(
                 &PhysicsBridge::addInstanced),
             py::arg("artic"), py::arg("handles"))
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
