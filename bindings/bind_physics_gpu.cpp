///
/// GPU physics Python bindings (PhysX only)
///

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#ifdef KANGENGINE_USE_PHYSX
#include "physics/physics.hpp"
#include "physics/physics_gpu_system.hpp"
#endif

namespace py = pybind11;

void bind_physics_gpu(py::module& m) {
#ifdef KANGENGINE_USE_PHYSX
    using namespace KE;

    py::class_<GpuPhysicsConfig>(
        m, "GpuPhysicsConfig",
        "Configuration for explicit GPU physics state synchronization.")
        .def(py::init<>())
        .def_readwrite("cuda_device_id", &GpuPhysicsConfig::cudaDeviceId);

    py::class_<PhysicsGpuStateViews>(
        m, "PhysicsGpuStateViews",
        "GPU buffer views used by PhysicsGpuSystem.")
        .def(py::init<>())
        .def_readwrite("rigid_data", &PhysicsGpuStateViews::rigidData)
        .def_readwrite("rigid_force", &PhysicsGpuStateViews::rigidForce)
        .def_readwrite("rigid_torque", &PhysicsGpuStateViews::rigidTorque)
        .def_readwrite("articulation_link_data",
                       &PhysicsGpuStateViews::articulationLinkData)
        .def_readwrite("articulation_joint_positions",
                       &PhysicsGpuStateViews::articulationJointPositions)
        .def_readwrite("articulation_joint_velocities",
                       &PhysicsGpuStateViews::articulationJointVelocities)
        .def_readwrite("articulation_joint_accelerations",
                       &PhysicsGpuStateViews::articulationJointAccelerations)
        .def_readwrite("articulation_joint_forces",
                       &PhysicsGpuStateViews::articulationJointForces)
        .def_readwrite(
            "articulation_target_joint_positions",
            &PhysicsGpuStateViews::articulationTargetJointPositions)
        .def_readwrite(
            "articulation_target_joint_velocities",
            &PhysicsGpuStateViews::articulationTargetJointVelocities)
        .def_readwrite(
            "articulation_link_incoming_joint_forces",
            &PhysicsGpuStateViews::articulationLinkIncomingJointForces);

    py::class_<PhysicsGpuSystem>(
        m, "PhysicsGpuSystem",
        "Explicit GPU physics synchronization surface for high-throughput "
        "simulation.")
        .def(py::init<PhysicsWorld*, GpuPhysicsConfig>(), py::arg("world"),
             py::arg("config") = GpuPhysicsConfig{}, py::keep_alive<1, 2>())
        .def("init", &PhysicsGpuSystem::init)
        .def("invalidate", &PhysicsGpuSystem::invalidate)
        .def("is_initialized", &PhysicsGpuSystem::isInitialized)
        .def("check_initialized", &PhysicsGpuSystem::checkInitialized)
        .def("set_cuda_stream", &PhysicsGpuSystem::setCudaStream,
             py::arg("stream_handle"))
        .def("cuda_stream", &PhysicsGpuSystem::cudaStream)
        .def("step_start", &PhysicsGpuSystem::stepStart)
        .def("step_finish", &PhysicsGpuSystem::stepFinish)
        .def("rigid_data", &PhysicsGpuSystem::rigidData,
             py::return_value_policy::reference_internal)
        .def("rigid_force", &PhysicsGpuSystem::rigidForce,
             py::return_value_policy::reference_internal)
        .def("rigid_torque", &PhysicsGpuSystem::rigidTorque,
             py::return_value_policy::reference_internal)
        .def("articulation_link_data",
             &PhysicsGpuSystem::articulationLinkData,
             py::return_value_policy::reference_internal)
        .def("articulation_joint_positions",
             &PhysicsGpuSystem::articulationJointPositions,
             py::return_value_policy::reference_internal)
        .def("articulation_joint_velocities",
             &PhysicsGpuSystem::articulationJointVelocities,
             py::return_value_policy::reference_internal)
        .def("articulation_joint_accelerations",
             &PhysicsGpuSystem::articulationJointAccelerations,
             py::return_value_policy::reference_internal)
        .def("articulation_joint_forces",
             &PhysicsGpuSystem::articulationJointForces,
             py::return_value_policy::reference_internal)
        .def("articulation_target_joint_positions",
             &PhysicsGpuSystem::articulationTargetJointPositions,
             py::return_value_policy::reference_internal)
        .def("articulation_target_joint_velocities",
             &PhysicsGpuSystem::articulationTargetJointVelocities,
             py::return_value_policy::reference_internal)
        .def("articulation_link_incoming_joint_forces",
             &PhysicsGpuSystem::articulationLinkIncomingJointForces,
             py::return_value_policy::reference_internal)
        .def("fetch_rigid_data", &PhysicsGpuSystem::fetchRigidData)
        .def("fetch_articulation_link_pose",
             &PhysicsGpuSystem::fetchArticulationLinkPose)
        .def("fetch_articulation_link_vel",
             &PhysicsGpuSystem::fetchArticulationLinkVel)
        .def("fetch_articulation_joint_positions",
             &PhysicsGpuSystem::fetchArticulationJointPositions)
        .def("fetch_articulation_joint_velocities",
             &PhysicsGpuSystem::fetchArticulationJointVelocities)
        .def("fetch_articulation_joint_accelerations",
             &PhysicsGpuSystem::fetchArticulationJointAccelerations)
        .def("fetch_articulation_target_joint_positions",
             &PhysicsGpuSystem::fetchArticulationTargetJointPositions)
        .def("fetch_articulation_target_joint_velocities",
             &PhysicsGpuSystem::fetchArticulationTargetJointVelocities)
        .def("fetch_articulation_link_incoming_joint_force",
             &PhysicsGpuSystem::fetchArticulationLinkIncomingJointForce)
        .def("apply_rigid_data", &PhysicsGpuSystem::applyRigidData,
             py::arg("indices") = nullptr)
        .def("apply_rigid_force", &PhysicsGpuSystem::applyRigidForce,
             py::arg("indices") = nullptr)
        .def("apply_rigid_torque", &PhysicsGpuSystem::applyRigidTorque,
             py::arg("indices") = nullptr)
        .def("apply_articulation_root_pose",
             &PhysicsGpuSystem::applyArticulationRootPose,
             py::arg("indices") = nullptr)
        .def("apply_articulation_root_vel",
             &PhysicsGpuSystem::applyArticulationRootVel,
             py::arg("indices") = nullptr)
        .def("apply_articulation_joint_positions",
             &PhysicsGpuSystem::applyArticulationJointPositions,
             py::arg("indices") = nullptr)
        .def("apply_articulation_joint_velocities",
             &PhysicsGpuSystem::applyArticulationJointVelocities,
             py::arg("indices") = nullptr)
        .def("apply_articulation_joint_forces",
             &PhysicsGpuSystem::applyArticulationJointForces,
             py::arg("indices") = nullptr)
        .def("apply_articulation_target_joint_positions",
             &PhysicsGpuSystem::applyArticulationTargetJointPositions,
             py::arg("indices") = nullptr)
        .def("apply_articulation_target_joint_velocities",
             &PhysicsGpuSystem::applyArticulationTargetJointVelocities,
             py::arg("indices") = nullptr)
        .def("update_articulation_kinematics",
             &PhysicsGpuSystem::updateArticulationKinematics)
        .def("sync_poses_gpu_to_cpu", &PhysicsGpuSystem::syncPosesGpuToCpu)
        .def("views",
             static_cast<PhysicsGpuStateViews& (PhysicsGpuSystem::*)()>(
                 &PhysicsGpuSystem::views),
             py::return_value_policy::reference_internal);
#else
    (void)m;
#endif
}
