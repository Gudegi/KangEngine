///
/// GPU physics Python bindings (PhysX only)
///

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#ifdef KANGENGINE_USE_PHYSX
#include "physics/articulation.hpp"
#include "physics/physics.hpp"
#include "physics/physics_gpu_system.hpp"
#ifdef KANGENGINE_USE_CUDA
#include "physics/physics_gpu_system_kernels.hpp"
#endif
#endif

namespace py = pybind11;

void bind_physics_gpu(py::module& m) {
#ifdef KANGENGINE_USE_PHYSX
    using namespace KE;

    py::module physics = py::reinterpret_borrow<py::module>(m.attr("physics"));

    py::class_<GpuPhysicsConfig>(
        physics, "GpuPhysicsConfig",
        "Configuration for explicit GPU physics state synchronization.")
        .def(py::init([](int cuda_device_id, uint32_t max_contact_pairs,
                         uint32_t max_contact_points) {
                 GpuPhysicsConfig config;
                 config.cudaDeviceId = cuda_device_id;
                 config.maxContactPairs = max_contact_pairs;
                 config.maxContactPoints = max_contact_points;
                 return config;
             }),
             py::kw_only(),
             py::arg("cuda_device_id") = GpuPhysicsConfig{}.cudaDeviceId,
             py::arg("max_contact_pairs") = GpuPhysicsConfig{}.maxContactPairs,
             py::arg("max_contact_points") =
                 GpuPhysicsConfig{}.maxContactPoints,
             "Create GPU synchronization configuration from keyword arguments.")
        .def_readwrite("cuda_device_id", &GpuPhysicsConfig::cudaDeviceId,
                       "CUDA device ordinal used by GPU physics.")
        .def_readwrite("max_contact_pairs", &GpuPhysicsConfig::maxContactPairs,
                       "Maximum mirrored contact pair count.")
        .def_readwrite("max_contact_points",
                       &GpuPhysicsConfig::maxContactPoints,
                       "Maximum mirrored contact point count.")
        .def("__repr__", [](const GpuPhysicsConfig& config) {
            return py::str("GpuPhysicsConfig(cuda_device_id={!r}, "
                           "max_contact_pairs={!r}, max_contact_points={!r})")
                .attr("format")(config.cudaDeviceId, config.maxContactPairs,
                                config.maxContactPoints);
        });

    py::class_<PhysicsGpuStateViews>(
        physics, "PhysicsGpuStateViews",
        "Backend-owned GPU state and command-buffer views. State buffers are "
        "refreshed by the corresponding fetch methods; command buffers are "
        "submitted by apply methods. The views become invalid when their "
        "PhysicsGpuSystem is released.")
        .def(py::init<>())
        .def_readwrite("rigid_data", &PhysicsGpuStateViews::rigidData,
                       "Rigid pose and velocity state refreshed by "
                       "fetch_rigid_data().")
        .def_readwrite("rigid_force", &PhysicsGpuStateViews::rigidForce,
                       "Rigid force command buffer submitted by "
                       "apply_rigid_force().")
        .def_readwrite("rigid_torque", &PhysicsGpuStateViews::rigidTorque,
                       "Rigid torque command buffer submitted by "
                       "apply_rigid_torque().")
        .def_readwrite("rigid_accelerations",
                       &PhysicsGpuStateViews::rigidAccelerations,
                       "Rigid COM linear/angular acceleration state.")
        .def_readwrite("articulation_link_data",
                       &PhysicsGpuStateViews::articulationLinkData,
                       "Articulation link pose and velocity state.")
        .def_readwrite("articulation_joint_positions",
                       &PhysicsGpuStateViews::articulationJointPositions,
                       "Articulation joint-position state.")
        .def_readwrite("articulation_joint_velocities",
                       &PhysicsGpuStateViews::articulationJointVelocities,
                       "Articulation joint-velocity state.")
        .def_readwrite("articulation_joint_accelerations",
                       &PhysicsGpuStateViews::articulationJointAccelerations,
                       "Articulation joint-acceleration state.")
        .def_readwrite("articulation_joint_forces",
                       &PhysicsGpuStateViews::articulationJointForces,
                       "Articulation joint-force state.")
        .def_readwrite("articulation_target_joint_positions",
                       &PhysicsGpuStateViews::articulationTargetJointPositions,
                       "Joint-position target command buffer.")
        .def_readwrite("articulation_target_joint_velocities",
                       &PhysicsGpuStateViews::articulationTargetJointVelocities,
                       "Joint-velocity target command buffer.")
        .def_readwrite(
            "articulation_link_incoming_joint_forces",
            &PhysicsGpuStateViews::articulationLinkIncomingJointForces,
            "Incoming joint-force state for articulation links.")
        .def_readwrite("articulation_link_accelerations",
                       &PhysicsGpuStateViews::articulationLinkAccelerations)
        .def_readwrite("articulation_link_forces",
                       &PhysicsGpuStateViews::articulationLinkForces)
        .def_readwrite("articulation_link_torques",
                       &PhysicsGpuStateViews::articulationLinkTorques)
        .def_readwrite("articulation_dense_jacobians",
                       &PhysicsGpuStateViews::articulationDenseJacobians)
        .def_readwrite("articulation_mass_matrices",
                       &PhysicsGpuStateViews::articulationMassMatrices)
        .def_readwrite("articulation_gravity_forces",
                       &PhysicsGpuStateViews::articulationGravityForces)
        .def_readwrite("articulation_coriolis_forces",
                       &PhysicsGpuStateViews::articulationCoriolisForces)
        .def_readwrite("articulation_com_world",
                       &PhysicsGpuStateViews::articulationComWorld)
        .def_readwrite("articulation_com_root",
                       &PhysicsGpuStateViews::articulationComRoot)
        .def_readwrite(
            "articulation_centroidal_momentum_matrices",
            &PhysicsGpuStateViews::articulationCentroidalMomentumMatrices)
        .def_readwrite("articulation_centroidal_bias_forces",
                       &PhysicsGpuStateViews::articulationCentroidalBiasForces)
        .def_readwrite("contact_pairs", &PhysicsGpuStateViews::contactPairs,
                       "Packed contact-pair state buffer.")
        .def_readwrite("contact_pair_count",
                       &PhysicsGpuStateViews::contactPairCount,
                       "Number of valid packed contact pairs.")
        .def_readwrite("contact_pair_headers",
                       &PhysicsGpuStateViews::contactPairHeaders,
                       "Headers describing packed contact-pair ranges.")
        .def_readwrite("contact_pair_body_refs",
                       &PhysicsGpuStateViews::contactPairBodyRefs,
                       "Body references for packed contact pairs.")
        .def_readwrite("contact_points", &PhysicsGpuStateViews::contactPoints,
                       "Packed contact-point state buffer.")
        .def_readwrite("contact_point_count",
                       &PhysicsGpuStateViews::contactPointCount,
                       "Number of valid packed contact points.")
        .def_readwrite("contact_point_pair_indices",
                       &PhysicsGpuStateViews::contactPointPairIndices,
                       "Contact-pair index for each packed contact point.");

    py::class_<PhysicsGpuSystem>(
        physics, "PhysicsGpuSystem",
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
        .def("rigid_row", &PhysicsGpuSystem::rigidRow, py::arg("rigid"),
             "Return the logical row used by PhysicsGpuSystem for a rigid.")
        .def(
            "articulation_row",
            [](const PhysicsGpuSystem& self, Articulation& articulation) {
                if (!articulation.raw())
                    throw std::runtime_error(
                        "articulation_row requires a valid articulation");
                return self.articulationRow(*articulation.raw());
            },
            py::arg("articulation"),
            "Return the logical row used by PhysicsGpuSystem for an "
            "articulation.")
        .def("articulation_link_count",
             &PhysicsGpuSystem::articulationLinkCount,
             py::arg("articulation_row"))
        .def("articulation_dof_count", &PhysicsGpuSystem::articulationDofCount,
             py::arg("articulation_row"))
        .def("articulation_count", &PhysicsGpuSystem::articulationCount)
        .def("articulation_max_links", &PhysicsGpuSystem::articulationMaxLinks)
        .def("articulation_max_dofs", &PhysicsGpuSystem::articulationMaxDofs)
        .def("articulation_generalized_dof_count",
             &PhysicsGpuSystem::articulationGeneralizedDofCount,
             py::arg("articulation_row"))
        .def("articulation_jacobian_row_count",
             &PhysicsGpuSystem::articulationJacobianRowCount,
             py::arg("articulation_row"))
        .def("articulation_max_generalized_dofs",
             &PhysicsGpuSystem::articulationMaxGeneralizedDofs)
        .def("articulation_max_jacobian_rows",
             &PhysicsGpuSystem::articulationMaxJacobianRows)
        .def("step_start", &PhysicsGpuSystem::stepStart)
        .def("step_finish", &PhysicsGpuSystem::stepFinish)
        .def("rigid_data", &PhysicsGpuSystem::rigidData,
             py::return_value_policy::reference_internal)
        .def("rigid_force", &PhysicsGpuSystem::rigidForce,
             py::return_value_policy::reference_internal)
        .def("rigid_torque", &PhysicsGpuSystem::rigidTorque,
             py::return_value_policy::reference_internal)
        .def("rigid_accelerations", &PhysicsGpuSystem::rigidAccelerations,
             py::return_value_policy::reference_internal)
        .def("articulation_link_data", &PhysicsGpuSystem::articulationLinkData,
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
        .def("articulation_link_accelerations",
             &PhysicsGpuSystem::articulationLinkAccelerations,
             py::return_value_policy::reference_internal)
        .def("articulation_link_forces",
             &PhysicsGpuSystem::articulationLinkForces,
             py::return_value_policy::reference_internal)
        .def("articulation_link_torques",
             &PhysicsGpuSystem::articulationLinkTorques,
             py::return_value_policy::reference_internal)
        .def("articulation_dense_jacobians",
             &PhysicsGpuSystem::articulationDenseJacobians,
             py::return_value_policy::reference_internal)
        .def("articulation_mass_matrices",
             &PhysicsGpuSystem::articulationMassMatrices,
             py::return_value_policy::reference_internal)
        .def("articulation_gravity_forces",
             &PhysicsGpuSystem::articulationGravityForces,
             py::return_value_policy::reference_internal)
        .def("articulation_coriolis_forces",
             &PhysicsGpuSystem::articulationCoriolisForces,
             py::return_value_policy::reference_internal)
        .def("articulation_com_world", &PhysicsGpuSystem::articulationComWorld,
             py::return_value_policy::reference_internal)
        .def("articulation_com_root", &PhysicsGpuSystem::articulationComRoot,
             py::return_value_policy::reference_internal)
        .def("articulation_centroidal_momentum_matrices",
             &PhysicsGpuSystem::articulationCentroidalMomentumMatrices,
             py::return_value_policy::reference_internal)
        .def("articulation_centroidal_bias_forces",
             &PhysicsGpuSystem::articulationCentroidalBiasForces,
             py::return_value_policy::reference_internal)
        .def("contact_pairs", &PhysicsGpuSystem::contactPairs,
             py::return_value_policy::reference_internal)
        .def("contact_pair_count", &PhysicsGpuSystem::contactPairCount,
             py::return_value_policy::reference_internal)
        .def("contact_pair_headers", &PhysicsGpuSystem::contactPairHeaders,
             py::return_value_policy::reference_internal)
        .def("contact_pair_body_refs", &PhysicsGpuSystem::contactPairBodyRefs,
             py::return_value_policy::reference_internal)
        .def("contact_points", &PhysicsGpuSystem::contactPoints,
             py::return_value_policy::reference_internal)
        .def("contact_point_count", &PhysicsGpuSystem::contactPointCount,
             py::return_value_policy::reference_internal)
        .def("contact_point_pair_indices",
             &PhysicsGpuSystem::contactPointPairIndices,
             py::return_value_policy::reference_internal)
        .def("fetch_rigid_data", &PhysicsGpuSystem::fetchRigidData)
        .def("fetch_rigid_accelerations",
             &PhysicsGpuSystem::fetchRigidAccelerations)
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
        .def("fetch_articulation_joint_forces",
             &PhysicsGpuSystem::fetchArticulationJointForces)
        .def("fetch_articulation_target_joint_positions",
             &PhysicsGpuSystem::fetchArticulationTargetJointPositions)
        .def("fetch_articulation_target_joint_velocities",
             &PhysicsGpuSystem::fetchArticulationTargetJointVelocities)
        .def("fetch_articulation_link_incoming_joint_force",
             &PhysicsGpuSystem::fetchArticulationLinkIncomingJointForce)
        .def("fetch_articulation_link_accelerations",
             &PhysicsGpuSystem::fetchArticulationLinkAccelerations)
        .def("prepare_articulation_link_commands",
             &PhysicsGpuSystem::prepareArticulationLinkCommands)
        .def("apply_articulation_link_forces",
             &PhysicsGpuSystem::applyArticulationLinkForces)
        .def("apply_articulation_link_torques",
             &PhysicsGpuSystem::applyArticulationLinkTorques)
        .def("compute_articulation_dense_jacobians",
             &PhysicsGpuSystem::computeArticulationDenseJacobians)
        .def("compute_articulation_mass_matrices",
             &PhysicsGpuSystem::computeArticulationMassMatrices)
        .def("compute_articulation_gravity_forces",
             &PhysicsGpuSystem::computeArticulationGravityForces)
        .def("compute_articulation_coriolis_forces",
             &PhysicsGpuSystem::computeArticulationCoriolisForces)
        .def("compute_articulation_com_world",
             &PhysicsGpuSystem::computeArticulationComWorld)
        .def("compute_articulation_com_root",
             &PhysicsGpuSystem::computeArticulationComRoot)
        .def("compute_articulation_centroidal_momentum",
             &PhysicsGpuSystem::computeArticulationCentroidalMomentum)
        .def("fetch_contact_pairs", &PhysicsGpuSystem::fetchContactPairs)
        .def("clear_contact_data", &PhysicsGpuSystem::clearContactData)
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
        .def("clear_rigid_commands", &PhysicsGpuSystem::clearRigidCommands,
             py::arg("indices") = nullptr)
        .def("clear_articulation_commands",
             &PhysicsGpuSystem::clearArticulationCommands,
             py::arg("indices") = nullptr)
        .def("update_articulation_kinematics",
             &PhysicsGpuSystem::updateArticulationKinematics)
        .def("sync_poses_gpu_to_cpu", &PhysicsGpuSystem::syncPosesGpuToCpu)
        .def("views",
             static_cast<PhysicsGpuStateViews& (PhysicsGpuSystem::*)()>(
                 &PhysicsGpuSystem::views),
             py::return_value_policy::reference_internal);
#ifdef KANGENGINE_USE_CUDA
    physics.def(
        "aggregate_contact_sensors_cuda",
        &PhysicsGpuKernels::aggregateContactSensorsCUDA,
        py::arg("contact_pair_body_refs"), py::arg("contact_pair_count"),
        py::arg("contact_points"), py::arg("contact_point_count"),
        py::arg("contact_point_pair_indices"), py::arg("sensor_descriptors"),
        py::arg("row_to_environment"), py::arg("body_to_slot"),
        py::arg("contact_count"), py::arg("in_contact"), py::arg("net_impulse"),
        "Aggregate contact-pair body refs into batched sensor outputs.");
#endif
#else
    (void)m;
#endif
}
