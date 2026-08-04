#include "physics/physics.hpp"
#include "physics/physics_gpu_system.hpp"
#include "sim/gpu_transform_kernels.hpp"

#include <cmath>
#include <fmt/base.h>
#include <PxActor.h>
#include <extensions/PxRigidActorExt.h>
#include <extensions/PxRigidBodyExt.h>
#include <algorithm>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <limits>
#include <stdexcept>
#include <vector>

#ifdef KANGENGINE_USE_CUDA
#include <cuda_runtime.h>
#endif

using namespace KE;

namespace {

glm::quat zQuat(float angle) {
    return glm::angleAxis(angle, glm::vec3(0.0f, 0.0f, 1.0f));
}

float absDiff(float a, float b) { return std::fabs(a - b); }

PxArticulationReducedCoordinate* createFixedTestArticulation(
    PhysicsWorld& world) {
    auto* articulation =
        world.getPhysics()->createArticulationReducedCoordinate();
    articulation->setArticulationFlag(PxArticulationFlag::eFIX_BASE, true);

    auto* root = articulation->createLink(
        nullptr, PxTransform(PxVec3(0.0f, 4.0f, 0.0f)));
    PxRigidActorExt::createExclusiveShape(
        *root, PxBoxGeometry(0.1f, 0.1f, 0.1f), *world.getMaterial());
    PxRigidBodyExt::updateMassAndInertia(*root, 1.0f);

    auto* child = articulation->createLink(
        root, PxTransform(PxVec3(0.0f, 4.0f, 1.0f)));
    PxRigidActorExt::createExclusiveShape(
        *child, PxBoxGeometry(0.1f, 0.1f, 0.1f), *world.getMaterial());
    PxRigidBodyExt::updateMassAndInertia(*child, 1.0f);
    auto* joint = static_cast<PxArticulationJointReducedCoordinate*>(
        child->getInboundJoint());
    joint->setJointType(PxArticulationJointType::eREVOLUTE);
    joint->setParentPose(PxTransform(PxVec3(0.0f, 0.0f, 0.5f)));
    joint->setChildPose(PxTransform(PxVec3(0.0f, 0.0f, -0.5f)));
    joint->setMotion(PxArticulationAxis::eTWIST,
                     PxArticulationMotion::eFREE);

    world.getScene()->addArticulation(*articulation);
    return articulation;
}

} // namespace

#define NUM_ACTORS 8

int main() {
    PhysicsConfig config = PhysicsConfig::zUp();
    config.enableGPU = true;
    config.gravity[0] = 0.0f;
    config.gravity[1] = 0.0f;
    config.gravity[2] = -9.8f;
    config.enableContactReports = false;

    PhysicsWorld world(config);
    std::vector<PxRigidDynamic*> actors;
    actors.reserve(NUM_ACTORS);
    for (int i = 0; i < NUM_ACTORS; ++i) {
        const float x = static_cast<float>(i - 4) * 1.25f;
        auto* actor = world.createDynamicBox(
            glm::vec3(0.18f, 0.16f, 0.14f),
            glm::vec3(x, 0.35f * static_cast<float>(i % 3),
                      2.0f + 0.15f * static_cast<float>(i)),
            zQuat(0.07f * static_cast<float>(i)), 1.0f);
        actor->setLinearVelocity(PxVec3(0.04f * static_cast<float>(i + 1),
                                        -0.03f * static_cast<float>(i % 2),
                                        0.02f));
        actor->setAngularVelocity(PxVec3(0.0f));
        actor->wakeUp();
        actors.push_back(actor);
    }
    auto* articulation = createFixedTestArticulation(world);

    GpuPhysicsConfig gpuConfig;
    gpuConfig.cudaDeviceId = 0;
    PhysicsGpuSystem gpuSystem(&world, gpuConfig);
    gpuSystem.init();

    gpuSystem.fetchRigidData();

    const auto& view = gpuSystem.rigidData();
    fmt::print("C++ PhysX GPU smoke\n");
    fmt::print("  rigid shape : [{}, {}]\n", view.shape.at(0),
               view.shape.at(1));
    fmt::print("  version     : {}\n", view.version);
    fmt::print("  ptr         : {}\n", view.data);

#ifdef KANGENGINE_USE_CUDA
    std::vector<float> host(static_cast<size_t>(view.shape.at(0)) *
                            static_cast<size_t>(view.shape.at(1)));
    if (cudaDeviceSynchronize() != cudaSuccess)
        throw std::runtime_error("cudaDeviceSynchronize failed");
    if (cudaMemcpy(host.data(), view.data, sizeof(float) * host.size(),
                   cudaMemcpyDeviceToHost) != cudaSuccess)
        throw std::runtime_error("cudaMemcpy rigid data failed");

    gpuSystem.fetchArticulationLinkPose();
    const auto& linkView = gpuSystem.articulationLinkData();
    if (linkView.shape.size() != 3 || linkView.shape.at(0) != 1 ||
        linkView.shape.at(1) != 2 || linkView.shape.at(2) != 13)
        throw std::runtime_error("unexpected articulation link data shape");
    const uint32_t articulationRow = gpuSystem.articulationRow(*articulation);
    if (articulationRow != 0 || gpuSystem.articulationCount() != 1 ||
        gpuSystem.articulationMaxLinks() != 2 ||
        gpuSystem.articulationLinkCount(articulationRow) != 2)
        throw std::runtime_error("unexpected articulation GPU row metadata");
    if (linkView.version != 1)
        throw std::runtime_error(
            "articulation link pose fetch did not advance the view version");
    if (cudaDeviceSynchronize() != cudaSuccess)
        throw std::runtime_error(
            "cudaDeviceSynchronize after articulation link fetch failed");

    std::vector<float> linkHost(2 * 13);
    if (cudaMemcpy(linkHost.data(), linkView.data,
                   sizeof(float) * linkHost.size(), cudaMemcpyDeviceToHost) !=
        cudaSuccess)
        throw std::runtime_error("cudaMemcpy articulation link data failed");
    const PxVec3 expectedLinkPositions[2] = {
        PxVec3(0.0f, 4.0f, 0.0f), PxVec3(0.0f, 4.0f, 1.0f)};
    float maxLinkPositionError = 0.0f;
    float maxLinkQuaternionError = 0.0f;
    for (size_t link = 0; link < 2; ++link) {
        const size_t row = link * 13;
        maxLinkPositionError = std::max(
            maxLinkPositionError,
            absDiff(linkHost[row + 0], expectedLinkPositions[link].x));
        maxLinkPositionError = std::max(
            maxLinkPositionError,
            absDiff(linkHost[row + 1], expectedLinkPositions[link].y));
        maxLinkPositionError = std::max(
            maxLinkPositionError,
            absDiff(linkHost[row + 2], expectedLinkPositions[link].z));
        maxLinkQuaternionError =
            std::max(maxLinkQuaternionError, std::fabs(linkHost[row + 3]));
        maxLinkQuaternionError =
            std::max(maxLinkQuaternionError, std::fabs(linkHost[row + 4]));
        maxLinkQuaternionError =
            std::max(maxLinkQuaternionError, std::fabs(linkHost[row + 5]));
        maxLinkQuaternionError = std::max(
            maxLinkQuaternionError, absDiff(linkHost[row + 6], 1.0f));
    }
    fmt::print("  articulation shape: [{}, {}, {}]\n",
               linkView.shape.at(0), linkView.shape.at(1),
               linkView.shape.at(2));
    fmt::print("  articulation pose err: {} / {}\n",
               maxLinkPositionError, maxLinkQuaternionError);
    if (maxLinkPositionError > 1e-4f || maxLinkQuaternionError > 1e-4f)
        throw std::runtime_error(
            "GPU articulation link pose does not match expected state");

    gpuSystem.fetchArticulationLinkVel();
    if (linkView.version != 2)
        throw std::runtime_error(
            "articulation link velocity fetch did not advance the view "
            "version");
    if (cudaDeviceSynchronize() != cudaSuccess)
        throw std::runtime_error(
            "cudaDeviceSynchronize after articulation link velocity fetch "
            "failed");
    if (cudaMemcpy(linkHost.data(), linkView.data,
                   sizeof(float) * linkHost.size(), cudaMemcpyDeviceToHost) !=
        cudaSuccess)
        throw std::runtime_error(
            "cudaMemcpy articulation link velocity data failed");
    float maxLinkVelocity = 0.0f;
    for (size_t link = 0; link < 2; ++link) {
        const size_t row = link * 13;
        for (size_t component = 7; component < 13; ++component)
            maxLinkVelocity =
                std::max(maxLinkVelocity, std::fabs(linkHost[row + component]));
    }
    fmt::print("  articulation max vel: {}\n", maxLinkVelocity);
    if (maxLinkVelocity > 1e-5f)
        throw std::runtime_error(
            "fixed articulation GPU link velocity is not zero");

    if (gpuSystem.articulationMaxDofs() != 1 ||
        gpuSystem.articulationDofCount(articulationRow) != 1)
        throw std::runtime_error("unexpected articulation GPU DOF metadata");
    gpuSystem.fetchArticulationJointPositions();
    gpuSystem.fetchArticulationJointVelocities();
    gpuSystem.fetchArticulationJointAccelerations();
    gpuSystem.fetchArticulationJointForces();
    gpuSystem.fetchArticulationTargetJointPositions();
    gpuSystem.fetchArticulationTargetJointVelocities();
    gpuSystem.fetchArticulationLinkIncomingJointForce();
    const auto& jointPositionView =
        gpuSystem.articulationJointPositions();
    const auto& jointVelocityView =
        gpuSystem.articulationJointVelocities();
    const auto& jointAccelerationView =
        gpuSystem.articulationJointAccelerations();
    const auto& jointForceView = gpuSystem.articulationJointForces();
    const auto& targetJointPositionView =
        gpuSystem.articulationTargetJointPositions();
    const auto& targetJointVelocityView =
        gpuSystem.articulationTargetJointVelocities();
    const auto& incomingJointForceView =
        gpuSystem.articulationLinkIncomingJointForces();
    if (jointPositionView.shape != std::vector<int64_t>({1, 1}) ||
        jointVelocityView.shape != std::vector<int64_t>({1, 1}) ||
        jointAccelerationView.shape != std::vector<int64_t>({1, 1}) ||
        jointForceView.shape != std::vector<int64_t>({1, 1}) ||
        targetJointPositionView.shape != std::vector<int64_t>({1, 1}) ||
        targetJointVelocityView.shape != std::vector<int64_t>({1, 1}) ||
        incomingJointForceView.shape != std::vector<int64_t>({1, 2, 6}))
        throw std::runtime_error("unexpected articulation joint view shape");
    if (jointPositionView.version != 1 || jointVelocityView.version != 1 ||
        jointAccelerationView.version != 1 || jointForceView.version != 1 ||
        targetJointPositionView.version != 1 ||
        targetJointVelocityView.version != 1 ||
        incomingJointForceView.version != 1)
        throw std::runtime_error(
            "articulation joint fetch did not advance view versions");
    if (cudaDeviceSynchronize() != cudaSuccess)
        throw std::runtime_error(
            "cudaDeviceSynchronize after articulation joint fetch failed");
    float jointPosition = 0.0f;
    float jointVelocity = 0.0f;
    float jointAcceleration = 0.0f;
    float jointForce = 0.0f;
    float targetJointPosition = 0.0f;
    float targetJointVelocity = 0.0f;
    if (cudaMemcpy(&jointPosition, jointPositionView.data, sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(&jointVelocity, jointVelocityView.data, sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(&jointAcceleration, jointAccelerationView.data,
                   sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(&jointForce, jointForceView.data, sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(&targetJointPosition, targetJointPositionView.data,
                   sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(&targetJointVelocity, targetJointVelocityView.data,
                   sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
        throw std::runtime_error("cudaMemcpy articulation joint state failed");
    std::vector<float> incomingJointForce(2 * 6);
    if (cudaMemcpy(incomingJointForce.data(), incomingJointForceView.data,
                   sizeof(float) * incomingJointForce.size(),
                   cudaMemcpyDeviceToHost) != cudaSuccess)
        throw std::runtime_error(
            "cudaMemcpy articulation incoming joint force failed");
    fmt::print("  articulation qpos/qvel: {} / {}\n", jointPosition,
               jointVelocity);
    fmt::print("  articulation qacc/qf/target: {} / {} / {} / {}\n",
               jointAcceleration, jointForce, targetJointPosition,
               targetJointVelocity);
    if (std::fabs(jointPosition) > 1e-5f ||
        std::fabs(jointVelocity) > 1e-5f ||
        !std::isfinite(jointAcceleration) || !std::isfinite(jointForce) ||
        !std::isfinite(targetJointPosition) ||
        !std::isfinite(targetJointVelocity) ||
        std::any_of(incomingJointForce.begin(), incomingJointForce.end(),
                    [](float value) { return !std::isfinite(value); }))
        throw std::runtime_error(
            "articulation joint GPU state does not match expected state");

    jointForce = 1.5f;
    targetJointPosition = 0.125f;
    targetJointVelocity = -0.25f;
    if (cudaMemcpy(jointForceView.data, &jointForce, sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(targetJointPositionView.data, &targetJointPosition,
                   sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(targetJointVelocityView.data, &targetJointVelocity,
                   sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
        throw std::runtime_error(
            "cudaMemcpy articulation joint command buffers failed");
    gpuSystem.applyArticulationJointForces();
    gpuSystem.applyArticulationTargetJointPositions();
    gpuSystem.applyArticulationTargetJointVelocities();
    gpuSystem.fetchArticulationJointForces();
    gpuSystem.fetchArticulationTargetJointPositions();
    gpuSystem.fetchArticulationTargetJointVelocities();
    if (jointForceView.version != 3 || targetJointPositionView.version != 3 ||
        targetJointVelocityView.version != 3)
        throw std::runtime_error(
            "articulation joint command apply/fetch did not advance versions");
    if (cudaDeviceSynchronize() != cudaSuccess)
        throw std::runtime_error(
            "cudaDeviceSynchronize after articulation command apply failed");
    float appliedJointForce = 0.0f;
    float appliedTargetJointPosition = 0.0f;
    float appliedTargetJointVelocity = 0.0f;
    if (cudaMemcpy(&appliedJointForce, jointForceView.data, sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(&appliedTargetJointPosition, targetJointPositionView.data,
                   sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(&appliedTargetJointVelocity, targetJointVelocityView.data,
                   sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
        throw std::runtime_error(
            "cudaMemcpy applied articulation joint command state failed");
    fmt::print("  articulation command echo: {} / {} / {}\n",
               appliedJointForce, appliedTargetJointPosition,
               appliedTargetJointVelocity);
    if (absDiff(appliedJointForce, jointForce) > 1e-5f ||
        absDiff(appliedTargetJointPosition, targetJointPosition) > 1e-5f ||
        absDiff(appliedTargetJointVelocity, targetJointVelocity) > 1e-5f)
        throw std::runtime_error(
            "articulation joint command GPU echo does not match");

    constexpr uint32_t sparseArticulation = 0;
    uint32_t* sparseArticulationData = nullptr;
    if (cudaMalloc(&sparseArticulationData, sizeof(uint32_t)) != cudaSuccess)
        throw std::runtime_error(
            "cudaMalloc sparse articulation indices failed");
    if (cudaMemcpy(sparseArticulationData, &sparseArticulation,
                   sizeof(uint32_t), cudaMemcpyHostToDevice) != cudaSuccess)
        throw std::runtime_error(
            "cudaMemcpy sparse articulation indices failed");
    Sim::GpuArrayView sparseArticulationIndices;
    sparseArticulationIndices.data = sparseArticulationData;
    sparseArticulationIndices.memoryType = Sim::SimMemoryType::CUDADevice;
    sparseArticulationIndices.dtype = Sim::SimDType::UInt32;
    sparseArticulationIndices.deviceId = gpuConfig.cudaDeviceId;
    sparseArticulationIndices.shape = {1};
    sparseArticulationIndices.strides = {1};
    sparseArticulationIndices.streamHandle = gpuSystem.cudaStream();
    sparseArticulationIndices.name =
        "physx_gpu_step_smoke_sparse_articulation_indices";

    jointForce = 2.5f;
    targetJointPosition = 0.375f;
    targetJointVelocity = -0.5f;
    if (cudaMemcpy(jointForceView.data, &jointForce, sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(targetJointPositionView.data, &targetJointPosition,
                   sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(targetJointVelocityView.data, &targetJointVelocity,
                   sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
        throw std::runtime_error(
            "cudaMemcpy sparse articulation joint command buffers failed");
    gpuSystem.applyArticulationJointForces(&sparseArticulationIndices);
    gpuSystem.applyArticulationTargetJointPositions(
        &sparseArticulationIndices);
    gpuSystem.applyArticulationTargetJointVelocities(
        &sparseArticulationIndices);
    if (cudaDeviceSynchronize() != cudaSuccess)
        throw std::runtime_error(
            "cudaDeviceSynchronize after sparse articulation command apply "
            "failed");
    cudaFree(sparseArticulationData);
    gpuSystem.fetchArticulationJointForces();
    gpuSystem.fetchArticulationTargetJointPositions();
    gpuSystem.fetchArticulationTargetJointVelocities();
    if (jointForceView.version != 5 || targetJointPositionView.version != 5 ||
        targetJointVelocityView.version != 5)
        throw std::runtime_error(
            "sparse articulation command apply/fetch did not advance "
            "versions");
    if (cudaDeviceSynchronize() != cudaSuccess)
        throw std::runtime_error(
            "cudaDeviceSynchronize after sparse articulation command fetch "
            "failed");
    if (cudaMemcpy(&appliedJointForce, jointForceView.data, sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(&appliedTargetJointPosition, targetJointPositionView.data,
                   sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(&appliedTargetJointVelocity, targetJointVelocityView.data,
                   sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
        throw std::runtime_error(
            "cudaMemcpy sparse articulation joint command state failed");
    fmt::print("  sparse articulation command echo: {} / {} / {}\n",
               appliedJointForce, appliedTargetJointPosition,
               appliedTargetJointVelocity);
    if (absDiff(appliedJointForce, jointForce) > 1e-5f ||
        absDiff(appliedTargetJointPosition, targetJointPosition) > 1e-5f ||
        absDiff(appliedTargetJointVelocity, targetJointVelocity) > 1e-5f)
        throw std::runtime_error(
            "sparse articulation joint command GPU echo does not match");

    const float rootCommand[13] = {0.25f, 4.5f, 0.75f, 0.0f, 0.0f,
                                   0.0f,  1.0f, 0.2f, 0.0f, -0.1f,
                                   0.0f,  0.3f, 0.0f};
    if (cudaMemcpy(linkView.data, rootCommand, sizeof(rootCommand),
                   cudaMemcpyHostToDevice) != cudaSuccess)
        throw std::runtime_error(
            "cudaMemcpy articulation root command state failed");
    gpuSystem.applyArticulationRootPose();
    gpuSystem.applyArticulationRootVel();
    gpuSystem.updateArticulationKinematics();
    gpuSystem.fetchArticulationLinkPose();
    gpuSystem.fetchArticulationLinkVel();
    if (cudaDeviceSynchronize() != cudaSuccess)
        throw std::runtime_error(
            "cudaDeviceSynchronize after articulation root apply failed");
    if (cudaMemcpy(linkHost.data(), linkView.data,
                   sizeof(float) * linkHost.size(), cudaMemcpyDeviceToHost) !=
        cudaSuccess)
        throw std::runtime_error(
            "cudaMemcpy articulation root applied state failed");
    float rootPoseError = 0.0f;
    float rootVelError = 0.0f;
    for (size_t i = 0; i < 7; ++i)
        rootPoseError =
            std::max(rootPoseError, absDiff(linkHost[i], rootCommand[i]));
    for (size_t i = 7; i < 13; ++i)
        rootVelError =
            std::max(rootVelError, absDiff(linkHost[i], rootCommand[i]));
    fmt::print("  articulation root apply err: {} / {}\n", rootPoseError,
               rootVelError);
    if (rootPoseError > 1e-5f || rootVelError > 1e-5f)
        throw std::runtime_error(
            "articulation root GPU apply does not match fetched state");

    const PxU32 actorCount = static_cast<PxU32>(view.shape.at(0));
    if (actorCount != NUM_ACTORS)
        throw std::runtime_error("unexpected rigid actor count");
    std::vector<bool> seenRows(actorCount, false);
    for (auto* actor : actors) {
        const uint32_t row = gpuSystem.rigidRow(*actor);
        if (row >= actorCount)
            throw std::runtime_error("rigid row is out of range");
        if (seenRows[row])
            throw std::runtime_error("rigid row mapping contains duplicates");
        seenRows[row] = true;
    }
    const auto& forceView = gpuSystem.rigidForce();
    const auto& torqueView = gpuSystem.rigidTorque();
    if (forceView.shape.size() != 2 || forceView.shape.at(0) != actorCount ||
        forceView.shape.at(1) != 3)
        throw std::runtime_error("unexpected rigid force shape");
    if (torqueView.shape.size() != 2 || torqueView.shape.at(0) != actorCount ||
        torqueView.shape.at(1) != 3)
        throw std::runtime_error("unexpected rigid torque shape");
    if (cudaMemset(forceView.data, 0, sizeof(float) * actorCount * 3) !=
        cudaSuccess)
        throw std::runtime_error("cudaMemset rigid force failed");
    if (cudaMemset(torqueView.data, 0, sizeof(float) * actorCount * 3) !=
        cudaSuccess)
        throw std::runtime_error("cudaMemset rigid torque failed");
    gpuSystem.applyRigidForce();
    gpuSystem.applyRigidTorque();
    if (cudaDeviceSynchronize() != cudaSuccess)
        throw std::runtime_error(
            "cudaDeviceSynchronize after rigid force/torque apply failed");
    if (forceView.version != 1 || torqueView.version != 1)
        throw std::runtime_error(
            "rigid force/torque apply did not update command versions");

    float maxPositionError = 0.0f;
    float maxQuatError = 0.0f;
    float maxVelocityError = 0.0f;
    for (PxU32 i = 0; i < actorCount; ++i) {
        const size_t row = static_cast<size_t>(i) * 13;
        const PxVec3 initialPosition(
            static_cast<float>(static_cast<int>(i) - 4) * 1.25f,
            0.35f * static_cast<float>(i % 3),
            2.0f + 0.15f * static_cast<float>(i));
        const PxVec3 initialVelocity(0.04f * static_cast<float>(i + 1),
                                     -0.03f * static_cast<float>(i % 2), 0.02f);
        const PxVec3 expectedVelocity(
            initialVelocity.x + config.gravity[0] * config.dt,
            initialVelocity.y + config.gravity[1] * config.dt,
            initialVelocity.z + config.gravity[2] * config.dt);
        const PxVec3 expectedPosition =
            initialPosition + expectedVelocity * config.dt;
        const glm::quat expectedRotation = zQuat(0.07f * static_cast<float>(i));
        maxPositionError = std::max(maxPositionError,
                                    absDiff(host[row + 0], expectedPosition.x));
        maxPositionError = std::max(maxPositionError,
                                    absDiff(host[row + 1], expectedPosition.y));
        maxPositionError = std::max(maxPositionError,
                                    absDiff(host[row + 2], expectedPosition.z));
        maxQuatError =
            std::max(maxQuatError, absDiff(host[row + 3], expectedRotation.x));
        maxQuatError =
            std::max(maxQuatError, absDiff(host[row + 4], expectedRotation.y));
        maxQuatError =
            std::max(maxQuatError, absDiff(host[row + 5], expectedRotation.z));
        maxQuatError =
            std::max(maxQuatError, absDiff(host[row + 6], expectedRotation.w));
        maxVelocityError = std::max(maxVelocityError,
                                    absDiff(host[row + 7], expectedVelocity.x));
        maxVelocityError = std::max(maxVelocityError,
                                    absDiff(host[row + 8], expectedVelocity.y));
        maxVelocityError = std::max(maxVelocityError,
                                    absDiff(host[row + 9], expectedVelocity.z));
    }
    Sim::CUDAExternalTransformBuffer transformBuffer(
        static_cast<int>(actorCount), gpuConfig.cudaDeviceId,
        "physx_gpu_step_smoke_transforms");
    Sim::launchRigidStateToMat4CUDA(view, transformBuffer.view(),
                                    static_cast<int>(actorCount));
    transformBuffer.incrementVersion();
    if (cudaDeviceSynchronize() != cudaSuccess)
        throw std::runtime_error(
            "cudaDeviceSynchronize after rigid state to mat4 failed");

    std::vector<float> transformHost(static_cast<size_t>(actorCount) * 16);
    if (cudaMemcpy(transformHost.data(), transformBuffer.view().data,
                   sizeof(float) * transformHost.size(),
                   cudaMemcpyDeviceToHost) != cudaSuccess)
        throw std::runtime_error("cudaMemcpy transform data failed");

    float maxTransformError = 0.0f;
    for (PxU32 i = 0; i < actorCount; ++i) {
        const size_t row = static_cast<size_t>(i) * 13;
        const glm::vec3 pos(host[row + 0], host[row + 1], host[row + 2]);
        const glm::quat rot(host[row + 6], host[row + 3], host[row + 4],
                            host[row + 5]);
        const glm::mat4 expected =
            glm::translate(glm::mat4(1.0f), pos) * glm::mat4_cast(rot);
        const float* expectedData = glm::value_ptr(expected);
        const size_t matrix = static_cast<size_t>(i) * 16;
        for (size_t component = 0; component < 16; ++component) {
            maxTransformError = std::max(
                maxTransformError, absDiff(transformHost[matrix + component],
                                           expectedData[component]));
        }
    }
    fmt::print("  mat4 err    : {}\n", maxTransformError);
    if (maxTransformError > 1e-4f)
        throw std::runtime_error(
            "CUDA rigid state to Mat4 transform does not match CPU layout");

    fmt::print("  max pos err : {}\n", maxPositionError);
    fmt::print("  max quat err: {}\n", maxQuatError);
    fmt::print("  max vel err : {}\n", maxVelocityError);
    if (maxPositionError > 1e-4f || maxQuatError > 1e-4f ||
        maxVelocityError > 1e-4f) {
        fmt::print("  first gpu row: [{}, {}, {}, {}, {}, {}, {}]\n", host[0],
                   host[1], host[2], host[3], host[4], host[5], host[6]);
        throw std::runtime_error(
            "GPU rigid data does not match CPU actor state");
    }

    std::vector<float> applied(host.size(), 0.0f);
    for (PxU32 i = 0; i < actorCount; ++i) {
        const size_t row = static_cast<size_t>(i) * 13;
        const glm::quat rotation = zQuat(0.11f * static_cast<float>(i));
        applied[row + 0] = -8.0f + 2.0f * static_cast<float>(i);
        applied[row + 1] = 1.0f + 0.2f * static_cast<float>(i);
        applied[row + 2] = 4.0f + 0.1f * static_cast<float>(i);
        applied[row + 3] = rotation.x;
        applied[row + 4] = rotation.y;
        applied[row + 5] = rotation.z;
        applied[row + 6] = rotation.w;
        applied[row + 7] = 0.1f * static_cast<float>(i + 1);
        applied[row + 8] = -0.04f * static_cast<float>(i);
        applied[row + 9] = 0.03f;
    }

    if (cudaMemcpy(view.data, applied.data(), sizeof(float) * applied.size(),
                   cudaMemcpyHostToDevice) != cudaSuccess)
        throw std::runtime_error("cudaMemcpy applied rigid data failed");
    gpuSystem.applyRigidData();
    if (cudaDeviceSynchronize() != cudaSuccess)
        throw std::runtime_error("cudaDeviceSynchronize after apply failed");

    world.step();
    gpuSystem.fetchRigidData();
    if (cudaDeviceSynchronize() != cudaSuccess)
        throw std::runtime_error("cudaDeviceSynchronize after fetch failed");
    if (cudaMemcpy(host.data(), view.data, sizeof(float) * host.size(),
                   cudaMemcpyDeviceToHost) != cudaSuccess)
        throw std::runtime_error("cudaMemcpy applied rigid result failed");

    float maxAppliedPositionError = 0.0f;
    float maxAppliedQuatError = 0.0f;
    float maxAppliedVelocityError = 0.0f;
    for (PxU32 i = 0; i < actorCount; ++i) {
        const size_t row = static_cast<size_t>(i) * 13;
        for (size_t axis = 0; axis < 3; ++axis) {
            const float expectedVelocity =
                applied[row + 7 + axis] + config.gravity[axis] * config.dt;
            const float expectedPosition =
                applied[row + axis] + expectedVelocity * config.dt;
            maxAppliedPositionError =
                std::max(maxAppliedPositionError,
                         absDiff(host[row + axis], expectedPosition));
            maxAppliedVelocityError =
                std::max(maxAppliedVelocityError,
                         absDiff(host[row + 7 + axis], expectedVelocity));
            maxAppliedVelocityError = std::max(
                maxAppliedVelocityError,
                absDiff(host[row + 10 + axis], applied[row + 10 + axis]));
        }
        for (size_t component = 3; component < 7; ++component)
            maxAppliedQuatError = std::max(
                maxAppliedQuatError,
                absDiff(host[row + component], applied[row + component]));
    }

    fmt::print("  apply pos err: {}\n", maxAppliedPositionError);
    fmt::print("  apply quat err: {}\n", maxAppliedQuatError);
    fmt::print("  apply vel err: {}\n", maxAppliedVelocityError);
    if (maxAppliedPositionError > 1e-4f || maxAppliedQuatError > 1e-4f ||
        maxAppliedVelocityError > 1e-4f)
        throw std::runtime_error("GPU rigid apply/step/fetch round trip did "
                                 "not match expected state");

    for (PxU32 i = 0; i < actorCount; ++i) {
        const size_t row = static_cast<size_t>(i) * 13;
        applied[row + 0] = -4.0f + static_cast<float>(i);
        applied[row + 1] = -2.0f;
        applied[row + 2] = 6.0f + 0.05f * static_cast<float>(i);
        applied[row + 3] = 0.0f;
        applied[row + 4] = 0.0f;
        applied[row + 5] = 0.0f;
        applied[row + 6] = 1.0f;
        applied[row + 7] = 0.0f;
        applied[row + 8] = 0.0f;
        applied[row + 9] = 0.0f;
        applied[row + 10] = 0.0f;
        applied[row + 11] = 0.0f;
        applied[row + 12] = 0.0f;
    }
    if (cudaMemcpy(view.data, applied.data(), sizeof(float) * applied.size(),
                   cudaMemcpyHostToDevice) != cudaSuccess)
        throw std::runtime_error(
            "cudaMemcpy command-effect reset rigid data failed");
    gpuSystem.applyRigidData();
    if (cudaDeviceSynchronize() != cudaSuccess)
        throw std::runtime_error(
            "cudaDeviceSynchronize after command-effect reset failed");

    std::vector<float> forces(static_cast<size_t>(actorCount) * 3, 0.0f);
    std::vector<float> torques(static_cast<size_t>(actorCount) * 3, 0.0f);
    for (PxU32 i = 0; i < actorCount; ++i) {
        const size_t row = static_cast<size_t>(i) * 3;
        forces[row + 2] = 100.0f;
        torques[row + 2] = 10.0f;
    }
    if (cudaMemcpy(forceView.data, forces.data(), sizeof(float) * forces.size(),
                   cudaMemcpyHostToDevice) != cudaSuccess)
        throw std::runtime_error("cudaMemcpy nonzero rigid force failed");
    if (cudaMemcpy(torqueView.data, torques.data(),
                   sizeof(float) * torques.size(),
                   cudaMemcpyHostToDevice) != cudaSuccess)
        throw std::runtime_error("cudaMemcpy nonzero rigid torque failed");
    gpuSystem.applyRigidForce();
    gpuSystem.applyRigidTorque();
    if (cudaDeviceSynchronize() != cudaSuccess)
        throw std::runtime_error(
            "cudaDeviceSynchronize after nonzero force/torque apply failed");

    world.step();
    gpuSystem.fetchRigidData();
    if (cudaDeviceSynchronize() != cudaSuccess)
        throw std::runtime_error(
            "cudaDeviceSynchronize after command-effect fetch failed");
    if (cudaMemcpy(host.data(), view.data, sizeof(float) * host.size(),
                   cudaMemcpyDeviceToHost) != cudaSuccess)
        throw std::runtime_error("cudaMemcpy command-effect result failed");

    const float gravityOnlyVz = config.gravity[2] * config.dt;
    float minForceVelocityDelta = std::numeric_limits<float>::max();
    float maxTorqueAngularZ = 0.0f;
    for (PxU32 i = 0; i < actorCount; ++i) {
        const size_t row = static_cast<size_t>(i) * 13;
        minForceVelocityDelta =
            std::min(minForceVelocityDelta, host[row + 9] - gravityOnlyVz);
        maxTorqueAngularZ =
            std::max(maxTorqueAngularZ, std::fabs(host[row + 12]));
    }
    fmt::print("  force dv z  : {}\n", minForceVelocityDelta);
    fmt::print("  torque wz   : {}\n", maxTorqueAngularZ);
    if (minForceVelocityDelta <= 0.01f || maxTorqueAngularZ <= 0.01f)
        throw std::runtime_error(
            "nonzero rigid force/torque command did not affect dynamics");

    for (PxU32 i = 0; i < actorCount; ++i) {
        const size_t row = static_cast<size_t>(i) * 13;
        applied[row + 0] = -4.0f + static_cast<float>(i);
        applied[row + 1] = 2.0f;
        applied[row + 2] = 8.0f + 0.05f * static_cast<float>(i);
        applied[row + 3] = 0.0f;
        applied[row + 4] = 0.0f;
        applied[row + 5] = 0.0f;
        applied[row + 6] = 1.0f;
        applied[row + 7] = 0.0f;
        applied[row + 8] = 0.0f;
        applied[row + 9] = 0.0f;
        applied[row + 10] = 0.0f;
        applied[row + 11] = 0.0f;
        applied[row + 12] = 0.0f;
    }
    if (cudaMemcpy(view.data, applied.data(), sizeof(float) * applied.size(),
                   cudaMemcpyHostToDevice) != cudaSuccess)
        throw std::runtime_error(
            "cudaMemcpy sparse command reset rigid data failed");
    gpuSystem.applyRigidData();

    std::fill(forces.begin(), forces.end(), 0.0f);
    constexpr uint32_t sparseActor = 2;
    forces[static_cast<size_t>(sparseActor) * 3 + 2] = 120.0f;
    if (cudaMemcpy(forceView.data, forces.data(), sizeof(float) * forces.size(),
                   cudaMemcpyHostToDevice) != cudaSuccess)
        throw std::runtime_error("cudaMemcpy sparse rigid force failed");

    uint32_t* sparseIndicesData = nullptr;
    if (cudaMalloc(&sparseIndicesData, sizeof(uint32_t)) != cudaSuccess)
        throw std::runtime_error("cudaMalloc sparse rigid indices failed");
    if (cudaMemcpy(sparseIndicesData, &sparseActor, sizeof(uint32_t),
                   cudaMemcpyHostToDevice) != cudaSuccess)
        throw std::runtime_error("cudaMemcpy sparse rigid indices failed");
    Sim::GpuArrayView sparseIndices;
    sparseIndices.data = sparseIndicesData;
    sparseIndices.memoryType = Sim::SimMemoryType::CUDADevice;
    sparseIndices.dtype = Sim::SimDType::UInt32;
    sparseIndices.deviceId = gpuConfig.cudaDeviceId;
    sparseIndices.shape = {1};
    sparseIndices.strides = {1};
    sparseIndices.streamHandle = gpuSystem.cudaStream();
    sparseIndices.name = "physx_gpu_step_smoke_sparse_indices";

    gpuSystem.applyRigidForce(&sparseIndices);
    if (cudaDeviceSynchronize() != cudaSuccess)
        throw std::runtime_error(
            "cudaDeviceSynchronize after sparse force apply failed");
    cudaFree(sparseIndicesData);

    world.step();
    gpuSystem.fetchRigidData();
    if (cudaDeviceSynchronize() != cudaSuccess)
        throw std::runtime_error(
            "cudaDeviceSynchronize after sparse command fetch failed");
    if (cudaMemcpy(host.data(), view.data, sizeof(float) * host.size(),
                   cudaMemcpyDeviceToHost) != cudaSuccess)
        throw std::runtime_error("cudaMemcpy sparse command result failed");

    float sparseSelectedDelta = 0.0f;
    float sparseOtherMaxDelta = 0.0f;
    for (PxU32 i = 0; i < actorCount; ++i) {
        const size_t row = static_cast<size_t>(i) * 13;
        const float delta = host[row + 9] - gravityOnlyVz;
        if (i == sparseActor)
            sparseSelectedDelta = delta;
        else
            sparseOtherMaxDelta =
                std::max(sparseOtherMaxDelta, std::fabs(delta));
    }
    fmt::print("  sparse dv z : {} (other max {})\n", sparseSelectedDelta,
               sparseOtherMaxDelta);
    if (sparseSelectedDelta <= 0.01f || sparseOtherMaxDelta > 1e-4f)
        throw std::runtime_error(
            "sparse rigid force command did not affect only the selected row");

    if (cudaMemset(forceView.data, 0, sizeof(float) * actorCount * 3) !=
        cudaSuccess)
        throw std::runtime_error("cudaMemset clear rigid force failed");
    if (cudaMemset(torqueView.data, 0, sizeof(float) * actorCount * 3) !=
        cudaSuccess)
        throw std::runtime_error("cudaMemset clear rigid torque failed");
    gpuSystem.applyRigidForce();
    gpuSystem.applyRigidTorque();

    for (PxU32 i = 0; i < actorCount; ++i) {
        const size_t row = static_cast<size_t>(i) * 13;
        applied[row + 0] = 100.0f + static_cast<float>(i);
        applied[row + 1] = -50.0f - static_cast<float>(i);
        applied[row + 2] = 12.0f;
        applied[row + 3] = 0.0f;
        applied[row + 4] = 0.0f;
        applied[row + 5] = 0.0f;
        applied[row + 6] = 1.0f;
        applied[row + 7] = 0.0f;
        applied[row + 8] = 0.0f;
        applied[row + 9] = 0.0f;
        applied[row + 10] = 0.0f;
        applied[row + 11] = 0.0f;
        applied[row + 12] = 0.0f;
    }
    if (cudaMemcpy(view.data, applied.data(), sizeof(float) * applied.size(),
                   cudaMemcpyHostToDevice) != cudaSuccess)
        throw std::runtime_error("cudaMemcpy sparse rigid data source failed");

    constexpr uint32_t sparseStateActor = 5;
    if (cudaMalloc(&sparseIndicesData, sizeof(uint32_t)) != cudaSuccess)
        throw std::runtime_error("cudaMalloc sparse rigid data indices failed");
    if (cudaMemcpy(sparseIndicesData, &sparseStateActor, sizeof(uint32_t),
                   cudaMemcpyHostToDevice) != cudaSuccess)
        throw std::runtime_error("cudaMemcpy sparse rigid data indices failed");
    sparseIndices.data = sparseIndicesData;
    sparseIndices.dtype = Sim::SimDType::UInt32;
    sparseIndices.shape = {1};
    sparseIndices.strides = {1};
    sparseIndices.name = "physx_gpu_step_smoke_sparse_state_indices";

    gpuSystem.applyRigidData(&sparseIndices);
    if (cudaDeviceSynchronize() != cudaSuccess)
        throw std::runtime_error(
            "cudaDeviceSynchronize after sparse rigid data apply failed");
    cudaFree(sparseIndicesData);

    world.step();
    gpuSystem.fetchRigidData();
    if (cudaDeviceSynchronize() != cudaSuccess)
        throw std::runtime_error(
            "cudaDeviceSynchronize after sparse rigid data fetch failed");
    if (cudaMemcpy(host.data(), view.data, sizeof(float) * host.size(),
                   cudaMemcpyDeviceToHost) != cudaSuccess)
        throw std::runtime_error("cudaMemcpy sparse rigid data result failed");

    float sparseStatePositionError = 0.0f;
    float sparseStateOtherMaxX = 0.0f;
    for (PxU32 i = 0; i < actorCount; ++i) {
        const size_t row = static_cast<size_t>(i) * 13;
        if (i == sparseStateActor) {
            const float expectedZ =
                applied[row + 2] + config.gravity[2] * config.dt * config.dt;
            sparseStatePositionError =
                std::max(sparseStatePositionError,
                         absDiff(host[row + 0], applied[row + 0]));
            sparseStatePositionError =
                std::max(sparseStatePositionError,
                         absDiff(host[row + 1], applied[row + 1]));
            sparseStatePositionError = std::max(
                sparseStatePositionError, absDiff(host[row + 2], expectedZ));
        } else {
            sparseStateOtherMaxX =
                std::max(sparseStateOtherMaxX, std::fabs(host[row + 0]));
        }
    }
    fmt::print("  sparse state err: {} (other max x {})\n",
               sparseStatePositionError, sparseStateOtherMaxX);
    if (sparseStatePositionError > 1e-4f || sparseStateOtherMaxX > 50.0f)
        throw std::runtime_error(
            "sparse rigid data apply did not affect only the selected row");
#endif

    fmt::print(
        "PASS: C++ PhysX GPU rigid fetch/apply and articulation link/joint "
        "state/command completed\n");
    return 0;
}
