#include "physics_gpu_system.hpp"

#include "physics.hpp"
#include "physics/physx_compat.hpp"
#include "physics/physics_gpu_system_kernels.hpp"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#ifdef KANGENGINE_USE_CUDA
#include <cuda_runtime.h>
#include <cudamanager/PxCudaContext.h>
#ifdef KANGENGINE_HAS_PHYSX_DIRECT_GPU_API
#include <PxContact.h>
#include <PxDirectGPUAPI.h>
#endif
#endif

namespace KE {

PhysicsGpuSystem::PhysicsGpuSystem(PhysicsWorld* world, GpuPhysicsConfig config)
    : _world(world), _config(config) {}

PhysicsGpuSystem::~PhysicsGpuSystem() { releaseGpuBuffers(); }

namespace {

#if defined(KANGENGINE_USE_CUDA) && defined(KANGENGINE_HAS_PHYSX_DIRECT_GPU_API)
constexpr int32_t kContactRefUnknown = -1;
constexpr int32_t kContactRefRigid = 0;
constexpr int32_t kContactRefArticulation = 1;
constexpr int kArticulationReadJointPosition =
    PxArticulationGPUAPIReadType::eJOINT_POSITION;
constexpr int kArticulationReadJointVelocity =
    PxArticulationGPUAPIReadType::eJOINT_VELOCITY;
constexpr int kArticulationReadJointAcceleration =
    PxArticulationGPUAPIReadType::eJOINT_ACCELERATION;
constexpr int kArticulationReadJointForce =
    PxArticulationGPUAPIReadType::eJOINT_FORCE;
constexpr int kArticulationReadTargetJointVelocity =
    PxArticulationGPUAPIReadType::eJOINT_TARGET_VELOCITY;
constexpr int kArticulationReadTargetJointPosition =
    PxArticulationGPUAPIReadType::eJOINT_TARGET_POSITION;
constexpr int kArticulationWriteJointPosition =
    PxArticulationGPUAPIWriteType::eJOINT_POSITION;
constexpr int kArticulationWriteJointVelocity =
    PxArticulationGPUAPIWriteType::eJOINT_VELOCITY;
constexpr int kArticulationWriteJointForce =
    PxArticulationGPUAPIWriteType::eJOINT_FORCE;
constexpr int kArticulationWriteTargetJointVelocity =
    PxArticulationGPUAPIWriteType::eJOINT_TARGET_VELOCITY;
constexpr int kArticulationWriteTargetJointPosition =
    PxArticulationGPUAPIWriteType::eJOINT_TARGET_POSITION;
constexpr auto kArticulationWriteRootPose =
    PxArticulationGPUAPIWriteType::eROOT_GLOBAL_POSE;
constexpr auto kArticulationWriteRootLinearVelocity =
    PxArticulationGPUAPIWriteType::eROOT_LINEAR_VELOCITY;
constexpr auto kArticulationWriteRootAngularVelocity =
    PxArticulationGPUAPIWriteType::eROOT_ANGULAR_VELOCITY;
constexpr auto kArticulationComputeUpdateKinematic =
    PxArticulationGPUAPIComputeType::eUPDATE_KINEMATIC;
#else
constexpr int kArticulationReadJointPosition = 0;
constexpr int kArticulationReadJointVelocity = 0;
constexpr int kArticulationReadJointAcceleration = 0;
constexpr int kArticulationReadJointForce = 0;
constexpr int kArticulationReadTargetJointVelocity = 0;
constexpr int kArticulationReadTargetJointPosition = 0;
constexpr int kArticulationWriteJointPosition = 0;
constexpr int kArticulationWriteJointVelocity = 0;
constexpr int kArticulationWriteJointForce = 0;
constexpr int kArticulationWriteTargetJointVelocity = 0;
constexpr int kArticulationWriteTargetJointPosition = 0;
#endif

#ifdef KANGENGINE_USE_CUDA
void checkCuda(PxCUresult result, const char* operation) {
    if (result.value != 0)
        throw std::runtime_error(std::string(operation) +
                                 " failed with CUDA driver error " +
                                 std::to_string(result.value));
}

void checkCuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(result));
}

CUdeviceptr toDevicePtr(void* ptr) {
    return static_cast<CUdeviceptr>(reinterpret_cast<uintptr_t>(ptr));
}

#ifdef KANGENGINE_HAS_PHYSX_DIRECT_GPU_API
void setContactBodyRef(std::vector<int32_t>& refs, PxNodeIndex node,
                       int32_t kind, int32_t row, int32_t body) {
    if (!node.isValid())
        return;
    const uint32_t nodeIndex = node.index();
    const size_t offset = static_cast<size_t>(nodeIndex) * 3;
    if (refs.size() < offset + 3)
        refs.resize(offset + 3, kContactRefUnknown);
    refs[offset + 0] = kind;
    refs[offset + 1] = row;
    refs[offset + 2] = body;
}
#endif
#endif

void setFloatCudaView(Sim::GpuArrayView& view, void* data, int deviceId,
                      uint32_t rows, uint32_t cols, uint64_t streamHandle,
                      uint64_t readyEventHandle, const char* name) {
    view.data = data;
    view.memoryType = Sim::SimMemoryType::CUDADevice;
    view.dtype = Sim::SimDType::Float32;
    view.lifetime = Sim::SimLifetimePolicy::ExternalOwner;
    view.deviceId = deviceId;
    view.shape = {static_cast<int64_t>(rows), static_cast<int64_t>(cols)};
    view.strides = {static_cast<int64_t>(cols), 1};
    view.streamHandle = streamHandle;
    view.readyEventHandle = readyEventHandle;
    view.name = name;
}

void setFloatCudaView3D(Sim::GpuArrayView& view, void* data, int deviceId,
                        uint32_t dim0, uint32_t dim1, uint32_t dim2,
                        uint64_t streamHandle, uint64_t readyEventHandle,
                        const char* name) {
    view.data = data;
    view.memoryType = Sim::SimMemoryType::CUDADevice;
    view.dtype = Sim::SimDType::Float32;
    view.lifetime = Sim::SimLifetimePolicy::ExternalOwner;
    view.deviceId = deviceId;
    view.shape = {static_cast<int64_t>(dim0), static_cast<int64_t>(dim1),
                  static_cast<int64_t>(dim2)};
    view.strides = {static_cast<int64_t>(dim1) * dim2,
                    static_cast<int64_t>(dim2), 1};
    view.streamHandle = streamHandle;
    view.readyEventHandle = readyEventHandle;
    view.name = name;
}

void setUintCudaView(Sim::GpuArrayView& view, void* data, int deviceId,
                     Sim::SimDType dtype, std::vector<int64_t> shape,
                     std::vector<int64_t> strides, uint64_t streamHandle,
                     uint64_t readyEventHandle, const char* name) {
    view.data = data;
    view.memoryType = Sim::SimMemoryType::CUDADevice;
    view.dtype = dtype;
    view.lifetime = Sim::SimLifetimePolicy::ExternalOwner;
    view.deviceId = deviceId;
    view.shape = std::move(shape);
    view.strides = std::move(strides);
    view.streamHandle = streamHandle;
    view.readyEventHandle = readyEventHandle;
    view.name = name;
}

void validateDenseRigidVec3View(const Sim::GpuArrayView& view,
                                const void* expectedData, int deviceId,
                                uint32_t rows, const char* name) {
    if (view.data != expectedData)
        throw std::runtime_error(std::string(name) +
                                 " view does not point to its owned buffer");
    if (view.memoryType != Sim::SimMemoryType::CUDADevice)
        throw std::runtime_error(std::string(name) + " must be a CUDA view");
    if (view.dtype != Sim::SimDType::Float32)
        throw std::runtime_error(std::string(name) + " must use float32");
    if (view.deviceId != deviceId)
        throw std::runtime_error(std::string(name) +
                                 " device_id does not match PhysicsGpuSystem");
    if (view.shape.size() != 2 || view.shape[0] != static_cast<int64_t>(rows) ||
        view.shape[1] != 3)
        throw std::runtime_error(std::string(name) +
                                 " must have shape [rigid_count, 3]");
    if (!view.strides.empty() && (view.strides.size() != 2 ||
                                  view.strides[0] != 3 || view.strides[1] != 1))
        throw std::runtime_error(std::string(name) +
                                 " must be contiguous with strides [3, 1]");
}

void validateDenseArticulationDofView(const Sim::GpuArrayView& view,
                                      const void* expectedData, int deviceId,
                                      uint32_t articulationCount,
                                      uint32_t maxDofs, const char* name) {
    if (view.data != expectedData)
        throw std::runtime_error(std::string(name) +
                                 " view does not point to its owned buffer");
    if (view.memoryType != Sim::SimMemoryType::CUDADevice)
        throw std::runtime_error(std::string(name) + " must be a CUDA view");
    if (view.dtype != Sim::SimDType::Float32)
        throw std::runtime_error(std::string(name) + " must use float32");
    if (view.deviceId != deviceId)
        throw std::runtime_error(std::string(name) +
                                 " device_id does not match PhysicsGpuSystem");
    if (view.shape.size() != 2 ||
        view.shape[0] != static_cast<int64_t>(articulationCount) ||
        view.shape[1] != static_cast<int64_t>(maxDofs))
        throw std::runtime_error(std::string(name) +
                                 " must have shape [articulation_count, "
                                 "max_dofs]");
    if (!view.strides.empty() &&
        (view.strides.size() != 2 ||
         view.strides[0] != static_cast<int64_t>(maxDofs) ||
         view.strides[1] != 1))
        throw std::runtime_error(std::string(name) +
                                 " must be contiguous with strides "
                                 "[max_dofs, 1]");
}

} // namespace

void PhysicsGpuSystem::init() {
    if (!_world)
        throw std::runtime_error("PhysicsGpuSystem requires a PhysicsWorld");
    if (!_world->isGpuEnabled())
        throw std::runtime_error(
            "PhysicsGpuSystem requires a PhysX GPU-enabled PhysicsWorld");
    if (!_world->getScene() || !_world->getScene()->getCudaContextManager())
        throw std::runtime_error(
            "PhysicsGpuSystem could not find a scene CUDA context manager");

#ifdef KANGENGINE_USE_CUDA
#ifndef KANGENGINE_HAS_PHYSX_DIRECT_GPU_API
    throw std::runtime_error(
        "PhysicsGpuSystem requires PhysX Direct GPU API support");
#else
    releaseGpuBuffers();
    checkCuda(cudaSetDevice(_config.cudaDeviceId), "cudaSetDevice");

    // PhysX assigns stable internal GPU node indices during the first step.
    _world->step();

    PxScene* scene = _world->getScene();
    std::vector<int32_t> contactNodeBodyRefs;
    _rigidCount = scene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
    if (_rigidCount > 0) {
        std::vector<PxActor*> actors(_rigidCount);
        _rigidCount = scene->getActors(PxActorTypeFlag::eRIGID_DYNAMIC,
                                       actors.data(), _rigidCount);
        std::vector<PxRigidDynamicGPUIndex> actorIndices(_rigidCount);
        _rigidRows.clear();
        for (uint32_t i = 0; i < _rigidCount; ++i) {
            auto* body = actors[i] ? actors[i]->is<PxRigidDynamic>() : nullptr;
            if (!body)
                throw std::runtime_error(
                    "PhysicsGpuSystem found a non-rigid dynamic actor");
            _rigidRows[body] = i;
            actorIndices[i] = body->getGPUIndex();
            if (actorIndices[i] == 0xFFFFFFFFu)
                throw std::runtime_error(
                    "PhysicsGpuSystem found an invalid PhysX rigid GPU index");
            setContactBodyRef(contactNodeBodyRefs,
                              body->getInternalIslandNodeIndex(),
                              kContactRefRigid, static_cast<int32_t>(i), 0);
        }

        checkCuda(cudaMalloc(&_rigidIndexBuffer,
                             sizeof(PxRigidDynamicGPUIndex) * _rigidCount),
                  "cudaMalloc(rigid indices)");
        checkCuda(cudaMalloc(&_rigidScratchBuffer,
                             (sizeof(PxRigidDynamicGPUIndex) +
                              sizeof(PxTransform) + 2 * sizeof(PxVec3)) *
                                 _rigidCount),
                  "cudaMalloc(rigid scratch)");
        checkCuda(
            cudaMalloc(&_rigidMirrorBuffer, sizeof(float) * 13 * _rigidCount),
            "cudaMalloc(rigid mirror)");
        checkCuda(
            cudaMalloc(&_rigidForceBuffer, sizeof(float) * 3 * _rigidCount),
            "cudaMalloc(rigid force)");
        checkCuda(
            cudaMalloc(&_rigidTorqueBuffer, sizeof(float) * 3 * _rigidCount),
            "cudaMalloc(rigid torque)");
        checkCuda(
            cudaMemset(_rigidForceBuffer, 0, sizeof(float) * 3 * _rigidCount),
            "cudaMemset(rigid force)");
        checkCuda(
            cudaMemset(_rigidTorqueBuffer, 0, sizeof(float) * 3 * _rigidCount),
            "cudaMemset(rigid torque)");
        checkCuda(cudaMemcpy(_rigidIndexBuffer, actorIndices.data(),
                             sizeof(PxRigidDynamicGPUIndex) * _rigidCount,
                             cudaMemcpyHostToDevice),
                  "cudaMemcpy(rigid indices)");
    }

    _articulationCount = scene->getNbArticulations();
    if (_articulationCount > 0) {
        std::vector<PxArticulationReducedCoordinate*> articulations(
            _articulationCount);
        _articulationCount =
            scene->getArticulations(articulations.data(), _articulationCount);
        const PxArticulationGPUAPIMaxCounts maxCounts =
            scene->getDirectGPUAPI().getArticulationGPUAPIMaxCounts();
        _articulationMaxLinks = maxCounts.maxLinks;
        _articulationMaxDofs = maxCounts.maxDofs;
        if (_articulationMaxLinks == 0)
            throw std::runtime_error(
                "PhysX reported zero max links for GPU articulations");

        std::vector<PxArticulationGPUIndex> articulationIndices(
            _articulationCount);
        _articulationLinkCounts.resize(_articulationCount);
        _articulationDofCounts.resize(_articulationCount);
        _articulationRows.clear();
        for (uint32_t i = 0; i < _articulationCount; ++i) {
            auto* articulation = articulations[i];
            if (!articulation)
                throw std::runtime_error(
                    "PhysicsGpuSystem found a null articulation");
            _articulationRows[articulation] = i;
            _articulationLinkCounts[i] = articulation->getNbLinks();
            _articulationDofCounts[i] = articulation->getDofs();
            if (_articulationLinkCounts[i] > _articulationMaxLinks)
                throw std::runtime_error(
                    "articulation link count exceeds PhysX GPU max links");
            if (_articulationDofCounts[i] > _articulationMaxDofs)
                throw std::runtime_error(
                    "articulation DOF count exceeds PhysX GPU max DOFs");
            articulationIndices[i] = articulation->getGPUIndex();
            if (articulationIndices[i] == 0xFFFFFFFFu)
                throw std::runtime_error(
                    "PhysicsGpuSystem found an invalid articulation GPU index");
            std::vector<PxArticulationLink*> links(_articulationLinkCounts[i]);
            const PxU32 linkCount =
                articulation->getLinks(links.data(), _articulationLinkCounts[i]);
            for (PxU32 linkSlot = 0; linkSlot < linkCount; ++linkSlot) {
                PxArticulationLink* link = links[linkSlot];
                if (!link)
                    continue;
                setContactBodyRef(contactNodeBodyRefs,
                                  link->getInternalIslandNodeIndex(),
                                  kContactRefArticulation,
                                  static_cast<int32_t>(i),
                                  static_cast<int32_t>(link->getLinkIndex()));
            }
        }

        const size_t paddedLinkCount =
            static_cast<size_t>(_articulationCount) * _articulationMaxLinks;
        checkCuda(
            cudaMalloc(&_articulationIndexBuffer,
                       sizeof(PxArticulationGPUIndex) * _articulationCount),
            "cudaMalloc(articulation indices)");
        checkCuda(
            cudaMalloc(&_articulationLinkScratchBuffer,
                       (sizeof(PxTransform) + 2 * sizeof(PxVec3)) *
                               paddedLinkCount +
                           sizeof(PxArticulationGPUIndex) * _articulationCount),
            "cudaMalloc(articulation link scratch)");
        checkCuda(cudaMalloc(&_articulationLinkMirrorBuffer,
                             sizeof(float) * 13 * paddedLinkCount),
                  "cudaMalloc(articulation link mirror)");
        checkCuda(cudaMemset(_articulationLinkMirrorBuffer, 0,
                             sizeof(float) * 13 * paddedLinkCount),
                  "cudaMemset(articulation link mirror)");
        checkCuda(
            cudaMemcpy(_articulationIndexBuffer, articulationIndices.data(),
                       sizeof(PxArticulationGPUIndex) * _articulationCount,
                       cudaMemcpyHostToDevice),
            "cudaMemcpy(articulation indices)");
        if (_articulationMaxDofs > 0) {
            const size_t paddedDofCount =
                static_cast<size_t>(_articulationCount) * _articulationMaxDofs;
            checkCuda(cudaMalloc(&_articulationJointPositionBuffer,
                                 sizeof(float) * paddedDofCount),
                      "cudaMalloc(articulation joint positions)");
            checkCuda(cudaMalloc(&_articulationJointVelocityBuffer,
                                 sizeof(float) * paddedDofCount),
                      "cudaMalloc(articulation joint velocities)");
            checkCuda(cudaMalloc(&_articulationJointAccelerationBuffer,
                                 sizeof(float) * paddedDofCount),
                      "cudaMalloc(articulation joint accelerations)");
            checkCuda(cudaMalloc(&_articulationJointForceBuffer,
                                 sizeof(float) * paddedDofCount),
                      "cudaMalloc(articulation joint forces)");
            checkCuda(cudaMalloc(&_articulationTargetJointPositionBuffer,
                                 sizeof(float) * paddedDofCount),
                      "cudaMalloc(articulation target joint positions)");
            checkCuda(cudaMalloc(&_articulationTargetJointVelocityBuffer,
                                 sizeof(float) * paddedDofCount),
                      "cudaMalloc(articulation target joint velocities)");
            checkCuda(cudaMalloc(&_articulationDofScratchBuffer,
                                 (sizeof(PxArticulationGPUIndex) +
                                  sizeof(float) * _articulationMaxDofs) *
                                     _articulationCount),
                      "cudaMalloc(articulation DOF scratch)");
            checkCuda(cudaMemset(_articulationJointPositionBuffer, 0,
                                 sizeof(float) * paddedDofCount),
                      "cudaMemset(articulation joint positions)");
            checkCuda(cudaMemset(_articulationJointVelocityBuffer, 0,
                                 sizeof(float) * paddedDofCount),
                      "cudaMemset(articulation joint velocities)");
            checkCuda(cudaMemset(_articulationJointAccelerationBuffer, 0,
                                 sizeof(float) * paddedDofCount),
                      "cudaMemset(articulation joint accelerations)");
            checkCuda(cudaMemset(_articulationJointForceBuffer, 0,
                                 sizeof(float) * paddedDofCount),
                      "cudaMemset(articulation joint forces)");
            checkCuda(cudaMemset(_articulationTargetJointPositionBuffer, 0,
                                 sizeof(float) * paddedDofCount),
                      "cudaMemset(articulation target joint positions)");
            checkCuda(cudaMemset(_articulationTargetJointVelocityBuffer, 0,
                                 sizeof(float) * paddedDofCount),
                      "cudaMemset(articulation target joint velocities)");
        }
        checkCuda(cudaMalloc(&_articulationLinkIncomingJointForceBuffer,
                             sizeof(float) * 6 * paddedLinkCount),
                  "cudaMalloc(articulation link incoming joint forces)");
        checkCuda(cudaMemset(_articulationLinkIncomingJointForceBuffer, 0,
                             sizeof(float) * 6 * paddedLinkCount),
                  "cudaMemset(articulation link incoming joint forces)");
    }

    if (_config.maxContactPairs > 0) {
        checkCuda(cudaMalloc(&_contactPairBuffer,
                             sizeof(PxGpuContactPair) *
                                 _config.maxContactPairs),
                  "cudaMalloc(contact pairs)");
        checkCuda(cudaMalloc(&_contactPairCountBuffer, sizeof(PxU32)),
                  "cudaMalloc(contact pair count)");
        checkCuda(cudaMalloc(&_contactPairHeaderBuffer,
                             sizeof(uint64_t) * 6 *
                                 _config.maxContactPairs),
                  "cudaMalloc(contact pair headers)");
        checkCuda(cudaMalloc(&_contactPairBodyRefBuffer,
                             sizeof(int32_t) * 6 *
                                 _config.maxContactPairs),
                  "cudaMalloc(contact pair body refs)");
        checkCuda(cudaMemset(_contactPairBuffer, 0,
                             sizeof(PxGpuContactPair) *
                                 _config.maxContactPairs),
                  "cudaMemset(contact pairs)");
        checkCuda(cudaMemset(_contactPairCountBuffer, 0, sizeof(PxU32)),
                  "cudaMemset(contact pair count)");
        checkCuda(cudaMemset(_contactPairHeaderBuffer, 0,
                             sizeof(uint64_t) * 6 *
                                 _config.maxContactPairs),
                  "cudaMemset(contact pair headers)");
        checkCuda(cudaMemset(_contactPairBodyRefBuffer, 0xFF,
                             sizeof(int32_t) * 6 *
                                 _config.maxContactPairs),
                  "cudaMemset(contact pair body refs)");
        if (!contactNodeBodyRefs.empty()) {
            _contactNodeBodyRefCapacity =
                static_cast<uint32_t>(contactNodeBodyRefs.size() / 3);
            checkCuda(cudaMalloc(&_contactNodeBodyRefBuffer,
                                 sizeof(int32_t) * contactNodeBodyRefs.size()),
                      "cudaMalloc(contact node body refs)");
            checkCuda(cudaMemcpy(_contactNodeBodyRefBuffer,
                                 contactNodeBodyRefs.data(),
                                 sizeof(int32_t) * contactNodeBodyRefs.size(),
                                 cudaMemcpyHostToDevice),
                      "cudaMemcpy(contact node body refs)");
        }
    }
    if (_config.maxContactPoints > 0) {
        checkCuda(cudaMalloc(&_contactPointBuffer,
                             sizeof(float) * 10 * _config.maxContactPoints),
                  "cudaMalloc(contact points)");
        checkCuda(cudaMalloc(&_contactPointCountBuffer, sizeof(PxU32)),
                  "cudaMalloc(contact point count)");
        checkCuda(cudaMalloc(&_contactPointPairIndexBuffer,
                             sizeof(PxU32) * _config.maxContactPoints),
                  "cudaMalloc(contact point pair indices)");
        checkCuda(cudaMemset(_contactPointBuffer, 0,
                             sizeof(float) * 10 * _config.maxContactPoints),
                  "cudaMemset(contact points)");
        checkCuda(cudaMemset(_contactPointCountBuffer, 0, sizeof(PxU32)),
                  "cudaMemset(contact point count)");
        checkCuda(cudaMemset(_contactPointPairIndexBuffer, 0xFF,
                             sizeof(PxU32) * _config.maxContactPoints),
                  "cudaMemset(contact point pair indices)");
    }

    if (_rigidCount > 0 || _articulationCount > 0 ||
        _config.maxContactPairs > 0 || _config.maxContactPoints > 0) {
        cudaEvent_t copyEvent = nullptr;
        cudaEvent_t readyEvent = nullptr;
        checkCuda(cudaEventCreateWithFlags(&copyEvent, cudaEventDisableTiming),
                  "cudaEventCreate(copy)");
        checkCuda(cudaEventCreateWithFlags(&readyEvent, cudaEventDisableTiming),
                  "cudaEventCreate(ready)");
        _copyEvent = copyEvent;
        _readyEvent = readyEvent;
    }

    const uint64_t readyEventHandle = reinterpret_cast<uint64_t>(_readyEvent);
    setFloatCudaView(_views.rigidData, _rigidMirrorBuffer, _config.cudaDeviceId,
                     _rigidCount, 13, _streamHandle, readyEventHandle,
                     "physics_rigid_data");
    setFloatCudaView(_views.rigidForce, _rigidForceBuffer, _config.cudaDeviceId,
                     _rigidCount, 3, _streamHandle, readyEventHandle,
                     "physics_rigid_force");
    setFloatCudaView(_views.rigidTorque, _rigidTorqueBuffer,
                     _config.cudaDeviceId, _rigidCount, 3, _streamHandle,
                     readyEventHandle, "physics_rigid_torque");
    setFloatCudaView3D(
        _views.articulationLinkData, _articulationLinkMirrorBuffer,
        _config.cudaDeviceId, _articulationCount, _articulationMaxLinks, 13,
        _streamHandle, readyEventHandle, "physics_articulation_link_data");
    setFloatCudaView(_views.articulationJointPositions,
                     _articulationJointPositionBuffer, _config.cudaDeviceId,
                     _articulationCount, _articulationMaxDofs, _streamHandle,
                     readyEventHandle, "physics_articulation_joint_positions");
    setFloatCudaView(_views.articulationJointVelocities,
                     _articulationJointVelocityBuffer, _config.cudaDeviceId,
                     _articulationCount, _articulationMaxDofs, _streamHandle,
                     readyEventHandle, "physics_articulation_joint_velocities");
    setFloatCudaView(_views.articulationJointAccelerations,
                     _articulationJointAccelerationBuffer, _config.cudaDeviceId,
                     _articulationCount, _articulationMaxDofs, _streamHandle,
                     readyEventHandle,
                     "physics_articulation_joint_accelerations");
    setFloatCudaView(_views.articulationJointForces,
                     _articulationJointForceBuffer, _config.cudaDeviceId,
                     _articulationCount, _articulationMaxDofs, _streamHandle,
                     readyEventHandle, "physics_articulation_joint_forces");
    setFloatCudaView(_views.articulationTargetJointPositions,
                     _articulationTargetJointPositionBuffer,
                     _config.cudaDeviceId, _articulationCount,
                     _articulationMaxDofs, _streamHandle, readyEventHandle,
                     "physics_articulation_target_joint_positions");
    setFloatCudaView(_views.articulationTargetJointVelocities,
                     _articulationTargetJointVelocityBuffer,
                     _config.cudaDeviceId, _articulationCount,
                     _articulationMaxDofs, _streamHandle, readyEventHandle,
                     "physics_articulation_target_joint_velocities");
    setFloatCudaView3D(
        _views.articulationLinkIncomingJointForces,
        _articulationLinkIncomingJointForceBuffer, _config.cudaDeviceId,
        _articulationCount, _articulationMaxLinks, 6, _streamHandle,
        readyEventHandle, "physics_articulation_link_incoming_joint_forces");
    setUintCudaView(
        _views.contactPairs, _contactPairBuffer, _config.cudaDeviceId,
        Sim::SimDType::UInt8,
        {static_cast<int64_t>(sizeof(PxGpuContactPair)) *
         static_cast<int64_t>(_config.maxContactPairs)},
        {1}, _streamHandle, readyEventHandle, "physics_contact_pairs_raw");
    setUintCudaView(_views.contactPairCount, _contactPairCountBuffer,
                    _config.cudaDeviceId, Sim::SimDType::UInt32, {1}, {1},
                    _streamHandle, readyEventHandle,
                    "physics_contact_pair_count");
    setUintCudaView(_views.contactPairHeaders, _contactPairHeaderBuffer,
                    _config.cudaDeviceId, Sim::SimDType::UInt64,
                    {static_cast<int64_t>(_config.maxContactPairs), 6},
                    {6, 1}, _streamHandle, readyEventHandle,
                    "physics_contact_pair_headers");
    setUintCudaView(_views.contactPairBodyRefs, _contactPairBodyRefBuffer,
                    _config.cudaDeviceId, Sim::SimDType::Int32,
                    {static_cast<int64_t>(_config.maxContactPairs), 6},
                    {6, 1}, _streamHandle, readyEventHandle,
                    "physics_contact_pair_body_refs");
    setFloatCudaView(_views.contactPoints, _contactPointBuffer,
                     _config.cudaDeviceId, _config.maxContactPoints, 10,
                     _streamHandle, readyEventHandle,
                     "physics_contact_points");
    setUintCudaView(_views.contactPointCount, _contactPointCountBuffer,
                    _config.cudaDeviceId, Sim::SimDType::UInt32, {1}, {1},
                    _streamHandle, readyEventHandle,
                    "physics_contact_point_count");
    setUintCudaView(_views.contactPointPairIndices,
                    _contactPointPairIndexBuffer, _config.cudaDeviceId,
                    Sim::SimDType::UInt32,
                    {static_cast<int64_t>(_config.maxContactPoints)}, {1},
                    _streamHandle, readyEventHandle,
                    "physics_contact_point_pair_indices");
    _initialized = true;
#endif
#else
    throw std::runtime_error(
        "PhysicsGpuSystem requires a CUDA-enabled KangEngine build");
#endif
}

void PhysicsGpuSystem::invalidate() {
    _initialized = false;
    releaseGpuBuffers();
}

void PhysicsGpuSystem::checkInitialized() const {
    if (!_initialized)
        throw std::runtime_error("PhysicsGpuSystem is not initialized");
}

void PhysicsGpuSystem::setCudaStream(uint64_t streamHandle) {
    _streamHandle = streamHandle;
    _views.rigidData.streamHandle = streamHandle;
    _views.rigidForce.streamHandle = streamHandle;
    _views.rigidTorque.streamHandle = streamHandle;
    _views.articulationLinkData.streamHandle = streamHandle;
    _views.articulationJointPositions.streamHandle = streamHandle;
    _views.articulationJointVelocities.streamHandle = streamHandle;
    _views.articulationJointAccelerations.streamHandle = streamHandle;
    _views.articulationJointForces.streamHandle = streamHandle;
    _views.articulationTargetJointPositions.streamHandle = streamHandle;
    _views.articulationTargetJointVelocities.streamHandle = streamHandle;
    _views.articulationLinkIncomingJointForces.streamHandle = streamHandle;
    _views.contactPairs.streamHandle = streamHandle;
    _views.contactPairCount.streamHandle = streamHandle;
    _views.contactPairHeaders.streamHandle = streamHandle;
    _views.contactPairBodyRefs.streamHandle = streamHandle;
    _views.contactPoints.streamHandle = streamHandle;
    _views.contactPointCount.streamHandle = streamHandle;
    _views.contactPointPairIndices.streamHandle = streamHandle;
}

uint32_t PhysicsGpuSystem::rigidRow(const physx::PxRigidDynamic& rigid) const {
    checkInitialized();
    auto it = _rigidRows.find(&rigid);
    if (it == _rigidRows.end())
        throw std::runtime_error(
            "PhysicsGpuSystem could not find the rigid actor row");
    return it->second;
}

uint32_t PhysicsGpuSystem::articulationRow(
    const physx::PxArticulationReducedCoordinate& articulation) const {
    checkInitialized();
    auto it = _articulationRows.find(&articulation);
    if (it == _articulationRows.end())
        throw std::runtime_error(
            "PhysicsGpuSystem could not find the articulation row");
    return it->second;
}

uint32_t
PhysicsGpuSystem::articulationLinkCount(uint32_t articulationRow) const {
    checkInitialized();
    if (articulationRow >= _articulationLinkCounts.size())
        throw std::runtime_error("articulation row is out of range");
    return _articulationLinkCounts[articulationRow];
}

uint32_t
PhysicsGpuSystem::articulationDofCount(uint32_t articulationRow) const {
    checkInitialized();
    if (articulationRow >= _articulationDofCounts.size())
        throw std::runtime_error("articulation row is out of range");
    return _articulationDofCounts[articulationRow];
}

void PhysicsGpuSystem::stepStart() { checkInitialized(); }

void PhysicsGpuSystem::stepFinish() { checkInitialized(); }

void PhysicsGpuSystem::fetchRigidData() {
    checkInitialized();
#ifdef KANGENGINE_USE_CUDA
    if (_rigidCount == 0)
        return;

    checkCuda(cudaSetDevice(_config.cudaDeviceId), "cudaSetDevice");
    auto stream = reinterpret_cast<cudaStream_t>(_streamHandle);
    auto copyEvent = reinterpret_cast<cudaEvent_t>(_copyEvent);
    auto readyEvent = reinterpret_cast<cudaEvent_t>(_readyEvent);
    auto physxCopyEvent = reinterpret_cast<CUevent>(_copyEvent);

#ifndef KANGENGINE_HAS_PHYSX_DIRECT_GPU_API
    throw std::runtime_error(
        "PhysicsGpuSystem requires PhysX Direct GPU API support");
#else
    auto* scratch = static_cast<unsigned char*>(_rigidScratchBuffer);
    void* poseBuffer = scratch;
    void* linearVelocityBuffer = scratch + sizeof(PxTransform) * _rigidCount;
    void* angularVelocityBuffer =
        scratch + (sizeof(PxTransform) + sizeof(PxVec3)) * _rigidCount;
    PxDirectGPUAPI& directGpuApi = _world->getScene()->getDirectGPUAPI();
    auto* gpuIndices = static_cast<PxRigidDynamicGPUIndex*>(_rigidIndexBuffer);
    if (!directGpuApi.getRigidDynamicData(
            poseBuffer, gpuIndices, PxRigidDynamicGPUAPIReadType::eGLOBAL_POSE,
            _rigidCount, nullptr, physxCopyEvent))
        throw std::runtime_error(
            "PxDirectGPUAPI::getRigidDynamicData(pose) failed");
    if (!directGpuApi.getRigidDynamicData(
            linearVelocityBuffer, gpuIndices,
            PxRigidDynamicGPUAPIReadType::eLINEAR_VELOCITY, _rigidCount,
            nullptr, physxCopyEvent))
        throw std::runtime_error(
            "PxDirectGPUAPI::getRigidDynamicData(linear velocity) failed");
    if (!directGpuApi.getRigidDynamicData(
            angularVelocityBuffer, gpuIndices,
            PxRigidDynamicGPUAPIReadType::eANGULAR_VELOCITY, _rigidCount,
            nullptr, physxCopyEvent))
        throw std::runtime_error(
            "PxDirectGPUAPI::getRigidDynamicData(angular velocity) failed");
    checkCuda(cudaStreamWaitEvent(stream, copyEvent, 0),
              "cudaStreamWaitEvent(PhysX body copy)");

    auto* destination = static_cast<unsigned char*>(_rigidMirrorBuffer);
    constexpr size_t destinationPitch = sizeof(float) * 13;
    checkCuda(cudaMemcpy2DAsync(destination, destinationPitch,
                                static_cast<unsigned char*>(poseBuffer) +
                                    offsetof(PxTransform, p),
                                sizeof(PxTransform), sizeof(float) * 3,
                                _rigidCount, cudaMemcpyDeviceToDevice, stream),
              "cudaMemcpy2DAsync(rigid position)");
    checkCuda(cudaMemcpy2DAsync(destination + sizeof(float) * 3,
                                destinationPitch,
                                static_cast<unsigned char*>(poseBuffer) +
                                    offsetof(PxTransform, q),
                                sizeof(PxTransform), sizeof(float) * 4,
                                _rigidCount, cudaMemcpyDeviceToDevice, stream),
              "cudaMemcpy2DAsync(rigid rotation)");
    checkCuda(cudaMemcpy2DAsync(destination + sizeof(float) * 7,
                                destinationPitch, linearVelocityBuffer,
                                sizeof(PxVec3), sizeof(float) * 3, _rigidCount,
                                cudaMemcpyDeviceToDevice, stream),
              "cudaMemcpy2DAsync(rigid linear velocity)");
    checkCuda(cudaMemcpy2DAsync(destination + sizeof(float) * 10,
                                destinationPitch, angularVelocityBuffer,
                                sizeof(PxVec3), sizeof(float) * 3, _rigidCount,
                                cudaMemcpyDeviceToDevice, stream),
              "cudaMemcpy2DAsync(rigid angular velocity)");
    checkCuda(cudaEventRecord(readyEvent, stream),
              "cudaEventRecord(rigid data ready)");
    ++_views.rigidData.version;
#endif
#else
    notImplemented("fetchRigidData");
#endif
}

void PhysicsGpuSystem::fetchArticulationLinkPose() {
    checkInitialized();
#ifdef KANGENGINE_USE_CUDA
    if (_articulationCount == 0)
        return;

    checkCuda(cudaSetDevice(_config.cudaDeviceId), "cudaSetDevice");
    auto stream = reinterpret_cast<cudaStream_t>(_streamHandle);
    auto copyEvent = reinterpret_cast<cudaEvent_t>(_copyEvent);
    auto readyEvent = reinterpret_cast<cudaEvent_t>(_readyEvent);
    auto physxCopyEvent = reinterpret_cast<CUevent>(_copyEvent);

#ifndef KANGENGINE_HAS_PHYSX_DIRECT_GPU_API
    throw std::runtime_error(
        "PhysicsGpuSystem requires PhysX Direct GPU API support");
#else
    PxDirectGPUAPI& directGpuApi = _world->getScene()->getDirectGPUAPI();
    auto* gpuIndices =
        static_cast<PxArticulationGPUIndex*>(_articulationIndexBuffer);
    if (!directGpuApi.getArticulationData(
            _articulationLinkScratchBuffer, gpuIndices,
            PxArticulationGPUAPIReadType::eLINK_GLOBAL_POSE, _articulationCount,
            nullptr, physxCopyEvent))
        throw std::runtime_error(
            "PxDirectGPUAPI::getArticulationData(link pose) failed");
    checkCuda(cudaStreamWaitEvent(stream, copyEvent, 0),
              "cudaStreamWaitEvent(PhysX articulation link pose copy)");

    const size_t paddedLinkCount =
        static_cast<size_t>(_articulationCount) * _articulationMaxLinks;
    auto* destination =
        static_cast<unsigned char*>(_articulationLinkMirrorBuffer);
    auto* source = static_cast<unsigned char*>(_articulationLinkScratchBuffer);
    constexpr size_t destinationPitch = sizeof(float) * 13;
    checkCuda(cudaMemcpy2DAsync(destination, destinationPitch,
                                source + offsetof(PxTransform, p),
                                sizeof(PxTransform), sizeof(float) * 3,
                                paddedLinkCount, cudaMemcpyDeviceToDevice,
                                stream),
              "cudaMemcpy2DAsync(articulation link position)");
    checkCuda(
        cudaMemcpy2DAsync(destination + sizeof(float) * 3, destinationPitch,
                          source + offsetof(PxTransform, q),
                          sizeof(PxTransform), sizeof(float) * 4,
                          paddedLinkCount, cudaMemcpyDeviceToDevice, stream),
        "cudaMemcpy2DAsync(articulation link rotation)");
    checkCuda(cudaEventRecord(readyEvent, stream),
              "cudaEventRecord(articulation link pose ready)");
    ++_views.articulationLinkData.version;
#endif
#else
    notImplemented("fetchArticulationLinkPose");
#endif
}

void PhysicsGpuSystem::fetchArticulationLinkVel() {
    checkInitialized();
#ifdef KANGENGINE_USE_CUDA
    if (_articulationCount == 0)
        return;

    checkCuda(cudaSetDevice(_config.cudaDeviceId), "cudaSetDevice");
    auto stream = reinterpret_cast<cudaStream_t>(_streamHandle);
    auto copyEvent = reinterpret_cast<cudaEvent_t>(_copyEvent);
    auto readyEvent = reinterpret_cast<cudaEvent_t>(_readyEvent);
    auto physxCopyEvent = reinterpret_cast<CUevent>(_copyEvent);

#ifndef KANGENGINE_HAS_PHYSX_DIRECT_GPU_API
    throw std::runtime_error(
        "PhysicsGpuSystem requires PhysX Direct GPU API support");
#else
    const size_t paddedLinkCount =
        static_cast<size_t>(_articulationCount) * _articulationMaxLinks;
    auto* scratch = static_cast<unsigned char*>(_articulationLinkScratchBuffer);
    void* linearVelocityBuffer =
        scratch + sizeof(PxTransform) * paddedLinkCount;
    void* angularVelocityBuffer =
        static_cast<unsigned char*>(linearVelocityBuffer) +
        sizeof(PxVec3) * paddedLinkCount;

    PxDirectGPUAPI& directGpuApi = _world->getScene()->getDirectGPUAPI();
    auto* gpuIndices =
        static_cast<PxArticulationGPUIndex*>(_articulationIndexBuffer);
    if (!directGpuApi.getArticulationData(
            linearVelocityBuffer, gpuIndices,
            PxArticulationGPUAPIReadType::eLINK_LINEAR_VELOCITY,
            _articulationCount, nullptr, physxCopyEvent))
        throw std::runtime_error(
            "PxDirectGPUAPI::getArticulationData(link linear velocity) "
            "failed");
    if (!directGpuApi.getArticulationData(
            angularVelocityBuffer, gpuIndices,
            PxArticulationGPUAPIReadType::eLINK_ANGULAR_VELOCITY,
            _articulationCount, nullptr, physxCopyEvent))
        throw std::runtime_error(
            "PxDirectGPUAPI::getArticulationData(link angular velocity) "
            "failed");
    checkCuda(cudaStreamWaitEvent(stream, copyEvent, 0),
              "cudaStreamWaitEvent(PhysX articulation link velocity copy)");

    auto* destination =
        static_cast<unsigned char*>(_articulationLinkMirrorBuffer);
    constexpr size_t destinationPitch = sizeof(float) * 13;
    checkCuda(cudaMemcpy2DAsync(
                  destination + sizeof(float) * 7, destinationPitch,
                  linearVelocityBuffer, sizeof(PxVec3), sizeof(float) * 3,
                  paddedLinkCount, cudaMemcpyDeviceToDevice, stream),
              "cudaMemcpy2DAsync(articulation link linear velocity)");
    checkCuda(cudaMemcpy2DAsync(
                  destination + sizeof(float) * 10, destinationPitch,
                  angularVelocityBuffer, sizeof(PxVec3), sizeof(float) * 3,
                  paddedLinkCount, cudaMemcpyDeviceToDevice, stream),
              "cudaMemcpy2DAsync(articulation link angular velocity)");
    checkCuda(cudaEventRecord(readyEvent, stream),
              "cudaEventRecord(articulation link velocity ready)");
    ++_views.articulationLinkData.version;
#endif
#else
    notImplemented("fetchArticulationLinkVel");
#endif
}

void PhysicsGpuSystem::fetchArticulationDofBuffer(
    void* buffer, Sim::GpuArrayView& view, int readType, const char* dataName,
    const char* waitOperation, const char* readyOperation) {
    checkInitialized();
#ifdef KANGENGINE_USE_CUDA
    if (_articulationCount == 0 || _articulationMaxDofs == 0 || !buffer)
        return;

    checkCuda(cudaSetDevice(_config.cudaDeviceId), "cudaSetDevice");
    auto stream = reinterpret_cast<cudaStream_t>(_streamHandle);
    auto copyEvent = reinterpret_cast<cudaEvent_t>(_copyEvent);
    auto readyEvent = reinterpret_cast<cudaEvent_t>(_readyEvent);
    auto physxCopyEvent = reinterpret_cast<CUevent>(_copyEvent);

#ifndef KANGENGINE_HAS_PHYSX_DIRECT_GPU_API
    throw std::runtime_error(
        "PhysicsGpuSystem requires PhysX Direct GPU API support");
#else
    PxDirectGPUAPI& directGpuApi = _world->getScene()->getDirectGPUAPI();
    auto* gpuIndices =
        static_cast<PxArticulationGPUIndex*>(_articulationIndexBuffer);
    if (!directGpuApi.getArticulationData(
            buffer, gpuIndices,
            static_cast<PxArticulationGPUAPIReadType::Enum>(readType),
            _articulationCount, nullptr, physxCopyEvent)) {
        throw std::runtime_error(
            std::string("PxDirectGPUAPI::getArticulationData(") + dataName +
            ") failed");
    }
    checkCuda(cudaStreamWaitEvent(stream, copyEvent, 0), waitOperation);
    checkCuda(cudaEventRecord(readyEvent, stream), readyOperation);
    ++view.version;
#endif
#else
    (void)buffer;
    (void)view;
    (void)readType;
    (void)dataName;
    (void)waitOperation;
    (void)readyOperation;
    notImplemented("fetchArticulationDofBuffer");
#endif
}

void PhysicsGpuSystem::fetchArticulationJointPositions() {
    fetchArticulationDofBuffer(
        _articulationJointPositionBuffer, _views.articulationJointPositions,
        kArticulationReadJointPosition, "joint position",
        "cudaStreamWaitEvent(PhysX articulation joint position copy)",
        "cudaEventRecord(articulation joint position ready)");
}

void PhysicsGpuSystem::fetchArticulationJointVelocities() {
    fetchArticulationDofBuffer(
        _articulationJointVelocityBuffer, _views.articulationJointVelocities,
        kArticulationReadJointVelocity, "joint velocity",
        "cudaStreamWaitEvent(PhysX articulation joint velocity copy)",
        "cudaEventRecord(articulation joint velocity ready)");
}

void PhysicsGpuSystem::fetchArticulationJointAccelerations() {
    fetchArticulationDofBuffer(
        _articulationJointAccelerationBuffer,
        _views.articulationJointAccelerations,
        kArticulationReadJointAcceleration, "joint acceleration",
        "cudaStreamWaitEvent(PhysX articulation joint acceleration copy)",
        "cudaEventRecord(articulation joint acceleration ready)");
}

void PhysicsGpuSystem::fetchArticulationJointForces() {
    fetchArticulationDofBuffer(
        _articulationJointForceBuffer, _views.articulationJointForces,
        kArticulationReadJointForce, "joint force",
        "cudaStreamWaitEvent(PhysX articulation joint force copy)",
        "cudaEventRecord(articulation joint force ready)");
}

void PhysicsGpuSystem::fetchArticulationTargetJointPositions() {
    fetchArticulationDofBuffer(
        _articulationTargetJointPositionBuffer,
        _views.articulationTargetJointPositions,
        kArticulationReadTargetJointPosition, "target joint position",
        "cudaStreamWaitEvent(PhysX articulation target joint position copy)",
        "cudaEventRecord(articulation target joint position ready)");
}

void PhysicsGpuSystem::fetchArticulationTargetJointVelocities() {
    fetchArticulationDofBuffer(
        _articulationTargetJointVelocityBuffer,
        _views.articulationTargetJointVelocities,
        kArticulationReadTargetJointVelocity, "target joint velocity",
        "cudaStreamWaitEvent(PhysX articulation target joint velocity copy)",
        "cudaEventRecord(articulation target joint velocity ready)");
}

void PhysicsGpuSystem::fetchArticulationLinkIncomingJointForce() {
    checkInitialized();
#ifdef KANGENGINE_USE_CUDA
    if (_articulationCount == 0 || _articulationMaxLinks == 0 ||
        !_articulationLinkIncomingJointForceBuffer)
        return;

    checkCuda(cudaSetDevice(_config.cudaDeviceId), "cudaSetDevice");
    auto stream = reinterpret_cast<cudaStream_t>(_streamHandle);
    auto copyEvent = reinterpret_cast<cudaEvent_t>(_copyEvent);
    auto readyEvent = reinterpret_cast<cudaEvent_t>(_readyEvent);
    auto physxCopyEvent = reinterpret_cast<CUevent>(_copyEvent);

#ifndef KANGENGINE_HAS_PHYSX_DIRECT_GPU_API
    throw std::runtime_error(
        "PhysicsGpuSystem requires PhysX Direct GPU API support");
#else
    PxDirectGPUAPI& directGpuApi = _world->getScene()->getDirectGPUAPI();
    auto* gpuIndices =
        static_cast<PxArticulationGPUIndex*>(_articulationIndexBuffer);
    if (!directGpuApi.getArticulationData(
            _articulationLinkIncomingJointForceBuffer, gpuIndices,
            PxArticulationGPUAPIReadType::eLINK_INCOMING_JOINT_FORCE,
            _articulationCount, nullptr, physxCopyEvent))
        throw std::runtime_error(
            "PxDirectGPUAPI::getArticulationData(link incoming joint force) "
            "failed");
    checkCuda(
        cudaStreamWaitEvent(stream, copyEvent, 0),
        "cudaStreamWaitEvent(PhysX articulation link incoming joint force "
        "copy)");
    checkCuda(cudaEventRecord(readyEvent, stream),
              "cudaEventRecord(articulation link incoming joint force ready)");
    ++_views.articulationLinkIncomingJointForces.version;
#endif
#else
    notImplemented("fetchArticulationLinkIncomingJointForce");
#endif
}

void PhysicsGpuSystem::fetchContactPairs() {
    checkInitialized();
#ifdef KANGENGINE_USE_CUDA
    if (!_contactPairBuffer || !_contactPairCountBuffer ||
        _config.maxContactPairs == 0)
        return;

    checkCuda(cudaSetDevice(_config.cudaDeviceId), "cudaSetDevice");
    auto stream = reinterpret_cast<cudaStream_t>(_streamHandle);
    auto copyEvent = reinterpret_cast<cudaEvent_t>(_copyEvent);
    auto readyEvent = reinterpret_cast<cudaEvent_t>(_readyEvent);
    auto physxStartEvent = reinterpret_cast<CUevent>(_readyEvent);
    auto physxCopyEvent = reinterpret_cast<CUevent>(_copyEvent);

#ifndef KANGENGINE_HAS_PHYSX_DIRECT_GPU_API
    throw std::runtime_error(
        "PhysicsGpuSystem requires PhysX Direct GPU API support");
#else
    checkCuda(cudaMemsetAsync(_contactPairCountBuffer, 0, sizeof(PxU32),
                              stream),
              "cudaMemsetAsync(contact pair count)");
    checkCuda(cudaEventRecord(readyEvent, stream),
              "cudaEventRecord(contact pair clear)");
    PxDirectGPUAPI& directGpuApi = _world->getScene()->getDirectGPUAPI();
    if (!directGpuApi.copyContactData(
            _contactPairBuffer, static_cast<PxU32*>(_contactPairCountBuffer),
            _config.maxContactPairs, physxStartEvent, physxCopyEvent))
        throw std::runtime_error("PxDirectGPUAPI::copyContactData failed");
    checkCuda(cudaStreamWaitEvent(stream, copyEvent, 0),
              "cudaStreamWaitEvent(PhysX contact pair copy)");
    PhysicsGpuKernels::flattenContactPairsCUDA(
        _contactPairBuffer, _contactPairCountBuffer, _contactPairHeaderBuffer,
        _contactNodeBodyRefBuffer, _contactNodeBodyRefCapacity,
        _contactPairBodyRefBuffer, _contactPointBuffer,
        _contactPointCountBuffer, _contactPointPairIndexBuffer,
        _config.maxContactPairs, _config.maxContactPoints, _streamHandle);
    checkCuda(cudaEventRecord(readyEvent, stream),
              "cudaEventRecord(contact pair ready)");
    ++_views.contactPairs.version;
    ++_views.contactPairCount.version;
    ++_views.contactPairHeaders.version;
    ++_views.contactPairBodyRefs.version;
    ++_views.contactPoints.version;
    ++_views.contactPointCount.version;
    ++_views.contactPointPairIndices.version;
#endif
#else
    notImplemented("fetchContactPairs");
#endif
}

void PhysicsGpuSystem::clearContactData() {
    checkInitialized();
#ifdef KANGENGINE_USE_CUDA
    checkCuda(cudaSetDevice(_config.cudaDeviceId), "cudaSetDevice");
    auto stream = reinterpret_cast<cudaStream_t>(_streamHandle);
    if (_contactPairCountBuffer) {
        checkCuda(cudaMemsetAsync(_contactPairCountBuffer, 0, sizeof(PxU32),
                                  stream),
                  "cudaMemsetAsync(clear contact pair count)");
    }
    if (_contactPointCountBuffer) {
        checkCuda(cudaMemsetAsync(_contactPointCountBuffer, 0, sizeof(PxU32),
                                  stream),
                  "cudaMemsetAsync(clear contact point count)");
    }
    if (_contactPairBodyRefBuffer && _config.maxContactPairs > 0) {
        checkCuda(cudaMemsetAsync(_contactPairBodyRefBuffer, 0xFF,
                                  sizeof(int32_t) *
                                      static_cast<size_t>(
                                          _config.maxContactPairs) *
                                      6,
                                  stream),
                  "cudaMemsetAsync(clear contact pair body refs)");
    }
    if (_contactPointPairIndexBuffer && _config.maxContactPoints > 0) {
        checkCuda(cudaMemsetAsync(_contactPointPairIndexBuffer, 0xFF,
                                  sizeof(PxU32) *
                                      static_cast<size_t>(
                                          _config.maxContactPoints),
                                  stream),
                  "cudaMemsetAsync(clear contact point pair indices)");
    }
    checkCuda(cudaEventRecord(reinterpret_cast<cudaEvent_t>(_readyEvent),
                              stream),
              "cudaEventRecord(clear contact data ready)");
    ++_views.contactPairCount.version;
    ++_views.contactPairBodyRefs.version;
    ++_views.contactPointCount.version;
    ++_views.contactPointPairIndices.version;
#else
    notImplemented("clearContactData");
#endif
}

void PhysicsGpuSystem::applyRigidData(const Sim::GpuArrayView* indices) {
    checkInitialized();
#ifdef KANGENGINE_USE_CUDA
    if (_rigidCount == 0)
        return;

    checkCuda(cudaSetDevice(_config.cudaDeviceId), "cudaSetDevice");
    auto stream = reinterpret_cast<cudaStream_t>(_streamHandle);
    auto packEvent = reinterpret_cast<cudaEvent_t>(_copyEvent);
    auto readyEvent = reinterpret_cast<cudaEvent_t>(_readyEvent);
    auto physxStartEvent = reinterpret_cast<CUevent>(_copyEvent);
    auto physxFinishEvent = reinterpret_cast<CUevent>(_readyEvent);

#ifndef KANGENGINE_HAS_PHYSX_DIRECT_GPU_API
    throw std::runtime_error(
        "PhysicsGpuSystem requires PhysX Direct GPU API support");
#else
    auto* source = static_cast<unsigned char*>(_rigidMirrorBuffer);
    auto* scratch = static_cast<unsigned char*>(_rigidScratchBuffer);
    void* poseBuffer = scratch;
    void* linearVelocityBuffer = scratch + sizeof(PxTransform) * _rigidCount;
    void* angularVelocityBuffer =
        scratch + (sizeof(PxTransform) + sizeof(PxVec3)) * _rigidCount;
    auto* gpuIndices = static_cast<PxRigidDynamicGPUIndex*>(_rigidIndexBuffer);
    uint32_t applyCount = _rigidCount;

    if (indices) {
        if (indices->deviceId != _config.cudaDeviceId)
            throw std::runtime_error(
                "sparse rigid data indices device_id does not match "
                "PhysicsGpuSystem");
        const int64_t indexCount = indices->numel();
        if (indexCount < 0 || indexCount > static_cast<int64_t>(_rigidCount))
            throw std::runtime_error(
                "sparse rigid data index count exceeds rigid count");
        if (indexCount == 0)
            return;
        applyCount = static_cast<uint32_t>(indexCount);
        gpuIndices = reinterpret_cast<PxRigidDynamicGPUIndex*>(scratch);
        poseBuffer = scratch + sizeof(PxRigidDynamicGPUIndex) * applyCount;
        linearVelocityBuffer = static_cast<unsigned char*>(poseBuffer) +
                               sizeof(PxTransform) * applyCount;
        angularVelocityBuffer =
            static_cast<unsigned char*>(linearVelocityBuffer) +
            sizeof(PxVec3) * applyCount;
        PhysicsGpuKernels::packSparseRigidStateCUDA(
            *indices, _rigidIndexBuffer, _rigidMirrorBuffer, gpuIndices,
            poseBuffer, linearVelocityBuffer, angularVelocityBuffer,
            _rigidCount, _streamHandle, sizeof(PxTransform),
            offsetof(PxTransform, p), offsetof(PxTransform, q), sizeof(PxVec3));
    } else {
        constexpr size_t sourcePitch = sizeof(float) * 13;
        checkCuda(cudaMemcpy2DAsync(static_cast<unsigned char*>(poseBuffer) +
                                        offsetof(PxTransform, p),
                                    sizeof(PxTransform), source, sourcePitch,
                                    sizeof(float) * 3, _rigidCount,
                                    cudaMemcpyDeviceToDevice, stream),
                  "cudaMemcpy2DAsync(apply rigid position)");
        checkCuda(cudaMemcpy2DAsync(static_cast<unsigned char*>(poseBuffer) +
                                        offsetof(PxTransform, q),
                                    sizeof(PxTransform),
                                    source + sizeof(float) * 3, sourcePitch,
                                    sizeof(float) * 4, _rigidCount,
                                    cudaMemcpyDeviceToDevice, stream),
                  "cudaMemcpy2DAsync(apply rigid rotation)");
        checkCuda(cudaMemcpy2DAsync(linearVelocityBuffer, sizeof(PxVec3),
                                    source + sizeof(float) * 7, sourcePitch,
                                    sizeof(float) * 3, _rigidCount,
                                    cudaMemcpyDeviceToDevice, stream),
                  "cudaMemcpy2DAsync(apply rigid linear velocity)");
        checkCuda(cudaMemcpy2DAsync(angularVelocityBuffer, sizeof(PxVec3),
                                    source + sizeof(float) * 10, sourcePitch,
                                    sizeof(float) * 3, _rigidCount,
                                    cudaMemcpyDeviceToDevice, stream),
                  "cudaMemcpy2DAsync(apply rigid angular velocity)");
    }
    checkCuda(cudaEventRecord(packEvent, stream),
              "cudaEventRecord(apply rigid pack)");

    PxDirectGPUAPI& directGpuApi = _world->getScene()->getDirectGPUAPI();
    if (!directGpuApi.setRigidDynamicData(
            poseBuffer, gpuIndices, PxRigidDynamicGPUAPIWriteType::eGLOBAL_POSE,
            applyCount, physxStartEvent, physxFinishEvent))
        throw std::runtime_error(
            "PxDirectGPUAPI::setRigidDynamicData(pose) failed");
    checkCuda(cudaStreamWaitEvent(stream, readyEvent, 0),
              "cudaStreamWaitEvent(apply pose)");

    if (!directGpuApi.setRigidDynamicData(
            linearVelocityBuffer, gpuIndices,
            PxRigidDynamicGPUAPIWriteType::eLINEAR_VELOCITY, applyCount,
            physxStartEvent, physxFinishEvent))
        throw std::runtime_error(
            "PxDirectGPUAPI::setRigidDynamicData(linear velocity) failed");
    checkCuda(cudaStreamWaitEvent(stream, readyEvent, 0),
              "cudaStreamWaitEvent(apply linear velocity)");

    if (!directGpuApi.setRigidDynamicData(
            angularVelocityBuffer, gpuIndices,
            PxRigidDynamicGPUAPIWriteType::eANGULAR_VELOCITY, applyCount,
            physxStartEvent, physxFinishEvent))
        throw std::runtime_error(
            "PxDirectGPUAPI::setRigidDynamicData(angular velocity) failed");
    checkCuda(cudaStreamWaitEvent(stream, readyEvent, 0),
              "cudaStreamWaitEvent(apply angular velocity)");
    ++_views.rigidData.version;
#endif
#else
    notImplemented("applyRigidData");
#endif
}

void PhysicsGpuSystem::applyRigidForce(const Sim::GpuArrayView* indices) {
    applyRigidCommand(indices, false);
}

void PhysicsGpuSystem::applyRigidTorque(const Sim::GpuArrayView* indices) {
    applyRigidCommand(indices, true);
}

void PhysicsGpuSystem::applyArticulationRootPose(
    const Sim::GpuArrayView* indices) {
    checkInitialized();
#ifdef KANGENGINE_USE_CUDA
    if (_articulationCount == 0)
        return;

    checkCuda(cudaSetDevice(_config.cudaDeviceId), "cudaSetDevice");
    auto stream = reinterpret_cast<cudaStream_t>(_streamHandle);
    auto startEvent = reinterpret_cast<cudaEvent_t>(_copyEvent);
    auto finishEvent = reinterpret_cast<cudaEvent_t>(_readyEvent);
    auto physxStartEvent = reinterpret_cast<CUevent>(_copyEvent);
    auto physxFinishEvent = reinterpret_cast<CUevent>(_readyEvent);

#ifndef KANGENGINE_HAS_PHYSX_DIRECT_GPU_API
    throw std::runtime_error(
        "PhysicsGpuSystem requires PhysX Direct GPU API support");
#else
    void* poseBuffer = nullptr;
    void* linearVelocityBuffer = nullptr;
    void* angularVelocityBuffer = nullptr;
    void* gpuIndices = nullptr;
    uint32_t applyCount = 0;
    packArticulationRootState(indices, poseBuffer, linearVelocityBuffer,
                              angularVelocityBuffer, gpuIndices, applyCount);
    if (applyCount == 0)
        return;
    checkCuda(cudaEventRecord(startEvent, stream),
              "cudaEventRecord(apply articulation root pose)");

    PxDirectGPUAPI& directGpuApi = _world->getScene()->getDirectGPUAPI();
    if (!directGpuApi.setArticulationData(
            poseBuffer, static_cast<PxArticulationGPUIndex*>(gpuIndices),
            kArticulationWriteRootPose, applyCount, physxStartEvent,
            physxFinishEvent))
        throw std::runtime_error(
            "PxDirectGPUAPI::setArticulationData(root pose) failed");
    checkCuda(cudaStreamWaitEvent(stream, finishEvent, 0),
              "cudaStreamWaitEvent(apply articulation root pose)");
    ++_views.articulationLinkData.version;
#endif
#else
    (void)indices;
    notImplemented("applyArticulationRootPose");
#endif
}

void PhysicsGpuSystem::applyArticulationRootVel(
    const Sim::GpuArrayView* indices) {
    checkInitialized();
#ifdef KANGENGINE_USE_CUDA
    if (_articulationCount == 0)
        return;

    checkCuda(cudaSetDevice(_config.cudaDeviceId), "cudaSetDevice");
    auto stream = reinterpret_cast<cudaStream_t>(_streamHandle);
    auto startEvent = reinterpret_cast<cudaEvent_t>(_copyEvent);
    auto finishEvent = reinterpret_cast<cudaEvent_t>(_readyEvent);
    auto physxStartEvent = reinterpret_cast<CUevent>(_copyEvent);
    auto physxFinishEvent = reinterpret_cast<CUevent>(_readyEvent);

#ifndef KANGENGINE_HAS_PHYSX_DIRECT_GPU_API
    throw std::runtime_error(
        "PhysicsGpuSystem requires PhysX Direct GPU API support");
#else
    void* poseBuffer = nullptr;
    void* linearVelocityBuffer = nullptr;
    void* angularVelocityBuffer = nullptr;
    void* gpuIndices = nullptr;
    uint32_t applyCount = 0;
    packArticulationRootState(indices, poseBuffer, linearVelocityBuffer,
                              angularVelocityBuffer, gpuIndices, applyCount);
    if (applyCount == 0)
        return;
    checkCuda(cudaEventRecord(startEvent, stream),
              "cudaEventRecord(apply articulation root velocity)");

    PxDirectGPUAPI& directGpuApi = _world->getScene()->getDirectGPUAPI();
    if (!directGpuApi.setArticulationData(
            linearVelocityBuffer,
            static_cast<PxArticulationGPUIndex*>(gpuIndices),
            kArticulationWriteRootLinearVelocity, applyCount, physxStartEvent,
            physxFinishEvent))
        throw std::runtime_error(
            "PxDirectGPUAPI::setArticulationData(root linear velocity) "
            "failed");
    checkCuda(cudaStreamWaitEvent(stream, finishEvent, 0),
              "cudaStreamWaitEvent(apply articulation root linear velocity)");
    if (!directGpuApi.setArticulationData(
            angularVelocityBuffer,
            static_cast<PxArticulationGPUIndex*>(gpuIndices),
            kArticulationWriteRootAngularVelocity, applyCount, physxStartEvent,
            physxFinishEvent))
        throw std::runtime_error(
            "PxDirectGPUAPI::setArticulationData(root angular velocity) "
            "failed");
    checkCuda(cudaStreamWaitEvent(stream, finishEvent, 0),
              "cudaStreamWaitEvent(apply articulation root angular velocity)");
    ++_views.articulationLinkData.version;
#endif
#else
    (void)indices;
    notImplemented("applyArticulationRootVel");
#endif
}

void PhysicsGpuSystem::applyArticulationJointPositions(
    const Sim::GpuArrayView* indices) {
    applyArticulationDofBuffer(
        indices, _articulationJointPositionBuffer,
        _views.articulationJointPositions, kArticulationWriteJointPosition,
        "joint position",
        "cudaStreamWaitEvent(apply articulation joint position)",
        "cudaEventRecord(apply articulation joint position)");
}

void PhysicsGpuSystem::applyArticulationJointVelocities(
    const Sim::GpuArrayView* indices) {
    applyArticulationDofBuffer(
        indices, _articulationJointVelocityBuffer,
        _views.articulationJointVelocities, kArticulationWriteJointVelocity,
        "joint velocity",
        "cudaStreamWaitEvent(apply articulation joint velocity)",
        "cudaEventRecord(apply articulation joint velocity)");
}

void PhysicsGpuSystem::applyArticulationJointForces(
    const Sim::GpuArrayView* indices) {
    applyArticulationDofBuffer(
        indices, _articulationJointForceBuffer, _views.articulationJointForces,
        kArticulationWriteJointForce, "joint force",
        "cudaStreamWaitEvent(apply articulation joint force)",
        "cudaEventRecord(apply articulation joint force)");
}

void PhysicsGpuSystem::applyArticulationTargetJointPositions(
    const Sim::GpuArrayView* indices) {
    applyArticulationDofBuffer(
        indices, _articulationTargetJointPositionBuffer,
        _views.articulationTargetJointPositions,
        kArticulationWriteTargetJointPosition, "target joint position",
        "cudaStreamWaitEvent(apply articulation target joint position)",
        "cudaEventRecord(apply articulation target joint position)");
}

void PhysicsGpuSystem::applyArticulationTargetJointVelocities(
    const Sim::GpuArrayView* indices) {
    applyArticulationDofBuffer(
        indices, _articulationTargetJointVelocityBuffer,
        _views.articulationTargetJointVelocities,
        kArticulationWriteTargetJointVelocity, "target joint velocity",
        "cudaStreamWaitEvent(apply articulation target joint velocity)",
        "cudaEventRecord(apply articulation target joint velocity)");
}

void PhysicsGpuSystem::clearRigidCommands(const Sim::GpuArrayView* indices) {
    checkInitialized();
#ifdef KANGENGINE_USE_CUDA
    if (_rigidCount == 0 || !_rigidForceBuffer || !_rigidTorqueBuffer)
        return;

    checkCuda(cudaSetDevice(_config.cudaDeviceId), "cudaSetDevice");
    auto stream = reinterpret_cast<cudaStream_t>(_streamHandle);
    if (indices) {
        if (indices->deviceId != _config.cudaDeviceId)
            throw std::runtime_error(
                "sparse rigid command clear indices device_id does not match "
                "PhysicsGpuSystem");
        PhysicsGpuKernels::clearSparseRigidCommandsCUDA(
            *indices, _rigidForceBuffer, _rigidTorqueBuffer, _rigidCount,
            _streamHandle);
    } else {
        const size_t bytes = sizeof(float) * static_cast<size_t>(_rigidCount) * 3;
        checkCuda(cudaMemsetAsync(_rigidForceBuffer, 0, bytes, stream),
                  "cudaMemsetAsync(clear rigid force)");
        checkCuda(cudaMemsetAsync(_rigidTorqueBuffer, 0, bytes, stream),
                  "cudaMemsetAsync(clear rigid torque)");
    }
    applyRigidForce(indices);
    applyRigidTorque(indices);
#else
    (void)indices;
    notImplemented("clearRigidCommands");
#endif
}

void PhysicsGpuSystem::clearArticulationCommands(
    const Sim::GpuArrayView* indices) {
    checkInitialized();
#ifdef KANGENGINE_USE_CUDA
    if (_articulationCount == 0 || _articulationMaxDofs == 0 ||
        !_articulationJointPositionBuffer || !_articulationJointForceBuffer ||
        !_articulationTargetJointPositionBuffer ||
        !_articulationTargetJointVelocityBuffer)
        return;

    checkCuda(cudaSetDevice(_config.cudaDeviceId), "cudaSetDevice");
    auto stream = reinterpret_cast<cudaStream_t>(_streamHandle);
    const size_t bytes = sizeof(float) *
                         static_cast<size_t>(_articulationCount) *
                         _articulationMaxDofs;
    if (indices) {
        if (indices->deviceId != _config.cudaDeviceId)
            throw std::runtime_error(
                "sparse articulation command clear indices device_id does not "
                "match PhysicsGpuSystem");
        PhysicsGpuKernels::clearSparseArticulationCommandsCUDA(
            *indices, _articulationJointPositionBuffer,
            _articulationJointForceBuffer,
            _articulationTargetJointPositionBuffer,
            _articulationTargetJointVelocityBuffer, _articulationCount,
            _articulationMaxDofs, _streamHandle);
    } else {
        checkCuda(cudaMemcpyAsync(_articulationTargetJointPositionBuffer,
                                  _articulationJointPositionBuffer, bytes,
                                  cudaMemcpyDeviceToDevice, stream),
                  "cudaMemcpyAsync(clear articulation target joint position)");
        checkCuda(cudaMemsetAsync(_articulationTargetJointVelocityBuffer, 0,
                                  bytes, stream),
                  "cudaMemsetAsync(clear articulation target joint velocity)");
        checkCuda(cudaMemsetAsync(_articulationJointForceBuffer, 0, bytes,
                                  stream),
                  "cudaMemsetAsync(clear articulation joint force)");
    }
    applyArticulationTargetJointPositions(indices);
    applyArticulationTargetJointVelocities(indices);
    applyArticulationJointForces(indices);
#else
    (void)indices;
    notImplemented("clearArticulationCommands");
#endif
}

void PhysicsGpuSystem::updateArticulationKinematics() {
    checkInitialized();
#ifdef KANGENGINE_USE_CUDA
    if (_articulationCount == 0)
        return;

    checkCuda(cudaSetDevice(_config.cudaDeviceId), "cudaSetDevice");
    auto stream = reinterpret_cast<cudaStream_t>(_streamHandle);
    auto startEvent = reinterpret_cast<cudaEvent_t>(_copyEvent);
    auto finishEvent = reinterpret_cast<cudaEvent_t>(_readyEvent);
    auto physxStartEvent = reinterpret_cast<CUevent>(_copyEvent);
    auto physxFinishEvent = reinterpret_cast<CUevent>(_readyEvent);

#ifndef KANGENGINE_HAS_PHYSX_DIRECT_GPU_API
    throw std::runtime_error(
        "PhysicsGpuSystem requires PhysX Direct GPU API support");
#else
    checkCuda(cudaEventRecord(startEvent, stream),
              "cudaEventRecord(update articulation kinematics)");
    PxDirectGPUAPI& directGpuApi = _world->getScene()->getDirectGPUAPI();
    auto* gpuIndices =
        static_cast<PxArticulationGPUIndex*>(_articulationIndexBuffer);
    if (!directGpuApi.computeArticulationData(
            nullptr, gpuIndices, kArticulationComputeUpdateKinematic,
            _articulationCount, physxStartEvent, physxFinishEvent))
        throw std::runtime_error(
            "PxDirectGPUAPI::computeArticulationData(update kinematic) "
            "failed");
    checkCuda(cudaStreamWaitEvent(stream, finishEvent, 0),
              "cudaStreamWaitEvent(update articulation kinematics)");
#endif
#else
    notImplemented("updateArticulationKinematics");
#endif
}

void PhysicsGpuSystem::syncPosesGpuToCpu() {
    notImplemented("syncPosesGpuToCpu");
}

void PhysicsGpuSystem::notImplemented(const char* functionName) const {
    throw std::runtime_error(std::string("PhysicsGpuSystem::") + functionName +
                             " is not implemented yet");
}

void PhysicsGpuSystem::packArticulationRootState(
    const Sim::GpuArrayView* indices, void*& poseBuffer,
    void*& linearVelocityBuffer, void*& angularVelocityBuffer,
    void*& gpuIndices, uint32_t& applyCount) {
#ifdef KANGENGINE_USE_CUDA
    auto stream = reinterpret_cast<cudaStream_t>(_streamHandle);
    auto* scratch = static_cast<unsigned char*>(_articulationLinkScratchBuffer);
    if (!scratch)
        throw std::runtime_error(
            "articulation root apply requires the link scratch buffer");
    auto* source = static_cast<unsigned char*>(_articulationLinkMirrorBuffer);
    if (!source)
        throw std::runtime_error(
            "articulation root apply requires articulation link data");

    poseBuffer = scratch;
    linearVelocityBuffer = scratch + sizeof(PxTransform) * _articulationCount;
    angularVelocityBuffer =
        scratch + (sizeof(PxTransform) + sizeof(PxVec3)) * _articulationCount;
    gpuIndices = _articulationIndexBuffer;
    applyCount = _articulationCount;

    if (indices) {
        if (indices->deviceId != _config.cudaDeviceId)
            throw std::runtime_error(
                "sparse articulation root indices device_id does not match "
                "PhysicsGpuSystem");
        const int64_t indexCount = indices->numel();
        if (indexCount < 0 ||
            indexCount > static_cast<int64_t>(_articulationCount))
            throw std::runtime_error(
                "sparse articulation root index count exceeds articulation "
                "count");
        if (indexCount == 0) {
            applyCount = 0;
            return;
        }
        applyCount = static_cast<uint32_t>(indexCount);
        gpuIndices = scratch;
        poseBuffer = scratch + sizeof(PxArticulationGPUIndex) * applyCount;
        linearVelocityBuffer = static_cast<unsigned char*>(poseBuffer) +
                               sizeof(PxTransform) * applyCount;
        angularVelocityBuffer =
            static_cast<unsigned char*>(linearVelocityBuffer) +
            sizeof(PxVec3) * applyCount;
        PhysicsGpuKernels::packSparseArticulationRootStateCUDA(
            *indices, _articulationIndexBuffer, _articulationLinkMirrorBuffer,
            gpuIndices, poseBuffer, linearVelocityBuffer, angularVelocityBuffer,
            _articulationCount, _articulationMaxLinks, _streamHandle,
            sizeof(PxTransform), offsetof(PxTransform, p),
            offsetof(PxTransform, q), sizeof(PxVec3));
        return;
    }

    const size_t sourcePitch =
        sizeof(float) * static_cast<size_t>(_articulationMaxLinks) * 13;
    checkCuda(cudaMemcpy2DAsync(static_cast<unsigned char*>(poseBuffer) +
                                    offsetof(PxTransform, p),
                                sizeof(PxTransform), source, sourcePitch,
                                sizeof(float) * 3, _articulationCount,
                                cudaMemcpyDeviceToDevice, stream),
              "cudaMemcpy2DAsync(apply articulation root position)");
    checkCuda(cudaMemcpy2DAsync(static_cast<unsigned char*>(poseBuffer) +
                                    offsetof(PxTransform, q),
                                sizeof(PxTransform), source + sizeof(float) * 3,
                                sourcePitch, sizeof(float) * 4,
                                _articulationCount, cudaMemcpyDeviceToDevice,
                                stream),
              "cudaMemcpy2DAsync(apply articulation root rotation)");
    checkCuda(cudaMemcpy2DAsync(linearVelocityBuffer, sizeof(PxVec3),
                                source + sizeof(float) * 7, sourcePitch,
                                sizeof(float) * 3, _articulationCount,
                                cudaMemcpyDeviceToDevice, stream),
              "cudaMemcpy2DAsync(apply articulation root linear velocity)");
    checkCuda(cudaMemcpy2DAsync(angularVelocityBuffer, sizeof(PxVec3),
                                source + sizeof(float) * 10, sourcePitch,
                                sizeof(float) * 3, _articulationCount,
                                cudaMemcpyDeviceToDevice, stream),
              "cudaMemcpy2DAsync(apply articulation root angular velocity)");
#else
    (void)indices;
    (void)poseBuffer;
    (void)linearVelocityBuffer;
    (void)angularVelocityBuffer;
    (void)gpuIndices;
    (void)applyCount;
    notImplemented("packArticulationRootState");
#endif
}

void PhysicsGpuSystem::applyArticulationDofBuffer(
    const Sim::GpuArrayView* indices, void* buffer, Sim::GpuArrayView& view,
    int writeType, const char* dataName, const char* waitOperation,
    const char* readyOperation) {
    checkInitialized();
#ifdef KANGENGINE_USE_CUDA
    if (_articulationCount == 0 || _articulationMaxDofs == 0 || !buffer)
        return;

    validateDenseArticulationDofView(view, buffer, _config.cudaDeviceId,
                                     _articulationCount, _articulationMaxDofs,
                                     dataName);

    checkCuda(cudaSetDevice(_config.cudaDeviceId), "cudaSetDevice");
    auto stream = reinterpret_cast<cudaStream_t>(_streamHandle);
    auto startEvent = reinterpret_cast<cudaEvent_t>(_copyEvent);
    auto finishEvent = reinterpret_cast<cudaEvent_t>(_readyEvent);
    auto physxStartEvent = reinterpret_cast<CUevent>(_copyEvent);
    auto physxFinishEvent = reinterpret_cast<CUevent>(_readyEvent);
    auto* gpuIndices =
        static_cast<PxArticulationGPUIndex*>(_articulationIndexBuffer);
    uint32_t applyCount = _articulationCount;

#ifndef KANGENGINE_HAS_PHYSX_DIRECT_GPU_API
    throw std::runtime_error(
        "PhysicsGpuSystem requires PhysX Direct GPU API support");
#else
    if (indices) {
        if (indices->deviceId != _config.cudaDeviceId)
            throw std::runtime_error(
                "sparse articulation command indices device_id does not match "
                "PhysicsGpuSystem");
        const int64_t indexCount = indices->numel();
        if (indexCount < 0 ||
            indexCount > static_cast<int64_t>(_articulationCount))
            throw std::runtime_error(
                "sparse articulation command index count exceeds "
                "articulation count");
        if (indexCount == 0)
            return;
        if (!_articulationDofScratchBuffer)
            throw std::runtime_error(
                "sparse articulation command scratch buffer is not allocated");
        applyCount = static_cast<uint32_t>(indexCount);
        auto* scratch =
            static_cast<unsigned char*>(_articulationDofScratchBuffer);
        gpuIndices = reinterpret_cast<PxArticulationGPUIndex*>(scratch);
        buffer = scratch + sizeof(PxArticulationGPUIndex) * applyCount;
        PhysicsGpuKernels::packSparseArticulationDofCommandCUDA(
            *indices, _articulationIndexBuffer, view.data, gpuIndices, buffer,
            _articulationCount, _articulationMaxDofs, _streamHandle);
    }

    checkCuda(cudaEventRecord(startEvent, stream), readyOperation);

    PxDirectGPUAPI& directGpuApi = _world->getScene()->getDirectGPUAPI();
    if (!directGpuApi.setArticulationData(
            buffer, gpuIndices,
            static_cast<PxArticulationGPUAPIWriteType::Enum>(writeType),
            applyCount, physxStartEvent, physxFinishEvent)) {
        throw std::runtime_error(
            std::string("PxDirectGPUAPI::setArticulationData(") + dataName +
            ") failed");
    }
    checkCuda(cudaStreamWaitEvent(stream, finishEvent, 0), waitOperation);
    ++view.version;
#endif
#else
    (void)indices;
    (void)buffer;
    (void)view;
    (void)writeType;
    (void)dataName;
    (void)waitOperation;
    (void)readyOperation;
    notImplemented("applyArticulationDofBuffer");
#endif
}

void PhysicsGpuSystem::applyRigidCommand(const Sim::GpuArrayView* indices,
                                         bool torque) {
    checkInitialized();
#ifdef KANGENGINE_USE_CUDA
    if (_rigidCount == 0)
        return;

    Sim::GpuArrayView& view = torque ? _views.rigidTorque : _views.rigidForce;
    void* commandBuffer = torque ? _rigidTorqueBuffer : _rigidForceBuffer;
    const char* name = torque ? "physics_rigid_torque" : "physics_rigid_force";
    validateDenseRigidVec3View(view, commandBuffer, _config.cudaDeviceId,
                               _rigidCount, name);

    checkCuda(cudaSetDevice(_config.cudaDeviceId), "cudaSetDevice");
    auto stream = reinterpret_cast<cudaStream_t>(_streamHandle);
    auto startEvent = reinterpret_cast<cudaEvent_t>(_copyEvent);
    auto finishEvent = reinterpret_cast<cudaEvent_t>(_readyEvent);
    auto physxStartEvent = reinterpret_cast<CUevent>(_copyEvent);
    auto physxFinishEvent = reinterpret_cast<CUevent>(_readyEvent);

#ifndef KANGENGINE_HAS_PHYSX_DIRECT_GPU_API
    throw std::runtime_error(
        "PhysicsGpuSystem requires PhysX Direct GPU API support");
#else
    PxDirectGPUAPI& directGpuApi = _world->getScene()->getDirectGPUAPI();
    auto* gpuIndices = static_cast<PxRigidDynamicGPUIndex*>(_rigidIndexBuffer);
    uint32_t applyCount = _rigidCount;
    if (indices) {
        if (indices->deviceId != _config.cudaDeviceId)
            throw std::runtime_error(
                "sparse rigid command indices device_id does not match "
                "PhysicsGpuSystem");
        const int64_t indexCount = indices->numel();
        if (indexCount < 0 || indexCount > static_cast<int64_t>(_rigidCount))
            throw std::runtime_error(
                "sparse rigid command index count exceeds rigid count");
        if (indexCount == 0)
            return;
        applyCount = static_cast<uint32_t>(indexCount);
        auto* scratch = static_cast<unsigned char*>(_rigidScratchBuffer);
        gpuIndices = reinterpret_cast<PxRigidDynamicGPUIndex*>(scratch);
        commandBuffer = scratch + sizeof(PxRigidDynamicGPUIndex) * applyCount;
        PhysicsGpuKernels::packSparseRigidCommandCUDA(
            *indices, _rigidIndexBuffer,
            torque ? _rigidTorqueBuffer : _rigidForceBuffer, gpuIndices,
            commandBuffer, _rigidCount, _streamHandle);
    }

    checkCuda(cudaEventRecord(startEvent, stream),
              torque ? "cudaEventRecord(apply rigid torque)"
                     : "cudaEventRecord(apply rigid force)");

    const auto writeType = torque ? PxRigidDynamicGPUAPIWriteType::eTORQUE
                                  : PxRigidDynamicGPUAPIWriteType::eFORCE;
    if (!directGpuApi.setRigidDynamicData(commandBuffer, gpuIndices, writeType,
                                          applyCount, physxStartEvent,
                                          physxFinishEvent))
        throw std::runtime_error(
            torque ? "PxDirectGPUAPI::setRigidDynamicData(torque) failed"
                   : "PxDirectGPUAPI::setRigidDynamicData(force) failed");
    checkCuda(cudaStreamWaitEvent(stream, finishEvent, 0),
              torque ? "cudaStreamWaitEvent(apply rigid torque)"
                     : "cudaStreamWaitEvent(apply rigid force)");
    ++view.version;
#endif
#else
    notImplemented(torque ? "applyRigidTorque" : "applyRigidForce");
#endif
}

void PhysicsGpuSystem::releaseGpuBuffers() {
#ifdef KANGENGINE_USE_CUDA
    cudaSetDevice(_config.cudaDeviceId);
    if (_readyEvent)
        cudaEventDestroy(reinterpret_cast<cudaEvent_t>(_readyEvent));
    if (_copyEvent)
        cudaEventDestroy(reinterpret_cast<cudaEvent_t>(_copyEvent));
    if (_rigidMirrorBuffer)
        cudaFree(_rigidMirrorBuffer);
    if (_rigidForceBuffer)
        cudaFree(_rigidForceBuffer);
    if (_rigidTorqueBuffer)
        cudaFree(_rigidTorqueBuffer);
    if (_rigidScratchBuffer)
        cudaFree(_rigidScratchBuffer);
    if (_rigidIndexBuffer)
        cudaFree(_rigidIndexBuffer);
    if (_articulationLinkMirrorBuffer)
        cudaFree(_articulationLinkMirrorBuffer);
    if (_articulationLinkScratchBuffer)
        cudaFree(_articulationLinkScratchBuffer);
    if (_articulationDofScratchBuffer)
        cudaFree(_articulationDofScratchBuffer);
    if (_articulationIndexBuffer)
        cudaFree(_articulationIndexBuffer);
    if (_articulationJointVelocityBuffer)
        cudaFree(_articulationJointVelocityBuffer);
    if (_articulationJointPositionBuffer)
        cudaFree(_articulationJointPositionBuffer);
    if (_articulationJointAccelerationBuffer)
        cudaFree(_articulationJointAccelerationBuffer);
    if (_articulationJointForceBuffer)
        cudaFree(_articulationJointForceBuffer);
    if (_articulationTargetJointPositionBuffer)
        cudaFree(_articulationTargetJointPositionBuffer);
    if (_articulationTargetJointVelocityBuffer)
        cudaFree(_articulationTargetJointVelocityBuffer);
    if (_articulationLinkIncomingJointForceBuffer)
        cudaFree(_articulationLinkIncomingJointForceBuffer);
    if (_contactPairBuffer)
        cudaFree(_contactPairBuffer);
    if (_contactPairCountBuffer)
        cudaFree(_contactPairCountBuffer);
    if (_contactPairHeaderBuffer)
        cudaFree(_contactPairHeaderBuffer);
    if (_contactNodeBodyRefBuffer)
        cudaFree(_contactNodeBodyRefBuffer);
    if (_contactPairBodyRefBuffer)
        cudaFree(_contactPairBodyRefBuffer);
    if (_contactPointBuffer)
        cudaFree(_contactPointBuffer);
    if (_contactPointCountBuffer)
        cudaFree(_contactPointCountBuffer);
    if (_contactPointPairIndexBuffer)
        cudaFree(_contactPointPairIndexBuffer);
#endif
    _rigidCount = 0;
    _rigidIndexBuffer = nullptr;
    _rigidScratchBuffer = nullptr;
    _rigidMirrorBuffer = nullptr;
    _rigidForceBuffer = nullptr;
    _rigidTorqueBuffer = nullptr;
    _rigidRows.clear();
    _articulationCount = 0;
    _articulationMaxLinks = 0;
    _articulationMaxDofs = 0;
    _articulationIndexBuffer = nullptr;
    _articulationLinkScratchBuffer = nullptr;
    _articulationDofScratchBuffer = nullptr;
    _articulationLinkMirrorBuffer = nullptr;
    _articulationJointPositionBuffer = nullptr;
    _articulationJointVelocityBuffer = nullptr;
    _articulationJointAccelerationBuffer = nullptr;
    _articulationJointForceBuffer = nullptr;
    _articulationTargetJointPositionBuffer = nullptr;
    _articulationTargetJointVelocityBuffer = nullptr;
    _articulationLinkIncomingJointForceBuffer = nullptr;
    _contactPairBuffer = nullptr;
    _contactPairCountBuffer = nullptr;
    _contactPairHeaderBuffer = nullptr;
    _contactNodeBodyRefBuffer = nullptr;
    _contactNodeBodyRefCapacity = 0;
    _contactPairBodyRefBuffer = nullptr;
    _contactPointBuffer = nullptr;
    _contactPointCountBuffer = nullptr;
    _contactPointPairIndexBuffer = nullptr;
    _articulationLinkCounts.clear();
    _articulationDofCounts.clear();
    _articulationRows.clear();
    _copyEvent = nullptr;
    _readyEvent = nullptr;
    _views.rigidData = {};
    _views.rigidForce = {};
    _views.rigidTorque = {};
    _views.articulationLinkData = {};
    _views.articulationJointPositions = {};
    _views.articulationJointVelocities = {};
    _views.articulationJointAccelerations = {};
    _views.articulationJointForces = {};
    _views.articulationTargetJointPositions = {};
    _views.articulationTargetJointVelocities = {};
    _views.articulationLinkIncomingJointForces = {};
    _views.contactPairs = {};
    _views.contactPairCount = {};
    _views.contactPairHeaders = {};
    _views.contactPairBodyRefs = {};
    _views.contactPoints = {};
    _views.contactPointCount = {};
    _views.contactPointPairIndices = {};
}

} // namespace KE
