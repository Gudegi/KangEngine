#include "physics_gpu_system.hpp"

#include "physics.hpp"
#include "physics/physx_compat.hpp"
#include "physics/physics_gpu_system_kernels.hpp"

#include <cstdint>
#include <string>
#include <vector>

#ifdef KANGENGINE_USE_CUDA
#include <cuda_runtime.h>
#include <cudamanager/PxCudaContext.h>
#ifdef KANGENGINE_HAS_PHYSX_DIRECT_GPU_API
#include <PxDirectGPUAPI.h>
#endif
#endif

namespace KE {

PhysicsGpuSystem::PhysicsGpuSystem(PhysicsWorld* world, GpuPhysicsConfig config)
    : _world(world), _config(config) {}

PhysicsGpuSystem::~PhysicsGpuSystem() { releaseGpuBuffers(); }

namespace {

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
    if (!view.strides.empty() &&
        (view.strides.size() != 2 || view.strides[0] != 3 ||
         view.strides[1] != 1))
        throw std::runtime_error(std::string(name) +
                                 " must be contiguous with strides [3, 1]");
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
}

uint32_t PhysicsGpuSystem::rigidRow(const physx::PxRigidDynamic& rigid) const {
    checkInitialized();
    auto it = _rigidRows.find(&rigid);
    if (it == _rigidRows.end())
        throw std::runtime_error(
            "PhysicsGpuSystem could not find the rigid actor row");
    return it->second;
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
    notImplemented("fetchArticulationLinkPose");
}

void PhysicsGpuSystem::fetchArticulationLinkVel() {
    notImplemented("fetchArticulationLinkVel");
}

void PhysicsGpuSystem::fetchArticulationJointPositions() {
    notImplemented("fetchArticulationJointPositions");
}

void PhysicsGpuSystem::fetchArticulationJointVelocities() {
    notImplemented("fetchArticulationJointVelocities");
}

void PhysicsGpuSystem::fetchArticulationJointAccelerations() {
    notImplemented("fetchArticulationJointAccelerations");
}

void PhysicsGpuSystem::fetchArticulationTargetJointPositions() {
    notImplemented("fetchArticulationTargetJointPositions");
}

void PhysicsGpuSystem::fetchArticulationTargetJointVelocities() {
    notImplemented("fetchArticulationTargetJointVelocities");
}

void PhysicsGpuSystem::fetchArticulationLinkIncomingJointForce() {
    notImplemented("fetchArticulationLinkIncomingJointForce");
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
        linearVelocityBuffer =
            static_cast<unsigned char*>(poseBuffer) +
            sizeof(PxTransform) * applyCount;
        angularVelocityBuffer =
            static_cast<unsigned char*>(linearVelocityBuffer) +
            sizeof(PxVec3) * applyCount;
        PhysicsGpuKernels::packSparseRigidStateCUDA(
            *indices, _rigidIndexBuffer, _rigidMirrorBuffer, gpuIndices,
            poseBuffer, linearVelocityBuffer, angularVelocityBuffer,
            _rigidCount, _streamHandle, sizeof(PxTransform),
            offsetof(PxTransform, p), offsetof(PxTransform, q),
            sizeof(PxVec3));
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

void PhysicsGpuSystem::applyArticulationRootPose(const Sim::GpuArrayView*) {
    notImplemented("applyArticulationRootPose");
}

void PhysicsGpuSystem::applyArticulationRootVel(const Sim::GpuArrayView*) {
    notImplemented("applyArticulationRootVel");
}

void PhysicsGpuSystem::applyArticulationJointPositions(
    const Sim::GpuArrayView*) {
    notImplemented("applyArticulationJointPositions");
}

void PhysicsGpuSystem::applyArticulationJointVelocities(
    const Sim::GpuArrayView*) {
    notImplemented("applyArticulationJointVelocities");
}

void PhysicsGpuSystem::applyArticulationJointForces(const Sim::GpuArrayView*) {
    notImplemented("applyArticulationJointForces");
}

void PhysicsGpuSystem::applyArticulationTargetJointPositions(
    const Sim::GpuArrayView*) {
    notImplemented("applyArticulationTargetJointPositions");
}

void PhysicsGpuSystem::applyArticulationTargetJointVelocities(
    const Sim::GpuArrayView*) {
    notImplemented("applyArticulationTargetJointVelocities");
}

void PhysicsGpuSystem::updateArticulationKinematics() {
    notImplemented("updateArticulationKinematics");
}

void PhysicsGpuSystem::syncPosesGpuToCpu() {
    notImplemented("syncPosesGpuToCpu");
}

void PhysicsGpuSystem::notImplemented(const char* functionName) const {
    throw std::runtime_error(std::string("PhysicsGpuSystem::") + functionName +
                             " is not implemented yet");
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
#endif
    _rigidCount = 0;
    _rigidIndexBuffer = nullptr;
    _rigidScratchBuffer = nullptr;
    _rigidMirrorBuffer = nullptr;
    _rigidForceBuffer = nullptr;
    _rigidTorqueBuffer = nullptr;
    _rigidRows.clear();
    _copyEvent = nullptr;
    _readyEvent = nullptr;
    _views.rigidData = {};
    _views.rigidForce = {};
    _views.rigidTorque = {};
}

} // namespace KE
