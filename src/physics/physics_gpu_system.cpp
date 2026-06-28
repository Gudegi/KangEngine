#include "physics_gpu_system.hpp"

#include "physics.hpp"

#include <cstring>
#include <string>
#include <vector>

#ifdef KANGENGINE_USE_CUDA
#include <cuda_runtime.h>
#endif

namespace KE {

PhysicsGpuSystem::PhysicsGpuSystem(PhysicsWorld* world,
                                   GpuPhysicsConfig config)
    : _world(world), _config(config) {}

PhysicsGpuSystem::~PhysicsGpuSystem() { releaseGpuBuffers(); }

#ifdef KANGENGINE_USE_CUDA
namespace {

void checkCuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(result));
}

} // namespace
#endif

void PhysicsGpuSystem::init() {
    if (!_world)
        throw std::runtime_error("PhysicsGpuSystem requires a PhysicsWorld");
    if (!_config.enableDirectGpuApi)
        throw std::runtime_error(
            "PhysicsGpuSystem requires enableDirectGpuApi=true");
    if (!_world->isGpuEnabled())
        throw std::runtime_error(
            "PhysicsGpuSystem requires a PhysX GPU-enabled PhysicsWorld");
    if (!_world->getScene() || !_world->getScene()->getCudaContextManager())
        throw std::runtime_error(
            "PhysicsGpuSystem could not find a scene CUDA context manager");

#ifdef KANGENGINE_USE_CUDA
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
        std::vector<PxGpuActorPair> actorPairs(_rigidCount);
        for (uint32_t i = 0; i < _rigidCount; ++i) {
            auto* body = actors[i] ? actors[i]->is<PxRigidBody>() : nullptr;
            if (!body)
                throw std::runtime_error(
                    "PhysicsGpuSystem found a non-rigid dynamic actor");
            std::memset(&actorPairs[i], 0, sizeof(PxGpuActorPair));
            actorPairs[i].srcIndex = i;
            actorPairs[i].nodeIndex = body->getInternalIslandNodeIndex();
        }

        checkCuda(cudaMalloc(&_rigidIndexBuffer,
                             sizeof(PxGpuActorPair) * _rigidCount),
                  "cudaMalloc(rigid indices)");
        checkCuda(cudaMalloc(&_rigidScratchBuffer,
                             sizeof(PxGpuBodyData) * _rigidCount),
                  "cudaMalloc(rigid scratch)");
        checkCuda(cudaMalloc(&_rigidMirrorBuffer,
                             sizeof(float) * 13 * _rigidCount),
                  "cudaMalloc(rigid mirror)");
        checkCuda(cudaMemcpy(_rigidIndexBuffer, actorPairs.data(),
                             sizeof(PxGpuActorPair) * _rigidCount,
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

    _views.rigidData.data = _rigidMirrorBuffer;
    _views.rigidData.memoryType = Sim::SimMemoryType::CudaDevice;
    _views.rigidData.dtype = Sim::SimDType::Float32;
    _views.rigidData.lifetime = Sim::SimLifetimePolicy::ExternalOwner;
    _views.rigidData.deviceId = _config.cudaDeviceId;
    _views.rigidData.shape = {static_cast<int64_t>(_rigidCount), 13};
    _views.rigidData.strides = {13, 1};
    _views.rigidData.streamHandle = _streamHandle;
    _views.rigidData.readyEventHandle =
        reinterpret_cast<uint64_t>(_readyEvent);
    _views.rigidData.name = "physics_rigid_data";
    _initialized = true;
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

    _world->getScene()->copyBodyData(
        static_cast<PxGpuBodyData*>(_rigidScratchBuffer),
        static_cast<PxGpuActorPair*>(_rigidIndexBuffer), _rigidCount,
        _copyEvent);
    checkCuda(cudaStreamWaitEvent(stream, copyEvent, 0),
              "cudaStreamWaitEvent(PhysX body copy)");

    auto* source = static_cast<unsigned char*>(_rigidScratchBuffer);
    auto* destination = static_cast<unsigned char*>(_rigidMirrorBuffer);
    constexpr size_t sourcePitch = sizeof(PxGpuBodyData);
    constexpr size_t destinationPitch = sizeof(float) * 13;
    checkCuda(cudaMemcpy2DAsync(destination, destinationPitch,
                                source + offsetof(PxGpuBodyData, pos),
                                sourcePitch, sizeof(float) * 3, _rigidCount,
                                cudaMemcpyDeviceToDevice, stream),
              "cudaMemcpy2DAsync(rigid position)");
    checkCuda(cudaMemcpy2DAsync(destination + sizeof(float) * 3,
                                destinationPitch,
                                source + offsetof(PxGpuBodyData, quat),
                                sourcePitch, sizeof(float) * 4, _rigidCount,
                                cudaMemcpyDeviceToDevice, stream),
              "cudaMemcpy2DAsync(rigid rotation)");
    checkCuda(cudaMemcpy2DAsync(destination + sizeof(float) * 7,
                                destinationPitch,
                                source + offsetof(PxGpuBodyData, linVel),
                                sourcePitch, sizeof(float) * 3, _rigidCount,
                                cudaMemcpyDeviceToDevice, stream),
              "cudaMemcpy2DAsync(rigid linear velocity)");
    checkCuda(cudaMemcpy2DAsync(destination + sizeof(float) * 10,
                                destinationPitch,
                                source + offsetof(PxGpuBodyData, angVel),
                                sourcePitch, sizeof(float) * 3, _rigidCount,
                                cudaMemcpyDeviceToDevice, stream),
              "cudaMemcpy2DAsync(rigid angular velocity)");
    checkCuda(cudaEventRecord(readyEvent, stream),
              "cudaEventRecord(rigid data ready)");
    ++_views.rigidData.version;
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

void PhysicsGpuSystem::applyRigidData(const Sim::GpuArrayView*) {
    notImplemented("applyRigidData");
}

void PhysicsGpuSystem::applyRigidForce(const Sim::GpuArrayView*) {
    notImplemented("applyRigidForce");
}

void PhysicsGpuSystem::applyRigidTorque(const Sim::GpuArrayView*) {
    notImplemented("applyRigidTorque");
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

void PhysicsGpuSystem::applyArticulationJointForces(
    const Sim::GpuArrayView*) {
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
    throw std::runtime_error(std::string("PhysicsGpuSystem::") +
                             functionName + " is not implemented yet");
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
    if (_rigidScratchBuffer)
        cudaFree(_rigidScratchBuffer);
    if (_rigidIndexBuffer)
        cudaFree(_rigidIndexBuffer);
#endif
    _rigidCount = 0;
    _rigidIndexBuffer = nullptr;
    _rigidScratchBuffer = nullptr;
    _rigidMirrorBuffer = nullptr;
    _copyEvent = nullptr;
    _readyEvent = nullptr;
    _views.rigidData = {};
}

} // namespace KE
