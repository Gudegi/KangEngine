#ifndef _PHYSICS_GPU_SYSTEM_HPP_
#define _PHYSICS_GPU_SYSTEM_HPP_

#include "sim/gpu_array_view.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace KE {

class PhysicsWorld;

struct GpuPhysicsConfig {
    int cudaDeviceId = 0;
    bool enableDirectGpuApi = true;
    bool enableGpuDynamics = true;
    bool enableGpuBroadphase = true;
    bool requirePcm = true;
    size_t tempBufferCapacity = 0;
    size_t heapCapacity = 0;
    size_t foundLostPairsCapacity = 0;
};

struct PhysicsGpuStateViews {
    Sim::GpuArrayView rigidData;
    Sim::GpuArrayView rigidForce;
    Sim::GpuArrayView rigidTorque;
    Sim::GpuArrayView articulationLinkData;
    Sim::GpuArrayView articulationJointPositions;
    Sim::GpuArrayView articulationJointVelocities;
    Sim::GpuArrayView articulationJointAccelerations;
    Sim::GpuArrayView articulationJointForces;
    Sim::GpuArrayView articulationTargetJointPositions;
    Sim::GpuArrayView articulationTargetJointVelocities;
    Sim::GpuArrayView articulationLinkIncomingJointForces;
};

// Skeleton for SAPIEN-style explicit GPU PhysX state synchronization.
//
// This class intentionally does not hide GPU simulation behind PhysicsWorld's
// CPU getters/setters. GPU state should be synchronized through explicit
// init/fetch/apply/stepStart/stepFinish verbs.
class PhysicsGpuSystem {
  public:
    PhysicsGpuSystem(PhysicsWorld* world, GpuPhysicsConfig config = {});
    ~PhysicsGpuSystem();

    void init();
    void invalidate();
    bool isInitialized() const { return _initialized; }
    void checkInitialized() const;

    void setCudaStream(uint64_t streamHandle);
    uint64_t cudaStream() const { return _streamHandle; }

    void stepStart();
    void stepFinish();

    const Sim::GpuArrayView& rigidData() const {
        return _views.rigidData;
    }
    const Sim::GpuArrayView& articulationLinkData() const {
        return _views.articulationLinkData;
    }
    const Sim::GpuArrayView& rigidForce() const {
        return _views.rigidForce;
    }
    const Sim::GpuArrayView& rigidTorque() const {
        return _views.rigidTorque;
    }
    const Sim::GpuArrayView& articulationJointPositions() const {
        return _views.articulationJointPositions;
    }
    const Sim::GpuArrayView& articulationJointVelocities() const {
        return _views.articulationJointVelocities;
    }
    const Sim::GpuArrayView& articulationJointAccelerations() const {
        return _views.articulationJointAccelerations;
    }
    const Sim::GpuArrayView& articulationJointForces() const {
        return _views.articulationJointForces;
    }
    const Sim::GpuArrayView& articulationTargetJointPositions() const {
        return _views.articulationTargetJointPositions;
    }
    const Sim::GpuArrayView& articulationTargetJointVelocities() const {
        return _views.articulationTargetJointVelocities;
    }
    const Sim::GpuArrayView& articulationLinkIncomingJointForces() const {
        return _views.articulationLinkIncomingJointForces;
    }

    void fetchRigidData();
    void fetchArticulationLinkPose();
    void fetchArticulationLinkVel();
    void fetchArticulationJointPositions();
    void fetchArticulationJointVelocities();
    void fetchArticulationJointAccelerations();
    void fetchArticulationTargetJointPositions();
    void fetchArticulationTargetJointVelocities();
    void fetchArticulationLinkIncomingJointForce();

    void applyRigidData(const Sim::GpuArrayView* indices = nullptr);
    void applyRigidForce(const Sim::GpuArrayView* indices = nullptr);
    void applyRigidTorque(const Sim::GpuArrayView* indices = nullptr);
    void applyArticulationRootPose(const Sim::GpuArrayView* indices = nullptr);
    void applyArticulationRootVel(const Sim::GpuArrayView* indices = nullptr);
    void applyArticulationJointPositions(
        const Sim::GpuArrayView* indices = nullptr);
    void applyArticulationJointVelocities(
        const Sim::GpuArrayView* indices = nullptr);
    void applyArticulationJointForces(
        const Sim::GpuArrayView* indices = nullptr);
    void applyArticulationTargetJointPositions(
        const Sim::GpuArrayView* indices = nullptr);
    void applyArticulationTargetJointVelocities(
        const Sim::GpuArrayView* indices = nullptr);

    void updateArticulationKinematics();
    void syncPosesGpuToCpu();

    const PhysicsGpuStateViews& views() const { return _views; }
    PhysicsGpuStateViews& views() { return _views; }

  private:
    [[noreturn]] void notImplemented(const char* functionName) const;
    void releaseGpuBuffers();

    PhysicsWorld* _world = nullptr;
    GpuPhysicsConfig _config;
    PhysicsGpuStateViews _views;
    bool _initialized = false;
    uint64_t _streamHandle = 0;
    uint32_t _rigidCount = 0;
    void* _rigidIndexBuffer = nullptr;
    void* _rigidScratchBuffer = nullptr;
    void* _rigidMirrorBuffer = nullptr;
    void* _copyEvent = nullptr;
    void* _readyEvent = nullptr;
};

} // namespace KE

#endif
