#ifndef _PHYSICS_GPU_SYSTEM_HPP_
#define _PHYSICS_GPU_SYSTEM_HPP_

#include "sim/gpu_array_view.hpp"

#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace physx {
class PxArticulationReducedCoordinate;
class PxRigidDynamic;
} // namespace physx

namespace KE {

class PhysicsWorld;

struct GpuPhysicsConfig {
    static constexpr uint32_t DefaultMaxContactPairs = 65536;
    static constexpr uint32_t DefaultMaxContactPoints = 262144;

    int cudaDeviceId = 0;
    // Fixed-capacity mirrors avoid allocation during simulation. Override these
    // defaults for workloads whose contact density is known in advance.
    uint32_t maxContactPairs = DefaultMaxContactPairs;
    uint32_t maxContactPoints = DefaultMaxContactPoints;
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
    Sim::GpuArrayView contactPairs;
    Sim::GpuArrayView contactPairCount;
    Sim::GpuArrayView contactPairHeaders;
    Sim::GpuArrayView contactPairBodyRefs;
    Sim::GpuArrayView contactPoints;
    Sim::GpuArrayView contactPointCount;
    Sim::GpuArrayView contactPointPairIndices;
};

// SAPIEN-style explicit GPU PhysX state synchronization.
//
// This class intentionally does not hide GPU simulation behind PhysicsWorld's
// CPU getters/setters. GPU state should be synchronized through explicit
// init/fetch/apply/stepStart/stepFinish verbs.
//
// Buffer ownership:
// - rigidData/rigidForce/rigidTorque and articulationLinkData are owned by
//   PhysicsGpuSystem.
// - GpuArrayView objects are metadata views; do not free their ptr values.
// - Views become invalid after invalidate(), destruction, or a later init().
//
// Stream/sync:
// - setCudaStream() selects the stream used for CUDA pack/copy kernels.
// - fetch/apply record and wait on the exported ready_event_handle.
// - External CUDA producers should set their GpuArrayView stream/event metadata
//   before passing views as indices.
//
// Sparse indices:
// - apply* indices are logical rigid rows in these views, not PhysX GPU
// indices.
// - indices must be a contiguous CUDA int32/uint32 [count] view on
// cudaDeviceId.
// - nullptr means dense apply over all rigid rows.
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
    // TODO: Treat this as a bootstrap lookup for high-level row caches, not as
    // a per-frame path. KangSimWorld should precompute env/object -> rigid row
    // tables and reuse CUDA index buffers for batched sparse apply.
    uint32_t rigidRow(const physx::PxRigidDynamic& rigid) const;
    uint32_t articulationRow(
        const physx::PxArticulationReducedCoordinate& articulation) const;
    uint32_t articulationLinkCount(uint32_t articulationRow) const;
    uint32_t articulationDofCount(uint32_t articulationRow) const;
    uint32_t articulationCount() const { return _articulationCount; }
    uint32_t articulationMaxLinks() const { return _articulationMaxLinks; }
    uint32_t articulationMaxDofs() const { return _articulationMaxDofs; }

    void stepStart();
    void stepFinish();

    const Sim::GpuArrayView& rigidData() const { return _views.rigidData; }
    const Sim::GpuArrayView& articulationLinkData() const {
        return _views.articulationLinkData;
    }
    const Sim::GpuArrayView& rigidForce() const { return _views.rigidForce; }
    const Sim::GpuArrayView& rigidTorque() const { return _views.rigidTorque; }
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
    const Sim::GpuArrayView& contactPairs() const {
        return _views.contactPairs;
    }
    const Sim::GpuArrayView& contactPairCount() const {
        return _views.contactPairCount;
    }
    const Sim::GpuArrayView& contactPairHeaders() const {
        return _views.contactPairHeaders;
    }
    const Sim::GpuArrayView& contactPairBodyRefs() const {
        return _views.contactPairBodyRefs;
    }
    const Sim::GpuArrayView& contactPoints() const {
        return _views.contactPoints;
    }
    const Sim::GpuArrayView& contactPointCount() const {
        return _views.contactPointCount;
    }
    const Sim::GpuArrayView& contactPointPairIndices() const {
        return _views.contactPointPairIndices;
    }

    void fetchRigidData();
    void fetchArticulationLinkPose();
    void fetchArticulationLinkVel();
    void fetchArticulationJointPositions();
    void fetchArticulationJointVelocities();
    void fetchArticulationJointAccelerations();
    void fetchArticulationJointForces();
    void fetchArticulationTargetJointPositions();
    void fetchArticulationTargetJointVelocities();
    void fetchArticulationLinkIncomingJointForce();
    void fetchContactPairs();

    void applyRigidData(const Sim::GpuArrayView* indices = nullptr);
    void applyRigidForce(const Sim::GpuArrayView* indices = nullptr);
    void applyRigidTorque(const Sim::GpuArrayView* indices = nullptr);
    void applyArticulationRootPose(const Sim::GpuArrayView* indices = nullptr);
    void applyArticulationRootVel(const Sim::GpuArrayView* indices = nullptr);
    void
    applyArticulationJointPositions(const Sim::GpuArrayView* indices = nullptr);
    void applyArticulationJointVelocities(
        const Sim::GpuArrayView* indices = nullptr);
    void
    applyArticulationJointForces(const Sim::GpuArrayView* indices = nullptr);
    void applyArticulationTargetJointPositions(
        const Sim::GpuArrayView* indices = nullptr);
    void applyArticulationTargetJointVelocities(
        const Sim::GpuArrayView* indices = nullptr);
    void clearRigidCommands(const Sim::GpuArrayView* indices = nullptr);
    void clearArticulationCommands(const Sim::GpuArrayView* indices = nullptr);

    void updateArticulationKinematics();
    void syncPosesGpuToCpu();

    const PhysicsGpuStateViews& views() const { return _views; }
    PhysicsGpuStateViews& views() { return _views; }

  private:
    [[noreturn]] void notImplemented(const char* functionName) const;
    void fetchArticulationDofBuffer(void* buffer, Sim::GpuArrayView& view,
                                    int readType, const char* dataName,
                                    const char* waitOperation,
                                    const char* readyOperation);
    void applyArticulationDofBuffer(const Sim::GpuArrayView* indices,
                                    void* buffer, Sim::GpuArrayView& view,
                                    int writeType, const char* dataName,
                                    const char* waitOperation,
                                    const char* readyOperation);
    void packArticulationRootState(const Sim::GpuArrayView* indices,
                                   void*& poseBuffer,
                                   void*& linearVelocityBuffer,
                                   void*& angularVelocityBuffer,
                                   void*& gpuIndices, uint32_t& applyCount);
    void applyRigidCommand(const Sim::GpuArrayView* indices, bool torque);
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
    void* _rigidForceBuffer = nullptr;
    void* _rigidTorqueBuffer = nullptr;
    std::unordered_map<const void*, uint32_t> _rigidRows;
    uint32_t _articulationCount = 0;
    uint32_t _articulationMaxLinks = 0;
    uint32_t _articulationMaxDofs = 0;
    void* _articulationIndexBuffer = nullptr;
    void* _articulationLinkScratchBuffer = nullptr;
    void* _articulationDofScratchBuffer = nullptr;
    void* _articulationLinkMirrorBuffer = nullptr;
    void* _articulationJointPositionBuffer = nullptr;
    void* _articulationJointVelocityBuffer = nullptr;
    void* _articulationJointAccelerationBuffer = nullptr;
    void* _articulationJointForceBuffer = nullptr;
    void* _articulationTargetJointPositionBuffer = nullptr;
    void* _articulationTargetJointVelocityBuffer = nullptr;
    void* _articulationLinkIncomingJointForceBuffer = nullptr;
    void* _contactPairBuffer = nullptr;
    void* _contactPairCountBuffer = nullptr;
    void* _contactPairHeaderBuffer = nullptr;
    void* _contactNodeBodyRefBuffer = nullptr;
    uint32_t _contactNodeBodyRefCapacity = 0;
    void* _contactPairBodyRefBuffer = nullptr;
    void* _contactPointBuffer = nullptr;
    void* _contactPointCountBuffer = nullptr;
    void* _contactPointPairIndexBuffer = nullptr;
    std::vector<uint32_t> _articulationLinkCounts;
    std::vector<uint32_t> _articulationDofCounts;
    std::unordered_map<const void*, uint32_t> _articulationRows;
    void* _copyEvent = nullptr;
    void* _readyEvent = nullptr;
};

} // namespace KE

#endif
