#pragma once

#include "physics.hpp"
#include <array>
#include <map>
#include <extensions/PxD6Joint.h>

namespace KE {
using JointFrame = std::array<float, 7>; // xyz, xyzw

// A checked, non-owning reference. PhysX deletion callbacks only invalidate it.
class JointBody : public PxDeletionListener {
    PxRigidActor* _actor;
    PxPhysics* _physics;
    std::weak_ptr<bool> _lifetime;
  public:
    JointBody(PhysicsWorld& world, PxRigidActor* actor);
    ~JointBody();
    JointBody(const JointBody&) = delete;
    JointBody& operator=(const JointBody&) = delete;
    bool valid() const;
    PxRigidActor* actor() const;
    void onRelease(const PxBase*, void*, PxDeletionEventFlag::Enum) override;
};

enum class D6Axis { X, Y, Z, Twist, Swing1, Swing2 };
enum class D6Motion { Locked, Limited, Free };
enum class D6DriveAxis { X, Y, Z, Twist, Swing, Slerp };

struct D6DriveConfig {
    float stiffness = 0.f;
    float damping = 0.f;
    float forceLimit = PX_MAX_F32; // N (linear) or N*m (angular), not impulse.
    bool acceleration = false;
};

struct D6JointConfig {
    // Joint-local axis order: X, Y, Z, twist, swing1, swing2.
    std::array<D6Motion, 6> motions = {D6Motion::Locked, D6Motion::Locked,
        D6Motion::Locked, D6Motion::Free, D6Motion::Free, D6Motion::Free};
    std::array<std::array<float, 2>, 3> linearLimits = {{{-1.f, 1.f}, {-1.f, 1.f}, {-1.f, 1.f}}};
    std::array<float, 2> twistLimits = {-PxPi / 2, PxPi / 2};
    // Elliptical cone half-angles about Y and Z, in radians.
    std::array<float, 2> swingLimits = {PxPi / 4, PxPi / 4};
    std::map<D6DriveAxis, D6DriveConfig> drives;
    // Desired pose of frame1 relative to frame0. Velocities also use frame0 axes.
    JointFrame driveTarget = {0, 0, 0, 0, 0, 0, 1};
    std::array<float, 3> driveLinearVelocity = {0, 0, 0};
    std::array<float, 3> driveAngularVelocity = {0, 0, 0};
    float breakForce = PX_MAX_F32;
    float breakTorque = PX_MAX_F32;
    JointFrame frame0 = {0, 0, 0, 0, 0, 0, 1};
    JointFrame frame1 = {0, 0, 0, 0, 0, 0, 1};
    bool enabled = false;
    std::array<bool, 6> lockedAxes() const;
    void setLockedAxes(const std::array<bool, 6>& axes);
    void validate() const;
};

class D6Joint {
    friend class PhysicsGpuSystem;
    PxD6Joint* _joint = nullptr;
    PhysicsWorld* _world = nullptr;
    std::weak_ptr<bool> _worldLifetime;
    std::shared_ptr<JointBody> _body0, _body1;
    bool _enabled = false;
    D6JointConfig _config;
    PxU32 _createdAt = 0;
    // Lazy, reusable storage for synchronous host reads on Direct GPU scenes.
    void* _gpuReadback = nullptr;
    void requireValid() const;
    void wake();
    void applyDrives(const std::map<D6DriveAxis, D6DriveConfig>& drives);
  public:
    ~D6Joint() { release(); }
    D6Joint() = default;
    D6Joint(const D6Joint&) = delete;
    D6Joint& operator=(const D6Joint&) = delete;
    static std::shared_ptr<D6Joint> create(PhysicsWorld& world,
        std::shared_ptr<JointBody> body0, std::shared_ptr<JointBody> body1,
        const D6JointConfig& config);
    bool valid() const;
    bool enabled() const { return valid() && _enabled && !broken(); }
    bool broken() const;
    D6JointConfig config() const;
    void setMotion(D6Axis axis, D6Motion motion);
    void setLinearLimit(D6Axis axis, float lower, float upper);
    void setTwistLimit(float lower, float upper);
    void setSwingLimit(float yAngle, float zAngle);
    void setDrive(D6DriveAxis axis, const D6DriveConfig& config);
    void setDriveTarget(const JointFrame& pose);
    void setDriveVelocity(const std::array<float, 3>& linear,
                          const std::array<float, 3>& angular);
    void setBreakForce(float force, float torque);
    // World-space force and torque on endpoint 1, about its joint frame.
    // Last solver result, synchronously returned on the host.
    // Inherits PhysX TGS torque-reporting limitations; see docs/simulation/D6_JOINTS.md.
    std::array<float, 6> getWrench();
    void setEnabled(bool enabled);
    void setFrame0(const JointFrame& frame0);
    void setFrame1(const JointFrame& frame1);
    void setFrames(const JointFrame& frame0, const JointFrame& frame1);
    JointFrame frame0() const;
    JointFrame frame1() const;
    void release();
};
} // namespace KE
