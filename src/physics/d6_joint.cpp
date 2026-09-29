#include "d6_joint.hpp"
#include "physx_compat.hpp"
#include <cmath>
#include <stdexcept>
#include <string>
#include <extensions/PxJointLimit.h>

#if KANGENGINE_PHYSX_VERSION_AT_LEAST(5, 7) || (PX_PHYSICS_VERSION_MAJOR == 5 && PX_PHYSICS_VERSION_MINOR == 6 && PX_PHYSICS_VERSION_BUGFIX >= 1)
#define KE_D6_SEPARATE_SWING_DRIVES
#endif

#if defined(KANGENGINE_USE_CUDA) && KANGENGINE_PHYSX_VERSION_AT_LEAST(5, 6)
#include <cuda_runtime.h>
#include <cudamanager/PxCudaContextManager.h>
#include <PxDirectGPUAPI.h>
#endif

namespace KE {
namespace {
PxTransform frame(const JointFrame& v) {
    for (float x : v)
        if (!std::isfinite(x)) throw std::invalid_argument("Joint frame must be finite");
    PxQuat q(v[3], v[4], v[5], v[6]);
    if (std::abs(q.magnitudeSquared() - 1.f) > 1e-3f)
        throw std::invalid_argument("Joint frame quaternion must be normalized xyzw");
    return PxTransform(PxVec3(v[0], v[1], v[2]), q.getNormalized());
}
JointFrame values(const PxTransform& p) {
    return {p.p.x, p.p.y, p.p.z, p.q.x, p.q.y, p.q.z, p.q.w};
}
int axisIndex(D6Axis axis, int count = 6) {
    const int i = static_cast<int>(axis);
    if (i < 0 || i >= count) throw std::invalid_argument("Invalid D6 axis");
    return i;
}
void motionValid(D6Motion motion) {
    if (motion != D6Motion::Locked && motion != D6Motion::Limited && motion != D6Motion::Free)
        throw std::invalid_argument("Invalid D6 motion");
}
void nonnegative(float value, const char* name) {
    if (!std::isfinite(value) || value < 0.f)
        throw std::invalid_argument(std::string(name) + " must be finite and nonnegative");
}
PxVec3 vector(const std::array<float, 3>& v) {
    for (float x : v)
        if (!std::isfinite(x)) throw std::invalid_argument("Drive velocity must be finite");
    return PxVec3(v[0], v[1], v[2]);
}
PxJointLinearLimitPair linearLimit(float lo, float hi) {
    if (!std::isfinite(lo) || !std::isfinite(hi) || lo >= hi)
        throw std::invalid_argument("Linear limits require finite lower < upper (meters)");
    return PxJointLinearLimitPair(PxTolerancesScale(), lo, hi);
}
PxJointAngularLimitPair twistLimit(float lo, float hi) {
    if (!std::isfinite(lo) || !std::isfinite(hi) || lo >= hi || lo <= -PxTwoPi || hi >= PxTwoPi)
        throw std::invalid_argument("Twist limits require -2*pi < lower < upper < 2*pi");
    return PxJointAngularLimitPair(lo, hi);
}
PxJointLimitCone swingLimit(float y, float z) {
    if (!std::isfinite(y) || !std::isfinite(z) || y <= 0 || z <= 0 || y >= PxPi || z >= PxPi)
        throw std::invalid_argument("Swing cone half-angles must be in (0, pi)");
    return PxJointLimitCone(y, z);
}
bool activeDrive(const D6DriveConfig& drive) {
    return drive.forceLimit > 0 && (drive.stiffness > 0 || drive.damping > 0);
}
D6DriveConfig driveAt(const std::map<D6DriveAxis, D6DriveConfig>& drives, D6DriveAxis axis) {
    const auto it = drives.find(axis);
    return it == drives.end() ? D6DriveConfig{} : it->second;
}
void validateDrives(const std::map<D6DriveAxis, D6DriveConfig>& drives,
                    const std::array<D6Motion, 6>& motions) {
    for (const auto& [axis, drive] : drives) {
        const int i = static_cast<int>(axis);
        if (i < 0 || i > static_cast<int>(D6DriveAxis::Slerp))
            throw std::invalid_argument("Invalid D6 drive axis");
        nonnegative(drive.stiffness, "Drive stiffness");
        nonnegative(drive.damping, "Drive damping");
        nonnegative(drive.forceLimit, "Drive force limit");
    }
    if (activeDrive(driveAt(drives, D6DriveAxis::Slerp))) {
        if (activeDrive(driveAt(drives, D6DriveAxis::Swing)) ||
            activeDrive(driveAt(drives, D6DriveAxis::Twist)))
            throw std::invalid_argument("SLERP and swing/twist drives cannot be active together");
        for (int i = 3; i < 6; ++i)
            if (motions[i] == D6Motion::Locked)
                throw std::invalid_argument("SLERP requires all angular axes unlocked");
    }
}
bool directGpu(const PhysicsWorld& world) {
#ifdef KANGENGINE_HAS_PHYSX_DIRECT_GPU_API
    return world.getScene()->getFlags().isSet(PxSceneFlag::eENABLE_DIRECT_GPU_API);
#else
    return false;
#endif
}
void validateBreakSupport(const PhysicsWorld& world, float force, float torque) {
#ifndef KE_D6_SEPARATE_SWING_DRIVES
    if (directGpu(world) && (force < PX_MAX_F32 || torque < PX_MAX_F32))
        throw std::runtime_error("Direct GPU joint breakage requires PhysX 5.6.1 or newer");
#endif
}
#if defined(KANGENGINE_USE_CUDA) && KANGENGINE_PHYSX_VERSION_AT_LEAST(5, 6)
void checkCuda(cudaError_t result) {
    if (result != cudaSuccess)
        throw std::runtime_error(std::string("D6 readback: ") + cudaGetErrorString(result));
}
#endif

}

std::array<bool, 6> D6JointConfig::lockedAxes() const {
    std::array<bool, 6> result;
    for (int i = 0; i < 6; ++i) result[i] = motions[i] == D6Motion::Locked;
    return result;
}
void D6JointConfig::setLockedAxes(const std::array<bool, 6>& axes) {
    for (int i = 0; i < 6; ++i) motions[i] = axes[i] ? D6Motion::Locked : D6Motion::Free;
}
void D6JointConfig::validate() const {
    frame(frame0); frame(frame1); frame(driveTarget);
    vector(driveLinearVelocity); vector(driveAngularVelocity);
    for (auto motion : motions) motionValid(motion);
    for (const auto& limits : linearLimits) linearLimit(limits[0], limits[1]);
    twistLimit(twistLimits[0], twistLimits[1]);
    swingLimit(swingLimits[0], swingLimits[1]);
    nonnegative(breakForce, "Break force"); nonnegative(breakTorque, "Break torque");
    validateDrives(drives, motions);
}

JointBody::JointBody(PhysicsWorld& world, PxRigidActor* actor)
    : _actor(actor), _physics(world.getPhysics()), _lifetime(world.jointLifetime()) {
    if (!actor || actor->getScene() != world.getScene())
        throw std::invalid_argument("Joint body must belong to this world");
    _physics->registerDeletionListener(*this, PxDeletionEventFlag::eUSER_RELEASE, true);
    const PxBase* observed[] = {actor, actor};
    PxU32 count = 1;
    if (auto* link = actor->is<PxArticulationLink>()) {
        observed[1] = &link->getArticulation();
        count = 2;
    }
    _physics->registerDeletionListenerObjects(*this, observed, count);
}
JointBody::~JointBody() {
    auto life = _lifetime.lock();
    if (life && *life) _physics->unregisterDeletionListener(*this);
}
bool JointBody::valid() const {
    auto life = _lifetime.lock();
    return _actor && life && *life;
}
PxRigidActor* JointBody::actor() const {
    if (!valid()) throw std::runtime_error("Joint body was released");
    return _actor;
}
void JointBody::onRelease(const PxBase*, void*, PxDeletionEventFlag::Enum) {
    _actor = nullptr;
}

std::shared_ptr<D6Joint> D6Joint::create(PhysicsWorld& world,
    std::shared_ptr<JointBody> body0, std::shared_ptr<JointBody> body1,
    const D6JointConfig& config) {
    config.validate();
    validateBreakSupport(world, config.breakForce, config.breakTorque);
    auto* a = body0 ? body0->actor() : nullptr;
    auto* b = body1 ? body1->actor() : nullptr;
    if ((!a && !b) || a == b)
        throw std::invalid_argument("D6 needs two distinct endpoints, at least one body");
    for (auto* actor : {a, b})
        if (actor && actor->getScene() != world.getScene())
            throw std::invalid_argument("D6 endpoints must belong to this world");
    if ((!a || a->is<PxRigidStatic>()) && (!b || b->is<PxRigidStatic>()))
        throw std::invalid_argument("D6 needs a dynamic body or articulation link");
    const auto f0 = frame(config.frame0), f1 = frame(config.frame1);
    auto result = std::make_shared<D6Joint>();
    result->_body0 = std::move(body0);
    result->_body1 = std::move(body1);
    result->_joint = PxD6JointCreate(*world.getPhysics(), a, f0, b, f1);
    if (!result->_joint) throw std::runtime_error("PxD6JointCreate failed");
    result->_world = &world;
    result->_worldLifetime = world.jointLifetime();
    result->_config = config;
    result->_createdAt = world.getScene()->getTimestamp();
    for (int i = 0; i < 6; ++i)
        result->_joint->setMotion(static_cast<PxD6Axis::Enum>(i),
                                 static_cast<PxD6Motion::Enum>(config.motions[i]));
    for (int i = 0; i < 3; ++i)
        result->_joint->setLinearLimit(static_cast<PxD6Axis::Enum>(i),
            linearLimit(config.linearLimits[i][0], config.linearLimits[i][1]));
    result->_joint->setTwistLimit(twistLimit(config.twistLimits[0], config.twistLimits[1]));
    result->_joint->setSwingLimit(swingLimit(config.swingLimits[0], config.swingLimits[1]));
    result->_joint->setConstraintFlag(PxConstraintFlag::eDRIVE_LIMITS_ARE_FORCES, true);
    result->applyDrives(config.drives);
    result->_joint->setDrivePosition(frame(config.driveTarget), false);
    result->_joint->setDriveVelocity(vector(config.driveLinearVelocity),
                                    vector(config.driveAngularVelocity), false);
    result->_joint->setBreakForce(config.breakForce, config.breakTorque);
    // Apply the initial state even when it equals the cached default (false).
    result->_joint->setConstraintFlag(PxConstraintFlag::eDISABLE_CONSTRAINT, !config.enabled);
    result->_enabled = config.enabled;
    result->wake();
    world.trackD6Joint(result);
    return result;
}
bool D6Joint::valid() const {
    return _joint && (!_body0 || _body0->valid()) && (!_body1 || _body1->valid());
}
void D6Joint::requireValid() const {
    if (!valid()) throw std::runtime_error("D6 joint or one of its endpoints was released");
}
void D6Joint::wake() {
    PxArticulationReducedCoordinate* previous = nullptr;
    for (const auto& body : {_body0, _body1}) {
        if (!body) continue;
        auto* actor = body->actor();
        if (auto* link = actor->is<PxArticulationLink>()) {
            auto* articulation = &link->getArticulation();
            if (articulation != previous) articulation->wakeUp();
            previous = articulation;
        } else if (auto* rigid = actor->is<PxRigidDynamic>()) {
            // Direct GPU scenes disable sleeping; CPU wakeUp is disallowed there.
            bool directGpu = false;
#ifdef KANGENGINE_HAS_PHYSX_DIRECT_GPU_API
            directGpu = actor->getScene()->getFlags() & PxSceneFlag::eENABLE_DIRECT_GPU_API;
#endif
            if (!directGpu && !(rigid->getRigidBodyFlags() & PxRigidBodyFlag::eKINEMATIC))
                rigid->wakeUp();
        }
    }
}
void D6Joint::setEnabled(bool enabled) {
    requireValid();
    if (enabled && broken()) throw std::runtime_error("A broken D6 must be released and recreated");
    if (_enabled == enabled) return;
    _joint->setConstraintFlag(PxConstraintFlag::eDISABLE_CONSTRAINT, !enabled);
    _enabled = enabled;
    wake();
}
void D6Joint::setFrame0(const JointFrame& value) {
    requireValid();
    _joint->setLocalPose(PxJointActorIndex::eACTOR0, frame(value));
    wake();
}
void D6Joint::setFrame1(const JointFrame& value) {
    requireValid();
    _joint->setLocalPose(PxJointActorIndex::eACTOR1, frame(value));
    wake();
}
void D6Joint::setFrames(const JointFrame& frame0, const JointFrame& frame1) {
    requireValid();
    auto a = frame(frame0), b = frame(frame1);
    _joint->setLocalPose(PxJointActorIndex::eACTOR0, a);
    _joint->setLocalPose(PxJointActorIndex::eACTOR1, b);
    wake();
}
JointFrame D6Joint::frame0() const { requireValid(); return values(_joint->getLocalPose(PxJointActorIndex::eACTOR0)); }
JointFrame D6Joint::frame1() const { requireValid(); return values(_joint->getLocalPose(PxJointActorIndex::eACTOR1)); }
bool D6Joint::broken() const {
    requireValid();
    return _joint->getConstraintFlags().isSet(PxConstraintFlag::eBROKEN);
}
D6JointConfig D6Joint::config() const {
    requireValid();
    auto result = _config;
    result.frame0 = frame0(); result.frame1 = frame1(); result.enabled = enabled();
    return result;
}
void D6Joint::setMotion(D6Axis axis, D6Motion motion) {
    requireValid();
    const int i = axisIndex(axis);
    motionValid(motion);
    auto motions = _config.motions;
    motions[i] = motion;
    validateDrives(_config.drives, motions);
    _joint->setMotion(static_cast<PxD6Axis::Enum>(i), static_cast<PxD6Motion::Enum>(motion));
    _config.motions = motions;
    wake();
}
void D6Joint::setLinearLimit(D6Axis axis, float lower, float upper) {
    requireValid();
    const int i = axisIndex(axis, 3);
    auto limit = linearLimit(lower, upper);
    _joint->setLinearLimit(static_cast<PxD6Axis::Enum>(i), limit);
    _config.linearLimits[i] = {lower, upper};
    wake();
}
void D6Joint::setTwistLimit(float lower, float upper) {
    requireValid();
    _joint->setTwistLimit(twistLimit(lower, upper));
    _config.twistLimits = {lower, upper};
    wake();
}
void D6Joint::setSwingLimit(float yAngle, float zAngle) {
    requireValid();
    _joint->setSwingLimit(swingLimit(yAngle, zAngle));
    _config.swingLimits = {yAngle, zAngle};
    wake();
}
void D6Joint::applyDrives(const std::map<D6DriveAxis, D6DriveConfig>& drives) {
    const bool slerp = activeDrive(driveAt(drives, D6DriveAxis::Slerp));
#ifdef KE_D6_SEPARATE_SWING_DRIVES
    const auto model = slerp ? PxD6AngularDriveConfig::eSLERP : PxD6AngularDriveConfig::eSWING_TWIST;
    if (_joint->getAngularDriveConfig() != model) _joint->setAngularDriveConfig(model);
#endif
    auto set = [&](PxD6Drive::Enum nativeAxis, D6DriveAxis axis) {
        const auto config = driveAt(drives, axis);
        // A zero cap disables this motor, including legacy SLERP mode selection.
        PxD6JointDrive drive(config.forceLimit > 0 ? config.stiffness : 0.f,
                             config.forceLimit > 0 ? config.damping : 0.f,
                             config.forceLimit, config.acceleration);
#if KANGENGINE_PHYSX_VERSION_AT_LEAST(5, 5)
        // Readback and break thresholds include motor effort as well as hard constraints.
        drive.flags |= PxD6JointDriveFlag::eOUTPUT_FORCE;
#endif
        _joint->setDrive(nativeAxis, drive);
    };
    set(PxD6Drive::eX, D6DriveAxis::X);
    set(PxD6Drive::eY, D6DriveAxis::Y);
    set(PxD6Drive::eZ, D6DriveAxis::Z);
#ifdef KE_D6_SEPARATE_SWING_DRIVES
    if (slerp) set(PxD6Drive::eSLERP, D6DriveAxis::Slerp);
    else {
        set(PxD6Drive::eTWIST, D6DriveAxis::Twist);
        set(PxD6Drive::eSWING1, D6DriveAxis::Swing);
        set(PxD6Drive::eSWING2, D6DriveAxis::Swing);
    }
#else
    set(PxD6Drive::eTWIST, D6DriveAxis::Twist);
    set(PxD6Drive::eSWING, D6DriveAxis::Swing);
    set(PxD6Drive::eSLERP, D6DriveAxis::Slerp);
#endif
}
void D6Joint::setDrive(D6DriveAxis axis, const D6DriveConfig& config) {
    requireValid();
    auto drives = _config.drives;
    drives[axis] = config;
    validateDrives(drives, _config.motions);
    applyDrives(drives);
    _config.drives = std::move(drives);
    wake();
}
void D6Joint::setDriveTarget(const JointFrame& pose) {
    requireValid();
    _joint->setDrivePosition(frame(pose), false);
    _config.driveTarget = pose;
    wake();
}
void D6Joint::setDriveVelocity(const std::array<float, 3>& linear, const std::array<float, 3>& angular) {
    requireValid();
    auto v = vector(linear), w = vector(angular);
    _joint->setDriveVelocity(v, w, false);
    _config.driveLinearVelocity = linear; _config.driveAngularVelocity = angular;
    wake();
}
void D6Joint::setBreakForce(float force, float torque) {
    requireValid();
    nonnegative(force, "Break force"); nonnegative(torque, "Break torque");
    validateBreakSupport(*_world, force, torque);
    _joint->setBreakForce(force, torque);
    _config.breakForce = force; _config.breakTorque = torque;
    wake();
}
std::array<float, 6> D6Joint::getWrench() {
    requireValid();
    if (!_enabled || broken()) return {};
    if (_world->getScene()->getTimestamp() == _createdAt)
        throw std::runtime_error("D6 wrench requires a completed physics step after creation");
#if !KANGENGINE_PHYSX_VERSION_AT_LEAST(5, 5)
    for (const auto& item : _config.drives)
        if (activeDrive(item.second))
            throw std::runtime_error("Drive-inclusive D6 wrench requires PhysX 5.5 or newer");
#endif
    PxVec3 force(0), torque(0);
    if (directGpu(*_world)) {
#if defined(KANGENGINE_USE_CUDA) && KANGENGINE_PHYSX_VERSION_AT_LEAST(5, 6)
        const auto index = _joint->getGPUIndex();
        if (index == PX_INVALID_D6_JOINT_GPU_INDEX)
            throw std::runtime_error("D6 GPU index unavailable; complete a physics step first");
        PxScopedCudaLock lock(*_world->getCudaContextManager());
        constexpr size_t dataSize = 2 * sizeof(PxVec3);
        if (!_gpuReadback) checkCuda(cudaMalloc(&_gpuReadback, dataSize + sizeof(index)));
        auto* gpuIndex = reinterpret_cast<PxD6JointGPUIndex*>(static_cast<char*>(_gpuReadback) + dataSize);
        checkCuda(cudaMemcpy(gpuIndex, &index, sizeof(index), cudaMemcpyHostToDevice));
        auto* gpuTorque = static_cast<char*>(_gpuReadback) + sizeof(PxVec3);
        auto& api = _world->getScene()->getDirectGPUAPI();
        if (!api.getD6JointData(_gpuReadback, gpuIndex, PxD6JointGPUAPIReadType::eJOINT_FORCE, 1) ||
            !api.getD6JointData(gpuTorque, gpuIndex, PxD6JointGPUAPIReadType::eJOINT_TORQUE, 1))
            throw std::runtime_error("D6 Direct GPU readback failed");
        std::array<PxVec3, 2> values;
        checkCuda(cudaMemcpy(values.data(), _gpuReadback, dataSize, cudaMemcpyDeviceToHost));
        force = values[0]; torque = values[1];
#else
        throw std::runtime_error("D6 Direct GPU readback requires a CUDA build with PhysX 5.6 or newer");
#endif
    } else {
        _joint->getConstraint()->getForce(force, torque);
    }
    // PhysX writes the wrench on body 0, about body 1's attachment point.
    return {-force.x, -force.y, -force.z, -torque.x, -torque.y, -torque.z};
}

void D6Joint::release() {
#if defined(KANGENGINE_USE_CUDA) && KANGENGINE_PHYSX_VERSION_AT_LEAST(5, 6)
    if (_gpuReadback) {
        PxScopedCudaLock lock(*_world->getCudaContextManager());
        cudaFree(_gpuReadback);
        _gpuReadback = nullptr;
    }
#endif
    if (_world) {
        auto life = _worldLifetime.lock();
        if (life && *life) _world->untrackD6Joint(this);
        _world = nullptr;
        _worldLifetime.reset();
    }
    if (_joint) { _joint->release(); _joint = nullptr; }
    _enabled = false;
    _body0.reset(); _body1.reset();
}
} // namespace KE
