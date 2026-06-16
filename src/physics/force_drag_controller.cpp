#include "physics/force_drag_controller.hpp"
#include "physics/articulation.hpp"

#include <extensions/PxRigidBodyExt.h>
#include <algorithm>
#include <glm/geometric.hpp>

namespace KE {
namespace {

physx::PxVec3 pointVelocity(const physx::PxTransform& pose,
                            const physx::PxVec3& linearVelocity,
                            const physx::PxVec3& angularVelocity,
                            const physx::PxVec3& worldPoint) {
    const physx::PxVec3 r = worldPoint - pose.p; // rot radius
    return linearVelocity + angularVelocity.cross(r);
}

} // namespace

ForceDragController::ForceDragController() : _config(ForceDragConfig{}) {}

ForceDragController::ForceDragController(ForceDragConfig config)
    : _config(config) {}

void ForceDragController::registerArticulation(
    Articulation& articulation,
    const std::vector<RenderableHandle>& bodyHandles) {
    const int n =
        std::min(static_cast<int>(bodyHandles.size()), articulation.numLinks());
    for (int i = 0; i < n; ++i) {
        const RenderableHandle handle = bodyHandles[static_cast<size_t>(i)];
        if (handle == InvalidHandle)
            continue;
        _bindings[handle] = Binding{&articulation, nullptr, i};
    }
}

void ForceDragController::registerRigid(physx::PxRigidDynamic& rigid,
                                        RenderableHandle handle) {
    _bindings[handle] = Binding{nullptr, &rigid, -1};
}

void ForceDragController::clearBindings() {
    end();
    _bindings.clear();
}

bool ForceDragController::begin(const RayPickResult& pick,
                                const glm::vec3& target) {
    end();
    if (!pick.hit)
        return false;

    auto it = _bindings.find(pick.handle);
    if (it == _bindings.end())
        return false;

    _activeBinding = it->second;
    const physx::PxVec3 hitPoint = toPxVec3(pick.position);
    if (_activeBinding.articulation) {
        physx::PxArticulationLink* link =
            _activeBinding.articulation->link(_activeBinding.linkIndex);
        _localAnchor = link->getGlobalPose().transformInv(hitPoint);
    } else if (_activeBinding.rigid) {
        _localAnchor =
            _activeBinding.rigid->getGlobalPose().transformInv(hitPoint);
    }
    _active = true;
    applyForce(target);
    return true;
}

void ForceDragController::update(const glm::vec3& target) {
    if (!_active)
        return;
    applyForce(target);
}

void ForceDragController::end() {
    _active = false;
    _activeBinding = Binding{};
    _localAnchor = physx::PxVec3(0.0f);
    _lastForce = glm::vec3(0.0f);
}

glm::vec3 ForceDragController::clampForce(glm::vec3 force, float maxForce) {
    if (maxForce <= 0.0f)
        return force;
    const float len = glm::length(force);
    if (len <= maxForce || len <= 1e-6f)
        return force;
    return force * (maxForce / len);
}

void ForceDragController::applyForce(const glm::vec3& target) {
    glm::vec3 pos(0.0f);
    glm::vec3 anchorPos(0.0f);
    glm::vec3 vel(0.0f);
    physx::PxVec3 anchorWorld(0.0f);

    if (_activeBinding.articulation) {
        physx::PxArticulationLink* link =
            _activeBinding.articulation->link(_activeBinding.linkIndex);
        const physx::PxTransform pose = link->getGlobalPose();
        anchorWorld = pose.transform(_localAnchor);
        pos = pxToVec3(pose.p);
        anchorPos = pxToVec3(anchorWorld);
        vel = pxToVec3(pointVelocity(pose, link->getLinearVelocity(),
                                     link->getAngularVelocity(), anchorWorld));
    } else if (_activeBinding.rigid) {
        const physx::PxTransform pose = _activeBinding.rigid->getGlobalPose();
        anchorWorld = pose.transform(_localAnchor);
        pos = pxToVec3(pose.p);
        anchorPos = pxToVec3(anchorWorld);
        vel = pxToVec3(pointVelocity(
            pose, _activeBinding.rigid->getLinearVelocity(),
            _activeBinding.rigid->getAngularVelocity(), anchorWorld));
    } else {
        return;
    }

    _lastBodyPosition = pos;
    _lastAnchorPosition = anchorPos;
    _lastTarget = target;

    glm::vec3 force =
        (target - anchorPos) * _config.stiffness - vel * _config.damping;
    force = clampForce(force, _config.maxForce);
    _lastForce = force;

    if (_activeBinding.articulation) {
        _activeBinding.articulation->addLinkForceAtPosition(
            _activeBinding.linkIndex, toPxVec3(force), anchorWorld);
    } else if (_activeBinding.rigid) {
        physx::PxRigidBodyExt::addForceAtPos(*_activeBinding.rigid,
                                             toPxVec3(force), anchorWorld,
                                             physx::PxForceMode::eFORCE, true);
    }
}

} // namespace KE
