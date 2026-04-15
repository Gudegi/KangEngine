#include "physics.hpp"
#include "foundation/Px.h"

#include <Eigen/Geometry>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include "engine/scene/native/prim.hpp"

namespace KE {

void PhysicsWorld::step() {
    _scene->simulate(_dt); // _dt is already deltaTime (1/60)
    _scene->fetchResults(true);
}

void PhysicsWorld::fecthData() {
    // TODO: complete me
    PxU32 nbActors = _scene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
}

void PhysicsWorld::addDefaultGround() {
    PxVec3 normal = (_upAxis == UpAxis::Z) ? PxVec3(0, 0, 1) : PxVec3(0, 1, 0);
    PxRigidStatic* groundPlane =
        PxCreatePlane(*_physics, PxPlane(normal, 0), *_material);
    _scene->addActor(*groundPlane);
}

void PhysicsWorld::addBox(float x, float y, float z) {
    PxTransform boxPose(PxVec3(x, y, z));
    PxRigidDynamic* box = PxCreateDynamic(
        *_physics, boxPose, PxBoxGeometry(0.5f, 0.5f, 0.5f), *_material, 1.0f);
    _scene->addActor(*box);
}

void PhysicsWorld::syncAllVisuals() {
    PxTransform pose;
    for (auto& v : _rigidVisuals) {
        if (v.link && !v.rigid) {
            pose = v.link->getGlobalPose();
        } else if (v.rigid && !v.link) {
            pose = v.rigid->getGlobalPose();
        } else {
            fmt::print(
                "Failed to parsing rigid visual, use only 'link' or 'rigid'");
        }
        if (v.prim) {
            v.prim->setAttribute("xformOp:translate", pxToGlm(pose.p));
            v.prim->setAttribute("xformOp:rotateQuaternion", pxToGlm(pose.q));
        } else {
            fmt::print("Failed to parsing rigid visual, check 'prim'");
        }
    }
}

} // namespace KE
