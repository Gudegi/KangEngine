#include "physics_bridge.hpp"
#include "engine/core/app/app.hpp"
#include "engine/scene/native/prim.hpp"
#include "engine/scene/scene_backend.hpp"
#include "physics/articulation.hpp"
#include "physics/physics.hpp"
#include "skeleton_bridge.hpp"

#include <Eigen/Geometry>
#include <cassert>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

namespace KE {
namespace Bridge {

void PhysicsBridge::add(const Articulation& artic,
                        const SkeletonBridge& skelBridge) {
    int n = artic.numLinks();
    for (int i = 0; i < n; i++)
        _primVisuals.push_back({artic.link(i), skelBridge.bodyPrim(i)});
}

void PhysicsBridge::addInstanced(std::vector<Articulation*> artics,
                                 std::vector<MeshHandle> handles) {
    InstancedGroup group;
    group.handles = std::move(handles);
    group.artics.reserve(artics.size());
    for (auto* artic : artics)
        group.artics.push_back(artic);
    _instancedGroups.push_back(std::move(group));
}

void PhysicsBridge::addInstanced(const Articulation& artic,
                                 const std::vector<MeshHandle>& handles) {
    for (auto& group : _instancedGroups) {
        if (group.handles == handles) {
            group.artics.push_back(&artic);
            return;
        }
    }

    _instancedGroups.push_back(
        {std::vector<const Articulation*>{&artic}, handles});
}

void PhysicsBridge::sync() {
    // Prim-based: PhysX pose -> Prim xform attributes
    for (auto& v : _primVisuals) {
        physx::PxTransform pose = v.link->getGlobalPose();
        v.prim->setAttribute("xformOp:translate", pxToGlm(pose.p));
        v.prim->setAttribute("xformOp:rotateQuaternion", pxToGlm(pose.q));
    }

    // Handle-based instanced: collect N transforms per body, upload to VBO
    assert((_instancedGroups.empty() || _app) &&
           "App* required for instanced sync — pass it to PhysicsBridge ctor");
    for (auto& grp : _instancedGroups) {
        int numBodies = (int)grp.handles.size();
        int numRobots = (int)grp.artics.size();
        std::vector<glm::mat4> transforms(numRobots);
        for (int b = 0; b < numBodies; b++) {
            for (int i = 0; i < numRobots; i++)
                transforms[i] =
                    pxToMat4(grp.artics[i]->link(b)->getGlobalPose());
            _app->updateShapeTransforms(grp.handles[b], transforms);
        }
    }

    // Collision visuals: link pose * local offset
    for (auto& cv : _colVisuals) {
        physx::PxTransform pose = cv.link->getGlobalPose();
        glm::vec3 linkPos = pxToGlm(pose.p);
        glm::quat linkRot = pxToGlm(pose.q);
        cv.prim->setAttribute("xformOp:translate",
                              linkPos + linkRot * cv.localPos);
        cv.prim->setAttribute("xformOp:rotateQuaternion",
                              linkRot * cv.localQuat);
    }
}

std::vector<Scene::Prim*> PhysicsBridge::addCollisionVisuals(
    const Articulation& artic, Scene::SceneBackend* scene,
    const std::string& basePath, bool visibleByDefault) {
    std::vector<Scene::Prim*> result;

    const auto& links = artic.links();
    const auto& colGeoms = artic.colGeoms();

    for (auto& [bodyIdx, geoms] : colGeoms) {
        if (bodyIdx >= static_cast<int>(links.size()))
            continue;
        physx::PxArticulationLink* lnk = links[bodyIdx];

        for (int gi = 0; gi < static_cast<int>(geoms.size()); gi++) {
            const auto& geom = geoms[gi];
            std::string path = basePath + "/b" + std::to_string(bodyIdx) +
                               "_g" + std::to_string(gi);
            auto* prim = scene->definePrim(path, Scene::PrimType::Mesh);

            Scene::MeshData meshData;
            glm::vec3 localPos{0.f};
            glm::quat localQuat{1.f, 0.f, 0.f, 0.f};

            if (geom.hasFromTo) {
                Eigen::Vector3f center = (geom.from + geom.to) * 0.5f;
                Eigen::Vector3f axis = (geom.to - geom.from).normalized();
                float halfLen = (geom.to - geom.from).norm() * 0.5f;
                Eigen::Quaternionf eq = Eigen::Quaternionf::FromTwoVectors(
                    Eigen::Vector3f::UnitZ(), axis);
                localPos = glm::vec3(center.x(), center.y(), center.z());
                localQuat = glm::quat(eq.w(), eq.x(), eq.y(), eq.z());

                float r = geom.size[0];
                if (geom.type == Animation::CollisionGeom::Type::Capsule)
                    meshData = Scene::Prim::createCapsuleData(r, halfLen * 2.f,
                                                              UpAxis::Z, 12);
                else
                    meshData = Scene::Prim::createCylinderData(r, halfLen * 2.f,
                                                               UpAxis::Z, 12);
            } else {
                localPos = glm::vec3(geom.pos.x(), geom.pos.y(), geom.pos.z());
                localQuat = glm::quat(geom.quat.w(), geom.quat.x(),
                                      geom.quat.y(), geom.quat.z());

                switch (geom.type) {
                case Animation::CollisionGeom::Type::Sphere:
                    meshData =
                        Scene::Prim::createSphereData(geom.size[0], 12, 8);
                    break;
                case Animation::CollisionGeom::Type::Capsule:
                    meshData = Scene::Prim::createCapsuleData(
                        geom.size[0], geom.size[1] * 2.f, UpAxis::Z, 12);
                    break;
                case Animation::CollisionGeom::Type::Cylinder:
                    meshData = Scene::Prim::createCylinderData(
                        geom.size[0], geom.size[1] * 2.f, UpAxis::Z, 12);
                    break;
                case Animation::CollisionGeom::Type::Box:
                    meshData = Scene::Prim::createRectangleData(
                        geom.size[0] * 2.f, geom.size[1] * 2.f,
                        geom.size[2] * 2.f);
                    break;
                }
            }

            prim->setMeshData(
                std::make_shared<Scene::MeshData>(std::move(meshData)));
            prim->setDisplayColorAlpha(glm::vec4(1.f, 0.5f, 0.f, 0.8f));
            prim->addTranslateOp(localPos);
            prim->addRotateQuaternionOp(localQuat);
            prim->setVisible(visibleByDefault);

            _colVisuals.push_back({lnk, prim, localPos, localQuat});
            result.push_back(prim);
        }
    }
    return result;
}

void PhysicsBridge::setCollisionVisible(bool visible) {
    for (auto& cv : _colVisuals)
        cv.prim->setVisible(visible);
}

} // namespace Bridge
} // namespace KE
