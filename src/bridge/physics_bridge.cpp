#include "physics_bridge.hpp"
#include "engine/scene/native/prim.hpp"
#include "engine/scene/scene_backend.hpp"
#include "physics/articulation.hpp"
#include "physics/physics.hpp"
#include "skeleton_bridge.hpp"

#include <Eigen/Geometry>
#include <fmt/core.h>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

namespace KE {
namespace Bridge {

void PhysicsBridge::syncAllVisuals() {
    PxTransform pose;
    for (auto& v : _rigidVisuals) {
        if (v.link && !v.rigid) {
            pose = v.link->getGlobalPose();
        } else if (v.rigid && !v.link) {
            pose = v.rigid->getGlobalPose();
        } else {
            fmt::print(
                "Failed to parse rigid visual, use only 'link' or 'rigid'");
        }
        if (v.prim) {
            v.prim->setAttribute("xformOp:translate", pxToGlm(pose.p));
            v.prim->setAttribute("xformOp:rotateQuaternion", pxToGlm(pose.q));
        } else {
            fmt::print("Failed to parse rigid visual, check 'prim'");
        }
    }
}

std::vector<Scene::Prim*>
PhysicsBridge::buildCollisionVisuals(const Articulation& artic,
                                     Scene::SceneBackend* scene,
                                     const std::string& basePath) {
    std::vector<Scene::Prim*> result;
    _colVisuals.clear();

    const auto& links = artic.links();
    const auto& colGeoms = artic.colGeoms();

    for (auto& [bodyIdx, geoms] : colGeoms) {
        if (bodyIdx >= static_cast<int>(links.size()))
            continue;
        PxArticulationLink* lnk = links[bodyIdx];

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
                if (geom.type == Animation::MJCFCollisionGeom::Type::Capsule)
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
                case Animation::MJCFCollisionGeom::Type::Sphere:
                    meshData =
                        Scene::Prim::createSphereData(geom.size[0], 12, 8);
                    break;
                case Animation::MJCFCollisionGeom::Type::Capsule:
                    meshData = Scene::Prim::createCapsuleData(
                        geom.size[0], geom.size[1] * 2.f, UpAxis::Z, 12);
                    break;
                case Animation::MJCFCollisionGeom::Type::Cylinder:
                    meshData = Scene::Prim::createCylinderData(
                        geom.size[0], geom.size[1] * 2.f, UpAxis::Z, 12);
                    break;
                case Animation::MJCFCollisionGeom::Type::Box:
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
            prim->setVisible(false);

            _colVisuals.push_back({lnk, prim, localPos, localQuat});
            result.push_back(prim);
        }
    }
    return result;
}

void PhysicsBridge::syncCollisionVisuals() {
    for (auto& cv : _colVisuals) {
        PxTransform pose = cv.link->getGlobalPose();
        glm::vec3 linkPos = pxToGlm(pose.p);
        glm::quat linkRot = pxToGlm(pose.q);
        cv.prim->setAttribute("xformOp:translate",
                              linkPos + linkRot * cv.localPos);
        cv.prim->setAttribute("xformOp:rotateQuaternion",
                              linkRot * cv.localQuat);
    }
}

void PhysicsBridge::setCollisionVisible(bool visible) {
    for (auto& cv : _colVisuals)
        cv.prim->setVisible(visible);
}

void PhysicsBridge::registerArticulationVisuals(
    const Articulation& artic, const SkeletonBridge& skelBridge) {
    int n = artic.numLinks();
    for (int i = 0; i < n; i++) {
        RigidVisual rv;
        rv.link = artic.link(i);
        rv.prim = skelBridge.bodyPrim(i);
        _rigidVisuals.push_back(rv);
    }
}

} // namespace Bridge
} // namespace KE
