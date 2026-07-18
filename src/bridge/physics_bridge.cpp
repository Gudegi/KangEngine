#include "physics_bridge.hpp"
#include "engine/scene/component/articulation_binding_component.hpp"
#include "engine/scene/component/collision_shape_component.hpp"
#include "engine/scene/native/prim.hpp"
#include "engine/scene/scene_backend.hpp"
#include "physics/articulation.hpp"
#include "physics/physics.hpp"
#include "skeleton_bridge.hpp"

#include <Eigen/Geometry>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

namespace KE {
namespace Bridge {

namespace {

std::string fallbackBodyName(int bodyIdx) {
    return "body_" + std::to_string(bodyIdx);
}

Scene::CollisionShapeType
toCollisionShapeType(Animation::CollisionGeom::Type type) {
    switch (type) {
    case Animation::CollisionGeom::Type::Sphere:
        return Scene::CollisionShapeType::Sphere;
    case Animation::CollisionGeom::Type::Capsule:
        return Scene::CollisionShapeType::Capsule;
    case Animation::CollisionGeom::Type::Cylinder:
        return Scene::CollisionShapeType::Cylinder;
    case Animation::CollisionGeom::Type::Box:
        return Scene::CollisionShapeType::Box;
    }
    return Scene::CollisionShapeType::Sphere;
}

} // namespace

void PhysicsBridge::add(const Articulation& artic,
                        const SkeletonBridge& skelBridge) {
    int n = artic.numLinks();
    for (int i = 0; i < n; i++)
        _primVisuals.push_back({artic.link(i), skelBridge.bodyPrim(i)});
}

void PhysicsBridge::sync() {
    // Prim-based: PhysX pose -> Prim xform attributes
    for (auto& v : _primVisuals) {
        physx::PxTransform pose = v.link->getGlobalPose();
        v.prim->setWorldMatrix(pxToMat4(pose));
    }

    // Collision visuals: link pose * local offset
    for (auto& cv : _colVisuals) {
        physx::PxTransform pose = cv.link->getGlobalPose();
        glm::vec3 linkPos = pxToGlm(pose.p);
        glm::quat linkRot = pxToGlm(pose.q);
        const glm::vec3 worldPos = linkPos + linkRot * cv.localPos;
        const glm::quat worldRot = linkRot * cv.localQuat;
        cv.prim->setWorldMatrix(glm::translate(glm::mat4(1.0f), worldPos) *
                                glm::mat4_cast(worldRot));
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
        std::string bodyName = fallbackBodyName(bodyIdx);
        if (bodyIdx < static_cast<int>(artic.bodyNames().size()))
            bodyName = artic.bodyName(bodyIdx);
        std::string rootPath = basePath;
        for (const auto& visual : _primVisuals) {
            if (visual.link != lnk || !visual.prim)
                continue;
            if (auto binding = visual.prim->getArticulationBindingComponent()) {
                if (!binding->bodyName().empty())
                    bodyName = binding->bodyName();
                if (!binding->articulationRootPath().empty())
                    rootPath = binding->articulationRootPath();
            }
            break;
        }

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
            prim->addArticulationBindingComponent()->setBinding(
                Scene::ArticulationPrimRole::CollisionGeom, bodyIdx, bodyName,
                rootPath);
            auto collisionShape = prim->addCollisionShapeComponent();
            collisionShape->setShapeMetadata(
                toCollisionShapeType(geom.type),
                glm::vec3(geom.size[0], geom.size[1], geom.size[2]), localPos,
                localQuat, geom.physicsMaterial.staticFriction,
                geom.physicsMaterial.dynamicFriction,
                geom.physicsMaterial.restitution, geom.condim, geom.margin,
                geom.isFallback ? -1 : gi);
            if (geom.hasFromTo) {
                collisionShape->setFromTo(
                    glm::vec3(geom.from.x(), geom.from.y(), geom.from.z()),
                    glm::vec3(geom.to.x(), geom.to.y(), geom.to.z()));
            }

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
