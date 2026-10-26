#pragma once

#include "asset/geometry_desc.hpp"
#include <PxPhysicsAPI.h>
#include <stdexcept>
#include <vector>

namespace KE::Physics {

struct CollisionShapeInfo {
    int bodyIdx = 0;
    int geomIdx = -1; // -1 denotes a generated collider without an authored geom.
    std::string name;
    physx::PxTransform localPose{physx::PxIdentity};
};

inline std::vector<CollisionShapeInfo> collisionShapes(
    const physx::PxRigidActor& actor, int bodyIdx, const std::string& bodyName,
    const std::vector<Asset::CollisionGeomDesc>& geoms) {
    const auto count = actor.getNbShapes();
    // Current builders create one shape per geom, or one rigid fallback sphere.
    if (count != geoms.size() && !(geoms.empty() && count == 1))
        throw std::runtime_error("collision shape count differs from its source geometry");

    std::vector<physx::PxShape*> shapes(count);
    actor.getShapes(shapes.data(), count);
    std::vector<CollisionShapeInfo> result;
    result.reserve(count);
    for (physx::PxU32 shapeIdx = 0; shapeIdx < count; ++shapeIdx) {
        const auto* geom = geoms.empty() ? nullptr : &geoms[shapeIdx];
        const int geomIdx = geom && !geom->isFallback ? static_cast<int>(shapeIdx) : -1;
        const std::string name = geom && !geom->name.empty()
                                     ? geom->name
                                     : (geomIdx < 0 ? "__fallback_sphere"
                                                    : "geom_" + std::to_string(geomIdx));
        result.push_back({bodyIdx, geomIdx, bodyName + ":" + name,
                          shapes[shapeIdx]->getLocalPose()});
    }
    return result;
}

} // namespace KE::Physics
