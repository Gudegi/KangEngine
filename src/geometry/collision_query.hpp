#pragma once

// Pairwise overlap and penetration queries for two shapes at supplied poses.
// Uses PhysX geometry routines without a world, actors, or simulation state.
// Returns separation or overlap depth/direction, with explicit unavailable and
// unsupported statuses when those quantities cannot be determined.
// Limits: triangle mesh/heightfield penetration depths are approximate;
// mesh surface tests can miss shapes fully enclosed without surface overlap.
// Depth and separation direction do not provide contact points or forces.

#include <geometry/PxGeometry.h>
#include <geometry/PxGeometryQuery.h>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>

namespace KE::Geometry {

struct CollisionResult {
    std::string status;
    // NaN means unavailable, never a zero-depth assertion of safety.
    float depth = std::numeric_limits<float>::quiet_NaN();
    physx::PxVec3 direction{std::numeric_limits<float>::quiet_NaN()};
};

inline bool collisionGeometrySupported(physx::PxGeometryType::Enum type) {
    using physx::PxGeometryType;
    return type == PxGeometryType::eSPHERE || type == PxGeometryType::eCAPSULE ||
           type == PxGeometryType::eBOX || type == PxGeometryType::eCONVEXMESH ||
           type == PxGeometryType::ePLANE || type == PxGeometryType::eTRIANGLEMESH ||
           type == PxGeometryType::eHEIGHTFIELD;
}

inline bool collisionPairSupported(physx::PxGeometryType::Enum a, physx::PxGeometryType::Enum b) {
    using physx::PxGeometryType;
    auto surface = [](PxGeometryType::Enum type) {
        return type == PxGeometryType::ePLANE || type == PxGeometryType::eTRIANGLEMESH ||
               type == PxGeometryType::eHEIGHTFIELD;
    };
    // Restrict to the common overlap/MTD support set across supported PhysX versions.
    return collisionGeometrySupported(a) && collisionGeometrySupported(b) && !(surface(a) && surface(b));
}

// PhysX-backed geometry query: no world, actor, articulation, or simulation state.
// Mesh payloads are borrowed and must remain alive for this synchronous call.
// nullopt means separated; unsupported pairs return a result with unknown depth.
inline std::optional<CollisionResult> queryCollision(
    const physx::PxGeometry& a, const physx::PxTransform& poseA,
    const physx::PxGeometry& b, const physx::PxTransform& poseB) {
    using namespace physx;
    if (!poseA.isValid() || !poseB.isValid())
        throw std::invalid_argument("Collision poses require finite xyz and unit xyzw quaternions");
    CollisionResult result;
    const auto typeA = a.getType();
    const auto typeB = b.getType();
    if (!collisionPairSupported(typeA, typeB)) {
        result.status = "unsupported";
        return result;
    }
    if (!PxGeometryQuery::overlap(a, poseA, b, poseB))
        return std::nullopt;

    if (PxGeometryQuery::computePenetration(result.direction, result.depth, a, poseA, b, poseB)) {
        const bool approximate = typeA == PxGeometryType::eTRIANGLEMESH || typeB == PxGeometryType::eTRIANGLEMESH ||
                                 typeA == PxGeometryType::eHEIGHTFIELD || typeB == PxGeometryType::eHEIGHTFIELD;
        result.status = approximate ? "overlap_mtd_approximate" : "overlap_mtd";
    } else {
        result.depth = std::numeric_limits<float>::quiet_NaN();
        result.direction = PxVec3(result.depth);
        result.status = "overlap_depth_unavailable";
    }
    return result;
}

} // namespace KE::Geometry
