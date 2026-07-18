#ifndef _PHYSICS_COLLISION_MATERIAL_UTILS_HPP_
#define _PHYSICS_COLLISION_MATERIAL_UTILS_HPP_

#include "animation/character_description.hpp"

#include <memory>
#include <vector>

namespace KE {

inline bool collisionOverrideBodyMatches(
    const Animation::CollisionMaterialOverride& entry,
    std::shared_ptr<const Animation::SkeletonTree> tree, int bodyIndex) {
    if (entry.bodyIndex >= 0)
        return entry.bodyIndex == bodyIndex;
    if (!entry.bodyName.empty()) {
        if (!tree)
            return false;
        try {
            return tree->index(entry.bodyName) == bodyIndex;
        } catch (...) {
            return false;
        }
    }
    return true;
}

inline bool
collisionOverrideBodyMatches(const Animation::CollisionMaterialOverride& entry,
                             const std::vector<std::string>& bodyNames,
                             int bodyIndex) {
    if (entry.bodyIndex >= 0)
        return entry.bodyIndex == bodyIndex;
    if (!entry.bodyName.empty()) {
        if (bodyIndex < 0 || bodyIndex >= static_cast<int>(bodyNames.size()))
            return false;
        return bodyNames[static_cast<size_t>(bodyIndex)] == entry.bodyName;
    }
    return true;
}

inline bool
collisionOverrideGeomMatches(const Animation::CollisionMaterialOverride& entry,
                             int geomIndex,
                             const Animation::CollisionGeom& geom) {
    if (entry.geomIndex >= 0)
        return entry.geomIndex == geomIndex;
    if (!entry.geomName.empty())
        return entry.geomName == geom.name;
    return true;
}

inline Animation::PhysicsMaterialDesc resolveCollisionMaterial(
    const Animation::CollisionGeom& geom,
    const std::vector<Animation::CollisionMaterialOverride>& overrides,
    std::shared_ptr<const Animation::SkeletonTree> tree, int bodyIndex,
    int geomIndex) {
    Animation::PhysicsMaterialDesc material = geom.physicsMaterial;
    for (const auto& entry : overrides) {
        if (!collisionOverrideBodyMatches(entry, tree, bodyIndex))
            continue;
        if (!collisionOverrideGeomMatches(entry, geomIndex, geom))
            continue;
        material = entry.material;
    }
    return material;
}

inline Animation::PhysicsMaterialDesc resolveCollisionMaterial(
    const Animation::CollisionGeom& geom,
    const std::vector<Animation::CollisionMaterialOverride>& overrides,
    const std::vector<std::string>& bodyNames, int bodyIndex, int geomIndex) {
    Animation::PhysicsMaterialDesc material = geom.physicsMaterial;
    for (const auto& entry : overrides) {
        if (!collisionOverrideBodyMatches(entry, bodyNames, bodyIndex))
            continue;
        if (!collisionOverrideGeomMatches(entry, geomIndex, geom))
            continue;
        material = entry.material;
    }
    return material;
}

} // namespace KE

#endif
