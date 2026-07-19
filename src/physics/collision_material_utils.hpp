#ifndef _PHYSICS_COLLISION_MATERIAL_UTILS_HPP_
#define _PHYSICS_COLLISION_MATERIAL_UTILS_HPP_

#include "character/character_description.hpp"

#include <memory>
#include <vector>

namespace KE {

inline bool collisionOverrideBodyMatches(
    const Physics::CollisionMaterialOverride& entry,
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
collisionOverrideBodyMatches(const Physics::CollisionMaterialOverride& entry,
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
collisionOverrideGeomMatches(const Physics::CollisionMaterialOverride& entry,
                             int geomIndex,
                             const Character::CollisionGeomDesc& geom) {
    if (entry.geomIndex >= 0)
        return entry.geomIndex == geomIndex;
    if (!entry.geomName.empty())
        return entry.geomName == geom.name;
    return true;
}

inline Physics::PhysicsMaterialDesc resolveCollisionMaterial(
    const Character::CollisionGeomDesc& geom,
    const std::vector<Physics::CollisionMaterialOverride>& overrides,
    std::shared_ptr<const Animation::SkeletonTree> tree, int bodyIndex,
    int geomIndex) {
    Physics::PhysicsMaterialDesc material = geom.physicsMaterial;
    for (const auto& entry : overrides) {
        if (!collisionOverrideBodyMatches(entry, tree, bodyIndex))
            continue;
        if (!collisionOverrideGeomMatches(entry, geomIndex, geom))
            continue;
        material = entry.material;
    }
    return material;
}

inline Physics::PhysicsMaterialDesc resolveCollisionMaterial(
    const Character::CollisionGeomDesc& geom,
    const std::vector<Physics::CollisionMaterialOverride>& overrides,
    const std::vector<std::string>& bodyNames, int bodyIndex, int geomIndex) {
    Physics::PhysicsMaterialDesc material = geom.physicsMaterial;
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
