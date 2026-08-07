#ifndef _PHYSICS_MATERIAL_HPP_
#define _PHYSICS_MATERIAL_HPP_

#include <string>
#include <vector>

namespace KE {
namespace Physics {

struct PhysicsMaterialDesc {
    float staticFriction = 1.f;
    float dynamicFriction = 1.f;
    float restitution = 0.f;
};

struct CollisionMaterialOverride {
    // Match by index when non-negative. Names are intended for Python/user
    // facing APIs and are resolved against imported ArticulationDesc/SkeletonTree
    // at build time. Empty body name and bodyIndex < 0 means "all bodies".
    int bodyIndex = -1;
    std::string bodyName;

    // Match one geom by index/name, or every collision geom on the matched body
    // when both are unspecified.
    int geomIndex = -1;
    std::string geomName;

    PhysicsMaterialDesc material;
};

inline PhysicsMaterialDesc mjcfFrictionToPhysX(
    const std::vector<float>& friction,
    const PhysicsMaterialDesc& fallback = PhysicsMaterialDesc()) {
    PhysicsMaterialDesc material = fallback;
    if (friction.empty())
        return material;

    // MuJoCo friction[0] is sliding friction. PhysX exposes separate static
    // and dynamic friction coefficients, so KangEngine maps sliding friction
    // to both for now. MuJoCo torsional/rolling friction are intentionally not
    // mapped yet because PhysX material properties do not have direct scalar
    // equivalents.
    material.staticFriction = friction[0];
    material.dynamicFriction = friction[0];
    material.restitution = 0.f;
    return material;
}

} // namespace Physics
} // namespace KE

#endif
