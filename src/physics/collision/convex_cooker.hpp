#pragma once

#include "physics/collision/convex_collision.hpp"

#include <vector>

namespace KE {
namespace Physics {

class ConvexCollisionCooker {
  public:
    static std::vector<physx::PxConvexMesh*>
    cook(physx::PxPhysics& physics, const std::vector<ConvexMeshPart>& parts,
         const ConvexCookingOptions& options = {});
};

} // namespace Physics
} // namespace KE
