#ifndef _RIGID_DESC_HPP_
#define _RIGID_DESC_HPP_

#include "asset/geometry_desc.hpp"

#include <optional>
#include <vector>

namespace KE {
namespace Asset {

// One rigid body with independent appearance and compound collision shapes.
struct RigidDesc {
    std::string name = "rigid";
    std::string assetDir;
    std::vector<VisualGeomDesc> visualGeoms;
    std::vector<CollisionGeomDesc> collisionGeoms;
    std::optional<InertialDesc> inertial;
};

} // namespace Asset
} // namespace KE

#endif
