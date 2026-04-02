#ifndef _MJCF_LOADER_HPP_
#define _MJCF_LOADER_HPP_

#include "skeleton_tree.hpp"

#include <Eigen/Geometry>
#include <string>
#include <unordered_map>
#include <vector>

namespace KE {
namespace Animation {

struct MJCFMeshInfo {
    std::string bodyName; // Body this mesh belongs to
    std::string meshFile; // filename (e.g., "pelvis.stl")
    int bodyIndex;        // Index into SkeletonTree
};

struct MJCFLoadResult {
    SkeletonTree skeleton;
    std::vector<MJCFMeshInfo> meshInfos; // Visual mesh per body
    std::string meshDir;                 // Resolved mesh directory path
};

class MJCFLoader {
  public:
    // Load MJCF file and extract skeleton + mesh info
    static MJCFLoadResult load(const std::string& mjcfPath);
};

// ── Collision geometry
// ────────────────────────────────────────────────────────

struct MJCFCollisionGeom {
    enum class Type { Capsule, Cylinder, Sphere, Box };
    Type type = Type::Sphere;

    Eigen::Vector3f pos = Eigen::Vector3f::Zero();
    // Orientation of geom frame in body frame (MJCF: w x y z)
    Eigen::Quaternionf quat = Eigen::Quaternionf::Identity();

    // size interpretation:
    //   Capsule/Sphere : size[0] = radius
    //   Cylinder       : size[0] = radius, size[1] = half-length
    //   Box            : size[0..2] = half-extents
    float size[3] = {};

    // When hasFromTo is true, from/to define the axis endpoints in body frame
    // (overrides pos/quat for capsule and cylinder)
    bool hasFromTo = false;
    Eigen::Vector3f from = Eigen::Vector3f::Zero();
    Eigen::Vector3f to = Eigen::Vector3f::Zero();
};

// body index (SkeletonTree) -> collision geoms for that body
using CollisionGeomMap =
    std::unordered_map<int, std::vector<MJCFCollisionGeom>>;

// Parse all collision geoms from MJCF, respecting <default> class inheritance.
// Bodies with no collision geoms will not have an entry in the map.
CollisionGeomMap parseMJCFCollision(const std::string& mjcfPath,
                                    const SkeletonTree& tree);

} // namespace Animation
} // namespace KE

#endif
