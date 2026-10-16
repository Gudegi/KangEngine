#ifndef _ARTICULATION_DESC_HPP_
#define _ARTICULATION_DESC_HPP_

// Shared character description types — used by all format parsers (MJCF, URDF,
// etc.) so that downstream code (physics bridge, skeleton FK) is
// format-agnostic.

#include "animation/skeleton_tree.hpp"
#include "asset/geometry_desc.hpp"

#include <cfloat>
#include <Eigen/Geometry>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace KE {
namespace Asset {

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------


struct JointDesc {
    enum class Type { Revolute, Prismatic, Spherical, Fixed };
    Type type = Type::Revolute;
    std::string name;
    Eigen::Vector3f axis = Eigen::Vector3f::UnitZ();
    // Joint origin in child-link-local coordinates.
    // E.g. a centered 2 m Z-axis bar uses (0, 0, -1) for a hinge at its lower
    // end.
    Eigen::Vector3f jointOffset = Eigen::Vector3f::Zero();
    float loLimit = -3.14159f;
    float hiLimit = 3.14159f;
    float kp = 0.f;
    float kd = 0.f;
    float armature = 0.f;
    float effortLimit = FLT_MAX;
    float velocityLimit = FLT_MAX;
};

// Fixed relative to its owning body, not to the world. On imported assets,
// bodyIndex is a source skeleton node; templates resolve it to a physical body.
struct FixedFrameDesc {
    std::string name;
    int bodyIndex = -1;
    Eigen::Vector3f pos = Eigen::Vector3f::Zero();
    Eigen::Quaternionf quat = Eigen::Quaternionf::Identity();
};

// MJCF site display metadata shares the same authoritative attachment pose.
struct SiteDesc : FixedFrameDesc {
    enum class Type { Sphere, Capsule, Box };
    Type type = Type::Sphere;
    Eigen::Vector3f size = Eigen::Vector3f::Zero();
    Eigen::Vector4f rgba = Eigen::Vector4f(0.15f, 0.15f, 0.15f, 1.0f);
    // True when the site orientation was derived from MJCF zaxis.
    bool hasZAxis = false;
    Eigen::Vector3f zaxis = Eigen::Vector3f::UnitZ();
};


// ---------------------------------------------------------------------------
// Map aliases  (body index -> data)
// ---------------------------------------------------------------------------

using JointDescMap = std::unordered_map<int, std::vector<JointDesc>>;
using SiteDescMap = std::unordered_map<std::string, SiteDesc>;
using InertialDescMap = std::unordered_map<int, InertialDesc>;
using CollisionGeomDescMap =
    std::unordered_map<int, std::vector<CollisionGeomDesc>>;

// ---------------------------------------------------------------------------
// Aggregate output
// ---------------------------------------------------------------------------

struct ArticulationDesc {
    std::shared_ptr<const Animation::SkeletonTree> skeletonTree;
    // Body/joint traversal used to construct skeletonTree. MJCF currently
    // supports only "BFS" and "DFS".
    std::string traversalOrder;
    std::vector<VisualGeomDesc> visualGeoms;
    std::string assetDir;
    JointDescMap joints;
    SiteDescMap sites;
    std::vector<FixedFrameDesc> fixedFrames;
    CollisionGeomDescMap collisionGeoms;
    InertialDescMap inertials;
};

} // namespace Asset
} // namespace KE

#endif
