#ifndef _CHARACTER_DESCRIPTION_HPP_
#define _CHARACTER_DESCRIPTION_HPP_

// Shared character description types — used by all format parsers (MJCF, URDF,
// etc.) so that downstream code (physics bridge, skeleton FK) is
// format-agnostic.

#include "skeleton_tree.hpp"

#include <Eigen/Geometry>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace KE {
namespace Animation {

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

struct MeshInfo {
    std::string bodyName;
    std::string meshFile;
    int bodyIndex;
};

struct Joint {
    enum class Type { Revolute, Spherical, Fixed };
    Type type = Type::Revolute;
    std::string name;
    Eigen::Vector3f axis = Eigen::Vector3f::UnitZ();
    float loLimit = -3.14159f;
    float hiLimit = 3.14159f;
};

struct Inertial {
    float mass = 1.f;
    Eigen::Vector3f com = Eigen::Vector3f::Zero();
    Eigen::Quaternionf quat = Eigen::Quaternionf::Identity();
    Eigen::Vector3f diagInertia = Eigen::Vector3f(1e-4f, 1e-4f, 1e-4f);
};

struct CollisionGeom {
    enum class Type { Capsule, Cylinder, Sphere, Box };
    Type type = Type::Sphere;

    Eigen::Vector3f pos = Eigen::Vector3f::Zero();
    Eigen::Quaternionf quat = Eigen::Quaternionf::Identity();

    // size interpretation:
    //   Sphere          : size[0] = radius
    //   Capsule/Cylinder: size[0] = radius, size[1] = half-length
    //   Box             : size[0..2] = half-extents
    float size[3] = {};

    // When true, from/to define axis endpoints in body frame
    // (overrides pos/quat for capsule and cylinder)
    bool hasFromTo = false;
    Eigen::Vector3f from = Eigen::Vector3f::Zero();
    Eigen::Vector3f to = Eigen::Vector3f::Zero();
};

// ---------------------------------------------------------------------------
// Map aliases  (body index -> data)
// ---------------------------------------------------------------------------

using JointMap = std::unordered_map<int, std::vector<Joint>>;
using InertialMap = std::unordered_map<int, Inertial>;
using CollisionGeomMap = std::unordered_map<int, std::vector<CollisionGeom>>;

// ---------------------------------------------------------------------------
// Aggregate output
// ---------------------------------------------------------------------------

struct CharacterData {
    std::shared_ptr<const SkeletonTree> skeletonTree;
    std::vector<MeshInfo> meshInfos;
    std::string meshDir;
    JointMap joints;
    CollisionGeomMap collisionGeoms;
    InertialMap inertials;
};

} // namespace Animation
} // namespace KE

#endif
