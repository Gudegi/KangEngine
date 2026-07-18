#ifndef _CHARACTER_DESCRIPTION_HPP_
#define _CHARACTER_DESCRIPTION_HPP_

// Shared character description types — used by all format parsers (MJCF, URDF,
// etc.) so that downstream code (physics bridge, skeleton FK) is
// format-agnostic.

#include "skeleton_tree.hpp"

#include <cfloat>
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
    Eigen::Vector3f pos = Eigen::Vector3f::Zero();
    Eigen::Quaternionf quat = Eigen::Quaternionf::Identity();
    Eigen::Vector4f rgba = Eigen::Vector4f(0.15f, 0.15f, 0.15f, 1.0f);
};

struct Joint {
    enum class Type { Revolute, Spherical, Fixed };
    Type type = Type::Revolute;
    std::string name;
    Eigen::Vector3f axis = Eigen::Vector3f::UnitZ();
    float loLimit = -3.14159f;
    float hiLimit = 3.14159f;
    float kp = 0.f;
    float kd = 0.f;
    float effortLimit = FLT_MAX;
};

// Named body-local reference frame parsed from MJCF <site> elements.
struct Site {
    enum class Type { Sphere, Capsule, Box };
    Type type = Type::Sphere;
    std::string name;
    int bodyIndex = -1;
    Eigen::Vector3f pos = Eigen::Vector3f::Zero();
    Eigen::Quaternionf quat = Eigen::Quaternionf::Identity();
    Eigen::Vector3f size = Eigen::Vector3f::Zero();
    Eigen::Vector4f rgba = Eigen::Vector4f(0.15f, 0.15f, 0.15f, 1.0f);
    // True when the site orientation was derived from MJCF zaxis.
    bool hasZAxis = false;
    Eigen::Vector3f zaxis = Eigen::Vector3f::UnitZ();
};

struct Inertial {
    float mass = 1.f;
    Eigen::Vector3f com = Eigen::Vector3f::Zero();
    Eigen::Quaternionf quat = Eigen::Quaternionf::Identity();
    Eigen::Vector3f diagInertia = Eigen::Vector3f(1e-4f, 1e-4f, 1e-4f);
};

struct PhysicsMaterialDesc {
    float staticFriction = 1.f;
    float dynamicFriction = 1.f;
    float restitution = 0.f;
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

struct CollisionGeom {
    // Supported collision payloads are primitive-only for now. MJCF
    // type="mesh" collision geoms are intentionally not represented here yet;
    // dynamic/articulation mesh collision needs a separate convex-cooking path
    // rather than reusing visual MeshData directly.
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

    // MuJoCo friction[0] is sliding friction. PhysX uses separate static and
    // dynamic friction coefficients; KangEngine maps sliding friction to both.
    float friction = 1.f; // legacy, deprecated.
    PhysicsMaterialDesc physicsMaterial;

    // MuJoCo contact dimensionality. KangEngine parses it for diagnostics and
    // future matching, but does not alter PhysX material behavior by default.
    int condim = -1;

    // MuJoCo geom margin controls the distance at which contacts become active.
    // When present, KangEngine maps it to PhysX shape contactOffset.
    float margin = -1.f;

    // True when KangEngine synthesized this descriptor from ArticulationConfig
    // fallback boxes because the source body had no supported collision geom.
    // These descriptors make debug collision visuals match the actual PhysX
    // shapes without pretending they came from MJCF.
    bool isFallback = false;
};

// ---------------------------------------------------------------------------
// Map aliases  (body index -> data)
// ---------------------------------------------------------------------------

using JointMap = std::unordered_map<int, std::vector<Joint>>;
using SiteMap = std::unordered_map<std::string, Site>;
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
    SiteMap sites;
    CollisionGeomMap collisionGeoms;
    InertialMap inertials;
};

} // namespace Animation
} // namespace KE

#endif
