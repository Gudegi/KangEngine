#ifndef _ARTICULATION_DESC_HPP_
#define _ARTICULATION_DESC_HPP_

// Shared character description types — used by all format parsers (MJCF, URDF,
// etc.) so that downstream code (physics bridge, skeleton FK) is
// format-agnostic.

#include "animation/skeleton_tree.hpp"
#include "engine/scene/scene_backend.hpp"
#include "physics/physics_material.hpp"

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

struct VisualGeomDesc {
    std::string bodyName;
    std::string meshFile;
    int bodyIndex;
    Eigen::Vector3f pos = Eigen::Vector3f::Zero();
    Eigen::Quaternionf quat = Eigen::Quaternionf::Identity();
    Eigen::Vector4f rgba = Eigen::Vector4f(0.15f, 0.15f, 0.15f, 1.0f);
};

struct JointDesc {
    enum class Type { Revolute, Spherical, Fixed };
    Type type = Type::Revolute;
    std::string name;
    Eigen::Vector3f axis = Eigen::Vector3f::UnitZ();
    float loLimit = -3.14159f;
    float hiLimit = 3.14159f;
    float kp = 0.f;
    float kd = 0.f;
    float armature = 0.f;
    float effortLimit = FLT_MAX;
};

// Named body-local reference frame parsed from MJCF <site> elements.
struct SiteDesc {
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

struct InertialDesc {
    float mass = 1.f;
    Eigen::Vector3f com = Eigen::Vector3f::Zero();
    Eigen::Quaternionf quat = Eigen::Quaternionf::Identity();
    Eigen::Vector3f diagInertia = Eigen::Vector3f(1e-4f, 1e-4f, 1e-4f);
};

struct CollisionGeomDesc {
    enum class Type { Capsule, Cylinder, Sphere, Box, ConvexMesh };
    Type type = Type::Sphere;
    std::string name;

    Eigen::Vector3f pos = Eigen::Vector3f::Zero();
    Eigen::Quaternionf quat = Eigen::Quaternionf::Identity();

    // size interpretation:
    //   Sphere          : size[0] = radius
    //   Capsule/Cylinder: size[0] = radius, size[1] = half-length
    //   Box             : size[0..2] = half-extents
    float size[3] = {};

    // ConvexMesh stores the imported source mesh. Physics backends cook their
    // native convex resource from this payload and may cache it across shapes.
    std::string meshFile;
    std::shared_ptr<const Scene::MeshData> meshData;

    // When true, from/to define axis endpoints in body frame
    // (overrides pos/quat for capsule and cylinder)
    bool hasFromTo = false;
    Eigen::Vector3f from = Eigen::Vector3f::Zero();
    Eigen::Vector3f to = Eigen::Vector3f::Zero();

    // MuJoCo friction[0] is sliding friction. PhysX uses separate static and
    // dynamic friction coefficients; KangEngine maps sliding friction to both.
    float friction = 1.f; // legacy, deprecated.
    Physics::PhysicsMaterialDesc physicsMaterial;

    // MuJoCo contact dimensionality. KangEngine parses it for diagnostics and
    // future matching, but does not alter PhysX material behavior by default.
    int condim = -1;

    // MuJoCo geom margin controls the distance at which contacts become active.
    // When present, KangEngine maps it to PhysX shape contactOffset.
    float margin = -1.f;

    // True when KangEngine synthesized this descriptor from ArticulationConfig
    // because an authored collision mesh could not be loaded.
    // These descriptors make debug collision visuals match the actual PhysX
    // shapes without pretending they came from MJCF.
    bool isFallback = false;
};

// ---------------------------------------------------------------------------
// Map aliases  (body index -> data)
// ---------------------------------------------------------------------------

using JointDescMap = std::unordered_map<int, std::vector<JointDesc>>;
using SiteDescMap = std::unordered_map<std::string, SiteDesc>;
using InertialDescMap = std::unordered_map<int, InertialDesc>;
using CollisionGeomDescMap = std::unordered_map<int, std::vector<CollisionGeomDesc>>;

// ---------------------------------------------------------------------------
// Aggregate output
// ---------------------------------------------------------------------------

struct ArticulationDesc {
    std::shared_ptr<const Animation::SkeletonTree> skeletonTree;
    std::vector<VisualGeomDesc> visualGeoms;
    std::string assetDir;
    JointDescMap joints;
    SiteDescMap sites;
    CollisionGeomDescMap collisionGeoms;
    InertialDescMap inertials;
};

} // namespace Asset
} // namespace KE

#endif
