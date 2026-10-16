#ifndef _GEOMETRY_DESC_HPP_
#define _GEOMETRY_DESC_HPP_

// Context-free geometry, appearance, and inertia shared by model descriptions.
#include "engine/scene/scene_backend.hpp"
#include "physics/physics_material.hpp"

#include <Eigen/Geometry>
#include <memory>
#include <string>

namespace KE {
class Material;

namespace Asset {

struct VisualGeomDesc {
    std::string bodyName;
    std::string meshFile;
    int bodyIndex = 0;
    Eigen::Vector3f pos = Eigen::Vector3f::Zero();
    Eigen::Quaternionf quat = Eigen::Quaternionf::Identity();
    Eigen::Vector4f rgba = Eigen::Vector4f(0.15f, 0.15f, 0.15f, 1.0f);
    Eigen::Vector3f scale = Eigen::Vector3f::Ones();
    std::shared_ptr<const Scene::MeshData> meshData;
    std::shared_ptr<Material> material;
};

struct InertialDesc {
    float mass = 1.f;
    Eigen::Vector3f com = Eigen::Vector3f::Zero();
    Eigen::Quaternionf quat = Eigen::Quaternionf::Identity();
    Eigen::Vector3f diagInertia = Eigen::Vector3f(1e-4f, 1e-4f, 1e-4f);
};

struct CollisionGeomDesc {
    enum class Type { Capsule, Cylinder, Sphere, Box, ConvexMesh, TriangleMesh };
    Type type = Type::Sphere;
    std::string name;

    Eigen::Vector3f pos = Eigen::Vector3f::Zero();
    Eigen::Quaternionf quat = Eigen::Quaternionf::Identity();

    // size interpretation:
    //   Sphere          : size[0] = radius
    //   Capsule/Cylinder: size[0] = radius, size[1] = half-length
    //   Box             : size[0..2] = half-extents
    float size[3] = {};

    // Mesh types keep immutable source geometry shared across cooked shapes.
    // TriangleMesh is supported by static and kinematic rigid bodies only.
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

} // namespace Asset
} // namespace KE

#endif
