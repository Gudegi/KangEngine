#ifndef _MJCF_LOADER_HPP_
#define _MJCF_LOADER_HPP_

#include "skeleton_tree.hpp"

#include <Eigen/Geometry>
#include <memory>
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

//  Joint definitions
struct Joint {
    enum class Type { Revolute }; // TODO: support more types
    Type type = Type::Revolute;
    Eigen::Vector3f axis = Eigen::Vector3f::UnitZ();
    float loLimit = -3.14159f;
    float hiLimit = 3.14159f;
    std::string name;
};

// Inertial properties
struct MJCFInertial {
    float mass = 1.f;
    Eigen::Vector3f com =
        Eigen::Vector3f::Zero(); // center of mass in body frame
    Eigen::Quaternionf quat =
        Eigen::Quaternionf::Identity(); // inertia frame orientation
    Eigen::Vector3f diagInertia = Eigen::Vector3f(1e-4f, 1e-4f, 1e-4f);
};

// body index (SkeletonTree) -> inertial properties
using InertialMap = std::unordered_map<int, MJCFInertial>;

// Collision geometry

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

struct MJCFData {
    std::shared_ptr<const SkeletonTree> skeletonTree;
    std::vector<MJCFMeshInfo> meshInfos;
    std::string meshDir;
    std::vector<Joint> joints;
    CollisionGeomMap collisionGeoms;
    InertialMap inertials;
};

class MJCFLoader {
  public:
    // Full single-pass parse: returns all data needed for visual + physics.
    static MJCFData load(const std::string& mjcfPath, float scale = 1.0f,
                         const std::string& order = "DFS");

  private:
    std::shared_ptr<const SkeletonTree> _skeletonTree;
    std::vector<MJCFMeshInfo> _meshInfos;
    std::string _meshDir;
    std::vector<Joint> _joints;
    CollisionGeomMap _collisionGeoms;
    InertialMap _inertials;

    void parse(const std::string& mjcfPath, float scale,
               const std::string& order);
};

} // namespace Animation
} // namespace KE

#endif
