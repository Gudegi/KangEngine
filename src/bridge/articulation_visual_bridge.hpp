///
/// ArticulationVisualBridge — articulated rigid-link visual bridge.
///
/// FK/IK calculation belongs in animation/. This class owns the viewer-side
/// mapping from articulation/body poses to non-owning scene Prim visuals.
///

#ifndef _ARTICULATION_VISUAL_BRIDGE_HPP_
#define _ARTICULATION_VISUAL_BRIDGE_HPP_

#include "animation/skeleton_fk.hpp"
#include <Eigen/Geometry>
#include <memory>
#include <string>
#include <vector>

namespace KE {

namespace Scene {
class SceneBackend;
class Prim;
struct MeshData;
} // namespace Scene

namespace Bridge {

class ArticulationVisualBridgeAsset;

// Scene mutation adapter for articulated rigid-link visuals.
// FK/IK calculation belongs in animation/; this class applies computed poses
// or physics articulation poses to non-owning scene Prims.
class ArticulationVisualBridge {
  public:
    ArticulationVisualBridge() {}

    // Load MJCF: builds FK/link visual mapping + creates one scene Prim per
    // body.
    static ArticulationVisualBridge
    fromMJCF(const std::string& mjcfPath, Scene::SceneBackend* scene,
             const std::string& primBasePath = "/robot", float scale = 1.0f,
             const std::string& order = "DFS",
             const std::string& meshAssetBasePath = "");

    static ArticulationVisualBridge
    fromData(const Asset::ArticulationDesc& data, Scene::SceneBackend* scene,
             const std::string& primBasePath = "/robot", float scale = 1.0f,
             const std::string& meshAssetBasePath = "");

    // Sync FK global transforms > update all body Prim xformOp attributes.
    void applyPose();

    // Pose manipulation (delegates to SkeletonFK)
    void setJointRotation(int idx, const Eigen::Quaternionf& q);
    void setRootTranslation(const Eigen::Vector3f& t);
    void resetToZeroPose();

    // Accessors
    Animation::SkeletonFK& fk() { return _fk; }
    const Animation::SkeletonFK& fk() const { return _fk; }
    const std::vector<Scene::Prim*>& bodyPrims() const { return _bodyPrims; }
    const std::vector<Scene::Prim*>& renderPrims() const {
        return _renderPrims;
    }
    const std::vector<int>& renderPrimBodyIndices() const {
        return _renderPrimBodyIndices;
    }
    Scene::Prim* bodyPrim(int idx) const { return _bodyPrims[idx]; }
    int numBodies() const { return _fk.numBodies(); }

  private:
    friend class ArticulationVisualBridgeAsset;

    Animation::SkeletonFK _fk;
    std::vector<Scene::Prim*> _bodyPrims;   // non-owning, scene owns
    std::vector<Scene::Prim*> _renderPrims; // actual renderable mesh prims
    std::vector<int> _renderPrimBodyIndices;
};

class ArticulationVisualBridgeAsset {
  public:
    ArticulationVisualBridgeAsset() = default;

    static ArticulationVisualBridgeAsset
    fromMJCF(const std::string& mjcfPath, float scale = 1.0f,
             const std::string& order = "DFS");

    static ArticulationVisualBridgeAsset
    fromData(const Asset::ArticulationDesc& data, float scale = 1.0f);

    void defineMeshAssets(Scene::SceneBackend* scene,
                          const std::string& meshAssetBasePath,
                          bool splitVisualGeoms = false) const;

    ArticulationVisualBridge
    instantiate(Scene::SceneBackend* scene,
                const std::string& primBasePath = "/robot",
                const std::string& meshAssetBasePath = "",
                bool splitVisualGeoms = false) const;

    int numBodies() const {
        return _data.skeletonTree ? _data.skeletonTree->numJoints() : 0;
    }

  public:
    struct VisualGeomAsset {
        int bodyIndex = -1;
        std::shared_ptr<Scene::MeshData> mesh;
        Eigen::Vector4f color = Eigen::Vector4f(0.15f, 0.15f, 0.15f, 1.0f);
    };

  private:
    Asset::ArticulationDesc _data;
    float _scale = 1.0f;
    std::string _assetPath;
    // Source visual assets, kept at MJCF geom granularity.  Merged body meshes
    // are derived from this data only for the performance-oriented path.
    std::vector<VisualGeomAsset> _visualGeomAssets;
};

} // namespace Bridge
} // namespace KE

#endif
