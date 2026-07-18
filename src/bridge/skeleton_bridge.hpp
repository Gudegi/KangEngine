///
/// SkeletonBridge — adapter from SkeletonFK/SkeletonState to scene Prim xforms.
///
/// FK/IK calculation belongs in animation/. This class is the scene mutation
/// bridge that applies computed skeleton poses to non-owning Prim visuals.
///

#ifndef _SKELETON_BRIDGE_HPP_
#define _SKELETON_BRIDGE_HPP_

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

class SkeletonBridgeAsset;

// Scene mutation adapter for SkeletonFK/SkeletonState.
// FK/IK calculation belongs in animation/; this class applies computed poses
// to non-owning scene Prims.
class SkeletonBridge {
  public:
    SkeletonBridge() {}

    // Load MJCF: builds SkeletonFK + creates one scene Prim per body
    static SkeletonBridge fromMJCF(const std::string& mjcfPath,
                                   Scene::SceneBackend* scene,
                                   const std::string& primBasePath = "/robot",
                                   float scale = 1.0f,
                                   const std::string& order = "DFS",
                                   const std::string& meshAssetBasePath = "");

    static SkeletonBridge fromData(const Animation::CharacterData& data,
                                   Scene::SceneBackend* scene,
                                   const std::string& primBasePath = "/robot",
                                   float scale = 1.0f,
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
    friend class SkeletonBridgeAsset;

    Animation::SkeletonFK _fk;
    std::vector<Scene::Prim*> _bodyPrims;   // non-owning, scene owns
    std::vector<Scene::Prim*> _renderPrims; // actual renderable mesh prims
    std::vector<int> _renderPrimBodyIndices;
};

class SkeletonBridgeAsset {
  public:
    SkeletonBridgeAsset() = default;

    static SkeletonBridgeAsset fromMJCF(const std::string& mjcfPath,
                                        float scale = 1.0f,
                                        const std::string& order = "DFS");

    static SkeletonBridgeAsset fromData(const Animation::CharacterData& data,
                                        float scale = 1.0f);

    void defineMeshAssets(Scene::SceneBackend* scene,
                          const std::string& meshAssetBasePath,
                          bool splitVisualGeoms = false) const;

    SkeletonBridge instantiate(Scene::SceneBackend* scene,
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
    Animation::CharacterData _data;
    float _scale = 1.0f;
    std::string _assetPath;
    // Source visual assets, kept at MJCF geom granularity.  Merged body meshes
    // are derived from this data only for the performance-oriented path.
    std::vector<VisualGeomAsset> _visualGeomAssets;
};

} // namespace Bridge
} // namespace KE

#endif
