///
/// SkeletonBridge — connects SkeletonFK pose computation to scene Prim visuals.
///

#ifndef _SKELETON_BRIDGE_HPP_
#define _SKELETON_BRIDGE_HPP_

#include "animation/skeleton_fk.hpp"
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
    Scene::Prim* bodyPrim(int idx) const { return _bodyPrims[idx]; }
    int numBodies() const { return _fk.numBodies(); }

  private:
    friend class SkeletonBridgeAsset;

    Animation::SkeletonFK _fk;
    std::vector<Scene::Prim*> _bodyPrims; // non-owning, scene owns
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
                          const std::string& meshAssetBasePath) const;

    SkeletonBridge instantiate(Scene::SceneBackend* scene,
                               const std::string& primBasePath = "/robot",
                               const std::string& meshAssetBasePath = "") const;

    int numBodies() const { return static_cast<int>(_bodyMeshes.size()); }

  private:
    Animation::CharacterData _data;
    float _scale = 1.0f;
    std::vector<std::shared_ptr<Scene::MeshData>> _bodyMeshes;
};

} // namespace Bridge
} // namespace KE

#endif
